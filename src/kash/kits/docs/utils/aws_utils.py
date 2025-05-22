import logging
import mimetypes
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cachetools import TTLCache, cached
from prettyfmt import fmt_path

from kash.utils.common.url import Url, is_url, parse_s3_url

log = logging.getLogger(__file__)

log_message = log.warning


@dataclass(frozen=True)
class CloudFrontDistributionInfo:
    id: str
    domain_name: str
    comment: str | None = None
    status: str | None = None


def s3_upload_path(local_path: Path, bucket: str, prefix: str = "") -> list[Url]:
    """
    Uploads a local file or directory contents to an S3 bucket prefix.
    Returns list of S3 URLs for the uploaded files.

    Requires AWS credentials configured via environment variables (e.g.,
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION)
    or other standard boto3 credential methods.
    """
    import boto3

    if not local_path.exists():
        raise ValueError(f"local_path must exist: {local_path}")

    s3_client = boto3.client("s3")
    uploaded_s3_urls: list[Url] = []

    # Normalize prefix.
    prefix = prefix.strip("/")
    s3_base = f"s3://{bucket}"

    def _upload(file_path: Path):
        # For single file uploads directly, use the filename as the base key
        # For directory uploads, use the relative path
        if local_path.is_file():
            relative_path = file_path.name
        else:
            relative_path = file_path.relative_to(local_path)

        # Normalize Windows paths.
        relative_path_str = str(relative_path).replace("\\", "/")
        key = f"{prefix}/{relative_path_str}"
        s3_url = Url(f"{s3_base}/{key}")

        content_type, _ = mimetypes.guess_type(str(file_path))
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        log.info("Uploading: %s -> %s", fmt_path(file_path), s3_url)

        s3_client.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)
        uploaded_s3_urls.append(s3_url)

    if local_path.is_dir():
        log_message("Uploading directory: %s -> %s", fmt_path(local_path), s3_base)
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                _upload(file_path)
    elif local_path.is_file():
        log_message("Uploading file: %s -> %s", fmt_path(local_path), s3_base)
        _upload(local_path)
    else:
        # This case should ideally not be reached due to the exists() check,
        # but handles potential edge cases like broken symlinks.
        raise ValueError(f"local_path is neither a file nor a directory: {local_path}")

    return uploaded_s3_urls


@cached(cache=TTLCache(maxsize=1000, ttl=60))
def r53_records_for_cf(cf_domain_name: str) -> list[str]:
    """
    Searches Route 53 for DNS records (A/AAAA Alias or CNAME)
    pointing to the given CloudFront domain name.

    Requires route53:ListHostedZones and route53:ListResourceRecordSets permissions.
    """
    import boto3
    from botocore.exceptions import ClientError

    r53 = boto3.client("route53")
    found_dns_names: list[str] = []

    # Normalize the target CloudFront domain name (remove trailing dot if present)
    target_cf_domain = cf_domain_name.rstrip(".")

    zones_paginator = r53.get_paginator("list_hosted_zones")
    for zones_page in zones_paginator.paginate():
        for zone in zones_page.get("HostedZones", []):
            zone_id = zone.get("Id")
            if not zone_id:
                continue

            log.debug(f"Scanning zone: {zone.get('Name')} ({zone_id})")
            records_paginator = r53.get_paginator("list_resource_record_sets")
            try:
                for records_page in records_paginator.paginate(HostedZoneId=zone_id):
                    for record in records_page.get("ResourceRecordSets", []):
                        record_name = record.get("Name", "").rstrip(".")
                        record_type = record.get("Type")

                        # Check A/AAAA Alias records
                        if record_type in ["A", "AAAA"]:
                            alias_target = record.get("AliasTarget", {})
                            alias_dns_name = alias_target.get("DNSName", "").rstrip(".")
                            # Check if the alias target matches the CloudFront domain
                            if alias_dns_name.lower() == target_cf_domain.lower():
                                # Check if the alias target is for CloudFront
                                # HostedZoneId for CloudFront distributions is always Z2FDTNDATAQYW2
                                if alias_target.get("HostedZoneId") == "Z2FDTNDATAQYW2":
                                    log.info(
                                        f"Found Alias record: {record_name} -> {alias_dns_name}"
                                    )
                                    found_dns_names.append(record_name)

                        # Check CNAME records
                        elif record_type == "CNAME":
                            resource_records = record.get("ResourceRecords", [])
                            if resource_records:
                                cname_value = resource_records[0].get("Value", "").rstrip(".")
                                # Check if the CNAME value matches the CloudFront domain
                                if cname_value.lower() == target_cf_domain.lower():
                                    log.info(f"Found CNAME record: {record_name} -> {cname_value}")
                                    found_dns_names.append(record_name)

            except ClientError as e:
                # Handle potential access denied errors for specific zones if needed
                log.error(f"Could not list records for zone {zone_id}: {e}")
                continue  # Continue to the next zone

    # Return unique names
    return sorted(list(set(found_dns_names)))


@cached(cache=TTLCache(maxsize=1000, ttl=60))
def cf_r53_for_bucket(bucket_name: str) -> list[str]:
    """
    Finds custom DNS names (via Route53 Alias/CNAME records) pointing to
    CloudFront distributions that use the specified S3 bucket as an origin.
    Returns a sorted list of unique DNS names found.
    """
    log.info(f"Searching for DNS names associated with bucket: {bucket_name}")
    cf_distributions = cf_distros_for_bucket(bucket_name)
    all_dns_names: set[str] = set()

    if not cf_distributions:
        log.warning(f"No CloudFront distributions found for bucket: {bucket_name}")
        return []

    for dist_info in cf_distributions:
        log.info(
            f"Found CloudFront distribution {dist_info.id} ({dist_info.domain_name}), searching DNS..."
        )
        dns_names = r53_records_for_cf(dist_info.domain_name)
        if dns_names:
            log_message(f"Found DNS names for {dist_info.domain_name}: {dns_names}")
            all_dns_names.update(dns_names)
        else:
            log_message(f"No custom DNS names found for {dist_info.domain_name}")

    if not all_dns_names:
        log.warning(
            f"No custom DNS names found pointing to any CloudFront distributions for bucket: {bucket_name}"
        )

    return sorted(list(all_dns_names))


@cached(cache=TTLCache(maxsize=1000, ttl=60))
def cf_distros_for_bucket(bucket_name: str) -> list[CloudFrontDistributionInfo]:
    """
    Return a list of CloudFront distributions whose origins point at bucket_name.
    Simply uses a regex to match different S3 endpoint formats.
    """
    import boto3

    cf = boto3.client("cloudfront")
    # Regex to match various S3 endpoint formats associated with the bucket
    # Handles:
    # - bucket.s3.amazonaws.com
    # - bucket.s3.region.amazonaws.com
    # - bucket.s3-website.region.amazonaws.com
    # - bucket.s3-website-region.amazonaws.com
    # Use re.escape for bucket_name just in case it contains special characters.
    pattern = re.compile(
        rf"^{re.escape(bucket_name)}\.s3(\.amazonaws\.com|[\.-].+\.amazonaws\.com)$"
    )

    log.info(f"Checking for CloudFront origins matching regex: {pattern.pattern}")

    paginator = cf.get_paginator("list_distributions")
    matches: list[CloudFrontDistributionInfo] = []

    for page in paginator.paginate():
        distribution_list = page.get("DistributionList", {})
        if not distribution_list:
            continue
        for dist in distribution_list.get("Items", []):
            if not dist or "Origins" not in dist or "Items" not in dist["Origins"]:
                continue  # Skip malformed distribution data

            for origin in dist["Origins"]["Items"]:
                origin_domain = origin.get("DomainName")
                if origin_domain and pattern.match(origin_domain):
                    matches.append(
                        CloudFrontDistributionInfo(
                            id=dist["Id"],
                            domain_name=dist["DomainName"],
                            comment=dist.get("Comment"),
                            status=dist.get("Status"),
                        )
                    )
    return matches


def cf_distros_for_domain(domain: str) -> list[CloudFrontDistributionInfo]:
    """
    Finds CloudFront distributions associated with a domain name.
    Searches through all CloudFront distributions to find those that have
    the specified domain as an alternate domain name.
    """
    import boto3

    cf = boto3.client("cloudfront")
    matches: list[CloudFrontDistributionInfo] = []

    # Normalize domain for comparison
    normalized_domain = domain.lower().strip()

    log.info(f"Searching for CloudFront distributions with domain: {normalized_domain}")

    paginator = cf.get_paginator("list_distributions")
    for page in paginator.paginate():
        distribution_list = page.get("DistributionList", {})
        if not distribution_list:
            continue

        for dist in distribution_list.get("Items", []):
            # Check if this distribution is configured for our domain
            # First, check the default CloudFront domain
            if dist.get("DomainName", "").lower() == normalized_domain:
                matches.append(
                    CloudFrontDistributionInfo(
                        id=dist["Id"],
                        domain_name=dist["DomainName"],
                        comment=dist.get("Comment"),
                        status=dist.get("Status"),
                    )
                )
                continue

            # Next, check alternate domain names (CNAMEs)
            aliases = dist.get("Aliases", {}).get("Items", [])
            if any(alias.lower() == normalized_domain for alias in aliases):
                matches.append(
                    CloudFrontDistributionInfo(
                        id=dist["Id"],
                        domain_name=dist["DomainName"],
                        comment=dist.get("Comment"),
                        status=dist.get("Status"),
                    )
                )

    if matches:
        log_message("Found CloudFront distributions for domain %s: %s", domain, matches)
    return matches


def cf_s3_to_public_urls(s3_urls: list[Url]) -> dict[Url, Url | None]:
    """
    Maps a list of S3 URLs to their corresponding public HTTPS URLs by
    finding the primary custom domain name associated with each S3 bucket via
    CloudFront and Route53.
    Returns a dictionary mapping each input S3 URL to its corresponding public
    HTTPS URL (e.g., "https://domain.com/key/file.txt"). S3 URLs whose buckets
    do not have a resolvable public DNS name will be omitted.
    """
    if not s3_urls or not all(is_url(url, only_schemes=["s3"]) for url in s3_urls):
        raise ValueError(f"Must have one or more valid S3 URLs: {s3_urls}")

    buckets: set[str] = set()
    s3_details: dict[Url, tuple[str, str]] = {}  # Map S3 URL to (bucket, key)

    # First pass: Parse URLs and collect unique buckets
    for s3_url in s3_urls:
        bucket, key = parse_s3_url(s3_url)  # Let ValueError propagate if invalid
        buckets.add(bucket)
        s3_details[s3_url] = (bucket, key)

    # Find the primary DNS name for each unique bucket
    bucket_to_dns_map: dict[str, str] = {}
    for bucket in buckets:
        dns_names = cf_r53_for_bucket(bucket)
        if dns_names:
            public_domain = dns_names[0]
            bucket_to_dns_map[bucket] = public_domain
        else:
            log.warning(f"No public DNS name found for bucket: {bucket}")

    # Second pass: Construct the public URLs using the map
    s3_to_public_map: dict[Url, Url | None] = {}
    for s3_url, (bucket, key) in s3_details.items():
        public_domain = bucket_to_dns_map.get(bucket)
        if public_domain:
            public_url = Url(f"https://{public_domain}/{key}")
            s3_to_public_map[s3_url] = public_url
        else:
            s3_to_public_map[s3_url] = None

    return s3_to_public_map


def cf_invalidate_paths(distribution_id: str, paths: list[str]) -> str:
    """
    Creates a CloudFront invalidation request for the specified paths.
    Accepts a list of paths to invalidate (e.g., ['/images/*', '/index.html']).
    Paths must start with '/'. CloudFront supports the '*' wildcard
    character, but it must be the *last* character in the path.
    Returns the ID of the created invalidation request.
    Raises ValueError or boto3 exceptions on failure.
    """
    import boto3

    if not distribution_id:
        raise ValueError("Distribution ID cannot be empty.")
    if not paths:
        raise ValueError("Paths list cannot be empty for invalidation.")

    # Ensure paths start with '/' as required by the API
    formatted_paths = []
    for path in paths:
        if not path:  # Skip empty strings if any
            continue
        formatted_path = path if path.startswith("/") else f"/{path}"
        # Validate wildcard usage.
        if "*" in formatted_path[:-1]:
            raise ValueError(f"Wildcard '*' must be the last character in the path: {path}")
        formatted_paths.append(formatted_path)

    if not formatted_paths:
        raise ValueError("No valid paths provided after formatting.")

    cf_client = boto3.client("cloudfront")
    # Using a timestamp for the caller reference ensures idempotency for
    # identical requests made close together.
    caller_reference = f"invalidation-{distribution_id}-{int(time.time())}"

    log_message(
        "Requesting invalidation for %d paths in distribution %s (Ref: %s): %s",
        len(formatted_paths),
        distribution_id,
        caller_reference,
        formatted_paths,  # Log the actual paths being invalidated
    )

    # Let ClientError or other exceptions from boto3 propagate naturally
    response = cf_client.create_invalidation(
        DistributionId=distribution_id,
        InvalidationBatch={
            "Paths": {"Quantity": len(formatted_paths), "Items": formatted_paths},
            "CallerReference": caller_reference,
        },
    )

    invalidation_info = response.get("Invalidation", {})
    invalidation_id = invalidation_info.get("Id")
    invalidation_status = invalidation_info.get("Status")

    if not invalidation_id:
        # This case indicates an unexpected API response format if no exception was raised.
        raise RuntimeError(
            f"CloudFront API call succeeded but did not return an Invalidation ID for distribution {distribution_id}. Response: {response}"
        )

    log.info(
        "Created invalidation request %s for distribution %s. Status: %s",
        invalidation_id,
        distribution_id,
        invalidation_status,
    )
    return invalidation_id


def cf_invalidate_s3_urls(s3_urls: list[Url]) -> dict[str, str]:
    """
    Invalidates paths in CloudFront distributions corresponding to given S3 URLs.
    Groups S3 URLs by bucket, finds the first CloudFront distribution
    associated with each bucket, and creates an invalidation request for the
    corresponding paths (S3 keys prefixed with '/').
    Returns dictionary mapping CloudFront distribution IDs to the created
    invalidation IDs.
    """
    if not s3_urls or not all(is_url(url, only_schemes=["s3"]) for url in s3_urls):
        raise ValueError(f"Must have one or more valid S3 URLs: {s3_urls}")

    # Group paths by bucket
    bucket_to_paths: dict[str, list[str]] = defaultdict(list)
    processed_buckets: set[str] = set()  # Keep track of buckets successfully parsed

    for s3_url in s3_urls:
        bucket, key = parse_s3_url(s3_url)  # Validates Url
        processed_buckets.add(bucket)
        bucket_to_paths[bucket].append(f"/{key}")

    invalidation_results: dict[str, str] = {}

    # Process invalidations bucket by bucket
    for bucket, paths in bucket_to_paths.items():
        log.info(f"Processing invalidation for bucket: {bucket}")
        cf_distributions = cf_distros_for_bucket(bucket)

        if not cf_distributions:
            raise ValueError(f"No CloudFront distribution found for bucket: {bucket}")

        if len(cf_distributions) > 1:
            log.warning(
                "Multiple CloudFront distributions found for bucket %s (%s). Invalidating all.",
                bucket,
                [d.id for d in cf_distributions],
            )

        # Process all distributions for this bucket
        for dist_info in cf_distributions:
            distribution_id = dist_info.id

            # Check if we already created an invalidation for this distribution
            if distribution_id in invalidation_results:
                log.warning(
                    "Distribution %s already has an invalidation request (%s) created in this run.",
                    distribution_id,
                    invalidation_results[distribution_id],
                )
                continue

            # Call the underlying invalidation function and raise exceptions on failure.
            invalidation_id = cf_invalidate_paths(distribution_id, paths)
            invalidation_results[distribution_id] = invalidation_id
            log.info(
                f"Invalidation request {invalidation_id} submitted for bucket {bucket} via distribution {distribution_id}"
            )

    return invalidation_results


def cf_invalidate_urls(urls: list[str | Url]) -> dict[str, str]:
    """
    Invalidates CloudFront cache for public URLs (https://domain.com/path).

    Extracts domains from URLs, finds associated CloudFront distributions,
    and creates invalidation requests for the URL paths.
    Supports wildcards in paths as long as '*' is the last character.

    Returns mapping of CloudFront distribution IDs to invalidation IDs.
    """
    from urllib.parse import urlparse

    if not urls:
        raise ValueError("URL list cannot be empty")

    # Convert all URLs to strings if they're Url objects
    str_urls = [str(url) for url in urls]

    # Group paths by domain
    domain_to_paths: dict[str, list[str]] = defaultdict(list)
    for url_str in str_urls:
        parsed = urlparse(url_str)
        if not parsed.netloc or not parsed.scheme:
            raise ValueError(f"Invalid URL format: {url_str}")

        domain = parsed.netloc
        # CloudFront paths must start with '/', handle empty path
        path = parsed.path if parsed.path else "/"
        domain_to_paths[domain].append(path)

    invalidation_results: dict[str, str] = {}

    # Find CloudFront distributions for each domain and create invalidations
    for domain, paths in domain_to_paths.items():
        log.info(f"Finding CloudFront distribution for domain: {domain}")

        # Find the distribution ID for this domain
        distributions = cf_distros_for_domain(domain)

        if not distributions:
            raise ValueError(f"No CloudFront distribution found for domain: {domain}")

        # Use the first distribution if multiple exist
        if len(distributions) > 1:
            log.warning(
                f"Multiple CloudFront distributions found for domain {domain}. "
                f"Using the first one: {distributions[0].id}"
            )

        distribution_id = distributions[0].id

        # Check if we already created an invalidation for this distribution
        if distribution_id in invalidation_results:
            log.info(
                f"Distribution {distribution_id} already has an invalidation request "
                f"({invalidation_results[distribution_id]}) created in this run."
            )
            continue

        # Create the invalidation and store the result
        invalidation_id = cf_invalidate_paths(distribution_id, paths)
        invalidation_results[distribution_id] = invalidation_id
        log_message(
            f"CloudFront invalidation request {invalidation_id} submitted for "
            f"distribution {distribution_id} (domain {domain})"
        )

    return invalidation_results
