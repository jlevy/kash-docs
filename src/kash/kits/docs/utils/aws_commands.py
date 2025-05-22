from kash.exec import kash_command
from kash.kits.docs.utils import aws_utils
from kash.shell.output.shell_formatting import format_name_and_value
from kash.shell.output.shell_output import PrintHooks, cprint, print_h3


@kash_command
def cf_distros_for_bucket(bucket_name: str) -> None:
    """
    Lists CloudFront distributions using the specified S3 bucket as an origin.
    """
    distributions = aws_utils.cf_distros_for_bucket(bucket_name)
    print_h3(f"CloudFront Distributions for {bucket_name}")
    if not distributions:
        cprint("No CloudFront distributions found")
        return

    for dist in distributions:
        cprint(format_name_and_value("ID", dist.id))
        cprint(format_name_and_value("Domain", dist.domain_name))
        if dist.comment:
            cprint(format_name_and_value("Comment", dist.comment))
        if dist.status:
            cprint(format_name_and_value("Status", dist.status))
        PrintHooks.spacer()


@kash_command
def r53_records_for_cf(cf_domain: str) -> None:
    """
    Lists DNS records pointing to the specified CloudFront domain.
    """
    dns_names = aws_utils.r53_records_for_cf(cf_domain)
    print_h3(f"DNS Records for {cf_domain}")
    if dns_names:
        for name in dns_names:
            cprint("  %s", name)
    else:
        cprint("No DNS records found")


@kash_command
def cf_invalidate(*urls: str) -> None:
    """
    Invalidates CloudFront cache for the given URLs or wildcard URLs.
    Finds the relevant CloudFront distribution for each URL.
    """
    aws_utils.cf_invalidate_urls(list(urls))


@kash_command
def cf_invalidate_paths(distribution_id: str, *paths: str) -> None:
    """
    Creates a CloudFront invalidation for specific paths in a distribution.
    Paths must start with '/' and wildcards (*) must be the last character.
    """
    path_list = list(paths) or ["/*"]
    invalidation_id = aws_utils.cf_invalidate_paths(distribution_id, path_list)
    print_h3("CloudFront Invalidation")
    cprint("Created invalidation %s for distribution %s", invalidation_id, distribution_id)
