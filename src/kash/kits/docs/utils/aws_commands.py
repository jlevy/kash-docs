from kash.exec import kash_command
from kash.kits.docs.utils import aws_utils


@kash_command
def cf_invalidate(*urls: str) -> None:
    """
    Invalidates CloudFront cache for the given URLs or wildcard URLs.
    Finds the relevant CloudFront distribution for each URL.
    """
    aws_utils.invalidate_public_urls(list(urls))
