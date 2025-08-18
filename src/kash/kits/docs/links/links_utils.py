from __future__ import annotations

from urllib.parse import urlparse

from frontmatter_format import from_yaml_string
from ruamel.yaml.error import YAMLError

from kash.config.logger import get_logger
from kash.kits.docs.links.links_model import LinkResults
from kash.model import Item
from kash.utils.common.url import Url
from kash.utils.errors import InvalidInput

log = get_logger(__name__)

# TODO: Move to general data item serialization in items_model.py


def parse_links_results_item(item: Item) -> LinkResults:
    """
    Parse LinkResults from a links data item body.
    Raises InvalidInput if parsing fails or body is missing.
    """
    if not item.body:
        raise InvalidInput(f"Links item must have a body: {item}")

    try:
        data = from_yaml_string(item.body)
        return LinkResults.model_validate(data)
    except (KeyError, TypeError, YAMLError) as e:
        raise InvalidInput(f"Failed to parse links data: {e}")
    except ValueError as e:
        raise InvalidInput(f"Invalid links data format: {e}")


def bucket_for_url(url: str | Url) -> str:
    """
    Extract hostname from URL for rate limiting buckets.
    """
    parsed = urlparse(str(url))
    return parsed.hostname or "unknown"
