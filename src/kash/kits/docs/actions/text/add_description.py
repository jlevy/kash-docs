from chopdiff.html.html_in_md import div_wrapper

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_simple_text_body
from kash.kits.docs.actions.text.describe_briefly import describe_briefly
from kash.model import Format, Item, ItemType

log = get_logger(__name__)

DESCRIPTION = "description"
"""Class name for the description."""

ORIGINAL = "original"
"""Class name for the original content."""


@kash_action(precondition=has_simple_text_body)
def add_description(item: Item) -> Item:
    """
    Add a brief description (from `describe_briefly`) of the content above the full text of
    a document, with each wrapped in a div.
    """
    description_item = describe_briefly(item)

    assert description_item.body and item.body
    combined_body = (
        div_wrapper(class_name=DESCRIPTION)(description_item.body)
        + "\n\n"
        + div_wrapper(class_name=ORIGINAL)(item.body)
    )

    output_item = item.derived_copy(type=ItemType.doc, format=Format.md_html, body=combined_body)

    return output_item
