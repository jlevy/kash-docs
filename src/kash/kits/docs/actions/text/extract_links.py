from __future__ import annotations

from frontmatter_format import to_yaml_string

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_markdown_body
from kash.kits.docs.links.links_model import Link, LinkResults
from kash.model import Format, Item, ItemType
from kash.utils.errors import InvalidInput
from kash.utils.text_handling.markdown_utils import extract_links as extract_links_from_markdown

log = get_logger(__name__)


@kash_action(precondition=has_markdown_body)
def extract_links(item: Item) -> Item:
    """
    Extract links from markdown content and return a data item with the list of URLs.
    Returns a YAML data item with the extracted links.
    """
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    try:
        urls = extract_links_from_markdown(item.body, include_internal=False)
    except Exception as e:
        raise InvalidInput(f"Failed to parse markdown content: {e}")

    if not urls:
        log.message("No links found in content")

    links = [Link(url=url) for url in urls]

    results = LinkResults(links=links)
    yaml_content = to_yaml_string(results.model_dump())

    return item.derived_copy(type=ItemType.data, format=Format.yaml, body=yaml_content)


## Tests


def test_extract_links_no_links():
    item = Item(
        type=ItemType.doc,
        format=Format.markdown,
        body="This is just plain text with no links at all.",
    )
    result = extract_links(item)
    assert result.type == ItemType.data
    assert result.format == Format.yaml
    assert result.body is not None
    assert "links: []" in result.body


def test_extract_links_with_urls():
    """Test link extraction from markdown content."""
    from textwrap import dedent

    markdown_content = dedent("""
        # Test Document
        
        Check out [GitHub](https://github.com) for code repositories.
        
        You can also visit [Python.org](https://python.org) for documentation.
        """).strip()

    item = Item(
        type=ItemType.doc,
        format=Format.markdown,
        body=markdown_content,
    )

    result = extract_links(item)
    assert result.type == ItemType.data
    assert result.format == Format.yaml
    assert result.body is not None
    assert "https://github.com" in result.body
    assert "https://python.org" in result.body
