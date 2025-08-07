from __future__ import annotations

from prettyfmt import fmt_lines

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_html_body, has_markdown_body, has_markdown_with_html_body
from kash.kits.docs.actions.text.fetch_links import fetch_links
from kash.kits.docs.actions.text.markdownify_doc import markdownify_doc
from kash.kits.docs.links.links_model import Link
from kash.kits.docs.links.links_preconditions import is_links_data
from kash.kits.docs.links.links_utils import read_links_from_yaml_item
from kash.model import (
    ActionInput,
    ActionResult,
    Format,
    Item,
    ItemType,
    TitleTemplate,
)
from kash.utils.common.url import Url
from kash.utils.errors import InvalidInput
from kash.workspaces import current_ws

log = get_logger(__name__)


@kash_action(
    precondition=has_markdown_body | has_markdown_with_html_body | has_html_body | is_links_data,
    title_template=TitleTemplate("Links from {title}"),
    live_output=True,
)
def markdownify_doc_links(input: ActionInput) -> ActionResult:
    """
    Extract raw Markdown content of all links in a document.
    Extracts links and then converts the downloaded files to markdown format.
    """
    if not input.items:
        raise InvalidInput("No items provided")

    item = input.items[0]

    # If not already links data, call fetch_links to extract and download
    if not is_links_data(item):
        links_item = fetch_links(item)
    else:
        links_item = item

    # Read the links data
    links_data = read_links_from_yaml_item(links_item)

    if not links_data.links:
        log.message("No links found to process")
        return ActionResult(items=[])

    log.message("Converting %d links to markdown...", len(links_data.links))

    ws = current_ws()
    markdown_items: list[Item] = []
    error_links: list[Link] = []

    # Process each successfully fetched link.
    for i, link in enumerate(links_data.links, 1):
        if not link.status.have_content:
            log.debug("Skipping link with status %s: %s", link.status, link.url)
            continue

        log.message("Converting link %d/%d: %s", i, len(links_data.links), link.url)

        try:
            # Load the HTML resource that was saved by fetch_links
            # Re-import the URL as a resource to get the saved HTML
            store_path = ws.import_item(Url(link.url), as_type=ItemType.resource)
            content_item = ws.load(store_path)

            # Convert HTML to markdown
            markdown_item = markdownify_doc(content_item)
            markdown_items.append(markdown_item)

        except Exception as e:
            log.error("Failed to process link %s: %s", link.url, e)
            error_links.append(link)
            continue

    if markdown_items:
        log.message("Successfully converted %d links to markdown", len(markdown_items))
    else:
        log.warning("No links were successfully converted to markdown")

    if error_links:
        log.warning(
            "Failed to process %d links\n%s",
            len(error_links),
            fmt_lines(url for url in error_links),
        )

    return ActionResult(items=markdown_items)


## Tests


def test_markdownify_doc_links_preconditions():
    """Test that the action accepts various input types."""

    # Test markdown input
    markdown_item = Item(
        type=ItemType.doc,
        format=Format.markdown,
        body="# Test\n[Link](https://example.com)",
    )
    assert has_markdown_body(markdown_item)

    # Test HTML input
    html_item = Item(
        type=ItemType.doc,
        format=Format.html,
        body="<html><body><a href='https://example.com'>Link</a></body></html>",
    )
    assert has_html_body(html_item)

    # Test mixed markdown with HTML
    mixed_item = Item(
        type=ItemType.doc,
        format=Format.md_html,
        body="# Test\n<a href='https://example.com'>Link</a>",
    )
    assert has_markdown_with_html_body(mixed_item)


def test_markdownify_doc_links_empty_links():
    """Test handling of content with no links."""
    item = Item(
        type=ItemType.doc,
        format=Format.markdown,
        body="This is just plain text with no links.",
    )

    # The action should handle empty links gracefully
    # Note: Full integration test would require workspace setup
    _ = ActionInput(items=[item])


def test_markdownify_doc_links_yaml_structure():
    """Test that we can work with YAML links data structure."""
    from textwrap import dedent

    # Create a YAML links data item directly
    yaml_content = dedent("""
        links:
          - url: https://example.com
            title: Example Site
            status: fetched
          - url: https://test.org
            title: Test Org
            status: fetched
        """).strip()

    yaml_item = Item(
        type=ItemType.data,
        format=Format.yaml,
        body=yaml_content,
    )

    # Verify the precondition accepts it
    assert is_links_data(yaml_item)

    # Verify we can read the links
    links_data = read_links_from_yaml_item(yaml_item)
    assert len(links_data.links) == 2
    assert links_data.links[0].url == "https://example.com"
    assert links_data.links[1].url == "https://test.org"
