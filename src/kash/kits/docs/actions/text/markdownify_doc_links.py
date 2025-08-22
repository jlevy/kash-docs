from __future__ import annotations

from frontmatter_format import to_yaml_string
from prettyfmt import fmt_lines
from sidematter_format import Sidematter

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_html_body, has_markdown_body, has_markdown_with_html_body
from kash.kits.docs.actions.text.fetch_links import fetch_links
from kash.kits.docs.actions.text.markdownify_doc import markdownify_doc
from kash.kits.docs.links.links_model import Link
from kash.kits.docs.links.links_preconditions import is_links_data
from kash.kits.docs.links.links_utils import parse_links_results_item
from kash.model import (
    Format,
    Item,
    ItemType,
    TitleTemplate,
)
from kash.utils.common.url import Url
from kash.workspaces import current_ws

log = get_logger(__name__)


@kash_action(
    precondition=has_markdown_body | has_markdown_with_html_body | has_html_body | is_links_data,
    output_format=Format.yaml,
    title_template=TitleTemplate("Links from {title}"),
    live_output=True,
)
def markdownify_doc_links(item: Item) -> Item:
    """
    Extract raw Markdown content of all links in a document.
    Extracts links and then converts the downloaded files to markdown format.
    """

    # If not already links data, call fetch_links to extract and download
    if not is_links_data(item):
        links_item = fetch_links(item)
    else:
        links_item = item

    # Read the links data
    links_data = parse_links_results_item(links_item)

    if not links_data.links:
        log.message("No links found to process")
        return links_item

    log.message("Converting %d links to markdown...", len(links_data.links))

    ws = current_ws()
    markdown_items: list[Item] = []
    error_links: list[Link] = []

    # Process each successfully fetched link.
    for i, link in enumerate(links_data.links):
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

    # Add .md versions of the link content as assets to the original item
    assert item.store_path
    sm = Sidematter(ws.base_dir / item.store_path)
    for i, markdown_item in enumerate(markdown_items):
        assert markdown_item.store_path
        asset_path = sm.add_asset(ws.base_dir / markdown_item.store_path)
        links_data.links[i].content_md_path = str(asset_path.relative_to(sm.assets_dir.parent))

    # Return a new links item.
    new_links_item = links_item.derived_copy(
        format=Format.yaml, body=to_yaml_string(links_data.model_dump())
    )
    return new_links_item
