from __future__ import annotations

from kash.config.logger import get_logger
from kash.kits.docs.actions.text.markdownify_doc import markdownify_doc
from kash.kits.docs.analysis.analysis_model import (
    SourceUrl,
)
from kash.model import Item
from kash.utils.errors import InvalidOutput
from kash.workspaces import current_ws

log = get_logger(__name__)


def get_source_md_item(source_url: SourceUrl) -> Item:
    """
    Get the markdownified content of a source URL as an item.
    """
    ws = current_ws()

    log.message(f"Getting content for: {source_url.url}")
    url_item = ws.import_and_load(source_url.url)
    md_item = markdownify_doc(url_item)

    return md_item


def get_source_text(source_url: SourceUrl) -> str:
    """
    Get the converted markdown text from a source URL.
    """
    md_item = get_source_md_item(source_url)
    if not md_item.body:
        raise InvalidOutput(f"No body found for source URL: {source_url.url}")
    return md_item.body
