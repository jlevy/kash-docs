from __future__ import annotations

from chopdiff.docs import TextDoc

from kash.exec import kash_action
from kash.exec.preconditions import (
    has_div_chunks,
    has_markdown_body,
    has_markdown_with_html_body,
)
from kash.kits.docs.analysis.chunk_docs import chunk_doc_paragraphs
from kash.model import Format, Item, ItemType, Param
from kash.utils.errors import InvalidInput


@kash_action(
    precondition=(has_markdown_body | has_markdown_with_html_body) & ~has_div_chunks,
    params=(
        Param(
            "min_size",
            description="Minimum number of paragraphs per chunk",
            type=int,
            default_value=1,
        ),
    ),
)
def chunk_paragraphs(item: Item, min_size: int = 1) -> Item:
    """
    Chunk a document's paragraphs and wrap them in divs with stable chunk IDs.
    Produces output with "chunk" divs.
    """
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    doc = TextDoc.from_text(item.body)
    chunked_doc = chunk_doc_paragraphs(doc, min_size=min_size)
    final_body = chunked_doc.reassemble()

    return item.derived_copy(type=ItemType.doc, format=Format.md_html, body=final_body)
