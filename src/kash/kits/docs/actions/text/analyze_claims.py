from __future__ import annotations

from chopdiff.divs import div
from chopdiff.docs import TextDoc
from prettyfmt import fmt_lines
from sidematter_format import Sidematter

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import (
    has_div_chunks,
    has_html_body,
    has_simple_text_body,
    is_docx_resource,
    is_pdf_resource,
    is_url_resource,
)
from kash.kits.docs.actions.text.fetch_links import fetch_links
from kash.kits.docs.actions.text.markdownify_doc import markdownify_doc
from kash.kits.docs.analysis.analysis_types import ORIGINAL
from kash.kits.docs.analysis.claim_analysis import analyze_mapped_claims
from kash.kits.docs.analysis.claim_mapping import (
    TOP_K_RELATED,
    extract_mapped_claims,
)
from kash.kits.docs.analysis.doc_chunking import ChunkedDoc
from kash.kits.docs.links.links_model import LinkResults
from kash.llm_utils import LLM, LLMName
from kash.model import Format, Item, ItemType, Param, common_param
from kash.model.items_model import from_yaml_string
from kash.utils.errors import InvalidInput
from kash.workspaces.workspaces import current_ws

log = get_logger(__name__)


@kash_action(
    precondition=(
        is_url_resource | is_docx_resource | is_pdf_resource | has_html_body | has_simple_text_body
    )
    & ~has_div_chunks,
    params=(
        common_param("model"),
        Param(
            "granular_only",
            description="Only include granular claims, not key claims.",
            type=bool,
        ),
        Param(
            "key_only",
            description="Only include key claims, not granular claims.",
            type=bool,
        ),
        Param(
            "include_debug",
            description="Include debug info in output as divs with a debug class",
            type=bool,
        ),
    ),
    mcp_tool=True,
)
def analyze_claims(
    item: Item,
    model: LLMName = LLM.default_standard,
    include_debug: bool = False,
    key_only: bool = False,
    granular_only: bool = False,
) -> Item:
    """
    Analyze key claims in the document with related paragraphs found via embeddings.

    Returns an enhanced document with claims and their related context.
    """
    # Convert to markdown if needed.
    as_markdown = markdownify_doc(item)
    if not as_markdown.body:
        raise InvalidInput(f"Item must have a body: {item}")

    # Chunk the doc before mapping claims.
    log.message("Chunking document...")
    text_doc = TextDoc.from_text(as_markdown.body)
    chunked_doc = ChunkedDoc.from_text_doc(text_doc, min_size=1)

    # Fetch links so they are all fetched concurrently as much as possible.
    links_item = fetch_links(item)
    assert links_item.body
    source_links = LinkResults.model_validate(from_yaml_string(links_item.body))

    # Extract and map all claims.
    mapped_claims = extract_mapped_claims(
        chunked_doc,
        top_k=TOP_K_RELATED,
        include_key_claims=not granular_only,
        include_granular_claims=not key_only,
    )

    # Analyze the claims for support stances (using top 5 chunks per claim)
    log.message("Analyzing claims...")
    doc_analysis = analyze_mapped_claims(mapped_claims, source_links=source_links, top_k=5)

    # Format output with claims and their related chunks
    output_parts = []

    summary_div = doc_analysis.format_key_claims_div(include_debug)
    output_parts.append(summary_div)

    # Add the chunked body
    chunked_body = chunked_doc.reassemble()
    output_parts.append(div([ORIGINAL], chunked_body))

    # Add similarity statistics as metadata only if include_debug is True
    if include_debug:
        stats_content = mapped_claims.format_stats()
        output_parts.append(div(["debug"], stats_content))

    combined_body = "\n\n".join(output_parts)

    combined_item = item.derived_copy(
        type=ItemType.doc,
        format=Format.md_html,
        body=combined_body,
    )

    # Get workspace and assign store path
    ws = current_ws()
    result_path = ws.assign_store_path(combined_item)

    # Write sidematter metadata combining item metadata with doc_analysis
    sm = Sidematter(ws.base_dir / result_path)

    # Get the item's metadata
    metadata_dict = combined_item.metadata()

    # Add the doc_analysis data to metadata using Pydantic's model_dump
    analysis_metadata = {"doc_analysis": doc_analysis.model_dump()}

    # Merge the analysis metadata with item metadata
    metadata_dict = metadata_dict | analysis_metadata

    # Write both JSON and YAML sidematter metadata
    sm.write_meta(metadata_dict, formats="all")

    log.message(
        "Wrote sidematter metadata:\n%s",
        fmt_lines(
            [sm.meta_json_path.relative_to(ws.base_dir), sm.meta_yaml_path.relative_to(ws.base_dir)]
        ),
    )

    return combined_item
