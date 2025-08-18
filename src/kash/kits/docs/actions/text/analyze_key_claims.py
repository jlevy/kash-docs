from __future__ import annotations

from chopdiff.divs import div
from chopdiff.docs import TextDoc
from prettyfmt import fmt_lines
from sidematter_format import Sidematter

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_simple_text_body
from kash.kits.docs.analysis.analysis_model import CLAIM, CLAIM_MAPPING, KEY_CLAIMS, claim_id_str
from kash.kits.docs.analysis.chunk_docs import chunk_doc_paragraphs
from kash.kits.docs.analysis.claim_analysis import analyze_claims
from kash.kits.docs.analysis.claim_mapping import (
    TOP_K_RELATED,
    extract_mapped_claims,
)
from kash.llm_utils import LLM, LLMName
from kash.model import Format, Item, ItemType, Param, common_param
from kash.utils.errors import InvalidInput
from kash.workspaces.workspaces import current_ws

log = get_logger(__name__)


@kash_action(
    precondition=has_simple_text_body,
    params=(
        common_param("model"),
        Param(
            "include_debug",
            description="Include debug info in output as divs with a debug class",
            type=bool,
        ),
    ),
)
def analyze_key_claims(
    item: Item, model: LLMName = LLM.default_standard, include_debug: bool = False
) -> Item:
    """
    Analyze key claims in the document with related paragraphs found via embeddings.

    Returns an enhanced document with claims and their related context.
    """
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    text_doc = TextDoc.from_text(item.body)

    chunked_doc = chunk_doc_paragraphs(text_doc, min_size=1)

    mapped_claims = extract_mapped_claims(chunked_doc, top_k=TOP_K_RELATED)

    # Analyze the claims for support stances (using top 5 chunks per claim)
    doc_analysis = analyze_claims(mapped_claims, top_k=5)

    # Format output with claims and their related chunks
    output_parts = []

    # Add the key claims section with enhanced information
    claim_divs = []
    for i, related in enumerate(mapped_claims.key_claims):
        # Build claim content parts
        claim_content = [related.claim.text]

        # Only add debug info if include_debug is True
        if include_debug:
            # Get the full debug summary for this claim
            claim_debug = doc_analysis.get_key_claim_debug(i)
            claim_content.append(
                div(
                    [CLAIM_MAPPING, "debug"],
                    claim_debug,
                )
            )

        claim_divs.append(
            div(
                CLAIM,
                *claim_content,
                attrs={"id": claim_id_str(i)},
            )
        )

    claims_content = "\n\n".join(claim_divs)
    summary_div = div(KEY_CLAIMS, claims_content)
    output_parts.append(summary_div)

    # Add the chunked body
    chunked_body = mapped_claims.chunked_doc.reassemble()
    output_parts.append(chunked_body)

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
    sm.write_meta(metadata_dict, formats="all", make_parents=True)

    log.message(
        "Wrote sidematter metadata:\n%s",
        fmt_lines(
            [sm.meta_json_path.relative_to(ws.base_dir), sm.meta_yaml_path.relative_to(ws.base_dir)]
        ),
    )

    return combined_item
