from __future__ import annotations

from chopdiff.divs import div

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_simple_text_body
from kash.kits.docs.analysis.analysis_model import CLAIM, CLAIM_MAPPING, KEY_CLAIMS, claim_id
from kash.kits.docs.analysis.claim_analysis import analyze_claims
from kash.kits.docs.analysis.claim_mapping import TOP_K_RELATED, extract_mapped_claims
from kash.llm_utils import LLM, LLMName
from kash.model import Format, Item, ItemType, common_params

log = get_logger(__name__)


@kash_action(
    precondition=has_simple_text_body,
    params=common_params("model"),
)
def analyze_key_claims(item: Item, model: LLMName = LLM.default_standard) -> Item:
    """
    Analyze key claims in the document with related paragraphs found via embeddings.

    Returns an enhanced document with claims and their related context.
    """
    # Perform the claim extraction and mapping
    mapped_claims = extract_mapped_claims(item, top_k=TOP_K_RELATED, model=model)

    # Analyze the claims for support stances (using top 5 chunks per claim)
    doc_analysis = analyze_claims(mapped_claims, top_k=5)

    # Format output with claims and their related chunks
    output_parts = []

    # Add the key claims section with enhanced information
    claim_divs = []
    for i, related in enumerate(mapped_claims.related_chunks_list):
        # Get the full debug summary for this claim
        claim_debug = doc_analysis.get_claim_debug(i)

        claim_divs.append(
            div(
                CLAIM,
                related.claim_text,
                # Meta content for debugging etc:
                div(
                    [CLAIM_MAPPING, "debug"],
                    claim_debug,
                ),
                attrs={"id": claim_id(i)},
            )
        )

    claims_content = "\n\n".join(claim_divs)
    summary_div = div(KEY_CLAIMS, claims_content)
    output_parts.append(summary_div)

    # Add the chunked body
    chunked_body = mapped_claims.chunked_doc.reassemble()
    output_parts.append(chunked_body)

    # Add similarity statistics as metadata
    stats_content = mapped_claims.format_stats()
    output_parts.append(div(["debug"], stats_content))

    combined_body = "\n\n".join(output_parts)

    combined_item = item.derived_copy(
        type=ItemType.doc,
        format=Format.md_html,
        body=combined_body,
    )

    return combined_item
