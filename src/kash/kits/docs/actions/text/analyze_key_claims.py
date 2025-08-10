from __future__ import annotations

from chopdiff.divs import div

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_simple_text_body
from kash.kits.docs.analysis.analysis_model import CLAIM, CLAIM_MAPPING, KEY_CLAIMS, claim_id
from kash.kits.docs.analysis.claim_mapping import TOP_K_RELATED, extract_mapped_claims
from kash.llm_utils import LLM, LLMName
from kash.model import Format, Item, ItemType, common_params

log = get_logger(__name__)


def format_related_chunks(related_chunks: list[tuple[str, float]], top_k: int | None = None) -> str:
    """
    Format related chunks as a string of HTML links.

    Handles empty chunks and optionally limits to top_k results.
    """
    if not related_chunks:
        return "No related chunks found."

    chunks_to_format = related_chunks[:top_k] if top_k else related_chunks
    chunk_refs = ", ".join(
        f"[{chunk_id}](#{chunk_id}) ({score:.2f})" for chunk_id, score in chunks_to_format
    )
    return f"Related chunks: {chunk_refs}"


@kash_action(
    precondition=has_simple_text_body,
    params=common_params("model"),
)
def analyze_key_claims(item: Item, model: LLMName = LLM.default_standard) -> Item:
    """
    Analyze key claims in the document with related paragraphs found via embeddings.

    Returns an enhanced document with claims and their related context.
    """
    # Perform the claim analysis
    analysis = extract_mapped_claims(item, top_k=TOP_K_RELATED, model=model)

    # Format output with claims and their related chunks
    output_parts = []

    # Add the key claims section
    claim_divs = []
    for i, related in enumerate(analysis.related_chunks_list):
        claim_divs.append(
            div(
                CLAIM,
                related.claim_text,
                # Meta content for debugging etc:
                div(
                    [CLAIM_MAPPING, "debug"],
                    format_related_chunks(related.related_chunks, TOP_K_RELATED),
                ),
                attrs={"id": claim_id(i)},
            )
        )

    claims_content = "\n\n".join(claim_divs)
    summary_div = div(KEY_CLAIMS, claims_content)
    output_parts.append(summary_div)

    # Add the chunked body
    chunked_body = analysis.chunked_doc.reassemble()
    output_parts.append(chunked_body)

    # Add similarity statistics as metadata
    cache_stats = analysis.similarity_cache.cache_stats()
    stats_content = (
        f"Analysis complete: {len(analysis.claims)} claims, "
        f"{len(analysis.chunked_doc.chunks)} chunks, "
        f"{cache_stats['cached_pairs']} similarities computed"
    )
    output_parts.append(div("analysis-stats", stats_content))

    combined_body = "\n\n".join(output_parts)

    combined_item = item.derived_copy(
        type=ItemType.doc,
        format=Format.md_html,
        body=combined_body,
    )

    return combined_item
