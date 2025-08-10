from __future__ import annotations

from dataclasses import dataclass

from chopdiff.docs import TextDoc

from kash.config.logger import get_logger
from kash.embeddings.embeddings import Embeddings, EmbValue, KeyVal
from kash.kits.docs.actions.text.summarize_key_claims import summarize_key_claims
from kash.kits.docs.analysis.analysis_model import claim_id
from kash.kits.docs.analysis.chunk_docs import ChunkedTextDoc, chunk_paragraphs
from kash.kits.docs.concepts.similarity_cache import SimilarityCache
from kash.llm_utils import LLM, LLMName
from kash.model import Item
from kash.utils.text_handling.markdown_utils import extract_bullet_points

log = get_logger(__name__)


@dataclass
class RelatedChunks:
    """
    Chunks related to a claim with similarity scores.
    """

    claim_id: str
    claim_text: str
    related_chunks: list[tuple[str, float]]  # (chunk_id, similarity_score)


@dataclass
class MappedClaims:
    """
    Extracted key claims with related paragraphs from the document.

    This structure holds the extracted claims, the chunked document, embeddings
    for both claims and chunks, and mappings of which paragraphs relate to each claim.
    """

    claims: list[str]
    chunked_doc: ChunkedTextDoc
    embeddings: Embeddings
    similarity_cache: SimilarityCache
    related_chunks_list: list[RelatedChunks]

    def get_claim_with_context(self, claim_index: int, top_k: int = 3) -> str:
        """
        Get a claim with its top related paragraph chunks.
        """
        if claim_index >= len(self.related_chunks_list):
            raise IndexError(f"Claim index {claim_index} out of range")

        related = self.related_chunks_list[claim_index]
        result = f"**Claim:** {related.claim_text}\n\n"
        result += "**Related passages:**\n\n"

        for chunk_id, score in related.related_chunks[:top_k]:
            chunk_paras = self.chunked_doc.chunks[chunk_id]
            chunk_text = " ".join(p.reassemble() for p in chunk_paras)
            # Truncate long chunks for display
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + "..."
            result += f"\n- (similarity: {score:.3f}) {chunk_text}\n"

        return result

    def format_related_chunks_debug(self, claim_index: int, top_k: int | None = None) -> str:
        """
        Format related chunks for a claim as HTML with clickable links for debug output.

        Args:
            claim_index: Index of the claim
            top_k: Number of top chunks to include (None for all)

        Returns:
            HTML formatted string with chunk links and similarity scores
        """
        if claim_index >= len(self.related_chunks_list):
            return "Invalid claim index"

        related = self.related_chunks_list[claim_index]
        if not related.related_chunks:
            return "No related chunks found"

        chunks_to_format = related.related_chunks[:top_k] if top_k else related.related_chunks

        chunk_links = []
        for chunk_id, score in chunks_to_format:
            link = f'<a href="#{chunk_id}">{chunk_id}</a>'
            chunk_links.append(f"{link} ({score:.2f})")

        return "Related chunks: " + ", ".join(chunk_links)

    def format_stats(self) -> str:
        """
        Format analysis statistics for debug output.

        Returns:
            Formatted string with analysis statistics
        """
        cache_stats = self.similarity_cache.cache_stats()
        return (
            f"**Analysis complete:** {len(self.claims)} claims, "
            f"{len(self.chunked_doc.chunks)} chunks, "
            f"{cache_stats['cached_pairs']} similarities computed"
        )


TOP_K_RELATED = 8
"""Default number of top related chunks to find for each claim."""


def extract_mapped_claims(
    item: Item, top_k: int = TOP_K_RELATED, model: LLMName = LLM.default_standard
) -> MappedClaims:
    """
    Extract key claims in a document and find related paragraphs using embeddings.

    Args:
        item: The document to analyze
        top_k_support: Number of top related chunks to find for each claim
        model: LLM model to use for claim extraction

    Returns:
        ClaimRelatedChunks with claims, embeddings, and related paragraph mappings
    """
    # Extract key claims
    summary_item = summarize_key_claims(item, model=model)
    assert summary_item.body
    claims = extract_bullet_points(summary_item.body)

    # Chunk the document
    assert item.body
    doc = TextDoc.from_text(item.body)
    chunked_doc = chunk_paragraphs(doc, min_size=1)

    # Prepare embeddings for claims and chunks
    keyvals: list[KeyVal] = []

    # Add claims
    for i, claim in enumerate(claims):
        cid = claim_id(i)
        keyvals.append(
            KeyVal(
                key=cid,
                value=EmbValue(emb_text=claim, data={"type": "claim", "index": i}),
            )
        )

    # Add chunks
    for cid, paragraphs in chunked_doc.chunks.items():
        chunk_text = " ".join(para.reassemble() for para in paragraphs)
        keyvals.append(
            KeyVal(
                key=cid,
                value=EmbValue(
                    emb_text=chunk_text,
                    data={"type": "chunk", "num_paragraphs": len(paragraphs)},
                ),
            )
        )

    # Create embeddings and similarity cache
    log.info("Embedding %d claims and %d chunks", len(claims), len(chunked_doc.chunks))
    embeddings = Embeddings.embed(keyvals)
    similarity_cache = SimilarityCache(embeddings)

    # Find related chunks for each claim
    chunk_ids = list(chunked_doc.chunks.keys())
    related_chunks_list = []

    for i, claim in enumerate(claims):
        cid = claim_id(i)
        # Find most similar chunks to this claim
        similar_chunks = similarity_cache.most_similar(
            target_key=cid, n=top_k, candidates=chunk_ids
        )

        related_chunks_list.append(
            RelatedChunks(claim_id=cid, claim_text=claim, related_chunks=similar_chunks)
        )

    return MappedClaims(
        claims=claims,
        chunked_doc=chunked_doc,
        embeddings=embeddings,
        similarity_cache=similarity_cache,
        related_chunks_list=related_chunks_list,
    )
