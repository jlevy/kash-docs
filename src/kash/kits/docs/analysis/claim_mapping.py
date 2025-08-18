from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from kash.config.logger import get_logger
from kash.config.settings import global_settings
from kash.embeddings.embeddings import Embeddings, EmbValue, KeyVal
from kash.kits.docs.analysis.analysis_model import (
    ChunkId,
    ChunkScore,
    Claim,
    MappedClaim,
    chunk_id_str,
    claim_id_str,
)
from kash.kits.docs.analysis.chunk_docs import ChunkedTextDoc
from kash.kits.docs.analysis.claim_extraction import (
    extract_granular_claims_text,
    extract_key_claims_text,
)
from kash.kits.docs.concepts.similarity_cache import SimilarityCache
from kash.utils.api_utils.gather_limited import FuncTask, Limit
from kash.utils.api_utils.multitask_gather import multitask_gather

log = get_logger(__name__)


@dataclass
class MappedClaims:
    """
    Extracted key claims with related paragraphs from the document.

    This structure holds the extracted claims, the chunked document, embeddings
    for both claims and chunks, and mappings of which paragraphs relate to each claim.
    """

    chunked_doc: ChunkedTextDoc
    key_claims: list[MappedClaim]
    granular_claims: list[MappedClaim]
    embeddings: Embeddings | None = None
    similarity_cache: SimilarityCache | None = None

    def format_related_chunks_debug(self, claim_index: int, top_k: int | None = None) -> str:
        """
        Format related chunks for a claim as HTML with clickable links for debug output.

        Args:
            claim_index: Index of the claim
            top_k: Number of top chunks to include (None for all)

        Returns:
            HTML formatted string with chunk links and similarity scores
        """

        related = self.key_claims[claim_index]
        if not related.related_chunks:
            return "No related chunks found"

        chunks_to_format = related.related_chunks[:top_k] if top_k else related.related_chunks

        chunk_links = []
        for cs in chunks_to_format:
            link = f'<a href="#{cs.chunk_id}">{cs.chunk_id}</a>'
            chunk_links.append(f"{link} ({cs.score:.2f})")

        return "Related chunks: " + ", ".join(chunk_links)

    def format_stats(self) -> str:
        """
        Format analysis statistics for debug output.

        Returns:
            Formatted string with analysis statistics
        """
        if self.similarity_cache:
            cache_stats = self.similarity_cache.cache_stats()
            return (
                f"**Analysis complete:** {len(self.key_claims)} claims, "
                f"{len(self.chunked_doc.chunks)} chunks, "
                f"{cache_stats['cached_pairs']} similarities computed"
            )
        else:
            return (
                f"**Analysis complete:** {len(self.key_claims)} claims, "
                f"{len(self.chunked_doc.chunks)} chunks"
            )


TOP_K_RELATED = 8
"""Default number of top related chunks to find for each claim."""


def extract_mapped_claims(
    chunked_doc: ChunkedTextDoc, top_k: int = TOP_K_RELATED, include_granular_claims: bool = True
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
    embed_vals: list[KeyVal] = []

    # Extract key claims
    claims_result = extract_key_claims_text(chunked_doc.doc.reassemble())
    key_claims = claims_result.claims

    # Extract granular claims
    granular_claims_list = extract_granular_claims(chunked_doc)
    granular_claims: list[MappedClaim] = []
    for chunk_id, claim_list in granular_claims_list:
        for claim in claim_list:
            granular_claims.append(
                MappedClaim(
                    claim=claim,
                    related_chunks=[ChunkScore(chunk_id=ChunkId(chunk_id), score=1.0)],
                )
            )

    # Prepare embeddings for mapping key claims to chunks, first adding key claims and then chunks
    for i, claim in enumerate(key_claims):
        claim_id = claim_id_str(i)
        embed_vals.append(
            KeyVal(
                key=claim_id,
                value=EmbValue(emb_text=claim.text, data={"type": "claim", "index": i}),
            )
        )

    for chunk_id, paragraphs in chunked_doc.chunks.items():
        chunk_text = " ".join(para.reassemble() for para in paragraphs)
        embed_vals.append(
            KeyVal(
                key=chunk_id,
                value=EmbValue(
                    emb_text=chunk_text,
                    data={"type": "chunk", "num_paragraphs": len(paragraphs)},
                ),
            )
        )

    # Create embeddings and similarity cache
    log.info("Embedding %d claims and %d chunks", len(key_claims), len(chunked_doc.chunks))
    embeddings = Embeddings.embed(embed_vals)
    similarity_cache = SimilarityCache(embeddings)

    # TODO: Could embed granular claims here too.

    # Find related chunks for each key claim
    chunk_ids: list[str] = list(chunked_doc.chunks.keys())
    key_claims_related: list[MappedClaim] = []
    for i, claim in enumerate(key_claims):
        chunk_id = chunk_id_str(i)
        # Find most similar chunks to this claim
        similar_chunks = similarity_cache.most_similar(
            target_key=chunk_id, n=top_k, candidates=chunk_ids
        )

        key_claims_related.append(
            MappedClaim(
                claim=claim.with_id(chunk_id),
                related_chunks=[
                    ChunkScore(chunk_id=ChunkId(key), score=score) for key, score in similar_chunks
                ],
            )
        )

    return MappedClaims(
        chunked_doc=chunked_doc,
        key_claims=key_claims_related,
        granular_claims=granular_claims,
        embeddings=embeddings,
        similarity_cache=similarity_cache,
    )


def extract_granular_claims(
    chunked_doc: ChunkedTextDoc,
) -> list[tuple[ChunkId, list[Claim]]]:
    """
    Simplified granular claim mapping: extract per-chunk and map each claim to its own chunk.
    """
    return asyncio.run(extract_granular_claims_async(chunked_doc))


async def extract_granular_claims_async(
    chunked_doc: ChunkedTextDoc,
) -> list[tuple[ChunkId, list[Claim]]]:
    """
    Concurrent granular claim mapping: extract per-chunk and map each claim to its own chunk.
    """
    # Prepare (cid, text) pairs to avoid passing Paragraph objects to task workers
    chunk_texts: list[tuple[str, str]] = [
        (cid, " ".join(para.reassemble() for para in paragraphs))
        for cid, paragraphs in chunked_doc.chunks.items()
    ]

    def extract_for_chunk(chunk_id: ChunkId, text: str) -> tuple[ChunkId, list[Claim]]:
        result = extract_granular_claims_text(text, start_index=len(chunked_doc.chunks))
        return chunk_id, result.claims

    tasks = [FuncTask(extract_for_chunk, (cid, text)) for cid, text in chunk_texts]

    def labeler(i: int, spec: Any) -> str:
        if isinstance(spec, FuncTask) and len(spec.args) >= 2:
            cid = spec.args[0]
            if isinstance(cid, str):
                return f"Extract granular claims {i + 1}/{len(tasks)} for {cid}"
        return f"Extract granular claims {i + 1}/{len(tasks)}"

    limit = Limit(rps=global_settings().limit_rps, concurrency=global_settings().limit_concurrency)
    results: list[tuple[ChunkId, list[Claim]]] = await multitask_gather(
        tasks, labeler=labeler, limit=limit
    )

    return results
