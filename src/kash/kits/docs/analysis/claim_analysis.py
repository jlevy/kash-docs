from __future__ import annotations

import asyncio
from typing import Any, Literal, cast

from strif import abbrev_str

from kash.config.logger import get_logger
from kash.config.settings import global_settings
from kash.kits.docs.analysis.analysis_model import (
    ClaimAnalysis,
    ClaimSupport,
    DocAnalysis,
    MappedClaim,
    RigorAnalysis,
    RigorDimension,
    Stance,
)
from kash.kits.docs.analysis.analysis_types import INT_SCORE_INVALID, IntScore
from kash.kits.docs.analysis.claim_mapping import TOP_K_RELATED, MappedClaims
from kash.kits.docs.analysis.doc_chunking import ChunkedDoc
from kash.kits.docs.analysis.rigor_analysis import RIGOR_DIMENSION_OPTIONS, analyze_rigor_dimension
from kash.kits.docs.analysis.support_analysis import analyze_claim_support_original
from kash.utils.api_utils.gather_limited import FuncTask, Limit
from kash.utils.api_utils.multitask_gather import multitask_gather

log = get_logger(__name__)

TaskType = Literal["support", "rigor"]


async def analyze_claims_async(
    chunked_doc: ChunkedDoc,
    claims: list[MappedClaim],
    *,
    include_rigor: bool = False,
    top_k_chunks: int = TOP_K_RELATED,
) -> list[ClaimAnalysis]:
    """
    Analyze all claims concurrently to determine their support stances and rigor scores.

    Args:
        claims: The claims to analyze, mapped to their related chunks
        top_k_chunks: Number of top chunks to analyze per claim
    """
    claims_count = len(claims)
    log.message("Analyzing support and rigor for %d claims", claims_count)

    if not claims:
        log.warning("No claims included. Skipping claim analysis!")
        return []

    # Build tasks with aligned metadata (kind, claim_index, dimension)
    all_tasks: list[FuncTask[IntScore | list[ClaimSupport]]] = []
    task_meta: list[tuple[TaskType, int, RigorDimension | None]] = []

    # Support tasks
    for idx, related in enumerate(claims):
        all_tasks.append(
            FuncTask(
                analyze_claim_support_original,
                (related, chunked_doc, top_k_chunks),
            )
        )
        task_meta.append(("support", idx, None))

    # Rigor tasks
    if include_rigor:
        for dim, include_evidence, evidence_top_k in [
            (RigorDimension.clarity, False, 0),
            (RigorDimension.consistency, True, min(3, top_k_chunks)),
            (RigorDimension.completeness, True, min(3, top_k_chunks)),
            (RigorDimension.depth, True, min(3, top_k_chunks)),
        ]:
            llm_opts = RIGOR_DIMENSION_OPTIONS[dim]
            for idx, related in enumerate(claims):
                all_tasks.append(
                    FuncTask(
                        analyze_rigor_dimension,
                        (
                            related,
                            chunked_doc,
                            llm_opts,
                            dim.value,
                            include_evidence,
                            evidence_top_k,
                        ),
                    )
                )
                task_meta.append(("rigor", idx, dim))

    def analysis_labeler(i: int, spec: Any) -> str:
        kind, idx, dim = task_meta[i]
        claim_text = abbrev_str(claims[idx].claim.text, 30)
        claim_num = idx + 1
        tag = "support" if kind == "support" else (dim.value if dim else "rigor")
        return f"{tag.capitalize()} {claim_num}/{claims_count}: {repr(claim_text)}"

    # Execute all analysis tasks in parallel with rate limiting
    limit = Limit(rps=global_settings().limit_rps, concurrency=global_settings().limit_concurrency)

    gather_result = await multitask_gather(all_tasks, labeler=analysis_labeler, limit=limit)
    if len(gather_result.successes) == 0:
        raise RuntimeError("analyze_claims_async: no successful analysis tasks")

    results = gather_result.successes_or_none

    # Aggregate results per-claim
    support_by_claim: list[list[ClaimSupport]] = [[] for _ in range(claims_count)]
    rigor_scores: dict[RigorDimension, list[IntScore]] = {
        RigorDimension.clarity: [INT_SCORE_INVALID] * claims_count,
        RigorDimension.consistency: [INT_SCORE_INVALID] * claims_count,
        RigorDimension.completeness: [INT_SCORE_INVALID] * claims_count,
        RigorDimension.depth: [INT_SCORE_INVALID] * claims_count,
    }

    for res, (kind, idx, dim) in zip(results, task_meta, strict=False):
        if kind == "support":
            support_by_claim[idx] = cast(list[ClaimSupport], res) if res is not None else []
        else:
            assert dim is not None
            rigor_scores[dim][idx] = cast(IntScore, res) if res is not None else INT_SCORE_INVALID

    # Build ClaimAnalysis objects
    claim_analyses: list[ClaimAnalysis] = []
    for idx, related in enumerate(claims):
        # Get chunk IDs and scores from the related chunks
        relevant_chunks = related.related_chunks[:top_k_chunks]
        chunk_ids = [cs.chunk_id for cs in relevant_chunks]
        chunk_similarity = [cs.similarity for cs in relevant_chunks]

        assert related.claim.id

        rigor_analysis: RigorAnalysis | None = None
        if include_rigor:
            rigor_analysis = RigorAnalysis(
                clarity=rigor_scores[RigorDimension.clarity][idx],
                consistency=rigor_scores[RigorDimension.consistency][idx],
                completeness=rigor_scores[RigorDimension.completeness][idx],
                depth=rigor_scores[RigorDimension.depth][idx],
            )

        claim_analysis = ClaimAnalysis(
            claim=related.claim,
            chunk_ids=chunk_ids,
            chunk_similarity=chunk_similarity,
            source_urls=chunked_doc.get_source_urls(*chunk_ids),
            rigor_analysis=rigor_analysis,
            claim_support=support_by_claim[idx],
            labels=[],  # TODO: Not implemented yet
        )

        claim_analyses.append(claim_analysis)

        # Log summary
        support_counts: dict[Stance, int] = {}
        for cs in support_by_claim[idx]:
            support_counts[cs.stance] = support_counts.get(cs.stance, 0) + 1
        log.message(
            "Claim %s analysis: support: %s, rigor: %s",
            related.claim.id,
            ", ".join(f"{stance}={count}" for stance, count in support_counts.items()),
            rigor_analysis,
        )

    return claim_analyses


def analyze_mapped_claims(mapped_claims: MappedClaims, top_k: int = TOP_K_RELATED) -> DocAnalysis:
    """
    Analyze claims to determine their support stances and rigor scores from related document chunks.

    This function takes the mapped claims (claims with their related document chunks)
    and uses LLMs to analyze the stance each chunk takes toward its related claim,
    as well as evaluating each claim on multiple rigor dimensions (clarity, rigor,
    factuality, and depth).

    Args:
        mapped_claims: The mapped claims with related chunks from the document
        top_k: Number of top related chunks to analyze per claim (default: 8)

    Returns:
        DocAnalysis containing ClaimAnalysis for each claim with support stances and rigor scores
    """
    claim_analyses = asyncio.run(
        analyze_claims_async(
            mapped_claims.chunked_doc,
            mapped_claims.key_claims,
            include_rigor=True,
            top_k_chunks=top_k,
        )
    )

    granular_analyses = asyncio.run(
        analyze_claims_async(
            mapped_claims.chunked_doc,
            mapped_claims.granular_claims,
            include_rigor=False,
            top_k_chunks=top_k,
        )
    )

    return DocAnalysis(key_claims=claim_analyses, granular_claims=granular_analyses)
