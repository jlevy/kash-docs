from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

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
)
from kash.kits.docs.analysis.analysis_types import INT_SCORE_INVALID, IntScore
from kash.kits.docs.analysis.claim_mapping import TOP_K_RELATED, MappedClaims
from kash.kits.docs.analysis.doc_chunking import ChunkedDoc
from kash.kits.docs.analysis.rigor_analysis import RIGOR_DIMENSION_OPTIONS, analyze_rigor_dimension
from kash.kits.docs.analysis.support_analysis import analyze_claim_support_original
from kash.utils.api_utils.gather_limited import FuncTask, Limit
from kash.utils.api_utils.multitask_gather import multitask_gather

log = get_logger(__name__)


@dataclass
class ClaimAnalysisResults:
    """
    Holds all analysis results for a single claim.
    """

    claim_support: list[ClaimSupport]
    rigor_analysis: RigorAnalysis | None


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

    # Combine all tasks while keeping track of their types for labeling
    all_tasks = []
    task_types: list[tuple[str, MappedClaim]] = []

    # Keep track of where each task type's results will be in the final results list
    task_result_slices: dict[RigorDimension | Literal["support"], slice] = {}
    current_index = 0

    # Add support tasks
    support_tasks = [
        FuncTask(
            analyze_claim_support_original,
            (related, chunked_doc, top_k_chunks),
        )
        for related in claims
    ]
    all_tasks.extend(support_tasks)
    task_types.extend([("support", related) for related in claims])
    task_result_slices["support"] = slice(current_index, current_index + claims_count)
    current_index += claims_count

    if include_rigor:
        # Create tasks for each rigor dimension
        # Define rigor dimensions with their configurations
        rigor_tasks_by_dimension: dict[RigorDimension, list[FuncTask]] = {}
        for dimension, include_evidence, evidence_top_k in [
            (RigorDimension.clarity, False, 0),
            (RigorDimension.consistency, True, min(3, top_k_chunks)),
            (RigorDimension.completeness, True, min(3, top_k_chunks)),
            (RigorDimension.depth, True, min(3, top_k_chunks)),
        ]:
            llm_opts = RIGOR_DIMENSION_OPTIONS[dimension]  # noqa: F821
            rigor_tasks_by_dimension[dimension] = [
                FuncTask(
                    analyze_rigor_dimension,
                    (
                        related,
                        chunked_doc,
                        llm_opts,
                        dimension.value,  # Pass the string value for logging
                        include_evidence,
                        evidence_top_k,
                    ),
                )
                for related in claims
            ]

        for dimension in [
            RigorDimension.clarity,
            RigorDimension.consistency,
            RigorDimension.completeness,
            RigorDimension.depth,
        ]:
            all_tasks.extend(rigor_tasks_by_dimension[dimension])
            task_types.extend([(dimension.value, related) for related in claims])
            task_result_slices[dimension] = slice(current_index, current_index + claims_count)
            current_index += claims_count

    def analysis_labeler(i: int, spec: Any) -> str:
        if i < len(task_types):
            task_type, related = task_types[i]
            claim_text = abbrev_str(related.claim.text, 30)
            assert related.claim.id
            claim_num = int(related.claim.id.split("-")[1]) + 1
            return f"{task_type.capitalize()} {claim_num}/{claims_count}: {repr(claim_text)}"
        return f"Analyze task {i + 1}/{len(all_tasks)}"

    # Execute all analysis tasks in parallel with rate limiting
    limit = Limit(rps=global_settings().limit_rps, concurrency=global_settings().limit_concurrency)

    gather_result = await multitask_gather(all_tasks, labeler=analysis_labeler, limit=limit)
    if len(gather_result.successes) == 0:
        raise RuntimeError("analyze_key_claims_async: no successful analysis tasks")

    all_results = gather_result.successes_or_none

    # Extract and organize results for each claim
    claim_results_list: list[ClaimAnalysisResults] = []

    def score_for(dimension: RigorDimension) -> IntScore:
        return all_results[task_result_slices[dimension]][i] or INT_SCORE_INVALID

    for i in range(claims_count):
        rigor_analysis = None
        if include_rigor:
            rigor_analysis = RigorAnalysis(
                clarity=score_for(RigorDimension.clarity),
                consistency=score_for(RigorDimension.consistency),
                completeness=score_for(RigorDimension.completeness),
                depth=score_for(RigorDimension.depth),
            )
        claim_results = ClaimAnalysisResults(
            claim_support=all_results[task_result_slices["support"]][i] or [],
            rigor_analysis=rigor_analysis,
        )
        claim_results_list.append(claim_results)

    # Build ClaimAnalysis objects
    claim_analyses: list[ClaimAnalysis] = []
    for related, results in zip(claims, claim_results_list, strict=False):
        # Get chunk IDs and scores from the related chunks
        relevant_chunks = related.related_chunks[:top_k_chunks]
        chunk_ids = [cs.chunk_id for cs in relevant_chunks]
        chunk_similarity = [cs.similarity for cs in relevant_chunks]

        assert related.claim.id

        claim_analysis = ClaimAnalysis(
            claim=related.claim,
            chunk_ids=chunk_ids,
            chunk_similarity=chunk_similarity,
            source_urls=chunked_doc.get_source_urls(*chunk_ids),
            rigor_analysis=results.rigor_analysis,
            claim_support=results.claim_support,
            labels=[],  # Empty for now
        )

        claim_analyses.append(claim_analysis)

        # Log summary
        support_counts = {}
        for cs in results.claim_support:
            support_counts[cs.stance] = support_counts.get(cs.stance, 0) + 1

        log.info(
            "Claim %s analysis: support: %s, rigor: %s",
            related.claim.id,
            ", ".join(f"{stance}={count}" for stance, count in support_counts.items()),
            results.rigor_analysis,
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
