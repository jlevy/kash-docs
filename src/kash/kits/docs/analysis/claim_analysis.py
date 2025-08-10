from __future__ import annotations

import asyncio
from textwrap import dedent
from typing import Any

from strif import abbrev_str

from kash.config.logger import get_logger
from kash.config.settings import global_settings
from kash.exec.llm_transforms import llm_transform_str
from kash.kits.docs.analysis.analysis_model import (
    ClaimAnalysis,
    ClaimSupport,
    DocAnalysis,
    RigorAnalysis,
    Stance,
)
from kash.kits.docs.analysis.chunk_docs import ChunkedTextDoc
from kash.kits.docs.analysis.claim_mapping import TOP_K_RELATED, MappedClaims, RelatedChunks
from kash.llm_utils import Message, MessageTemplate, llm_template_completion
from kash.model import LLMOptions
from kash.shell.output.shell_output import multitask_status
from kash.utils.api_utils.gather_limited import FuncTask, Limit, gather_limited_sync

log = get_logger(__name__)

# LLM options for analyzing claim support
claim_support_options = LLMOptions(
    system_message=Message(
        """
        You are an expert editor and analyst who gives careful, unbiased assessments of
        statements, evidence, and factuality.
        You provide careful, nuanced assessments and careful checking of facts and logic.
        """
    ),
    body_template=MessageTemplate(
        """
        You are evaluating how a set of text passages relate to a specific claim.
        
        Your task is to determine the stance each passage takes with respect to the claim.
        
        {body}
        
        For each passage, evaluate its stance toward the claim using ONE of these categories:
        
        - **direct_support**: The passage clearly states or strongly implies the claim is true
        - **partial_support**: The passage provides evidence that partially supports the claim
        - **partial_refute**: The passage provides evidence that partially contradicts the claim  
        - **direct_refute**: The passage clearly states or strongly implies the claim is false
        - **background**: Relevant background information but not directly supporting or refuting
        - **mixed**: Contains both supporting and refuting evidence
        - **unrelated**: The passage is not relevant to evaluating the claim
        - **invalid**: The passage appears corrupted, unclear, or unusable
        
        Output your analysis as a simple list, one stance per line, in the format:
        passage_1: stance
        passage_2: stance
        
        For example:
        passage_1: direct_support
        passage_2: background
        passage_3: partial_refute
        
        Be precise and thoughtful. Consider:
        - Does the passage directly address the claim or just mention related topics?
        - Is the evidence definitive or qualified/partial?
        - Does the passage present multiple viewpoints?
        
        Output ONLY the stance labels, no additional commentary.
        """
    ),
)


def analyze_claim_support(
    related: RelatedChunks,
    chunked_doc: ChunkedTextDoc,
    top_k: int = TOP_K_RELATED,
) -> list[ClaimSupport]:
    """
    Analyze a claim and its related chunks.
    """
    # Take only the top K most relevant chunks
    relevant_chunks = related.related_chunks[:top_k]

    if not relevant_chunks:
        log.warning("No related chunks found for claim: %s", abbrev_str(related.claim_text, 50))
        return []

    # Format passages for the LLM
    passages_text = ""
    for i, (chunk_id, score) in enumerate(relevant_chunks, 1):
        # Get the actual chunk text
        if chunk_id in chunked_doc.chunks:
            chunk_paras = chunked_doc.chunks[chunk_id]
            chunk_text = " ".join(p.reassemble() for p in chunk_paras)
            # Truncate very long chunks for the LLM
            if len(chunk_text) > 1000:
                chunk_text = chunk_text[:1000] + "..."
        else:
            chunk_text = "[Chunk not found]"
            log.warning("Chunk %s not found in document", chunk_id)

        passages_text += f"\n**passage_{i}** (similarity: {score:.3f}):\n"
        passages_text += f"{chunk_text}\n"

    # Call LLM to analyze stances
    # Format the input body with the claim and passages
    input_body = dedent(f"""
        **The Claim:** {related.claim_text}

        **Related Passages:**
        {passages_text}
        """)

    llm_response = llm_template_completion(
        model=claim_support_options.model,
        system_message=claim_support_options.system_message,
        body_template=claim_support_options.body_template,
        input=input_body,
    ).content

    # Parse the response to extract stances
    claim_supports = []
    lines = llm_response.strip().split("\n")

    for i, (chunk_id, score) in enumerate(relevant_chunks, 1):
        # Parse stance from response
        stance = Stance.error  # Default if parsing fails

        for line in lines:
            if line.startswith(f"passage_{i}:"):
                stance_value = line.split(":", 1)[1].strip()
                try:
                    stance = Stance[stance_value]
                except (KeyError, ValueError):
                    log.warning("Invalid stance value: %s", stance_value)
                    stance = Stance.error
                break

        # Create ClaimSupport object
        support = ClaimSupport.create(ref_id=chunk_id, stance=stance)
        claim_supports.append(support)

        log.info(
            "Claim %s -> Chunk %s: %s (score: %d)",
            related.claim_id,
            chunk_id,
            stance,
            support.support_score,
        )

    return claim_supports


async def analyze_claims_async(
    mapped_claims: MappedClaims, top_k_chunks: int = TOP_K_RELATED
) -> DocAnalysis:
    """
    Analyze all claims concurrently to determine their support stances.

    Args:
        mapped_claims: The mapped claims with related chunks
        top_k_chunks: Number of top chunks to analyze per claim

    Returns:
        DocAnalysis with ClaimAnalysis for each claim
    """
    claims_count = len(mapped_claims.claims)
    log.message("Analyzing support for %d claims", claims_count)

    # Create tasks for analyzing each claim
    analysis_tasks = [
        FuncTask(
            analyze_claim_support,
            (claim_support_options, related, mapped_claims.chunked_doc, top_k_chunks),
        )
        for related in mapped_claims.related_chunks_list
    ]

    def analysis_labeler(i: int, spec: Any) -> str:
        if isinstance(spec, FuncTask) and len(spec.args) >= 2:
            related = spec.args[1]  # Second arg is the RelatedChunks
            if isinstance(related, RelatedChunks):
                claim_text = abbrev_str(related.claim_text, 40)
                return f"Analyze {i + 1}/{claims_count}: {repr(claim_text)}"
        return f"Analyze claim {i + 1}/{claims_count}"

    # Execute analysis in parallel with rate limiting
    limit = Limit(rps=global_settings().limit_rps, concurrency=global_settings().limit_concurrency)

    async with multitask_status() as status:
        claim_support_results = await gather_limited_sync(
            *analysis_tasks, limit=limit, status=status, labeler=analysis_labeler
        )

    # Build ClaimAnalysis objects
    claim_analyses = []
    for related, claim_supports in zip(
        mapped_claims.related_chunks_list, claim_support_results, strict=False
    ):
        # Get chunk IDs and scores from the related chunks
        relevant_chunks = related.related_chunks[:top_k_chunks]
        chunk_ids = [chunk_id for chunk_id, _ in relevant_chunks]
        chunk_scores = [score for _, score in relevant_chunks]

        # FIXME: stubbed rigor analysis for now
        rigor_analysis = RigorAnalysis(
            clarity=3,  # Stubbed mid-range value
            factuality=3,
            rigor=3,
            depth=3,
        )

        claim_analysis = ClaimAnalysis(
            claim_id=related.claim_id,
            claim=related.claim_text,
            chunk_ids=chunk_ids,
            chunk_scores=chunk_scores,
            rigor_analysis=rigor_analysis,
            claim_support=claim_supports,
            labels=[],  # Empty for now
        )

        claim_analyses.append(claim_analysis)

        # Log summary
        support_counts = {}
        for cs in claim_supports:
            support_counts[cs.stance] = support_counts.get(cs.stance, 0) + 1
        log.info(
            "Claim %s support summary: %s",
            related.claim_id,
            ", ".join(f"{stance}={count}" for stance, count in support_counts.items()),
        )

    return DocAnalysis(key_claims=claim_analyses)


def analyze_claims(mapped_claims: MappedClaims, top_k: int = TOP_K_RELATED) -> DocAnalysis:
    """
    Analyze claims to determine their support stances from related document chunks.

    This function takes the mapped claims (claims with their related document chunks)
    and uses an LLM to analyze the stance each chunk takes toward its related claim.

    Args:
        mapped_claims: The mapped claims with related chunks from the document
        top_k_chunks: Number of top related chunks to analyze per claim (default: 5)

    Returns:
        DocAnalysis containing ClaimAnalysis for each claim with support stances
    """
    return asyncio.run(analyze_claims_async(mapped_claims, top_k))
