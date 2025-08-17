from __future__ import annotations

import asyncio
from dataclasses import dataclass
from textwrap import dedent
from typing import Any

from strif import abbrev_str

from kash.config.logger import get_logger
from kash.config.settings import global_settings
from kash.kits.docs.analysis.analysis_model import (
    ClaimAnalysis,
    ClaimSupport,
    DocAnalysis,
    RigorAnalysis,
    RigorDimension,
    Stance,
)
from kash.kits.docs.analysis.chunk_docs import ChunkedTextDoc
from kash.kits.docs.analysis.claim_mapping import TOP_K_RELATED, MappedClaims, RelatedChunks
from kash.llm_utils import Message, MessageTemplate, llm_template_completion
from kash.model import LLMOptions
from kash.utils.api_utils.gather_limited import FuncTask, Limit
from kash.utils.api_utils.multitask_gather import multitask_gather

log = get_logger(__name__)


@dataclass
class ClaimAnalysisResults:
    """
    Holds all analysis results for a single claim.
    """

    claim_support: list[ClaimSupport]
    rigor_analysis: RigorAnalysis


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

# LLM options for analyzing clarity
clarity_options = LLMOptions(
    system_message=Message(
        """
        You are an expert editor evaluating the clarity of written claims.
        You assess how clearly and unambiguously ideas are expressed.
        """
    ),
    body_template=MessageTemplate(
        """
        Evaluate the clarity of this claim on a scale of 1 to 5:
        
        {body}
        
        **Scoring Guidelines:**
        - 5: Crystal clear, unambiguous, precisely stated with no room for misinterpretation
        - 4: Clear and well-stated with only minor ambiguities
        - 3: Generally clear but has some vague terms or could be more precise
        - 2: Somewhat unclear, contains ambiguous language or confusing phrasing
        - 1: Very unclear, highly ambiguous, difficult to understand the intended meaning
        
        Consider:
        - Is the claim specific or vague?
        - Are technical terms properly defined or used correctly?
        - Could the claim be misinterpreted?
        - Is the scope and context clear?
        
        Output ONLY a single integer from 1 to 5.
        """
    ),
)

# LLM options for analyzing rigor
rigor_options = LLMOptions(
    system_message=Message(
        """
        You are an expert analyst evaluating the logical rigor and analytical quality of claims.
        You assess the strength of reasoning and evidence presented.
        """
    ),
    body_template=MessageTemplate(
        """
        Evaluate the rigor of this claim and its supporting evidence on a scale of 1 to 5:
        
        {body}
        
        **Scoring Guidelines:**
        - 5: Highly rigorous with strong logical reasoning and robust evidence
        - 4: Good rigor with solid reasoning and adequate evidence
        - 3: Moderate rigor with acceptable reasoning but could use stronger evidence
        - 2: Weak rigor with questionable reasoning or insufficient evidence
        - 1: Very poor rigor with flawed logic or no supporting evidence
        
        Consider:
        - Is the reasoning logically sound?
        - Is evidence provided to support the claim?
        - Are potential counterarguments addressed?
        - Is the methodology (if applicable) appropriate?
        
        Output ONLY a single integer from 1 to 5.
        """
    ),
)

# LLM options for analyzing factuality
factuality_options = LLMOptions(
    system_message=Message(
        """
        You are a fact-checker evaluating the factual accuracy and verifiability of claims.
        You assess whether claims are supported by facts and evidence.
        """
    ),
    body_template=MessageTemplate(
        """
        Evaluate the factuality of this claim on a scale of 1 to 5:
        
        {body}
        
        **Scoring Guidelines:**
        - 5: Highly factual, verifiable with strong evidence from reliable sources
        - 4: Generally factual with good supporting evidence
        - 3: Partially factual or plausible but lacks complete verification
        - 2: Questionable factuality with weak or contradictory evidence
        - 1: Likely false or completely unverifiable
        
        Consider:
        - Can the claim be verified through evidence?
        - Are sources credible and relevant?
        - Does the evidence directly support the claim?
        - Are there known facts that contradict this claim?
        
        Output ONLY a single integer from 1 to 5.
        """
    ),
)

# LLM options for analyzing depth
depth_options = LLMOptions(
    system_message=Message(
        """
        You are an expert analyst evaluating the depth and thoroughness of analysis in claims.
        You assess how comprehensively topics are explored.
        """
    ),
    body_template=MessageTemplate(
        """
        Evaluate the depth of analysis for this claim on a scale of 1 to 5:
        
        {body}
        
        **Scoring Guidelines:**
        - 5: Very deep analysis with comprehensive exploration of nuances and implications
        - 4: Good depth with solid exploration of key aspects
        - 3: Moderate depth covering main points but missing some important aspects
        - 2: Shallow analysis that only scratches the surface
        - 1: Superficial or trivial with no meaningful analysis
        
        Consider:
        - Does the claim explore underlying causes and effects?
        - Are multiple perspectives considered?
        - Is the context and broader implications discussed?
        - Does it go beyond obvious observations?
        
        Output ONLY a single integer from 1 to 5.
        """
    ),
)

RIGOR_DIMENSION_OPTIONS = {
    RigorDimension.clarity: clarity_options,
    RigorDimension.rigor: rigor_options,
    RigorDimension.factuality: factuality_options,
    RigorDimension.depth: depth_options,
}


def analyze_rigor_dimension(
    related: RelatedChunks,
    chunked_doc: ChunkedTextDoc,
    llm_options: LLMOptions,
    dimension_name: str,
    include_evidence: bool = False,
    top_k: int = 3,
) -> int:
    """
    Analyze a single rigor dimension for a claim.

    Args:
        related: The claim and its related chunks
        chunked_doc: The chunked document
        llm_options: LLM configuration for this dimension
        dimension_name: Name of the dimension being analyzed (for logging)
        include_evidence: Whether to include supporting evidence in the prompt
        top_k: Number of top chunks to include as evidence

    Returns:
        Score from 1 to 5
    """
    input_body = f"**Claim:** {related.claim.text}"

    if include_evidence:
        # Include top chunks as context
        relevant_chunks = related.related_chunks[:top_k]
        evidence_text = ""

        for chunk_id, score in relevant_chunks:
            if chunk_id in chunked_doc.chunks:
                chunk_paras = chunked_doc.chunks[chunk_id]
                chunk_text = " ".join(p.reassemble() for p in chunk_paras)
                if len(chunk_text) > 500:
                    chunk_text = chunk_text[:500] + "..."
                evidence_text += f"\n- {chunk_text}\n"

        evidence_label = "Supporting Evidence" if dimension_name == "rigor" else "Related Evidence"
        if dimension_name == "depth":
            evidence_label = "Document Context"

        input_body = dedent(f"""
            **Claim:** {related.claim.text}
            
            **{evidence_label} from Document:**
            {evidence_text if evidence_text else "No evidence found"}
            """)

    llm_response = llm_template_completion(
        model=llm_options.model,
        system_message=llm_options.system_message,
        body_template=llm_options.body_template,
        input=input_body,
    ).content

    try:
        score = int(llm_response.strip())
        if 1 <= score <= 5:
            return score
    except (ValueError, TypeError):
        log.warning("Invalid %s score: %s", dimension_name, llm_response)

    return 3  # Default to mid-range if parsing fails


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
        log.warning("No related chunks found for claim: %s", abbrev_str(related.claim.text, 50))
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
        **The Claim:** {related.claim.text}

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
            related.claim.id,
            chunk_id,
            stance,
            support.support_score,
        )

    return claim_supports


async def analyze_claims_async(
    mapped_claims: MappedClaims, top_k_chunks: int = TOP_K_RELATED
) -> DocAnalysis:
    """
    Analyze all claims concurrently to determine their support stances and rigor scores.

    Args:
        mapped_claims: The mapped claims with related chunks
        top_k_chunks: Number of top chunks to analyze per claim

    Returns:
        DocAnalysis with ClaimAnalysis for each claim
    """
    claims_count = len(mapped_claims.claims)
    log.message("Analyzing support and rigor for %d claims", claims_count)

    # Create support tasks
    support_tasks = [
        FuncTask(
            analyze_claim_support,
            (related, mapped_claims.chunked_doc, top_k_chunks),
        )
        for related in mapped_claims.related_chunks_list
    ]

    # Define rigor dimensions with their configurations
    rigor_dimension_configs = [
        (RigorDimension.clarity, False, 0),
        (RigorDimension.rigor, True, min(3, top_k_chunks)),
        (RigorDimension.factuality, True, min(3, top_k_chunks)),
        (RigorDimension.depth, True, min(3, top_k_chunks)),
    ]

    # Create tasks for each rigor dimension
    rigor_tasks_by_dimension = {}
    for dimension, include_evidence, evidence_top_k in rigor_dimension_configs:
        llm_opts = RIGOR_DIMENSION_OPTIONS[dimension]
        rigor_tasks_by_dimension[dimension] = [
            FuncTask(
                analyze_rigor_dimension,
                (
                    related,
                    mapped_claims.chunked_doc,
                    llm_opts,
                    dimension.value,  # Pass the string value for logging
                    include_evidence,
                    evidence_top_k,
                ),
            )
            for related in mapped_claims.related_chunks_list
        ]

    # Combine all tasks while keeping track of their types for labeling
    all_tasks = []
    task_types: list[tuple[str, RelatedChunks]] = []

    # Keep track of where each task type's results will be in the final results list
    task_result_slices = {}
    current_index = 0

    # Add support tasks
    all_tasks.extend(support_tasks)
    task_types.extend([("support", related) for related in mapped_claims.related_chunks_list])
    task_result_slices["support"] = slice(current_index, current_index + claims_count)
    current_index += claims_count

    # Add rigor dimension tasks
    for dimension in [
        RigorDimension.clarity,
        RigorDimension.rigor,
        RigorDimension.factuality,
        RigorDimension.depth,
    ]:
        all_tasks.extend(rigor_tasks_by_dimension[dimension])
        task_types.extend(
            [(dimension.value, related) for related in mapped_claims.related_chunks_list]
        )
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

    all_results = await multitask_gather(all_tasks, labeler=analysis_labeler, limit=limit)

    # Extract and organize results for each claim
    claim_results_list: list[ClaimAnalysisResults] = []

    for i in range(claims_count):
        rigor_analysis = RigorAnalysis(
            clarity=all_results[task_result_slices[RigorDimension.clarity]][i],
            rigor=all_results[task_result_slices[RigorDimension.rigor]][i],
            factuality=all_results[task_result_slices[RigorDimension.factuality]][i],
            depth=all_results[task_result_slices[RigorDimension.depth]][i],
        )
        claim_results = ClaimAnalysisResults(
            claim_support=all_results[task_result_slices["support"]][i],
            rigor_analysis=rigor_analysis,
        )
        claim_results_list.append(claim_results)

    # Build ClaimAnalysis objects
    claim_analyses = []
    for related, results in zip(
        mapped_claims.related_chunks_list, claim_results_list, strict=False
    ):
        # Get chunk IDs and scores from the related chunks
        relevant_chunks = related.related_chunks[:top_k_chunks]
        chunk_ids = [chunk_id for chunk_id, _ in relevant_chunks]
        chunk_scores = [score for _, score in relevant_chunks]

        assert related.claim.id
        claim_analysis = ClaimAnalysis(
            claim_id=related.claim.id,
            claim=related.claim.text,
            chunk_ids=chunk_ids,
            chunk_scores=chunk_scores,
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

    return DocAnalysis(key_claims=claim_analyses)


def analyze_claims(mapped_claims: MappedClaims, top_k: int = TOP_K_RELATED) -> DocAnalysis:
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
    return asyncio.run(analyze_claims_async(mapped_claims, top_k))
