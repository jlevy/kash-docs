from __future__ import annotations

from textwrap import dedent

from kash.config.logger import get_logger
from kash.kits.docs.analysis.analysis_model import (
    MappedClaim,
    RigorDimension,
)
from kash.kits.docs.analysis.analysis_types import INT_SCORE_INVALID, IntScore
from kash.kits.docs.analysis.doc_chunking import ChunkedDoc
from kash.llm_utils import Message, MessageTemplate, llm_template_completion
from kash.model import LLMOptions

log = get_logger(__name__)


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

# LLM options for analyzing consistency
consistency_options = LLMOptions(
    system_message=Message(
        """
        You are an expert analyst evaluating the internal consistency of claims.
        You assess whether statements and related evidence align without contradiction.
        """
    ),
    body_template=MessageTemplate(
        """
        Evaluate the internal consistency of this claim with respect to itself and the provided evidence on a scale of 1 to 5:
        
        {body}
        
        **Scoring Guidelines:**
        - 5: Fully consistent with no contradictions or tensions
        - 4: Mostly consistent with only minor tensions or qualifications
        - 3: Mixed consistency; some aspects align while others conflict or are unclear
        - 2: Notably inconsistent; multiple statements or evidence elements conflict
        - 1: Highly inconsistent or self-contradictory
        
        Consider:
        - Do statements about the same facts align across passages?
        - Are there contradictions, hedges, or shifts in definitions/criteria?
        - Do qualifiers meaningfully resolve apparent conflicts?
        
        Output ONLY a single integer from 1 to 5.
        """
    ),
)

# LLM options for analyzing completeness
completeness_options = LLMOptions(
    system_message=Message(
        """
        You are an expert analyst evaluating the completeness of claims.
        You assess whether all key aspects, details, and necessary context are addressed.
        """
    ),
    body_template=MessageTemplate(
        """
        Evaluate the completeness of this claim with respect to the provided evidence and expected scope on a scale of 1 to 5:
        
        {body}
        
        **Scoring Guidelines:**
        - 5: Fully complete; covers all essential aspects with sufficient detail and citations
        - 4: Mostly complete; minor gaps but overall adequate coverage
        - 3: Partially complete; covers main points but misses important aspects or specificity
        - 2: Incomplete; significant gaps in reasoning, evidence, or necessary qualifiers
        - 1: Very incomplete; superficial or missing core elements
        
        Consider:
        - Are necessary assumptions, definitions, and caveats present?
        - Are key evidence and counterpoints addressed where relevant?
        - Is the scope appropriate and sufficiently supported?
        
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
    RigorDimension.consistency: consistency_options,
    RigorDimension.completeness: completeness_options,
    RigorDimension.depth: depth_options,
}


def analyze_rigor_dimension(
    related: MappedClaim,
    chunked_doc: ChunkedDoc,
    llm_options: LLMOptions,
    dimension_name: str,
    include_evidence: bool = False,
    top_k_chunks: int = 3,
) -> IntScore:
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
        relevant_chunks = related.related_chunks[:top_k_chunks]
        evidence_text = ""

        for cs in relevant_chunks:
            if cs.chunk_id in chunked_doc.chunks:
                chunk_paras = chunked_doc.chunks[cs.chunk_id]
                chunk_text = " ".join(p.reassemble() for p in chunk_paras)
                if len(chunk_text) > 500:
                    chunk_text = chunk_text[:500] + "..."
                evidence_text += f"\n- {chunk_text}\n"

        evidence_label = "Related Evidence"
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
            return IntScore(score)
    except (ValueError, TypeError):
        log.warning("Invalid %s score: %s", dimension_name, llm_response)

    return INT_SCORE_INVALID  # Fallback score when parsing fails
