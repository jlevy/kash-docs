from __future__ import annotations

from textwrap import dedent

from chopdiff.docs import TextDoc
from strif import abbrev_str

from kash.config.logger import get_logger
from kash.kits.docs.analysis.analysis_model import (
    ClaimSupport,
    MappedClaim,
    SourceUrl,
    Stance,
)
from kash.kits.docs.analysis.claim_mapping import TOP_K_RELATED
from kash.kits.docs.analysis.doc_chunking import ChunkedDoc
from kash.kits.docs.links.links_model import LinkResults
from kash.llm_utils import Message, MessageTemplate, llm_template_completion
from kash.model import LLMOptions

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
        
        Be precise and thoughtful. Consider:
        - Does the passage directly address the claim or just mention related topics?
        - Is the evidence definitive or qualified/partial?
        - Does the passage present multiple viewpoints?
        """
    ),
)


multi_passage_prompt = dedent(
    """
        Output your analysis as a simple list, one stance per line, in the format:
        passage_1: stance
        passage_2: stance
        
        For example:
        passage_1: direct_support
        passage_2: background
        passage_3: partial_refute
        
        Output ONLY the stance labels, no additional commentary.
        """
)

single_passage_prompt = dedent(
    """
    Output the analysis ONLY as the SINGLE WORD stance label, followed by
    a ONE-SENTENCE justification.

    Example 1: For the claim "Acme, Inc. had $120M in revenue in 2022" if the
    passage is a news article that confirms this value, then the output should be:

    direct_support: This is a reputable news source that confirms the revenue number of $120M for 2022.

    Example 2: For the claim "Acme, Inc. had $120M in revenue in 2022" if the
    passage is a news article that only says that Acme is a company that makes
    high-end cooking appliances, then the output should be:
    
    background: Acme is a company that makes high-end cooking appliances.

    The label must be one of the valid stance values: `direct_support`, `partial_support`,
    `partial_refute`, `direct_refute`, `background`, `mixed`, `unrelated`, or `invalid`.
    """
)


def analyze_claim_support_original(
    related: MappedClaim,
    chunked_doc: ChunkedDoc,
    top_k_chunks: int = TOP_K_RELATED,
) -> list[ClaimSupport]:
    """
    Analyze a claim and its related chunks from the original document.

    Args:
        related: The claim and its related chunks
        chunked_doc: The chunked document
        top_k_chunks: Number of top chunks to analyze
    """
    # Take only the top K most relevant chunks
    relevant_chunks = related.related_chunks[:top_k_chunks]

    if not relevant_chunks:
        log.warning("No related chunks found for claim: %s", abbrev_str(related.claim.text, 50))
        return []

    # Format passages for the LLM
    passages_text = ""
    for i, cs in enumerate(relevant_chunks, 1):
        # Get the actual chunk text
        if cs.chunk_id in chunked_doc.chunks:
            chunk_paras = chunked_doc.chunks[cs.chunk_id]
            chunk_text = " ".join(p.reassemble() for p in chunk_paras)
            # Truncate very long chunks for the LLM
            if len(chunk_text) > 1000:
                chunk_text = chunk_text[:1000] + "..."
        else:
            chunk_text = "[Chunk not found]"
            log.warning("Chunk %s not found in document", cs.chunk_id)

        passages_text += f"\n**passage_{i}** (similarity: {cs.similarity:.3f}):\n"
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
        body_template=MessageTemplate(
            claim_support_options.body_template.template + "\n\n" + multi_passage_prompt
        ),
        input=input_body,
    ).content

    # Parse the response to extract stances
    claim_supports = []
    lines = llm_response.strip().split("\n")

    for i, cs in enumerate(relevant_chunks, 1):
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
        support = ClaimSupport.create(ref_id=cs.chunk_id, stance=stance, justification=None)
        claim_supports.append(support)

        log.info(
            "Claim %s -> Chunk %s: %s (score: %d)",
            related.claim.id,
            cs.chunk_id,
            stance,
            support.support_score,
        )

    return claim_supports


def get_source_text(source_url: SourceUrl, links_results: LinkResults) -> str:
    """
    Get the converted markdown text from a source URL.
    This assumes we've already fetched all available links.
    """
    md_item = links_results.get_source_md_item(source_url.url)
    if not md_item.body:
        raise ValueError(f"No body found for source URL: {source_url.url}")
    return md_item.body


def analyze_claim_support_source(
    related: MappedClaim, links_results: LinkResults, source_url: SourceUrl
) -> ClaimSupport:
    """
    Analyze a claim against the content at a referenced `source_url`.

    The source content is fetched and converted to markdown text, then the LLM
    returns a single stance label with a brief justification.
    """
    source_text = None

    source_text = get_source_text(source_url, links_results)
    log.message(
        "Analyzing claim support for against source: %s in markdown extracted from: %s",
        TextDoc.from_text(source_text).size_summary(),
        source_url.url,
    )

    # Call LLM to analyze stances
    # Format the input body with the claim and passages
    input_body = dedent(f"""
        **The Claim:** {related.claim.text}

        **Related Source Text:**
        {source_text}
        """)

    llm_response = llm_template_completion(
        model=claim_support_options.model,
        system_message=claim_support_options.system_message,
        body_template=MessageTemplate(
            claim_support_options.body_template.template + "\n\n" + single_passage_prompt
        ),
        input=input_body,
    ).content

    # Parse the response
    result_str = llm_response.strip().strip("\"'`")
    stance = Stance.error  # Default if parsing fails
    justification = None
    try:
        label, justification = result_str.split(":", 1)
        stance = Stance[label]
        justification = justification.strip()
    except (KeyError, ValueError):
        log.error("Invalid stance value: %r", result_str)

    # Create ClaimSupport object
    return ClaimSupport.create(ref_id=source_url.ref_id, stance=stance, justification=justification)
