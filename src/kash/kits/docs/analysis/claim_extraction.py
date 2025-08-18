from __future__ import annotations

from dataclasses import dataclass

from kash.exec.llm_transforms import llm_transform_str
from kash.kits.docs.analysis.analysis_model import Claim, ClaimType, claim_id_str
from kash.llm_utils import Message, MessageTemplate
from kash.model import LLMOptions
from kash.utils.text_handling.markdown_utils import extract_bullet_points


@dataclass(frozen=True)
class ExtractedClaims:
    """
    Extracted claims from a text block.
    """

    text: str
    claims: list[Claim]


key_claims_options = LLMOptions(
    system_message=Message(
        """
        You are a careful and precise editor.
        You give exactly the results requested without additional commentary.
        """
    ),
    body_template=MessageTemplate(
        """
        You are to carefully summarize the most important key claims in the text below.

        The goal is to summarize key claims. These should be the
        *most important* overall claims or arguments made by the document.

        Each claim should be ONE SENTENCE. Keep them short!

        These should be very clear written with the same level of technical detail as the
        original text, including relevant terms and details.

        The goal should be to have **3 to 10** key claims, organized as bullet points,
        one sentence each.

        If the document makes many claims, pick the most important or key ones an expert
        analyst who understands the material would pick as most relevant.

        Input text:

        {body}

        Key claims:
        """
    ),
)


def extract_key_claims_text(text: str, start_index: int = 0) -> ExtractedClaims:
    """
    Extract high-level key claims from a (typically larger) block of text.
    """
    claims_md = llm_transform_str(key_claims_options, text)
    return ExtractedClaims(
        text=claims_md,
        claims=[
            Claim(text=c, claim_type=ClaimType.key, id=claim_id_str(i + start_index))
            for i, c in enumerate(extract_bullet_points(claims_md))
        ],
    )


granular_claims_llm_options = LLMOptions(
    system_message=Message(
        """
        You are a careful and precise editor.
        You give exactly the results requested without additional commentary.
        """
    ),
    body_template=MessageTemplate(
        """
        You are to carefully extract each specific, granular claim made in the text below.

        The goal is to read the document (one or more paragraphs) and extract each separate,
        individual claim made.

        Each claim should be ONE SENTENCE.
        These should be concise and very clear written with the same level of technical
        detail as the original text.

        If the text is factual, extract the factual statements. If it is opinionated, extract
        the opinions expressed individually.
        
        You should ignore any parts of the text that does not make any claims or assertions.
        If no claims or assertions are made, return "(No results)".

        Input text:

        {body}

        Key claims:
        """
    ),
)


def extract_granular_claims_text(text: str, start_index: int = 0) -> ExtractedClaims:
    """
    Extract granular/specific claims from a (typically small) block of text.
    """
    claims_md = llm_transform_str(granular_claims_llm_options, text)
    return ExtractedClaims(
        text=claims_md,
        claims=[
            Claim(text=c, claim_type=ClaimType.granular, id=claim_id_str(i + start_index))
            for i, c in enumerate(extract_bullet_points(claims_md))
        ],
    )
