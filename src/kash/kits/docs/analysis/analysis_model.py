from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Stance(StrEnum):
    """
    Stance a given document has with respect to supporting a statement or to a claim.
    """

    direct_refute = "direct_refute"
    partial_refute = "partial_refute"
    partial_support = "partial_support"
    direct_support = "direct_support"
    background = "background"
    mixed = "mixed"
    unrelated = "unrelated"
    invalid = "invalid"


class ClaimSupport(BaseModel):
    """
    A scored stance a reference takes with with respect to a claim.
    This reflects only stated support for a claim within the referenced source.
    It is not a judgment on the truthfulness or quality of the source.

    | Support Score | Stance | Description |
    |-------|---------------|-------------|
    | +2 | direct_support | Clear stance or statement that the claim is true |
    | +1 | partial_support | Stance that partially supports the claim |
    | -1 | partial_refute | Stance that partially contradicts the claim |
    | -2 | direct_refute | Clear stance or statement that the claim is false |
    | 0 | background | Background information that is relevant to the claim but not supporting or refuting it |
    | 0 | mixed | Contains both supporting and refuting evidence or an overview or synthesis of multiple views |
    | 0 | unrelated | Well-formed content that is off-topic or provides no probative content related to the claim |
    | 0 | invalid | Resource seems to be invalid, such as an invalid URL, malformed or unclear, hallucinated by an LLM, or otherwise unusable |
    """

    ref_id: str = Field(
        description="Claim identifier or reference identifier within the document (such as a footnote id in Markdown or span id in HTML)"
    )
    support_score: int = Field(description="Numeric support score (-2 to +2)")
    stance: Stance = Field(description="Type of evidence support")

    @classmethod
    def create(cls, ref_id: str, stance: Stance) -> ClaimSupport:
        """
        Create ClaimSupport with appropriate score for the stance.
        """
        score_mapping = {
            Stance.direct_refute: -2,
            Stance.partial_refute: -1,
            Stance.partial_support: 1,
            Stance.direct_support: 2,
            Stance.background: 0,
            Stance.mixed: 0,
            Stance.unrelated: 0,
            Stance.invalid: 0,
        }
        return cls(ref_id=ref_id, stance=stance, support_score=score_mapping[stance])


class DocAnalysis(BaseModel):
    """
    Structured data holding analysis of a document.
    """

    claim_support: list[ClaimSupport] = Field(
        description="List of claim support evidence from references", default_factory=list
    )
