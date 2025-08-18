from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import NewType

from pydantic import BaseModel, Field

## IDs and HTML Conventions


KEY_CLAIMS = "key-claims"
"""Class name for the key claims."""

CLAIM = "claim"
"""Class name for individual claims."""

CLAIM_MAPPING = "claim-mapping"
"""Class name for the mapping of a claim to its related chunks."""


ClaimId = NewType("ClaimId", str)

ChunkId = NewType("ChunkId", str)


def claim_id_str(index: int) -> ClaimId:
    """
    Generate a consistent claim ID from an index.
    """
    return ClaimId(f"claim-{index}")


def chunk_id_str(index: int) -> ChunkId:
    """
    Get the ID for a chunk (one or more paragraphs).
    """
    return ChunkId(f"chunk-{index}")


def format_chunk_link(chunk_id: ChunkId) -> str:
    """
    Format a chunk ID as a clickable HTML link.
    """
    return f'<a href="#{chunk_id}">{chunk_id}</a>'


def format_chunk_links(chunk_ids: list[ChunkId]) -> str:
    """
    Format a list of chunk IDs as clickable HTML links.
    """
    return ", ".join(format_chunk_link(cid) for cid in chunk_ids)


## Analysis Models and Rubrics


@dataclass(frozen=True)
class ChunkScore:
    """
    Similarity score for a specific chunk.
    """

    chunk_id: ChunkId
    score: float


@dataclass
class MappedClaim:
    """
    A claim along with a mapping to related chunks in the document.
    """

    claim: Claim
    related_chunks: list[ChunkScore]


class ClaimType(Enum):
    """
    Type of claim.
    """

    granular = "granular"
    """A granuar, specific claim."""

    sentence = "sentence"
    """A sentence from the original document."""

    key = "key"
    """An extracted "key" major claim from a document."""


@dataclass(frozen=True)
class Claim:
    """
    A claim or assertion, such as one extracted from a document.
    """

    text: str
    id: str | None = None
    claim_type: ClaimType = ClaimType.granular

    def with_id(self, claim_id: str) -> Claim:
        """
        Create a claim with an id.
        """
        return Claim(text=self.text, id=claim_id, claim_type=self.claim_type)


class Stance(StrEnum):
    """
    Stance a given document has with respect to supporting a statement or claim.
    Stance describes the position taken and does not imply truth or validity.
    """

    direct_refute = "direct_refute"
    partial_refute = "partial_refute"
    partial_support = "partial_support"
    direct_support = "direct_support"
    background = "background"
    mixed = "mixed"
    unrelated = "unrelated"
    invalid = "invalid"
    error = "error"


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
            Stance.error: 0,
        }
        return cls(ref_id=ref_id, stance=stance, support_score=score_mapping[stance])


class RigorDimension(Enum):
    """
    A dimension of rigor.
    """

    clarity = "clarity"
    """Is this clearly written and are statements well-expressed?"""

    consistency = "consistency"
    """Are the statements consistent with each other and are citations consistent with the statements?"""

    completeness = "completeness"
    """Are all the details relevant to the document's claimsincluded, cited, and addressed?"""

    depth = "depth"
    """Are the content and citations deep and comprehensive?"""


class RigorAnalysis(BaseModel):
    """
    Structured analysis of the rigor of the document.
    """

    clarity: int = Field(description="Clarity score (1 to 5)")
    consistency: int = Field(description="Consistency score (1 to 5)")
    completeness: int = Field(description="Completeness score (1 to 5)")
    depth: int = Field(description="Depth score (1 to 5)")


class ClaimLabel(StrEnum):
    """
    Label for a claim.
    """

    insightful = "insightful"
    """Something surprising or non-obvious that also seems likely to be true"""

    weak_support = "weak_support"
    """A claim that has weak supporting evidence"""

    inconsistent = "inconsistent"
    """A claim that appears to be inconsistent with other claims"""

    controversial = "controversial"
    """A claim that is controversial where there is varied evidence or conflictingexpert opinion"""


class ClaimAnalysis(BaseModel):
    """
    Structured analysis of a claim.
    """

    claim: Claim = Field(description="The claim")

    chunk_ids: list[ChunkId] = Field(
        description="List of ids to pieces of text in the document that are relevant"
    )

    chunk_scores: list[float] = Field(
        description="Similarity scores for each chunk in chunk_ids", default_factory=list
    )

    rigor_analysis: RigorAnalysis = Field(description="Rigor analysis of the claim")

    claim_support: list[ClaimSupport] = Field(
        description="List of claim support evidence from the doc or other sources",
        default_factory=list,
    )

    labels: list[ClaimLabel] = Field(
        description="List of labels for the claim", default_factory=list
    )

    def debug_summary(self) -> str:
        """
        Generate a debug summary for this individual claim.

        Returns formatted string with all claim analysis details.
        """
        parts = []

        # Claim text and related chunks
        parts.append(f"**Text:** {self.claim}")

        # Format related chunks with scores if available
        if self.chunk_scores and len(self.chunk_scores) == len(self.chunk_ids):
            chunk_links = []
            for chunk_id, score in zip(self.chunk_ids, self.chunk_scores, strict=False):
                link = format_chunk_link(chunk_id)
                chunk_links.append(f"{link} ({score:.2f})")
            parts.append(f"**Related chunks:** {', '.join(chunk_links)}")
        else:
            # Fallback if scores not available
            parts.append(f"**Related chunks:** {format_chunk_links(self.chunk_ids)}")

        # Support analysis
        if self.claim_support:
            stance_counts = {}
            for cs in self.claim_support:
                stance_counts[cs.stance] = stance_counts.get(cs.stance, 0) + 1

            # Summary of stances
            summary_items = []
            for stance, count in sorted(stance_counts.items(), key=lambda x: x[0].value):
                summary_items.append(f"{stance.value}: {count}")
            parts.append(
                f"**Support analysis ({len(self.claim_support)} chunks):** "
                f"{', '.join(summary_items)}"
            )

            # Detailed support with clickable links
            detail_items = []
            for cs in self.claim_support:
                # TODO: Update this to handle other claims, footnotes, crawled docs, etc.
                link = format_chunk_link(chunk_id=ChunkId(cs.ref_id))
                detail_items.append(f"{link}: {cs.stance.value} ({cs.support_score:+d})")
            parts.append(f"**Detailed support:** {', '.join(detail_items)}")
        else:
            parts.append("**Support analysis:** No support data")

        r = self.rigor_analysis
        parts.append(
            f"**Rigor scores:** clarity={r.clarity}, consistency={r.consistency}, "
            f"completeness={r.completeness}, depth={r.depth}"
        )

        # Labels if any
        if self.labels:
            label_list = ", ".join(label.value for label in self.labels)
            parts.append(f"**Labels:** {label_list}")

        return "\n\n".join(parts)


class DocAnalysis(BaseModel):
    """
    Structured analysis of a document.
    """

    key_claims: list[ClaimAnalysis] = Field(description="Key claims made in a document")

    granular_claims: list[MappedClaim] = Field(description="Granular claims made in a document")

    def debug_summary(self) -> str:
        """
        Generate a full debug summary of the document analysis.

        Assembles debug summaries from all individual claims.
        """
        sections = []

        # Header section
        sections.append("**Document Analysis Debug Summary**")
        sections.append(f"**Total claims analyzed:** {len(self.key_claims)}")

        # Add each claim's debug summary with its ID as header
        for claim_analysis in self.key_claims:
            claim_header = f"**{claim_analysis.claim.id}:**"
            claim_summary = claim_analysis.debug_summary()
            sections.append(f"{claim_header}\n\n{claim_summary}")

        for granular_claim in self.granular_claims:
            claim_header = f"**{granular_claim.claim.id}:**"
            sections.append(
                f"{claim_header} {granular_claim.claim.text} {granular_claim.related_chunks}"
            )

        return "\n\n".join(sections)

    def get_key_claim_debug(self, claim_index: int) -> str:
        """
        Get the debug summary for a specific claim by index.

        Args:
            claim_index: Index of the claim in the key_claims list

        Returns:
            Debug summary string for the claim, or empty string if index is invalid
        """
        if claim_index >= len(self.key_claims):
            return ""

        return self.key_claims[claim_index].debug_summary()


if __name__ == "__main__":
    import json
    import sys

    schema_dict = DocAnalysis.model_json_schema()

    json.dump(schema_dict, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
