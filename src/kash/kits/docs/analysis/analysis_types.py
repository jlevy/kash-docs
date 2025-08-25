from dataclasses import dataclass
from functools import cached_property
from typing import NewType

from kash.utils.common.url import Url
from kash.utils.text_handling.markdown_utils import extract_urls
from kash.web_content.canon_url import canonicalize_url

## ID types

ClaimId = NewType("ClaimId", str)
"""A claim ID, e.g. `claim-123`."""

ChunkId = NewType("ChunkId", str)
"""A chunk ID, e.g. `chunk-123`."""

FootnoteId = NewType("FootnoteId", str)
"""A footnote ID, e.g. `^123`."""

RefId = FootnoteId | ChunkId
"""A chunk id or other referenced id in the document, such as a footnote id."""

IntScore = NewType("IntScore", int)
"""
A score between 1 and 5, with 5 highest. 0 is used for invalid or missing data.
"""

INT_SCORE_INVALID = IntScore(0)


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


## Shared document types


@dataclass(frozen=True)
class Footnote:
    """
    Represents a footnote with its ID and content.
    """

    id: FootnoteId
    """The footnote ID (includes ^ prefix, e.g., "^123", "^foo")"""

    content: str
    """The footnote content/annotation text"""

    @cached_property
    def urls(self) -> tuple[Url, ...]:
        """
        Extract unique URLs from the footnote content.
        """
        return tuple(sorted(set(canonicalize_url(url) for url in extract_urls(self.content))))

    @property
    def primary_url(self) -> Url | None:
        """
        Extract the first URL from the footnote content. Useful for when we know footnotes
        have been structured with a most one URL.
        """
        return self.urls[0] if self.urls else None


@dataclass
class TextSpan:
    """
    Represents a span of text within a string.
    """

    start: int
    """Start position of the span in the text"""

    end: int
    """End position of the span in the text"""

    text: str
    """The actual text content of the span"""

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span: start={self.start}, end={self.end}")


## HTML Conventions

ORIGINAL = "original"
"""Class name for the original document."""

KEY_CLAIMS = "key-claims"
"""Class name for the key claims."""

CLAIM = "claim"
"""Class name for individual claims."""

CLAIM_MAPPING = "claim-mapping"
"""Class name for the mapping of a claim to its related chunks."""

CONCEPTS = "concepts"
"""Class name for the concepts."""

SUMMARY = "summary"
"""Class name for the summary."""

DESCRIPTION = "description"
"""Class name for a description."""
