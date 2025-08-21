from typing import NewType

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
