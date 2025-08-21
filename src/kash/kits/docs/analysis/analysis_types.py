from typing import NewType

## ID types

ClaimId = NewType("ClaimId", str)

ChunkId = NewType("ChunkId", str)

RefId = NewType("RefId", str)
"""A chunk id or other referenced id in the document, such as a footnote id."""


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


KEY_CLAIMS = "key-claims"
"""Class name for the key claims."""

CLAIM = "claim"
"""Class name for individual claims."""

CLAIM_MAPPING = "claim-mapping"
"""Class name for the mapping of a claim to its related chunks."""
