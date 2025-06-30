from __future__ import annotations

from pydantic import BaseModel


class Link(BaseModel):
    """A single link with metadata."""

    url: str
    title: str | None = None
    description: str | None = None


class LinkError(BaseModel):
    """An error that occurred while downloading a link."""

    url: str
    error_message: str


class LinkResults(BaseModel):
    """
    Collection of successfully downloaded links (for backward compatibility).
    """

    links: list[Link]


class LinkDownloadResult(BaseModel):
    """
    Result of downloading multiple links, including both successes and errors.
    """

    links: list[Link]
    errors: list[LinkError]

    @property
    def has_errors(self) -> bool:
        """Whether any errors occurred during download."""
        return len(self.errors) > 0

    @property
    def total_attempted(self) -> int:
        """Total number of links that were attempted to download."""
        return len(self.links) + len(self.errors)
