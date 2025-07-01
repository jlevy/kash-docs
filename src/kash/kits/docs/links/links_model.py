from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class LinkStatus(StrEnum):
    """
    The status of a link based on a fetch attempt.
    Status should be None for new links.
    """

    fetched = "fetched"
    not_found = "not_found"
    forbidden = "forbidden"
    fetch_error = "fetch_error"
    disabled = "disabled"

    @classmethod
    def from_status_code(cls, status_code: int) -> LinkStatus:
        """Create a LinkStatus from an HTTP status code."""
        if status_code == 200:
            return cls.fetched
        elif status_code == 404:
            return cls.not_found
        elif status_code == 403:
            return cls.forbidden
        else:
            return cls.fetch_error

    @property
    def is_error(self) -> bool:
        """Whether the link should not be reported as a success."""
        return self in (self.not_found, self.forbidden, self.fetch_error)

    @property
    def should_fetch(self) -> bool:
        """Whether the link should be fetched or retried."""
        return self in (self.forbidden, self.fetch_error)


class Link(BaseModel):
    """A single link with metadata."""

    url: str
    title: str | None = None
    description: str | None = None
    summary: str | None = None
    status: LinkStatus | None = None
    status_code: int | None = None


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
