from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class LinkStatus(StrEnum):
    """
    The status of a link based on a fetch attempt.
    """

    new = "new"
    fetched = "fetched"
    not_found = "not_found"
    forbidden = "forbidden"
    fetch_error = "fetch_error"
    # These are permanent errors:
    invalid = "invalid"
    disabled = "disabled"

    @classmethod
    def from_status_code(cls, status_code: int | None) -> LinkStatus:
        """
        Create a LinkStatus from an HTTP status code.
        """
        # Sanity check so we don't get confused by redirects.
        if status_code and (status_code >= 300 and status_code < 400):
            raise ValueError("Redirects should already be followed")

        if not status_code:
            return cls.invalid
        elif status_code == 200:
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
        return self in (self.not_found, self.forbidden, self.fetch_error, self.invalid)

    @property
    def should_fetch(self) -> bool:
        """
        Whether the link should be fetched or retried.
        """
        return self in (self.new, self.forbidden, self.fetch_error)

    @property
    def have_content(self) -> bool:
        """Whether we have the content of the link."""
        return self == self.fetched


class Link(BaseModel):
    """
    A single link with metadata and optionally a pointer to a path with extracted content.
    """

    url: str
    title: str | None = None
    description: str | None = None
    summary: str | None = None

    status: LinkStatus = LinkStatus.new
    status_code: int | None = None

    content_md_path: str | None = None
    """Points to the path of the Markdown content of the link."""


class LinkError(BaseModel):
    """
    An error that occurred while downloading a link.
    """

    url: str
    error_message: str


class LinkResults(BaseModel):
    """
    Collection of successfully downloaded links.
    """

    links: list[Link]


class LinkDownloadResult(BaseModel):
    """
    Result of downloading multiple links, including both successes and errors.
    """

    links: list[Link]
    errors: list[LinkError]

    @property
    def total_attempted(self) -> int:
        """Total number of links that were attempted to download."""
        return len(self.links)

    @property
    def total_errors(self) -> int:
        """Total number of links that were successfully downloaded."""
        return len([link for link in self.links if link.status.is_error])

    @property
    def total_successes(self) -> int:
        """Total number of links that were successfully downloaded."""
        return self.total_attempted - self.total_errors
