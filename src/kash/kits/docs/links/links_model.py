from __future__ import annotations

from enum import StrEnum
from functools import cached_property

from pydantic import BaseModel
from strif import abbrev_str, single_line

from kash.config.logger import get_logger
from kash.model import Item, StorePath
from kash.utils.common.url import Url
from kash.workspaces import current_ws

log = get_logger(__name__)


class FetchStatus(StrEnum):
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
    def from_status_code(cls, status_code: int | None) -> FetchStatus:
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

    status: FetchStatus = FetchStatus.new
    status_code: int | None = None

    content_orig_path: str | None = None
    """Points to the path of the original content of the link."""

    content_md_path: str | None = None
    """Points to the path of the Markdown content of the link."""


class FetchError(BaseModel):
    """
    An error that occurred while downloading a link.
    """

    url: str
    error_message: str


class LinkResults(BaseModel):
    """
    Collection of downloaded links.
    Parsing methods cache link info, so consider this immutable after creation.
    """

    links: list[Link]

    @cached_property
    def link_map(self) -> dict[Url, Link]:
        """Get a map of links by URL."""
        return {Url(link.url): link for link in self.links}

    def get_link(self, url: Url) -> Link | None:
        """Get a link by URL."""
        return self.link_map.get(url)

    def get_source_md_item(self, url: Url) -> Item:
        """
        Get the source Markdown item for a link, from the current workspace.
        show
        """
        ws = current_ws()

        link = self.get_link(url)
        if not link:
            raise ValueError(f"Link not found: {url}")
        if link.status != FetchStatus.fetched:
            raise ValueError(
                f"Link was not fetched successfully: {link.status} ({link.status_code}): {url}"
            )
        if not link.content_md_path:
            raise ValueError(f"Link has no content path: {url}")

        return ws.load(StorePath(link.content_md_path))

    def get_source_text(self, url: Url) -> str:
        """Get the source text for a link."""
        item = self.get_source_md_item(url)
        if not item.format or not item.format.is_markdown and not item.format.is_markdown_with_html:
            raise ValueError(f"Expected markdown for source item: {url}")
        if not item.body:
            raise ValueError(f"Source item has no body: {url}")

        log.message("Got source text for %s: %s", url, single_line(abbrev_str(item.body, 100)))
        return item.body


class LinkDownloadResult(BaseModel):
    """
    Result of downloading multiple links, including both successes and errors.
    """

    links: list[Link]
    """All links, with all status codes."""

    errors: list[FetchError]
    """Additional info about errors."""

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

    def histogram(self) -> dict[FetchStatus, int]:
        """
        Return counts of links grouped by fetch status.
        """
        counts: dict[FetchStatus, int] = {}
        for link in self.links:
            code = link.status
            counts[code] = counts.get(code, 0) + 1
        return counts
