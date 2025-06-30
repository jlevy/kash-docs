from __future__ import annotations

import asyncio
from typing import Any

from strif import abbrev_str

from kash.config.logger import get_logger
from kash.exec.fetch_url_items import fetch_url_item
from kash.kits.docs.links.links_model import Link, LinkDownloadResult, LinkError
from kash.shell.output.shell_output import multitask_status
from kash.utils.api_utils.gather_limited import FuncTask, Limit, gather_limited_sync
from kash.utils.common.url import Url
from kash.utils.text_handling.markdown_utils import extract_links

log = get_logger(__name__)


def download_single_link(url: str, *, save_content: bool = False, refetch: bool = False) -> Link:
    """
    Download a single URL and extract metadata.
    Raises exceptions on failure instead of handling them silently.
    """
    item = fetch_url_item(Url(url), save_content=save_content, refetch=refetch, cache=True)
    return Link(
        url=url,
        title=item.title,
        description=item.description,
    )


def bucket_for(url: Url) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(str(url))
    return parsed.hostname or "unknown"


OVERALL_LIMIT = Limit(rps=20, concurrency=20)
PER_HOST_LIMIT = Limit(rps=2, concurrency=2)


async def fetch_urls_async(urls: list[Url], show_progress: bool = True) -> LinkDownloadResult:
    """
    Download a list of URLs and return both successful results and errors.
    """
    if not urls:
        log.message("No URLs to download")
        return LinkDownloadResult(links=[], errors=[])

    log.message("Downloading %d links...", len(urls))

    download_tasks = [
        FuncTask(download_single_link, (url,), bucket=bucket_for(url)) for url in urls
    ]

    def labeler(i: int, spec: Any) -> str:
        if isinstance(spec, FuncTask) and len(spec.args) >= 1:
            url = spec.args[0]
            return f"Link {i + 1}/{len(urls)}: {abbrev_str(url, 50)}"
        return f"Link {i + 1}/{len(urls)}"

    log.message("Rate limits: overall %s, per host %s", OVERALL_LIMIT, PER_HOST_LIMIT)
    async with multitask_status() as status:
        task_results = await gather_limited_sync(
            *download_tasks,
            status=status,
            labeler=labeler,
            global_limit=OVERALL_LIMIT,
            bucket_limits={"*": PER_HOST_LIMIT},
        )

    successful_links = []
    errors = []

    for i, result in enumerate(task_results):
        url = urls[i]
        if isinstance(result, Exception):
            errors.append(LinkError(url=url, error_message=str(result)))
            log.warning("Failed to fetch URL %s: %s", url, result)
        else:
            successful_links.append(result)

    return LinkDownloadResult(links=successful_links, errors=errors)


## Tests


def test_fetch_links_with_mock_links():
    """Test the link extraction part without actually downloading URLs."""
    from textwrap import dedent

    markdown_content = dedent("""
        # Test Document
        
        Check out [GitHub](https://github.com) for code repositories.
        
        You can also visit [Python.org](https://python.org) for documentation.
        """).strip()

    links = extract_links(markdown_content, include_internal=False)
    assert len(links) == 2
    assert "https://github.com" in links
    assert "https://python.org" in links


def test_fetch_urls_async_empty_behavior():
    """Test fetch_urls_async handles empty input correctly."""
    result = asyncio.run(fetch_urls_async([]))
    assert len(result.links) == 0
    assert len(result.errors) == 0
    assert not result.has_errors
