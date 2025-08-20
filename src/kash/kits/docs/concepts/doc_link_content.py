from strif import abbrev_str

from kash.config.logger import get_logger
from kash.kits.docs.actions.text.markdownify_doc import markdownify_doc
from kash.utils.common.url import Url
from kash.workspaces import current_ws

log = get_logger(__name__)


def extract_link_contents(*urls: Url, max_body_size: int = 20 * 1024) -> str:
    """
    Get the markdownified content of all the provided links as a single string.

    Uses the current workspace to import and convert all content to markdown
    as resource items.

    This not parallel, but if content has been pre-fetched it will be fast.
    """
    ws = current_ws()
    md_docs: list[str] = []
    for url in urls:
        log.message(f"Getting content for: {url}")
        url_item = ws.import_and_load(url)
        md_item = markdownify_doc(url_item)

        title = f"Contents of: {url}"
        body = abbrev_str(md_item.body or "", max_body_size)

        md_docs.append(f"\n\n{title}\n{body}\n\n")

    return "\n\n\n".join(md_docs)
