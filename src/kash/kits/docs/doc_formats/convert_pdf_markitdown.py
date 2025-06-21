from __future__ import annotations

from pathlib import Path

from kash.kits.docs.doc_formats.markitdown_convert import MarkdownResult


def pdf_to_md_markitdown(pdf_path: Path) -> MarkdownResult:
    """
    Convert a PDF file to Markdown using MarkItDown.
    Does not normalize the Markdown.
    """

    from markitdown import MarkItDown

    mid = MarkItDown(enable_plugins=False)
    result = mid.convert(pdf_path)

    return MarkdownResult(markdown=result.markdown, raw_html=None, title=result.title)
