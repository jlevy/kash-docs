from __future__ import annotations

from pathlib import Path

from kash.kits.docs.doc_formats.markitdown_convert import MarkdownResult


def pdf_to_md_docling(pdf_path: Path) -> MarkdownResult:
    """
    Convert a PDF file to Markdown using docling (layout-aware models, tables,
    OCR). Requires the `pdf` extra: `pip install kash-docs[pdf]`.
    Does not normalize the Markdown.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()

    return MarkdownResult(markdown=markdown, raw_html=None, title=None)
