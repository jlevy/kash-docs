from __future__ import annotations

from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def pdf_to_md(pdf_path: Path) -> str:
    """
    Convert a PDF file to Markdown using Marker.
    Does not normalize the Markdown.
    """
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(str(pdf_path))
    markdown_text, _, _ = text_from_rendered(rendered)

    # Ensure we have a string for markdown content
    if isinstance(markdown_text, dict):
        # If it's a dict, extract the text content (this might need adjustment based on marker's actual output)
        markdown_content = str(markdown_text.get("text", "")) if markdown_text else ""
    else:
        markdown_content = str(markdown_text) if markdown_text else ""

    return markdown_content
