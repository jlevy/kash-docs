from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from prettyfmt import fmt_path
from strif import atomic_output_file

from kash.config.logger import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image

log = get_logger(__name__)


@dataclass(frozen=True)
class MarkerResult:
    markdown: str
    images: dict[str, Image]

    def write_images(self, output_dir: Path, make_parents: bool = True) -> None:
        """
        Save all images to the given output directory.
        Creates parent directories if `make_parents` is True.
        """
        if self.images:
            log.message(f"Writing {len(self.images)} images to {fmt_path(output_dir)}")
            for filename, image in self.images.items():
                with atomic_output_file(
                    output_dir / filename,
                    make_parents=make_parents,
                    tmp_suffix=Path(filename).suffix,  # PIL infers format from filename
                ) as f:
                    image.save(f)


def pdf_to_md_marker(pdf_path: Path) -> MarkerResult:
    """
    Convert a PDF file to Markdown using Marker.
    Does not normalize the Markdown.
    """
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(str(pdf_path))
    markdown_text, _, images = text_from_rendered(rendered)

    # Ensure we have a string for markdown content
    if isinstance(markdown_text, dict):
        # If it's a dict, extract the text content (this might need adjustment based on marker's actual output)
        markdown_content = str(markdown_text.get("text", "")) if markdown_text else ""
    else:
        markdown_content = str(markdown_text) if markdown_text else ""

    return MarkerResult(markdown_content, images)
