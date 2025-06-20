from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from markitdown._base_converter import DocumentConverterResult
from markitdown.converters._docx_converter import DocxConverter
from typing_extensions import override

from kash.utils.text_handling.markdownify_utils import (
    MARKDOWNIFY_OPTIONS,
    markdownify_postprocess,
    markdownify_preprocess,
)

if TYPE_CHECKING:
    from markitdown._stream_info import StreamInfo

log = logging.getLogger(__name__)

# Based on markitdown.converters._docx_converter.DocxConverter.

_dependency_exc_info = None


# Accepted types (copied exactly from original DocxConverter)
ACCEPTED_MIME_TYPE_PREFIXES = [
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]
ACCEPTED_FILE_EXTENSIONS = [".docx"]


class CustomDCResult(DocumentConverterResult):
    def __init__(self, *, html: str, md: str, title: str | None):
        super().__init__(markdown=md, title=title)
        self.html: str = html


class CustomDocxConverter(DocxConverter):
    """
    A custom DocxConverter derived from the original, modified to allow passing
    Markdownify options to the underlying Markdownify HtmlConverter. Also exposes
    the raw HTML, which is sometimes useful at least for debugging.

    See options:
    https://github.com/matthewwithanm/python-markdownify
    """

    def __init__(
        self,
        markdownify_options: dict[str, Any] | None = None,
        html_postprocess: Callable[[str], str] | None = None,
        md_postprocess: Callable[[str], str] | None = None,
    ):
        """
        Initializes the converter, storing custom markdownify options.
        """
        super().__init__()  # Call base class init (initializes self._html_converter)
        # Store custom options for markdownify
        self.markdownify_options: dict[str, Any] = (
            markdownify_options if markdownify_options is not None else {}
        )
        self.html_postprocess: Callable[[str], str] | None = html_postprocess
        self.md_postprocess: Callable[[str], str] | None = md_postprocess

    @override
    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options passed from MarkItDown.convert (e.g., llm_client)
    ) -> DocumentConverterResult:
        """
        Converts the docx stream using Mammoth, then converts the resulting
        HTML to Markdown using the internal HtmlConverter, passing along
        any stored markdownify options.
        """
        import mammoth
        from markitdown._exceptions import MISSING_DEPENDENCY_MESSAGE, MissingDependencyException

        # Same as original DocxConverter:
        if _dependency_exc_info is not None:
            raise MissingDependencyException(
                MISSING_DEPENDENCY_MESSAGE.format(
                    converter=type(self).__name__,
                    extension=".docx",
                    feature="docx",
                )
            ) from _dependency_exc_info[1].with_traceback(  # type: ignore[union-attr]  # pyright: ignore
                _dependency_exc_info[2]
            )

        # Customized form MarkItDown:

        # Extract mammoth-specific options if any are passed via kwargs.
        style_map = kwargs.get("style_map", None)

        html_result = mammoth.convert_to_html(file_stream, style_map=style_map)
        html_content = html_result.value

        if self.html_postprocess:
            log.info("Postprocessing HTML with %s", self.html_postprocess)
            html_content = self.html_postprocess(html_content)

        # Add custom markdownify options to the kwargs.
        combined_options = {**self.markdownify_options, **kwargs}

        result = self._html_converter.convert_string(
            html_content, url=stream_info.url, **combined_options
        )

        if self.md_postprocess:
            log.info("Postprocessing Markdown with %s", self.md_postprocess)
            result.markdown = self.md_postprocess(result.markdown)

        return CustomDCResult(
            html=html_content,
            md=result.markdown,
            title=result.title,
        )


@dataclass(frozen=True)
class MarkdownResult:
    markdown: str
    raw_html: str | None
    title: str | None


def docx_to_md(
    docx_path: Path,
    *,
    html_postprocess: Callable[[str], str] | None = markdownify_preprocess,
    md_postprocess: Callable[[str], str] | None = markdownify_postprocess,
) -> MarkdownResult:
    """
    Convert a docx file to clean markdown using MarkItDown, which wraps
    Mammoth and Markdownify. Does not normalize the Markdown.
    """

    from markitdown import MarkItDown

    # Preserve superscript and subscripts, which are important for
    # Gemini Deep Research report docx files.
    # https://github.com/matthewwithanm/python-markdownify
    docx_converter = CustomDocxConverter(
        markdownify_options=MARKDOWNIFY_OPTIONS,
        html_postprocess=html_postprocess,
        md_postprocess=md_postprocess,
    )
    mid = MarkItDown(enable_plugins=False)
    mid.register_converter(docx_converter)
    result = cast(CustomDCResult, mid.convert(docx_path))

    return MarkdownResult(markdown=result.markdown, raw_html=result.html, title=result.title)
