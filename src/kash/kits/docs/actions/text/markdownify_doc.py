from kash.actions.core.markdownify_html import markdownify_html
from kash.config.logger import get_logger
from kash.exec import fetch_url_item_content, kash_action
from kash.exec.preconditions import (
    has_fullpage_html_body,
    has_html_body,
    has_simple_text_body,
    is_docx_resource,
    is_pdf_resource,
    is_url_resource,
)
from kash.kits.docs.actions.text.docx_to_md import docx_to_md
from kash.kits.docs.actions.text.pdf_to_md import pdf_to_md
from kash.model import Format, Item, Param
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


@kash_action(
    precondition=is_url_resource
    | is_docx_resource
    | is_pdf_resource
    | has_html_body
    | has_simple_text_body,
    output_format=Format.markdown,
    params=(
        Param(
            name="pdf_converter",
            description="The converter to use to convert the PDF to Markdown.",
            type=str,
            default_value="marker",
            valid_str_values=["markitdown", "marker"],
        ),
    ),
    mcp_tool=True,
)
def markdownify_doc(item: Item, pdf_converter: str = "marker") -> Item:
    """
    A more flexible `markdownify` action that converts documents of multiple formats
    to Markdown, handling HTML as well as PDF and .docx files.
    """
    if is_url_resource(item):
        log.message("Converting URL to Markdown with custom Markdownify...")
        content_result = fetch_url_item_content(item)
        result_item = markdownify_html(content_result.item)
    elif has_fullpage_html_body(item):
        log.message("Converting to Markdown with custom Markdownify...")
        # Web formats should be converted to Markdown.
        result_item = markdownify_html(item)
    elif is_docx_resource(item):
        log.message("Converting docx to Markdown with custom MarkItDown/Mammoth/Markdownify...")
        # First do basic conversion to markdown.
        result_item = docx_to_md(item)
    elif is_pdf_resource(item):
        log.message("Converting PDF to Markdown with custom MarkItDown/WeasyPrint/Markdownify...")
        result_item = pdf_to_md(item, converter=pdf_converter)
    elif has_simple_text_body(item):
        log.message("Document already simple text so not converting further.")
        result_item = item
    else:
        raise InvalidInput(f"Don't know how to convert this content to Markdown: {item}")

    return result_item
