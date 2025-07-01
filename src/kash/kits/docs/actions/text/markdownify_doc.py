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
from kash.model import ActionInput, ActionResult, Item, Param
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


def markdownify_item(item: Item, pdf_converter: str = "markitdown") -> Item:
    """
    Convert an item with content already present to Markdown.
    """
    if has_fullpage_html_body(item):
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
        raise InvalidInput(f"Don't know how to convert this content to Markdown: {item.type}")

    return result_item


@kash_action(
    precondition=is_url_resource
    | is_docx_resource
    | is_pdf_resource
    | has_html_body
    | has_simple_text_body,
    params=(
        Param(
            name="pdf_converter",
            description="The converter to use to convert the PDF to Markdown.",
            type=str,
            default_value="marker",
            valid_str_values=["markitdown", "marker"],
        ),
    ),
)
def markdownify_doc(input: ActionInput, pdf_converter: str = "marker") -> ActionResult:
    """
    A more flexible `markdownify` action that converts documents of multiple formats
    to Markdown, handling HTML as well as PDF and .docx files.
    """
    item = input.items[0]

    try:
        result_item = markdownify_item(item, pdf_converter=pdf_converter)
    except InvalidInput:
        if is_url_resource(item):
            log.message("Converting URL to Markdown with custom Markdownify...")
            content_result = fetch_url_item_content(item)
            try:
                result_item = markdownify_item(content_result.item, pdf_converter=pdf_converter)
            except InvalidInput as e:
                raise InvalidInput(
                    f"Downloaded content doesn't seem to be a format we can convert to Markdown: {content_result.item}"
                ) from e
        else:
            raise InvalidInput(f"Not a recognized format or URL we can convert to Markdown: {item}")

    return ActionResult(items=[result_item])
