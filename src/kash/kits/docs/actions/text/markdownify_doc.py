from kash.actions.core.markdownify import markdownify
from kash.config.logger import get_logger
from kash.exec import kash_action
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
from kash.model import ActionInput, ActionResult
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


@kash_action(
    precondition=is_url_resource
    | is_docx_resource
    | is_pdf_resource
    | has_html_body
    | has_simple_text_body
)
def markdownify_doc(input: ActionInput) -> ActionResult:
    """
    A more flexible `markdownify` action that converts documents of multiple formats
    to Markdown, handling HTML as well as PDF and .docx files.
    """
    item = input.items[0]
    if is_url_resource(item) or has_fullpage_html_body(item):
        log.message("Converting to Markdown with custom Markdownify...")
        # Web formats should be converted to Markdown.
        result_item = markdownify(item)
    elif is_docx_resource(item):
        log.message("Converting docx to Markdown with custom MarkItDown/Mammoth/Markdownify...")
        # First do basic conversion to markdown.
        result_item = docx_to_md(item)
    elif is_pdf_resource(item):
        log.message("Converting PDF to Markdown with custom MarkItDown/WeasyPrint/Markdownify...")
        result_item = pdf_to_md(item)
    elif has_simple_text_body(item):
        log.message("Document already simple text so not converting further.")
        result_item = item
    else:
        raise InvalidInput(f"Don't know how to convert item to HTML: {item.type}")

    return ActionResult(items=[result_item])
