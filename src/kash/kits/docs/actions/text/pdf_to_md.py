from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import is_pdf_resource
from kash.model import Format, Item, ItemType, Param
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


@kash_action(
    precondition=is_pdf_resource,
    mcp_tool=True,
    params=(
        Param(
            name="converter",
            description="The converter to use to convert the PDF to Markdown.",
            type=str,
            default_value="markitdown",
            valid_str_values=["markitdown", "marker"],
        ),
    ),
)
def pdf_to_md(item: Item, converter: str = "markitdown") -> Item:
    """
    Convert a PDF file to clean Markdown using MarkItDown.

    This is a lower-level action. You may also use `markdownify_doc`, which
    auto-detects formats and calls this action for PDFs.

    :param converter: The converter to use to convert the PDF to Markdown
    (markitdown or marker)
    """

    log.message(f"Using PDF converter: {converter}")

    if converter == "markitdown":
        from kash.kits.docs.doc_formats.convert_pdf_markitdown import pdf_to_md_markitdown

        result = pdf_to_md_markitdown(item.absolute_path())
        title = result.title
        body = result.markdown
    elif converter == "marker":
        from kash.kits.docs.doc_formats.convert_pdf_marker import pdf_to_md_marker

        title = None
        body = pdf_to_md_marker(item.absolute_path())
    else:
        raise InvalidInput(f"Invalid converter: {converter}")

    return item.derived_copy(
        type=ItemType.doc,
        format=Format.markdown,
        title=title or item.title,  # Preserve original title (or none).
        body=body,
    )
