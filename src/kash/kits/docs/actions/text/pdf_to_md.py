from kash.exec import kash_action
from kash.exec.preconditions import is_pdf_resource
from kash.model import Format, Item, ItemType, Param
from kash.utils.errors import InvalidInput


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
    uses this action, to convert documents of multiple formats to Markdown.

    :param converter: The converter to use to convert the PDF to Markdown
    (markitdown or marker)
    """

    if converter == "markitdown":
        from kash.kits.docs.doc_formats import convert_pdf_markitdown

        result = convert_pdf_markitdown.pdf_to_md(item.absolute_path())
        title = result.title
        body = result.markdown
    elif converter == "marker":
        from kash.kits.docs.doc_formats import convert_pdf_marker

        title = None
        body = convert_pdf_marker.pdf_to_md(item.absolute_path())
    else:
        raise InvalidInput(f"Invalid converter: {converter}")

    return item.derived_copy(
        type=ItemType.doc,
        format=Format.markdown,
        title=title or item.title,  # Preserve original title (or none).
        body=body,
    )
