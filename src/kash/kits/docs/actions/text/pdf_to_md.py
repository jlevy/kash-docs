from kash.exec import kash_action
from kash.exec.preconditions import is_pdf_resource
from kash.kits.docs.doc_formats import markitdown_convert
from kash.model import Format, Item, ItemType


@kash_action(precondition=is_pdf_resource, mcp_tool=True)
def pdf_to_md(item: Item) -> Item:
    """
    Convert a PDF file to clean Markdown using MarkItDown.

    This is a lower-level action. You may also use `markdownify_doc`, which
    uses this action, to convert documents of multiple formats to Markdown.
    """

    result = markitdown_convert.pdf_to_md(item.absolute_path())

    return item.derived_copy(
        type=ItemType.doc,
        format=Format.markdown,
        title=result.title or item.title,  # Preserve original title (or none).
        body=result.markdown,
    )
