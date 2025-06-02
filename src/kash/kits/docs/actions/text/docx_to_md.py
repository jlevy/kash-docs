from kash.exec import kash_action
from kash.exec.preconditions import is_docx_resource
from kash.kits.docs.doc_formats import markitdown_convert
from kash.kits.docs.doc_formats.doc_cleanups import gemini_cleanups
from kash.kits.docs.doc_formats.markdown_footnotes import convert_endnotes_to_footnotes
from kash.model import Format, Item, ItemType


@kash_action(precondition=is_docx_resource, mcp_tool=True)
def docx_to_md(item: Item) -> Item:
    """
    Convert a docx file to clean Markdown, hopefully in good enough shape
    to publish. Uses MarkItDown/Mammoth/Markdownify and a few additional
    cleanups.

    This works well to convert docx files from Gemini Deep Research
    output: click to export a report to Google Docs, then select `File >
    Download > Microsoft Word (.docx)`.

    This is a lower-level action. You may also use `markdownify_doc`, which
    uses this action, to convert documents of multiple formats to Markdown.
    """

    result = markitdown_convert.docx_to_md(item.absolute_path())

    # Cleanups for Gemini reports. Should be fine on other files too.
    body = gemini_cleanups(result.markdown)
    final_body = convert_endnotes_to_footnotes(body)

    return item.derived_copy(
        type=ItemType.doc,
        format=Format.markdown,
        title=result.title or item.title,  # Preserve original title (or none).
        body=final_body,
    )
