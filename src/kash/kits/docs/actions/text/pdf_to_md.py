from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import is_pdf_resource
from kash.model import Format, Item, ItemType, Param
from kash.utils.errors import InvalidInput
from kash.workspaces import current_ws

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
    live_output=True,  # Marker shows progress bars.
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

        return item.derived_copy(
            type=ItemType.doc,
            format=Format.markdown,
            title=title or item.title,  # Preserve original title (or none).
            body=body,
        )
    elif converter == "marker":
        from sidematter_format import Sidematter

        from kash.kits.docs.doc_formats.convert_pdf_marker import pdf_to_md_marker

        title = None
        marker_result = pdf_to_md_marker(item.absolute_path())
        body = marker_result.markdown

        result = item.derived_copy(
            type=ItemType.doc,
            format=Format.markdown,
            title=title or item.title,  # Preserve original title (or none).
            body=body,
        )

        # Manually write images to the sidematter assets directory.
        ws = current_ws()
        assets_dir = Sidematter(ws.assign_store_path(result)).assets_dir
        marker_result.write_images(assets_dir)

        return result
    else:
        raise InvalidInput(f"Invalid converter: {converter}")
