from kash.exec import kash_action
from kash.exec.preconditions import has_markdown_body
from kash.kits.docs.doc_formats.markdown_footnotes import convert_endnotes_to_footnotes
from kash.model import Format, Item, ItemType


@kash_action(precondition=has_markdown_body)
def endnotes_to_footnotes(item: Item) -> Item:
    """
    Remove endnotes from a Markdown document and replace them with footnotes.
    Looks for <sup>n</sup> tags and and an enumerated list of notes and replaces
    the list items with Markdown footnotes.

    This is the format of Gemini Deep Research reports. Should be safe for any doc.
    """
    if not item.body:
        raise ValueError("Item has no body")

    new_body = convert_endnotes_to_footnotes(item.body)

    return item.derived_copy(
        type=ItemType.doc,
        format=Format.markdown,
        title=item.title,  # Preserve original title (or none).
        body=new_body,
    )
