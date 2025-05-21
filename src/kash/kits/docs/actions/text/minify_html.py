from kash.exec import kash_action
from kash.exec.preconditions import has_fullpage_html_body
from kash.model import Format, Item
from kash.utils.errors import InvalidInput


@kash_action(precondition=has_fullpage_html_body)
def minify_html(item: Item) -> Item:
    """
    Minify an HTML item's content using minify_html, a modern Rust-based minifier.
    """
    from minify_html import minify

    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    minified_content = minify(
        item.body,
        minify_js=True,
        minify_css=True,
        remove_processing_instructions=True,
        # keep_comments=True,  # Keeps frontmatter format comments.
    )

    return item.derived_copy(type=item.type, format=Format.html, body=minified_content)
