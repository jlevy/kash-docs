from kash.exec import kash_action
from kash.exec.preconditions import has_fullpage_html_body
from kash.model import Format, Item, Param
from kash.utils.errors import InvalidInput


@kash_action(
    precondition=has_fullpage_html_body,
    params=(
        Param("no_js_min", "Disable JS minification", bool),
        Param("no_css_min", "Disable CSS minification", bool),
    ),
)
def minify_html_fast(item: Item, no_js_min: bool = False, no_css_min: bool = False) -> Item:
    """
    Minify an HTML item's content using minify_html, a fast, modern Rust-based minifier.

    This is convenient but seems to have a few bugs with JS minification:
    https://github.com/wilsonzlin/minify-html/issues/236
    """
    from minify_html import minify

    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    minified_content = minify(
        item.body,
        minify_js=not no_js_min,
        minify_css=not no_css_min,
        remove_processing_instructions=True,
        # keep_comments=True,  # Keeps frontmatter format comments.
    )

    return item.derived_copy(type=item.type, format=Format.html, body=minified_content)
