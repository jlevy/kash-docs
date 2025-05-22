from typing import Any

from bs4 import BeautifulSoup
from bs4.dammit import EntitySubstitution
from bs4.element import PageElement, Tag
from bs4.formatter import HTMLFormatter
from typing_extensions import override

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_html_body
from kash.model import Item, ItemType
from kash.utils.errors import InvalidInput

log = get_logger(__name__)

# Common inline elements (adjust as needed)
# Based on MDN's list of inline elements
INLINE_ELEMENTS = {
    "a",
    "abbr",
    "acronym",
    "b",
    "bdi",
    "bdo",
    "big",
    "br",
    "button",
    "cite",
    "code",
    "data",
    "datalist",
    "del",
    "dfn",
    "em",
    "embed",
    "i",
    "iframe",
    "img",
    "input",
    "ins",
    "kbd",
    "label",
    "map",
    "mark",
    "meter",
    "noscript",
    "object",
    "output",
    "picture",
    "progress",
    "q",
    "ruby",
    "s",
    "samp",
    "select",
    "slot",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "svg",
    "template",
    "textarea",
    "time",
    "u",
    "tt",
    "var",
    "video",
    "wbr",
}

SMALL_ELEMENTS = {"td"}


NEWLINE_FREE_ELEMENTS = INLINE_ELEMENTS | SMALL_ELEMENTS


class SelectiveFormatter(HTMLFormatter):
    """
    A custom BS4 formatter that properly indents block elements while avoiding
    excessive newlines around inline elements.
    """

    # FIXME: Class names seem to be serializing incorrectly, e.g.
    #  <div class="['gb_Fa', 'gb_Kd', 'gb_3d']" id="gb">

    def __init__(self, *args: Any, indent: int = 4, **kwargs: Any):
        # Use HTML5 defaults
        kwargs.setdefault("entity_substitution", EntitySubstitution.substitute_html5)
        kwargs.setdefault("void_element_close_prefix", "")
        kwargs.setdefault("empty_attributes_are_booleans", True)

        # Base indent as string (matching formatter.py)
        indent_str = " " * indent if isinstance(indent, int) and indent > 0 else ""
        super().__init__(*args, indent=indent_str, **kwargs)

    @override
    def substitute(self, ns: str) -> str:
        """Delegate entity substitution to the parent implementation."""
        return super().substitute(ns)

    def prettify(self, soup: BeautifulSoup | Tag) -> str:
        """Return formatted HTML without the top-level [document] wrapper."""
        result: list[str] = []
        # Skip the synthetic '[document]' tag emitted by BeautifulSoup
        for child in soup.contents:
            self._prettify(child, result, 0)
        return "".join(result)

    def _prettify(self, node: PageElement, result: list[str], level: int) -> None:
        """Recursive method to prettify a node with custom indentation rules."""
        if isinstance(node, Tag):
            is_inline = node.name in NEWLINE_FREE_ELEMENTS

            # Only indent block elements (non-inline)
            if not is_inline and level > 0:
                result.append("\n" + self.indent * level)

            # Opening tag + attributes
            result.append(f"<{node.name}")
            for key, val in self.attributes(node):
                if val is None:
                    result.append(f" {key}")
                else:
                    result.append(f' {key}="{self.attribute_value(str(val))}"')

            if node.is_empty_element:
                result.append(f"{self.void_element_close_prefix}>")
                return  # No children or closing tag

            # Non-empty element
            result.append(">")

            # Process children
            content_level = level + (0 if is_inline else 1)
            has_block_content = False
            for child in node.children:
                # BeautifulSoup types can be Tag or NavigableString
                if isinstance(child, Tag) and child.name not in NEWLINE_FREE_ELEMENTS:
                    has_block_content = True
                self._prettify(child, result, content_level)

            # Indent closing tag for block elements that had block content
            if not is_inline and has_block_content:
                result.append("\n" + self.indent * level)

            result.append(f"</{node.name}>")
        else:
            # NavigableString (or derivative)
            text = self.substitute(str(node))
            if text.strip():
                result.append(text)
            elif " " in text and "\n" not in text:
                # Preserve a single space between inline elements
                result.append(" ")


@kash_action(precondition=has_html_body)
def prettify_html(item: Item) -> Item:
    """
    Parses HTML, corrects errors via html5lib, and formats using
    a custom formatter to provide readable structure. Can change whitespace and layout!
    But good for human readability.
    """

    if not item.body:
        raise InvalidInput("Item must have a body")

    soup = BeautifulSoup(item.body, "html5lib")
    custom_formatter = SelectiveFormatter(indent=2)

    # Use our custom prettify directly
    formatted_body = custom_formatter.prettify(soup)
    formatted_body = formatted_body.strip()

    return item.derived_copy(type=ItemType.doc, body=formatted_body)
