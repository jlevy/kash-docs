import logging
import re

from chopdiff.docs import diff_wordtoks, wordtokenize

log = logging.getLogger(__name__)


def _fix_literal_sups_bug(body: str) -> str:
    """
    Gemini sometimes puts explicit `sup` tags around superscripts (in addition to them
    already being superscripts).
    """

    return (
        body.replace("&lt;sup&gt;<sup>", "<sup>")
        .replace("&lt;sup><sup>", "<sup>")
        .replace("</sup>&lt;/sup&gt;", "</sup>")
        .replace("</sup>&lt;/sup>", "</sup>")
    )


_sup_space_pat = re.compile(r" +<sup>")


def _fix_sup_space(html: str) -> str:
    """
    Google Gemini has a bad habit of putting extra space before superscript
    footnotes in docx exports.
    """
    return _sup_space_pat.sub("<sup>", html)


def _fix_works_cited(body: str) -> str:
    """
    Gemini puts "Works cited" as an h4 for some reason.
    Convert any "Works cited" header to h2 level like other main sections.
    """
    return re.sub(r"#{1,6}\s+(works\s+cited)", r"## Works Cited", body, flags=re.IGNORECASE)


def gemini_cleanups(body: str) -> str:
    """
    Extra modifications to clean up Gemini Deep Research output.
    Rare replacements that should be safe for other docs as well.
    """

    new_body = _fix_literal_sups_bug(body)
    new_body = _fix_sup_space(new_body)
    new_body = _fix_works_cited(new_body)

    diff = diff_wordtoks(wordtokenize(body), wordtokenize(new_body))
    if diff.stats().nchanges() > 0:
        log.warning(
            "Seems to be a Gemini doc output that needed cleanups: %s word tokens", diff.stats()
        )

    return new_body
