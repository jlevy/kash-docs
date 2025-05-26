import re

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
    Should be safe for other docs as well.
    """

    body = _fix_sup_space(body)
    body = _fix_works_cited(body)

    return body
