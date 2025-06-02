import logging
import re

log = logging.getLogger(__name__)
_FOOTNOTE_INDENT = "    "


def convert_endnotes_to_footnotes(text: str, strict: bool = False) -> str:
    """
    Detects and converts docx-style endnotes (superscript footnotes marked with
    `<sup>n</sup>` tags and an enumerated list of notes) to GitHub-style footnotes.
    It identifies the *last* contiguous numbered list starting from 1
    as the potential endnote block. Ensures no content outside the original endnote
    list block is dropped.

    Returns original text if there are no endnotes (no <sup> tags are found).
    Two issues can occur:
    1. Failure to find a suitable endnote list block.
    2. Mismatch between the numbers in `<sup>` tags and the numbers in the found block.
    If `strict=True`, these issues raise a `ValueError`.
    If `strict=False`, these issues log a warning.
    """
    # Gather superscript markers
    sups = re.findall(r"<sup>(\d+)</sup>", text)
    sup_nums = sorted({int(n) for n in sups})
    if not sup_nums:
        return text

    # Find all numbered list items
    lines = text.splitlines()
    numbered = [
        (i, int(m.group(1)))
        for i, line in enumerate(lines)
        if (m := re.match(r"^\s*(\d+)\.\s+", line))
    ]

    # Locate the rightmost valid block starting at 1
    endnote_start = None
    endnote_nums = []
    for idx in range(len(numbered) - 1, -1, -1):
        _i, num = numbered[idx]
        if num == 1:
            cand = numbered[idx:]
            nums_cand = [n for _, n in cand]
            if nums_cand == list(range(1, len(nums_cand) + 1)):
                endnote_start = cand[0][0]
                endnote_nums = nums_cand
                break

    # Handle block-finding failure
    if endnote_start is None:
        msg = "Detected <sup> tags but could not find a valid endnote block"
        if strict:
            raise ValueError(msg)
        log.warning(msg + " (returning original text)")
        return text

    # Compare superscript vs endnote numbers
    if sup_nums != endnote_nums:
        msg = (
            f"Superscript numbers do not match detected endnote list numbers: found {len(sup_nums)} sup nums and {len(endnote_nums)} endnotes:\n"
            f"    missing endnotes: {sorted(list(set(sup_nums) - set(endnote_nums)))}\n"
            f"    missing sup nums: {sorted(list(set(endnote_nums) - set(sup_nums)))}"
        )
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    use_nums = set(sup_nums if strict else endnote_nums)

    # Parse notes block
    notes = {}
    current = None
    end_idx = endnote_start
    for j in range(endnote_start, len(lines)):
        line = lines[j]
        m = re.match(r"\s*(\d+)\.\s+(.*)", line)
        if m and (n := int(m.group(1))) in endnote_nums:
            current = n
            notes[n] = [m.group(2).rstrip()]
            end_idx = j
        elif current and (not line.strip() or line.startswith(" ")):
            notes[current].append(line.strip())
            end_idx = j
        else:
            break

    # Build footnote definitions
    footnote_defs: list[str] = []
    for n in endnote_nums:
        if n in notes:
            first, *rest = notes[n]
            body = [first] + [f"{(_FOOTNOTE_INDENT + line).rstrip()}" for line in rest]
            footnote_defs.append(f"[^{n}]: " + "\n".join(body))
        elif n in use_nums:
            log.warning(f"Missing note {n}")

    # Replace all <sup>n</sup> with [^n]
    replaced = re.sub(r"<sup>(\d+)</sup>", lambda m: f"[^{m.group(1)}]", text)
    mlines = replaced.splitlines()

    # Slice out original notes
    before = "\n".join(mlines[:endnote_start]).rstrip()
    after = "\n".join(mlines[end_idx + 1 :])

    # Assemble final document
    parts = [before]
    if footnote_defs:
        parts.append("\n\n".join(footnote_defs))
    if after:
        parts.append(after)

    result = "\n\n".join(filter(None, parts)).rstrip() + "\n"
    return result


## Tests


def test_endnotes_conversion():
    from textwrap import dedent

    #  Simple endnotes detection & conversion
    md_simple_long = dedent("""
    Hello<sup>1</sup>, world<sup>2</sup> and again<sup>3</sup>.
    More text.
    1. First note
    2. Second note
    3. Third note
    """)
    converted_long = convert_endnotes_to_footnotes(md_simple_long)
    assert "Hello[^1]" in converted_long
    assert "world[^2]" in converted_long
    assert "again[^3]" in converted_long
    assert "[^1]: First note" in converted_long
    assert "[^2]: Second note" in converted_long
    assert "[^3]: Third note" in converted_long
    assert "<sup>" not in converted_long  # Check all sup gone

    # No endnotes (no <sup> tags)
    plain = "Just some text without endnotes."
    assert convert_endnotes_to_footnotes(plain) == plain

    # Mismatch between <sup> and list numbers
    bad_mismatch = dedent("""
    Oops<sup>1</sup><sup>3</sup>
    1. Def one
    2. Def two
    3. Def three
    """)
    # Strict -> ValueError
    try:
        convert_endnotes_to_footnotes(bad_mismatch, strict=True)
    except ValueError as e:
        assert "do not match detected endnote" in str(e)
        assert "missing sup nums: [2]" in str(e)
        assert "missing endnotes: []" in str(e)
    else:
        raise AssertionError("Expected ValueError for mismatch (strict)")
    # Non-strict -> warning, proceeds with conversion based on list nums [1, 2, 3]
    converted_mismatch_nonstrict = convert_endnotes_to_footnotes(bad_mismatch, strict=False)
    assert "Oops[^1][^3]" in converted_mismatch_nonstrict  # Superscripts still replaced
    assert "[^1]: Def one" in converted_mismatch_nonstrict  # Definitions based on list
    assert "[^2]: Def two" in converted_mismatch_nonstrict
    assert "[^3]: Def three" in converted_mismatch_nonstrict

    # --- Block Finding Failures ---

    # Non-contiguous list numbers -> No valid block found
    bad_non_contig_list = dedent("""
    Oops<sup>1</sup><sup>2</sup>
    1. Def one
    3. Def three
    """)
    # Strict -> ValueError
    try:
        convert_endnotes_to_footnotes(bad_non_contig_list, strict=True)
    except ValueError as e:
        assert "could not find a valid endnote block" in str(e)
    else:
        raise AssertionError("Expected ValueError for non-contiguous list (strict)")
    # Non-strict -> warning, return original
    assert convert_endnotes_to_footnotes(bad_non_contig_list, strict=False) == bad_non_contig_list

    # List doesn't start at 1 -> No valid block found
    bad_start_list = dedent("""
    Oops<sup>2</sup><sup>3</sup>
    2. Def two
    3. Def three
    4. Def four
    """)
    # Strict -> ValueError
    try:
        convert_endnotes_to_footnotes(bad_start_list, strict=True)
    except ValueError as e:
        assert "could not find a valid endnote block" in str(e)
    else:
        raise AssertionError("Expected ValueError for list not starting at 1 (strict)")
    # Non-strict -> warning, return original
    assert convert_endnotes_to_footnotes(bad_start_list, strict=False) == bad_start_list

    # <sup> tags present, but NO numbered list exists -> No valid block found
    no_list_at_all = dedent("""
    Text<sup>1</sup> with sup tags.

    No valid list follows.
    """)
    # Strict -> ValueError
    try:
        convert_endnotes_to_footnotes(no_list_at_all, strict=True)
    except ValueError as e:
        assert "could not find a valid endnote block" in str(e)
    else:
        raise AssertionError("Expected ValueError for missing list (strict)")
    # Non-strict -> warning, return original
    assert convert_endnotes_to_footnotes(no_list_at_all, strict=False) == no_list_at_all

    # --- Formatting and Preservation ---

    # Test with header preservation
    md_with_header = dedent("""
    Some text<sup>1</sup> and <sup>2</sup> and <sup>3</sup>.

    ## Notes

    1. The first note.
    2. The second note.
    3. The third note.
    """)
    converted_header = convert_endnotes_to_footnotes(md_with_header)
    assert "Some text[^1] and [^2] and [^3]." in converted_header
    assert "\n\n## Notes\n\n" in converted_header  # Header preserved
    assert "[^1]: The first note." in converted_header
    assert "[^3]: The third note." in converted_header
    assert "\n1. The first note." not in converted_header  # Original list removed

    # 9) Test multiline notes (with improved indent handling)
    md_multiline = dedent("""
    Point<sup>1</sup> and point<sup>2</sup> and point<sup>3</sup>.

    1. This is the first line.
       This is indented second line.

       This is the third line after blank.
    2. Note two.
    3. Note three starts here.
      Indented continuation.
        And more.
    """)
    converted_multiline = convert_endnotes_to_footnotes(md_multiline)
    print("Multiline converted:\n", converted_multiline)
    expected_note1 = "[^1]: This is the first line.\n    This is indented second line.\n\n    This is the third line after blank."
    expected_note2 = "[^2]: Note two."
    expected_note3 = "[^3]: Note three starts here.\n    Indented continuation.\n    And more."
    assert expected_note1 in converted_multiline
    assert expected_note2 in converted_multiline
    assert expected_note3 in converted_multiline
    # Ensure original list is gone
    assert "\n1. This is the first line." not in converted_multiline
    assert "\n2. Note two." not in converted_multiline
    assert "\n3. Note three starts here." not in converted_multiline

    # Tests conversion when other numbered lists precede/follow the endnotes.
    md = dedent("""
    # Document Title

    Here is an introductory list:
    1. First item.
    2. Second item.
    3. Third item.

    Some text with a footnote<sup>1</sup>. More text<sup>2</sup>. Final point<sup>3</sup>.

    Another unrelated list:
    1. Apple
    2. Banana

    ## Notes Section

    1. This is the actual first note.
       It can span multiple lines.

       Includes blank lines.
    2. This is the second note.
    3. And the third note.

    Some text *after* the notes list.
    This should be preserved.
    """)

    converted = convert_endnotes_to_footnotes(md)
    print("Multiple lists converted:\n", converted)

    # Check first list is preserved
    assert "1. First item." in converted
    assert "2. Second item." in converted
    assert "3. Third item." in converted
    # Check second list is preserved
    assert "1. Apple" in converted
    assert "2. Banana" in converted
    # Check footnotes converted
    assert "footnote[^1]" in converted
    assert "text[^2]" in converted
    # Check header preserved
    assert "\n## Notes Section\n" in converted
    # Check definitions are correct
    expected_note1 = "[^1]: This is the actual first note.\n    It can span multiple lines.\n\n    Includes blank lines."
    assert expected_note1 in converted
    assert "[^2]: This is the second note." in converted
    assert "[^3]: And the third note." in converted
    # Check original endnote list numbers are removed
    assert "\n1. This is the actual first note." not in converted
    assert "\n2. This is the second note." not in converted
    assert "\n3. And the third note." not in converted
    # Check text after notes IS PRESERVED
    assert "Some text *after* the notes list." in converted
    assert "This should be preserved." in converted
