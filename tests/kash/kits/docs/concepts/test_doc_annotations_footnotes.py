"""
Comprehensive tests for doc_annotations module with footnote support.
"""

from textwrap import dedent

from chopdiff.docs.text_doc import Paragraph, TextDoc

from kash.kits.docs.concepts.doc_annotations import (
    AnnotatedDoc,
    AnnotatedPara,
    FootnoteReference,
    TextSpan,
)
from kash.utils.text_handling.markdown_footnotes import MarkdownFootnotes


def test_text_span_basic() -> None:
    """Test TextSpan validation and creation."""
    # Valid span
    span = TextSpan(start=0, end=5, text="hello")
    assert span.start == 0
    assert span.end == 5
    assert span.text == "hello"

    # Invalid span - negative start
    try:
        TextSpan(start=-1, end=5, text="hello")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Invalid span" in str(e)

    # Invalid span - end before start
    try:
        TextSpan(start=10, end=5, text="hello")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Invalid span" in str(e)


def test_footnote_reference_basic() -> None:
    """Test FootnoteReference creation."""
    span = TextSpan(start=10, end=15, text="[^1]")
    ref = FootnoteReference(footnote_id="1", span=span, sentence_index=0)

    assert ref.footnote_id == "1"
    assert ref.span.start == 10
    assert ref.span.end == 15
    assert ref.span.text == "[^1]"
    assert ref.sentence_index == 0


def test_extract_footnote_references() -> None:
    """Test extracting footnote references from paragraph text."""
    # Note: This entire text is parsed as a single sentence by the sentence splitter
    para = Paragraph.from_text(
        "This is a sentence with a footnote[^1]. "
        "This has two footnotes[^2][^abc]. "
        "This has no footnotes."
    )
    ann_para = AnnotatedPara.from_para(para)

    refs = ann_para.extract_footnote_references()

    assert len(refs) == 3

    # All references are in the single sentence (IDs now include ^)
    assert refs[0].footnote_id == "^1"
    assert refs[0].sentence_index == 0
    assert refs[0].span.text == "[^1]"
    assert refs[0].span.start == 34  # Position in the text

    assert refs[1].footnote_id == "^2"
    assert refs[1].sentence_index == 0  # Same sentence
    assert refs[1].span.text == "[^2]"
    assert refs[1].span.start == 62  # Position in the text

    assert refs[2].footnote_id == "^abc"
    assert refs[2].sentence_index == 0  # Same sentence
    assert refs[2].span.text == "[^abc]"
    assert refs[2].span.start == 66  # Position after [^2]


def test_from_para_with_footnotes_basic() -> None:
    """Test creating AnnotatedPara with footnotes."""
    # Create markdown content with footnotes
    markdown_content = dedent("""
        This is a paragraph with a footnote[^1]. Another sentence[^2].
        
        [^1]: First footnote content
        [^2]: Second footnote content
        """)

    # Parse footnotes
    footnotes = MarkdownFootnotes.from_markdown(markdown_content)

    # Create paragraph (without the footnote definitions)
    # Note: This text is parsed as a single sentence by the sentence splitter
    para = Paragraph.from_text("This is a paragraph with a footnote[^1]. Another sentence[^2].")

    # Create annotated paragraph with footnotes
    ann_para = AnnotatedPara.from_para_with_footnotes(para, footnotes)

    # Check annotations were created
    assert ann_para.annotation_count() == 2

    # Both footnotes are in the single sentence
    sent0_annotations = ann_para.get_sentence_annotations(0)
    assert len(sent0_annotations) == 2
    assert "First footnote content" in sent0_annotations[0]
    assert "Second footnote content" in sent0_annotations[1]


def test_from_para_with_footnotes_multiple_refs_same_footnote() -> None:
    """Test handling multiple references to the same footnote."""
    markdown_content = dedent("""
        First reference[^note]. Second reference[^note]. Different note[^other].
        
        [^note]: Shared footnote content
        [^other]: Other footnote content
        """)

    footnotes = MarkdownFootnotes.from_markdown(markdown_content)
    para = Paragraph.from_text(
        "First reference[^note]. Second reference[^note]. Different note[^other]."
    )

    ann_para = AnnotatedPara.from_para_with_footnotes(para, footnotes)

    # Should have 2 unique annotations (not 3) - all in same sentence
    assert ann_para.annotation_count() == 2

    # All footnotes are in the single sentence
    sent0_annotations = ann_para.get_sentence_annotations(0)
    assert len(sent0_annotations) == 2
    assert "Shared footnote content" in sent0_annotations[0]
    assert "Other footnote content" in sent0_annotations[1]


def test_from_para_with_footnotes_missing_definition() -> None:
    """Test handling footnote references without definitions."""
    markdown_content = dedent("""
        Text with reference[^exists].
        
        [^exists]: This footnote exists
        """)

    footnotes = MarkdownFootnotes.from_markdown(markdown_content)

    # Paragraph has a reference to a non-existent footnote
    para = Paragraph.from_text("Text with reference[^exists]. Missing reference[^missing].")

    ann_para = AnnotatedPara.from_para_with_footnotes(para, footnotes)

    # Should only have annotation for existing footnote
    assert ann_para.annotation_count() == 1

    sent0_annotations = ann_para.get_sentence_annotations(0)
    assert len(sent0_annotations) == 1
    assert "This footnote exists" in sent0_annotations[0]

    # Second sentence should have no annotations
    sent1_annotations = ann_para.get_sentence_annotations(1)
    assert len(sent1_annotations) == 0


def test_from_para_with_footnotes_complex_content() -> None:
    """Test footnotes with complex markdown content."""
    markdown_content = dedent("""
        Main text with footnote[^complex].
        
        [^complex]: This footnote has **bold**, *italic*, and [a link](https://example.com).
            
            It even has multiple paragraphs!
            
            - And a list item
            - Another item
        """)

    footnotes = MarkdownFootnotes.from_markdown(markdown_content)
    para = Paragraph.from_text("Main text with footnote[^complex].")

    ann_para = AnnotatedPara.from_para_with_footnotes(para, footnotes)

    assert ann_para.annotation_count() == 1
    annotations = ann_para.get_sentence_annotations(0)
    assert len(annotations) == 1

    # Check that complex content is preserved (rendered as markdown)
    content = annotations[0]
    assert "**bold**" in content
    assert "*italic*" in content
    assert "[a link](https://example.com)" in content
    assert "multiple paragraphs" in content
    assert "- And a list item" in content


def test_from_para_with_footnotes_preserve_ids() -> None:
    """Test that original footnote IDs are always preserved."""
    markdown_content = dedent("""
        Text with custom IDs[^foo][^bar][^123].
        
        [^foo]: Foo footnote
        [^bar]: Bar footnote
        [^123]: Numeric footnote
        """)

    footnotes = MarkdownFootnotes.from_markdown(markdown_content)
    para = Paragraph.from_text("Text with custom IDs[^foo][^bar][^123].")

    # Original IDs are always preserved
    ann_para = AnnotatedPara.from_para_with_footnotes(para, footnotes)
    assert ann_para.annotation_count() == 3

    # When rendered, should use original IDs
    rendered = ann_para.as_markdown_footnotes()
    assert "[^foo]:" in rendered
    assert "[^bar]:" in rendered
    assert "[^123]:" in rendered
    assert "Foo footnote" in rendered
    assert "Bar footnote" in rendered
    assert "Numeric footnote" in rendered


def test_from_para_with_footnotes_in_document() -> None:
    """Test footnotes in a full document context."""
    markdown_content = dedent("""
        # Document Title
        
        First paragraph with a note[^1].
        
        Second paragraph with two notes[^2][^3].
        
        Third paragraph with no notes.
        
        [^1]: First note content
        [^2]: Second note content
        [^3]: Third note content
        """)

    # Parse footnotes
    footnotes = MarkdownFootnotes.from_markdown(markdown_content)

    # Create document
    text_doc = TextDoc.from_text(
        dedent("""
        First paragraph with a note[^1].
        
        Second paragraph with two notes[^2][^3].
        
        Third paragraph with no notes.
        """).strip()
    )

    # Process each paragraph
    ann_paras = []
    for para in text_doc.paragraphs:
        ann_para = AnnotatedPara.from_para_with_footnotes(para, footnotes)
        ann_paras.append(ann_para)

    # Consolidate into document
    ann_doc = AnnotatedDoc.consolidate_annotations(ann_paras)

    assert ann_doc.total_annotation_count() == 3
    assert len(ann_doc.footnote_mapping) == 3

    # Render with footnotes
    rendered = ann_doc.as_markdown_with_footnotes()

    # Check that footnotes are included
    assert "[^1]: First note content" in rendered
    assert "[^2]: Second note content" in rendered
    assert "[^3]: Third note content" in rendered


def test_footnote_dataclass_in_annotations() -> None:
    """Test that Footnote dataclass is properly used in annotations."""
    from kash.kits.docs.concepts.doc_annotations import Footnote

    para = Paragraph.from_text("Test paragraph.")
    ann_para = AnnotatedPara.from_para(para)

    # Add an annotation
    ann_para.add_annotation_with_id(0, "custom_id", "Custom content")

    # Get the annotations with IDs (should return Footnote objects)
    footnotes = ann_para.get_sentence_annotations_with_ids(0)
    assert len(footnotes) == 1
    assert isinstance(footnotes[0], Footnote)
    assert footnotes[0].id == "^custom_id"  # IDs are now normalized to include ^
    assert footnotes[0].content == "Custom content"

    # Get just the content
    contents = ann_para.get_sentence_annotations(0)
    assert len(contents) == 1
    assert contents[0] == "Custom content"


def test_mixed_footnotes_and_manual_annotations() -> None:
    """Test combining footnotes from markdown with manually added annotations."""
    markdown_content = dedent("""
        Text with footnote[^md].
        
        [^md]: Markdown footnote
        """)

    footnotes = MarkdownFootnotes.from_markdown(markdown_content)
    para = Paragraph.from_text("Text with footnote[^md]. Another sentence.")

    # Create with footnotes
    ann_para = AnnotatedPara.from_para_with_footnotes(para, footnotes)

    # Add manual annotation
    ann_para.add_annotation(1, "Manual annotation for second sentence")

    assert ann_para.annotation_count() == 2

    # Check both types of annotations
    sent0_annotations = ann_para.get_sentence_annotations(0)
    assert len(sent0_annotations) == 1
    assert "Markdown footnote" in sent0_annotations[0]

    sent1_annotations = ann_para.get_sentence_annotations(1)
    assert len(sent1_annotations) == 1
    assert "Manual annotation" in sent1_annotations[0]

    # Render should include both (preserving original ID for markdown footnote)
    rendered = ann_para.as_markdown_footnotes()
    assert "[^md]: Markdown footnote" in rendered  # Original ID preserved
    assert "[^2]: Manual annotation" in rendered  # Manual annotation gets generated ID


def test_footnotes_edge_cases() -> None:
    """Test edge cases in footnote handling."""
    # Empty paragraph
    empty_para = Paragraph.from_text("")
    footnotes = MarkdownFootnotes.from_markdown("[^1]: Some content")
    ann_para = AnnotatedPara.from_para_with_footnotes(empty_para, footnotes)
    assert ann_para.annotation_count() == 0

    # No footnotes object
    para = Paragraph.from_text("Text with reference[^1].")
    ann_para = AnnotatedPara.from_para_with_footnotes(para, None)
    assert ann_para.annotation_count() == 0

    # Malformed footnote references - should not be picked up
    para = Paragraph.from_text("Text with [^ spaces] and [^] empty.")
    footnotes = MarkdownFootnotes.from_markdown("")
    ann_para = AnnotatedPara.from_para_with_footnotes(para, footnotes)

    refs = ann_para.extract_footnote_references()
    # The tighter regex r"\[\^([\w.-]+)\]" won't match either malformed reference
    # [^ spaces] has a space, [^] has no content
    assert len(refs) == 0

    # Valid footnote references with hyphens and dots (IDs include ^)
    para2 = Paragraph.from_text("Valid refs: [^note-1] and [^ref.2] and [^foo_bar].")
    ann_para2 = AnnotatedPara.from_para(para2)
    refs2 = ann_para2.extract_footnote_references()
    assert len(refs2) == 3
    assert refs2[0].footnote_id == "^note-1"
    assert refs2[1].footnote_id == "^ref.2"
    assert refs2[2].footnote_id == "^foo_bar"


def test_footnotes_after_sentences() -> None:
    """Test footnotes that appear after sentence boundaries."""
    # Footnote immediately after period
    para = Paragraph.from_text("First sentence.[^1] Second sentence.")
    ann_para = AnnotatedPara.from_para(para)
    refs = ann_para.extract_footnote_references()

    assert len(refs) == 1
    assert refs[0].footnote_id == "^1"
    # Should be attached to first sentence (preceding sentence)
    assert refs[0].sentence_index == 0

    # Footnote with space after period
    para = Paragraph.from_text("First sentence. [^2] Second sentence.")
    ann_para = AnnotatedPara.from_para(para)
    refs = ann_para.extract_footnote_references()

    assert len(refs) == 1
    assert refs[0].footnote_id == "^2"
    # Should be attached to first sentence
    assert refs[0].sentence_index == 0

    # Multiple footnotes after sentence
    para = Paragraph.from_text("First sentence.[^a][^b] Second sentence.")
    ann_para = AnnotatedPara.from_para(para)
    refs = ann_para.extract_footnote_references()

    assert len(refs) == 2
    assert refs[0].footnote_id == "^a"
    assert refs[0].sentence_index == 0
    assert refs[1].footnote_id == "^b"
    assert refs[1].sentence_index == 0


def test_paragraph_starting_with_footnote() -> None:
    """Test paragraphs that start with footnotes."""
    # Paragraph starting with footnote
    para = Paragraph.from_text("[^start] This is the first sentence. Second sentence.")
    ann_para = AnnotatedPara.from_para(para)
    refs = ann_para.extract_footnote_references()

    assert len(refs) == 1
    assert refs[0].footnote_id == "^start"
    # Should be attached to first sentence
    assert refs[0].sentence_index == 0
    assert refs[0].span.start == 0  # At the beginning

    # Multiple footnotes at start
    para = Paragraph.from_text("[^1][^2][^3] First sentence here.")
    ann_para = AnnotatedPara.from_para(para)
    refs = ann_para.extract_footnote_references()

    assert len(refs) == 3
    for ref in refs:
        assert ref.sentence_index == 0


def test_paragraph_with_only_footnotes() -> None:
    """Test paragraphs containing only footnotes."""
    # Single footnote only
    para = Paragraph.from_text("[^only]")
    ann_para = AnnotatedPara.from_para(para)
    refs = ann_para.extract_footnote_references()

    assert len(refs) == 1
    assert refs[0].footnote_id == "^only"
    assert refs[0].sentence_index == 0

    # Multiple footnotes only
    para = Paragraph.from_text("[^a][^b][^c]")
    ann_para = AnnotatedPara.from_para(para)
    refs = ann_para.extract_footnote_references()

    assert len(refs) == 3
    assert all(ref.sentence_index == 0 for ref in refs)

    # Footnotes with spaces
    para = Paragraph.from_text("[^x] [^y] [^z]")
    ann_para = AnnotatedPara.from_para(para)
    refs = ann_para.extract_footnote_references()

    assert len(refs) == 3
    assert all(ref.sentence_index == 0 for ref in refs)


def test_complex_footnote_positioning() -> None:
    """Test complex scenarios with footnotes in various positions."""
    # Footnotes in middle, end, and after sentences
    # Note: The sentence splitter treats "End of first[^2]. [^3] Second sentence[^4]." as one sentence
    para = Paragraph.from_text(
        "Start[^1] middle of first. End of first[^2]. [^3] Second sentence[^4]."
    )
    ann_para = AnnotatedPara.from_para(para)
    refs = ann_para.extract_footnote_references()

    assert len(refs) == 4

    # [^1] in middle of first sentence (IDs include ^)
    assert refs[0].footnote_id == "^1"
    assert refs[0].sentence_index == 0
    assert refs[0].span.text == "[^1]"

    # [^2], [^3], and [^4] are all in the second sentence
    assert refs[1].footnote_id == "^2"
    assert refs[1].sentence_index == 1

    assert refs[2].footnote_id == "^3"
    assert refs[2].sentence_index == 1

    assert refs[3].footnote_id == "^4"
    assert refs[3].sentence_index == 1


def test_footnotes_with_content_integration() -> None:
    """Test the full integration with markdown footnotes content."""
    markdown_content = dedent("""
        [^intro] First paragraph with note[^1].
        
        Second para[^2]. [^3]
        
        [^only_footnotes]
        
        [^intro]: Introduction footnote
        [^1]: First note
        [^2]: Second note
        [^3]: Third note
        [^only_footnotes]: Standalone footnote
        """)

    footnotes = MarkdownFootnotes.from_markdown(markdown_content)

    # First paragraph
    para1 = Paragraph.from_text("[^intro] First paragraph with note[^1].")
    ann_para1 = AnnotatedPara.from_para_with_footnotes(para1, footnotes)

    assert ann_para1.annotation_count() == 2
    annotations = ann_para1.get_sentence_annotations(0)
    assert len(annotations) == 2
    assert "Introduction footnote" in annotations[0]
    assert "First note" in annotations[1]

    # Second paragraph
    para2 = Paragraph.from_text("Second para[^2]. [^3]")
    ann_para2 = AnnotatedPara.from_para_with_footnotes(para2, footnotes)

    assert ann_para2.annotation_count() == 2
    annotations = ann_para2.get_sentence_annotations(0)
    assert len(annotations) == 2
    assert "Second note" in annotations[0]
    assert "Third note" in annotations[1]

    # Paragraph with only footnotes
    para3 = Paragraph.from_text("[^only_footnotes]")
    ann_para3 = AnnotatedPara.from_para_with_footnotes(para3, footnotes)

    assert ann_para3.annotation_count() == 1
    annotations = ann_para3.get_sentence_annotations(0)
    assert len(annotations) == 1
    assert "Standalone footnote" in annotations[0]


def test_real_world_example() -> None:
    """Test with a realistic example of research text with citations."""
    markdown_content = dedent("""
        # Research on Ketamine Therapy
        
        Ketamine has shown promise for treatment-resistant depression[^109].
        Recent studies indicate rapid onset of action[^110][^111].
        
        The mechanism involves NMDA receptor antagonism[^112], though
        the full therapeutic pathway remains under investigation.
        
        [^109]: What Is The Future Of Ketamine Therapy For Mental Health Treatment?
            - The Ko-Op, accessed June 28, 2025,
              <https://psychedelictherapists.co/blog/the-future-of-ketamine-assisted-psychotherapy/>
        
        [^110]: Smith et al. (2024). "Rapid antidepressant effects of ketamine."
            Journal of Psychiatry, 45(3), 234-245.
        
        [^111]: Johnson, M. (2024). "Ketamine mechanisms in depression treatment."
            Neuropsychopharmacology Reviews, 12(1), 78-92.
        
        [^112]: NMDA receptor modulation leads to increased BDNF expression
            and synaptic plasticity enhancement (Davis, 2023).
        """)

    footnotes = MarkdownFootnotes.from_markdown(markdown_content)

    # Create document without footnote definitions
    doc_text = dedent("""
        Ketamine has shown promise for treatment-resistant depression[^109].
        Recent studies indicate rapid onset of action[^110][^111].
        
        The mechanism involves NMDA receptor antagonism[^112], though
        the full therapeutic pathway remains under investigation.
        """).strip()

    text_doc = TextDoc.from_text(doc_text)

    # Process paragraphs (preserving original footnote IDs)
    ann_paras = []
    for i, para in enumerate(text_doc.paragraphs):
        ann_para = AnnotatedPara.from_para_with_footnotes(
            para, footnotes, fn_prefix=f"p{i + 1}_", fn_start=1
        )
        ann_paras.append(ann_para)

    # First paragraph should have 3 annotations
    assert ann_paras[0].annotation_count() == 3

    # Second paragraph should have 1 annotation
    assert ann_paras[1].annotation_count() == 1

    # Consolidate
    ann_doc = AnnotatedDoc.consolidate_annotations(ann_paras)

    # Render
    rendered = ann_doc.as_markdown_with_footnotes(footnote_header="## References")

    # Check structure - IDs are preserved with prefixes
    assert "## References" in rendered
    assert "[^p1_109]:" in rendered  # First para, first footnote (preserves ID 109)
    assert "[^p1_110]:" in rendered  # First para, second footnote (preserves ID 110)
    assert "[^p1_111]:" in rendered  # First para, third footnote (preserves ID 111)
    assert "[^p2_112]:" in rendered  # Second para, first footnote (preserves ID 112)

    # Check content preservation
    assert "Ko-Op" in rendered
    assert "Smith et al." in rendered
    assert "NMDA receptor modulation" in rendered
