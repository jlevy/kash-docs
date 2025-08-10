from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType

from chopdiff.docs.text_doc import Paragraph, SentIndex, TextDoc

from kash.utils.common.testing import enable_if
from kash.utils.text_handling.markdown_footnotes import MarkdownFootnotes

FootnoteId = NewType("FootnoteId", str)

# Valid footnote ID pattern: Unicode word characters (letters, digits, underscore), period, or hyphen
_FOOTNOTE_ID_PATTERN = re.compile(r"^[\w.-]+$")

# Pattern to match footnote references in text: [^id] where id contains no whitespace
# Matches: [^1], [^foo], [^note-1], [^ref.2]
# Does not match: [^ spaces], [^], [^foo bar]
_FOOTNOTE_REF_PATTERN = re.compile(r"\[\^([\w.-]+)\]")


def _normalize_footnote_id(footnote_id: str) -> str:
    """
    Normalize a footnote ID to always include the ^ prefix.
    """
    return footnote_id if footnote_id.startswith("^") else f"^{footnote_id}"


@dataclass(frozen=True)
class Footnote:
    """
    Represents a footnote with its ID and content.
    """

    id: str
    """The footnote ID (includes ^ prefix, e.g., "^123", "^foo")"""

    content: str
    """The footnote content/annotation text"""


@dataclass
class TextSpan:
    """
    Represents a span of text within a string.
    """

    start: int
    """Start position of the span in the text"""

    end: int
    """End position of the span in the text"""

    text: str
    """The actual text content of the span"""

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span: start={self.start}, end={self.end}")


@dataclass
class FootnoteReference:
    """
    Represents a footnote reference found in text.
    """

    footnote_id: str
    """The ID with the caret (e.g., "^123", "^foo")"""

    span: TextSpan
    """The location of the reference in the text"""

    sentence_index: int
    """Which sentence contains this reference"""


def check_fn_id(footnote_id: str) -> FootnoteId:
    """
    Validate and return a footnote ID. IDs should include the ^ prefix.
    """
    # Normalize to include ^ if not present
    normalized_id = _normalize_footnote_id(footnote_id)

    # Sanity check
    if len(normalized_id) > 50:
        raise ValueError(
            f"Not a valid footnote id (must be <=50 chars excluding ^): '{footnote_id!r}' ({len(normalized_id) - 1} chars)"
        )

    # Validate the part after the ^
    id_part = normalized_id[1:]  # Remove ^ for pattern matching
    if not _FOOTNOTE_ID_PATTERN.match(id_part):
        raise ValueError(
            f"Not a valid footnote id (must contain only word chars, period, or hyphen): '{footnote_id!r}'"
        )

    return FootnoteId(normalized_id)


@dataclass
class AnnotatedPara:
    """
    A paragraph with annotations that can be rendered as markdown footnotes.

    Wraps a `Paragraph` from chopdiff and adds annotation functionality.
    Annotations are stored as a mapping by sentence index in this paragraph.
    """

    paragraph: Paragraph

    annotations: dict[int, list[Footnote]]
    """
    Mapping from sentence indices to lists of Footnote objects.
    The Footnote.id preserves the original ID from the source (without ^).
    """

    fn_prefix: str = ""
    """Prefix for footnote ids."""

    fn_start: int = 1
    """Starting number for footnotes."""

    @classmethod
    def from_para(
        cls, paragraph: Paragraph, fn_prefix: str = "", fn_start: int = 1
    ) -> AnnotatedPara:
        """Create an AnnotatedParagraph from an existing Paragraph."""
        return cls(paragraph=paragraph, annotations={}, fn_prefix=fn_prefix, fn_start=fn_start)

    @classmethod
    def from_para_with_footnotes(
        cls,
        paragraph: Paragraph,
        markdown_footnotes: MarkdownFootnotes | None = None,
        fn_prefix: str = "",
        fn_start: int = 1,
    ) -> AnnotatedPara:
        """
        Create an AnnotatedParagraph from an existing Paragraph, automatically
        extracting footnote references and creating annotations from them.

        Always preserves original footnote IDs. Footnote IDs are normalized
        to always include the ^ prefix when looking them up.

        Args:
            paragraph: The paragraph to annotate
            markdown_footnotes: Optional MarkdownFootnotes object containing footnote definitions
            fn_prefix: Prefix for footnote IDs (currently unused - preserves original IDs)
            fn_start: Starting number for footnotes (currently unused - preserves original IDs)

        Returns:
            AnnotatedPara with footnotes extracted as annotations
        """
        ann_para = cls(paragraph=paragraph, annotations={}, fn_prefix=fn_prefix, fn_start=fn_start)

        if markdown_footnotes is None:
            return ann_para

        # Extract footnote references from the paragraph
        footnote_refs = ann_para.extract_footnote_references()

        # Add annotations for each unique footnote reference
        # We preserve original IDs and track which we've already processed
        processed_footnotes: set[str] = set()
        for ref in footnote_refs:
            if ref.footnote_id not in processed_footnotes:
                processed_footnotes.add(ref.footnote_id)

                # Look up the footnote content (ref.footnote_id already has ^)
                footnote_info = markdown_footnotes.get(ref.footnote_id)
                if footnote_info:
                    # Add the footnote content as an annotation for this sentence
                    # Store with ID including ^ and content
                    ann_para.add_annotation_with_id(
                        ref.sentence_index, ref.footnote_id, footnote_info.content
                    )

        return ann_para

    def extract_footnote_references(self) -> list[FootnoteReference]:
        """
        Extract all footnote references from the paragraph text.

        Handles footnotes that appear:
        - Within sentences
        - After sentences (attached to preceding sentence)
        - At the beginning of paragraphs (attached to first sentence)
        - As the only content (treated as first sentence)

        Returns:
            List of FootnoteReference objects with positions and IDs
        """
        footnote_refs: list[FootnoteReference] = []

        # Process each sentence individually
        for sent_index, sentence in enumerate(self.paragraph.sentences):
            # Find all footnote references in this sentence
            for match in _FOOTNOTE_REF_PATTERN.finditer(sentence.text):
                footnote_id = match.group(1)

                # Check if footnote is at the very beginning of this sentence
                # If so, and this isn't the first sentence, attach to previous
                if match.start() == 0 and sent_index > 0:
                    # Footnote at start of sentence should be attached to previous
                    target_sentence_index = sent_index - 1
                    # Calculate position as if it were at the end of previous sentence
                    prev_sent_len = len(self.paragraph.sentences[target_sentence_index].text)
                    span = TextSpan(
                        start=prev_sent_len,  # Position at end of previous sentence
                        end=prev_sent_len + len(match.group(0)),
                        text=match.group(0),
                    )
                else:
                    # Footnote belongs to current sentence
                    target_sentence_index = sent_index
                    span = TextSpan(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(0),
                    )

                footnote_refs.append(
                    FootnoteReference(
                        footnote_id=f"^{footnote_id}",
                        span=span,
                        sentence_index=target_sentence_index,
                    )
                )

        return footnote_refs

    def add_annotation(self, sentence_index: int, annotation: str) -> None:
        """Add an annotation to a specific sentence with auto-generated ID."""
        # Generate a sequential ID for manual annotations
        next_id = str(self.fn_start + self.annotation_count())
        self.add_annotation_with_id(sentence_index, next_id, annotation)

    def add_annotation_with_id(
        self, sentence_index: int, footnote_id: str, annotation: str
    ) -> None:
        """Add an annotation to a specific sentence with a specific footnote ID."""
        if sentence_index not in self.annotations:
            self.annotations[sentence_index] = []
        # Always normalize the ID to include ^
        normalized_id = _normalize_footnote_id(footnote_id)
        self.annotations[sentence_index].append(Footnote(id=normalized_id, content=annotation))

    def as_markdown_footnotes(self) -> str:
        """
        Reassemble the paragraph with annotations rendered as markdown footnotes.

        Each sentence with annotations gets footnote references appended,
        and footnotes are listed at the end of the paragraph.
        """
        if not self.annotations:
            return self.paragraph.reassemble()

        # Build footnote references and definitions
        footnote_refs: dict[int, list[str]] = {}  # sentence_index -> list of footnote IDs
        footnotes: list[str] = []  # list of footnote texts

        # Process annotations preserving original IDs
        for sentence_index in sorted(self.annotations.keys()):
            footnote_refs[sentence_index] = []
            for footnote in self.annotations[sentence_index]:
                # Use the preserved footnote ID with prefix if needed
                # footnote.id already has ^, so strip it before adding prefix
                base_id = footnote.id[1:] if footnote.id.startswith("^") else footnote.id
                full_id = _normalize_footnote_id(
                    f"{self.fn_prefix}{base_id}" if self.fn_prefix else footnote.id
                )
                footnote_refs[sentence_index].append(full_id)
                footnotes.append(f"[{full_id}]: {footnote.content}")

        # Build the paragraph with footnote references
        annotated_sentences: list[str] = []
        for i, sentence in enumerate(self.paragraph.sentences):
            sentence_text = sentence.text
            if i in footnote_refs:
                # Add footnote references to this sentence (IDs already have ^)
                refs = "".join(f"[{fid}]" for fid in footnote_refs[i])
                sentence_text = sentence_text.rstrip() + refs
            annotated_sentences.append(sentence_text)

        # Combine sentences and add footnotes at the end
        paragraph_text = " ".join(annotated_sentences)
        if footnotes:
            paragraph_text += "\n\n" + "\n\n".join(footnotes)

        return paragraph_text

    def has_annotations(self) -> bool:
        """Check if this paragraph has any annotations."""
        return bool(self.annotations)

    def annotation_count(self) -> int:
        """Get the total number of annotations across all sentences."""
        return sum(len(annotations) for annotations in self.annotations.values())

    def get_sentence_annotations(self, sentence_index: int) -> list[str]:
        """Get all annotation texts for a specific sentence."""
        # Return just the annotation texts, not the IDs
        return [footnote.content for footnote in self.annotations.get(sentence_index, [])]

    def get_sentence_annotations_with_ids(self, sentence_index: int) -> list[Footnote]:
        """Get all annotations as Footnote objects for a specific sentence."""
        return self.annotations.get(sentence_index, [])

    def clear_annotations_for_sentence(self, sentence_index: int) -> None:
        """Remove all annotations for a specific sentence."""
        if sentence_index in self.annotations:
            del self.annotations[sentence_index]

    def footnote_id(self, index: int) -> FootnoteId:
        """Get the footnote id for a specific annotation."""
        base_id = str(index)
        full_id = f"{self.fn_prefix}{base_id}" if self.fn_prefix else base_id
        return check_fn_id(_normalize_footnote_id(full_id))

    def next_footnote_number(self) -> int:
        """Get the next footnote number after all current annotations."""
        return self.fn_start + self.annotation_count()


@dataclass
class AnnotatedDoc:
    """
    A document with annotations that can be rendered with consolidated footnotes.

    Wraps a TextDoc and stores annotations indexed by SentIndex, avoiding
    duplication of document structure.
    """

    text_doc: TextDoc

    annotations: dict[SentIndex, list[str]]
    """Mapping from sentence index to list of annotations for that sentence."""

    footnote_mapping: dict[FootnoteId, str]
    """Mapping from footnote ID to annotation text."""

    @classmethod
    def from_text_doc(cls, text_doc: TextDoc) -> AnnotatedDoc:
        """Create an AnnotatedDoc from a TextDoc with no annotations."""
        return cls(text_doc=text_doc, annotations={}, footnote_mapping={})

    @staticmethod
    def consolidate_annotations(ann_paras: list[AnnotatedPara]) -> AnnotatedDoc:
        """
        Consolidate a list of AnnotatedPara objects into an AnnotatedDoc.

        Handles footnote ID uniquing by tracking used IDs per prefix and
        renumbering as needed to ensure all footnote IDs are unique.
        """
        if not ann_paras:
            return AnnotatedDoc(text_doc=TextDoc([]), annotations={}, footnote_mapping={})

        footnote_mapping: dict[FootnoteId, str] = {}
        annotations: dict[SentIndex, list[str]] = {}

        # Build TextDoc from paragraphs
        paragraphs = [ann_para.paragraph for ann_para in ann_paras]
        text_doc = TextDoc(paragraphs)

        # Process annotations preserving original IDs
        for para_index, ann_para in enumerate(ann_paras):
            if not ann_para.has_annotations():
                continue

            prefix = ann_para.fn_prefix

            # Process each sentence's annotations
            for sentence_index, sentence_annotations in ann_para.annotations.items():
                sent_index = SentIndex(para_index, sentence_index)

                for footnote in sentence_annotations:
                    # Use the original footnote ID with prefix if provided
                    # footnote.id already has ^, so strip it before adding prefix
                    base_id = footnote.id.lstrip("^")
                    full_footnote_id = _normalize_footnote_id(
                        f"{prefix}{base_id}" if prefix else footnote.id
                    )

                    # Create validated footnote ID
                    footnote_id = check_fn_id(full_footnote_id)

                    # Check if this ID is already used and make unique if necessary
                    base_id = full_footnote_id
                    counter = 1
                    while footnote_id in footnote_mapping:
                        # If the same ID appears multiple times, append a counter
                        full_footnote_id = f"{base_id}_{counter}"
                        footnote_id = check_fn_id(full_footnote_id)
                        counter += 1

                    # Store the mapping
                    footnote_mapping[footnote_id] = footnote.content

                    # Store annotation by SentIndex
                    if sent_index not in annotations:
                        annotations[sent_index] = []
                    annotations[sent_index].append(footnote.content)

        return AnnotatedDoc(
            text_doc=text_doc, annotations=annotations, footnote_mapping=footnote_mapping
        )

    def as_markdown_with_footnotes(
        self,
        footnote_header: str | None = None,
    ) -> str:
        """
        Render the entire document as markdown with consolidated footnotes.

        Each paragraph is rendered with its footnote references, and all
        footnotes are consolidated at the end of the document.
        """
        if not self.annotations:
            return self.text_doc.reassemble()

        # Render each paragraph with its annotations
        para_texts = []
        for para_index, paragraph in enumerate(self.text_doc.paragraphs):
            # Build footnote references for this paragraph
            para_footnote_refs: dict[int, list[FootnoteId]] = {}

            # Collect footnote IDs for sentences in this paragraph
            for sentence_index, sentence in enumerate(paragraph.sentences):
                sent_index = SentIndex(para_index, sentence_index)
                if sent_index in self.annotations:
                    para_footnote_refs[sentence_index] = []
                    # Find the corresponding footnote IDs for this sentence's annotations
                    sentence_annotations = self.annotations[sent_index]
                    for annotation in sentence_annotations:
                        # Find the footnote ID for this annotation
                        for footnote_id, stored_annotation in self.footnote_mapping.items():
                            if stored_annotation == annotation:
                                para_footnote_refs[sentence_index].append(footnote_id)
                                break

            # Build the paragraph text with footnote references
            annotated_sentences: list[str] = []
            for sentence_index, sentence in enumerate(paragraph.sentences):
                sentence_text = sentence.text
                if sentence_index in para_footnote_refs:
                    # Add footnote references to this sentence (IDs already have ^)
                    refs = "".join(
                        f"[{footnote_id}]" for footnote_id in para_footnote_refs[sentence_index]
                    )
                    sentence_text = sentence_text.rstrip() + refs
                annotated_sentences.append(sentence_text)

            para_texts.append(" ".join(annotated_sentences))

        # Build output
        if self.footnote_mapping:
            # Preserve insertion order of footnotes (IDs already have ^)
            footnote_lines = [
                f"[{footnote_id}]: {self.footnote_mapping[footnote_id]}"
                for footnote_id in self.footnote_mapping.keys()
            ]

            # Append optional header before footnotes
            if footnote_header and footnote_header.strip():
                para_texts.append(footnote_header.strip())

            para_texts.extend(footnote_lines)

        # Join all sections as separate Markdown paragraphs
        return "\n\n".join(para_texts)

    def add_annotation(self, sent_index: SentIndex, annotation: str, fn_prefix: str = "") -> None:
        """Add an annotation to a specific sentence."""
        # Validate SentIndex is within bounds
        if sent_index.para_index >= len(self.text_doc.paragraphs):
            raise IndexError(f"Paragraph index {sent_index.para_index} out of range")
        para = self.text_doc.paragraphs[sent_index.para_index]
        if sent_index.sent_index >= len(para.sentences):
            raise IndexError(
                f"Sentence index {sent_index.sent_index} out of range in paragraph {sent_index.para_index}"
            )

        # Add annotation to sentence
        if sent_index not in self.annotations:
            self.annotations[sent_index] = []
        self.annotations[sent_index].append(annotation)

        # Update footnote mapping: find next available footnote number
        # Handle IDs that have ^ prefix
        prefix_with_caret = f"^{fn_prefix}"
        used_nums = set()
        for fid in self.footnote_mapping.keys():
            if fid.startswith(prefix_with_caret):
                num_part = fid[len(prefix_with_caret) :]
                if num_part.isdigit():
                    used_nums.add(int(num_part))

        next_num = 1
        while next_num in used_nums:
            next_num += 1

        # Add prefix if provided and normalize
        full_id = f"{fn_prefix}{next_num}"
        footnote_id = check_fn_id(_normalize_footnote_id(full_id))
        self.footnote_mapping[footnote_id] = annotation

    def get_sentence_annotations(self, sent_index: SentIndex) -> list[str]:
        """Get all annotations for a specific sentence."""
        return self.annotations.get(sent_index, [])

    def clear_annotations_for_sentence(self, sent_index: SentIndex) -> None:
        """Remove all annotations for a specific sentence."""
        if sent_index in self.annotations:
            # Save the annotations before deleting them
            annotations_to_remove = self.annotations[sent_index]
            del self.annotations[sent_index]

            # Also remove from footnote mapping
            footnote_ids_to_remove = []
            for footnote_id, annotation in self.footnote_mapping.items():
                if annotation in annotations_to_remove:
                    footnote_ids_to_remove.append(footnote_id)
            for footnote_id in footnote_ids_to_remove:
                del self.footnote_mapping[footnote_id]

    def total_annotation_count(self) -> int:
        """Get the total number of annotations across all sentences."""
        return sum(len(sentence_annotations) for sentence_annotations in self.annotations.values())

    def has_annotations(self) -> bool:
        """Check if this document has any annotations."""
        return bool(self.annotations)


def map_notes_with_embeddings(
    paragraph: Paragraph, notes: list[str], fn_prefix: str = "", fn_start: int = 1
) -> AnnotatedPara:
    """
    Map research notes to sentences using embedding-based similarity.
    Each note is mapped to exactly one best-fitting sentence.

    Args:
        paragraph: The paragraph to annotate
        notes: List of annotation strings
        fn_prefix: Prefix for footnote IDs
        fn_start: Starting number for footnotes

    Returns:
        AnnotatedParagraph with notes mapped to most similar sentences
    """
    from kash.embeddings.embeddings import EmbValue, KeyVal
    from kash.kits.docs.concepts.similarity_cache import create_similarity_cache

    # Filter out empty notes and "(No results)" placeholder
    filtered_notes = [
        note.strip() for note in notes if note.strip() and note.strip() != "(No results)"
    ]

    annotated_para = AnnotatedPara.from_para(paragraph, fn_prefix=fn_prefix, fn_start=fn_start)

    if not filtered_notes:
        return annotated_para

    # Get sentence texts from paragraph
    sentence_texts = [sent.text for sent in paragraph.sentences if sent.text.strip()]
    if not sentence_texts:
        return annotated_para

    # Create similarity cache with all sentences and notes
    sentence_keyvals = [
        KeyVal(f"sent_{i}", EmbValue(text)) for i, text in enumerate(sentence_texts)
    ]
    note_keyvals = [KeyVal(f"note_{i}", EmbValue(note)) for i, note in enumerate(filtered_notes)]

    all_keyvals = sentence_keyvals + note_keyvals
    similarity_cache = create_similarity_cache(all_keyvals)

    # Find most related sentence for each note (each note maps to exactly one sentence)
    sentence_keys = [f"sent_{i}" for i in range(len(sentence_texts))]

    for note_idx, note in enumerate(filtered_notes):
        note_key = f"note_{note_idx}"

        # Find the most similar sentence for this note
        most_similar = similarity_cache.most_similar(note_key, n=1, candidates=sentence_keys)

        if most_similar:
            best_sentence_key, _ = most_similar[0]
            best_sentence_idx = int(best_sentence_key.split("_")[1])
            annotated_para.add_annotation(best_sentence_idx, note)

    return annotated_para


## Tests


@enable_if("online")
def test_map_notes_with_embeddings() -> None:
    para = Paragraph.from_text("Python is great for AI. Java is verbose but reliable.")
    notes = ["Python is popular for machine learning", "Java enterprise applications"]

    annotated = map_notes_with_embeddings(para, notes)

    assert annotated.annotation_count() == 2
    # Each note should map to exactly one sentence
    total_annotations = sum(
        len(annotated.get_sentence_annotations(i)) for i in range(len(para.sentences))
    )
    assert total_annotations == 2


def test_annotated_paragraph_basic() -> None:
    para = Paragraph.from_text("First sentence. Second sentence. Third sentence.")
    annotated = AnnotatedPara.from_para(para)

    # Test basic functionality
    assert not annotated.has_annotations()
    assert annotated.annotation_count() == 0
    assert annotated.as_markdown_footnotes() == para.reassemble()

    # Add annotations
    annotated.add_annotation(0, "Note about first sentence")
    annotated.add_annotation(1, "Note about second sentence")
    annotated.add_annotation(1, "Another note about second sentence")

    assert annotated.has_annotations()
    assert annotated.annotation_count() == 3
    assert len(annotated.get_sentence_annotations(0)) == 1
    assert len(annotated.get_sentence_annotations(1)) == 2
    assert len(annotated.get_sentence_annotations(2)) == 0


def test_markdown_footnotes() -> None:
    para = Paragraph.from_text("First sentence. Second sentence.")
    annotated = AnnotatedPara.from_para(para)

    annotated.add_annotation(0, "First note")
    annotated.add_annotation(1, "Second note")
    annotated.add_annotation(1, "Third note")

    result = annotated.as_markdown_footnotes()

    # Should contain footnote references (with ^)
    assert "[^1]" in result
    assert "[^2]" in result
    assert "[^3]" in result

    # Should contain footnote definitions (with ^)
    assert "[^1]: First note" in result
    assert "[^2]: Second note" in result
    assert "[^3]: Third note" in result

    # Footnotes should be at the end
    lines = result.split("\n")
    footnote_lines = [line for line in lines if line.startswith("[^")]
    assert len(footnote_lines) == 3


def test_annotated_doc_basic() -> None:
    """Test basic AnnotatedDoc functionality."""
    text_doc = TextDoc.from_text("First paragraph.\n\nSecond paragraph.")
    ann_doc = AnnotatedDoc.from_text_doc(text_doc)

    assert len(ann_doc.text_doc.paragraphs) == 2
    assert not ann_doc.has_annotations()
    assert ann_doc.total_annotation_count() == 0
    assert ann_doc.as_markdown_with_footnotes() == text_doc.reassemble()


def test_annotated_doc_add_annotation() -> None:
    """Test adding annotations to AnnotatedDoc."""
    text_doc = TextDoc.from_text("First paragraph.\n\nSecond paragraph.")
    ann_doc = AnnotatedDoc.from_text_doc(text_doc)

    # Add annotations using SentIndex
    ann_doc.add_annotation(SentIndex(0, 0), "Note about first paragraph")
    ann_doc.add_annotation(SentIndex(1, 0), "Note about second paragraph")

    assert ann_doc.has_annotations()
    assert ann_doc.total_annotation_count() == 2
    assert len(ann_doc.footnote_mapping) == 2


def test_consolidate_ann_paras_basic() -> None:
    """Test basic consolidation of annotated paragraphs."""
    para1 = Paragraph.from_text("First paragraph.")
    para2 = Paragraph.from_text("Second paragraph.")

    ann_para1 = AnnotatedPara.from_para(para1)
    ann_para1.add_annotation(0, "Note 1")

    ann_para2 = AnnotatedPara.from_para(para2)
    ann_para2.add_annotation(0, "Note 2")

    ann_doc = AnnotatedDoc.consolidate_annotations([ann_para1, ann_para2])

    assert len(ann_doc.text_doc.paragraphs) == 2
    assert ann_doc.total_annotation_count() == 2
    assert len(ann_doc.footnote_mapping) == 2

    # Check annotations are stored by SentIndex
    assert SentIndex(0, 0) in ann_doc.annotations
    assert SentIndex(1, 0) in ann_doc.annotations


def test_consolidate_ann_paras_with_prefixes() -> None:
    """Test consolidation with different footnote prefixes."""
    para1 = Paragraph.from_text("First paragraph.")
    para2 = Paragraph.from_text("Second paragraph.")
    para3 = Paragraph.from_text("Third paragraph.")

    # Different prefixes
    ann_para1 = AnnotatedPara.from_para(para1, fn_prefix="a", fn_start=1)
    ann_para1.add_annotation(0, "Note A1")
    ann_para1.add_annotation(0, "Note A2")

    ann_para2 = AnnotatedPara.from_para(para2, fn_prefix="b", fn_start=1)
    ann_para2.add_annotation(0, "Note B1")

    ann_para3 = AnnotatedPara.from_para(para3, fn_prefix="a", fn_start=1)
    ann_para3.add_annotation(0, "Note A3")

    ann_doc = AnnotatedDoc.consolidate_annotations([ann_para1, ann_para2, ann_para3])

    assert len(ann_doc.text_doc.paragraphs) == 3
    assert ann_doc.total_annotation_count() == 4
    assert len(ann_doc.footnote_mapping) == 4

    # Check that we have the expected footnote IDs (all now start with ^)
    footnote_ids = set(ann_doc.footnote_mapping.keys())
    a_ids = [fid for fid in footnote_ids if fid.startswith("^a")]
    b_ids = [fid for fid in footnote_ids if fid.startswith("^b")]

    assert len(a_ids) == 3  # Three 'a' prefixed annotations
    assert len(b_ids) == 1  # One 'b' prefixed annotation


def test_consolidate_ann_paras_uniquing() -> None:
    """Test footnote ID uniquing when there are conflicts."""
    para1 = Paragraph.from_text("First paragraph.")
    para2 = Paragraph.from_text("Second paragraph.")

    # Both start with same prefix and fn_start
    ann_para1 = AnnotatedPara.from_para(para1, fn_prefix="", fn_start=1)
    ann_para1.add_annotation(0, "Note 1")
    ann_para1.add_annotation(0, "Note 2")

    ann_para2 = AnnotatedPara.from_para(para2, fn_prefix="", fn_start=1)
    ann_para2.add_annotation(0, "Note 3")
    ann_para2.add_annotation(0, "Note 4")

    ann_doc = AnnotatedDoc.consolidate_annotations([ann_para1, ann_para2])

    assert ann_doc.total_annotation_count() == 4
    assert len(ann_doc.footnote_mapping) == 4

    # All footnote IDs should be unique
    footnote_ids = list(ann_doc.footnote_mapping.keys())
    assert len(footnote_ids) == len(set(footnote_ids))


def test_consolidate_empty_list() -> None:
    """Test consolidation of empty list."""
    ann_doc = AnnotatedDoc.consolidate_annotations([])

    assert len(ann_doc.text_doc.paragraphs) == 0
    assert ann_doc.total_annotation_count() == 0
    assert len(ann_doc.footnote_mapping) == 0
    assert not ann_doc.has_annotations()


def test_consolidate_ann_paras_no_annotations() -> None:
    """Test consolidation of paragraphs with no annotations."""
    para1 = Paragraph.from_text("First paragraph.")
    para2 = Paragraph.from_text("Second paragraph.")

    ann_para1 = AnnotatedPara.from_para(para1)
    ann_para2 = AnnotatedPara.from_para(para2)

    ann_doc = AnnotatedDoc.consolidate_annotations([ann_para1, ann_para2])

    assert len(ann_doc.text_doc.paragraphs) == 2
    assert ann_doc.total_annotation_count() == 0
    assert len(ann_doc.footnote_mapping) == 0
    assert not ann_doc.has_annotations()


def test_markdown_with_footnotes_consolidated() -> None:
    """Test markdown rendering with consolidated footnotes."""
    para1 = Paragraph.from_text("First paragraph.")
    para2 = Paragraph.from_text("Second paragraph.")

    ann_para1 = AnnotatedPara.from_para(para1, fn_prefix="ref", fn_start=1)
    ann_para1.add_annotation(0, "Reference 1")

    ann_para2 = AnnotatedPara.from_para(para2, fn_prefix="ref", fn_start=1)
    ann_para2.add_annotation(0, "Reference 2")

    ann_doc = AnnotatedDoc.consolidate_annotations([ann_para1, ann_para2])
    result = ann_doc.as_markdown_with_footnotes()

    # Should contain footnote references in text
    assert "[^ref" in result

    # Should contain footnote definitions
    assert "]: Reference 1" in result
    assert "]: Reference 2" in result

    # Should have paragraph separation
    lines = result.split("\n")
    assert len([line for line in lines if line.strip()]) >= 4  # 2 paras + 2 footnotes


def test_sentence_index_operations() -> None:
    """Test operations using SentIndex directly."""
    text_doc = TextDoc.from_text(
        "First sentence of first para. Second sentence.\n\nFirst sentence of second para."
    )
    ann_doc = AnnotatedDoc.from_text_doc(text_doc)

    # Add annotations using SentIndex
    ann_doc.add_annotation(SentIndex(0, 0), "Note on first sentence of first para")
    ann_doc.add_annotation(SentIndex(0, 1), "Note on second sentence of first para")
    ann_doc.add_annotation(SentIndex(1, 0), "Note on first sentence of second para")

    assert ann_doc.total_annotation_count() == 3

    # Check specific sentence annotations
    assert len(ann_doc.get_sentence_annotations(SentIndex(0, 0))) == 1
    assert len(ann_doc.get_sentence_annotations(SentIndex(0, 1))) == 1
    assert len(ann_doc.get_sentence_annotations(SentIndex(1, 0))) == 1
    assert len(ann_doc.get_sentence_annotations(SentIndex(0, 2))) == 0  # Non-existent sentence

    # Test clearing annotations
    ann_doc.clear_annotations_for_sentence(SentIndex(0, 1))
    assert ann_doc.total_annotation_count() == 2
    assert len(ann_doc.get_sentence_annotations(SentIndex(0, 1))) == 0


def test_footnote_id_validation() -> None:
    """Test footnote ID validation function."""
    # Valid IDs (normalized to include ^)
    assert check_fn_id("abc123") == FootnoteId("^abc123")
    assert check_fn_id("ref_1") == FootnoteId("^ref_1")
    assert check_fn_id("note-1") == FootnoteId("^note-1")
    assert check_fn_id("fn.1") == FootnoteId("^fn.1")
    assert check_fn_id("αβγ123") == FootnoteId("^αβγ123")  # Unicode letters
    assert check_fn_id("中文1") == FootnoteId("^中文1")  # Chinese characters

    # IDs that already have ^ should stay the same
    assert check_fn_id("^abc123") == FootnoteId("^abc123")
    assert check_fn_id("^ref_1") == FootnoteId("^ref_1")

    # Valid ID with exactly 20 characters (plus ^)
    assert check_fn_id("a" * 20) == FootnoteId("^" + "a" * 20)

    # Invalid IDs - invalid characters
    try:
        check_fn_id("invalid@char")
        raise AssertionError("Expected ValueError for invalid character @")
    except ValueError as e:
        assert "word chars, period, or hyphen" in str(e)

    try:
        check_fn_id("invalid space")
        raise AssertionError("Expected ValueError for invalid character space")
    except ValueError as e:
        assert "word chars, period, or hyphen" in str(e)

    try:
        check_fn_id("invalid#char")
        raise AssertionError("Expected ValueError for invalid character #")
    except ValueError as e:
        assert "word chars, period, or hyphen" in str(e)


def test_annotated_para_footnote_id_validation() -> None:
    """Test that AnnotatedPara validates footnote IDs."""
    para = Paragraph.from_text("Test sentence.")

    # Valid prefix
    annotated = AnnotatedPara.from_para(para, fn_prefix="ref_", fn_start=1)
    footnote_id = annotated.footnote_id(1)
    assert footnote_id == FootnoteId("^ref_1")

    # Invalid prefix should raise error when creating footnote ID
    annotated_invalid = AnnotatedPara.from_para(para, fn_prefix="invalid@prefix", fn_start=1)
    try:
        annotated_invalid.footnote_id(1)
        raise AssertionError("Expected ValueError for invalid footnote prefix")
    except ValueError:
        pass  # Expected


## Tests for footnote header


def test_markdown_with_footnotes_header() -> None:
    """Ensure footnote_header is inserted correctly above consolidated footnotes."""
    para = Paragraph.from_text("Some text.")
    ann_para = AnnotatedPara.from_para(para, fn_prefix="ref", fn_start=1)
    ann_para.add_annotation(0, "Reference note")

    ann_doc = AnnotatedDoc.consolidate_annotations([ann_para])
    header_text = "## Footnotes"
    result = ann_doc.as_markdown_with_footnotes(footnote_header=header_text)

    # Header should appear exactly once and above footnotes
    assert header_text in result
    header_index = result.find(header_text)
    footnote_index = result.find("[^ref1]:")
    assert 0 <= header_index < footnote_index, "Header must precede footnotes"


def test_markdown_footnote_order() -> None:
    """Ensure footnotes retain order of appearance, not lexicographic order."""
    para1 = Paragraph.from_text("P1.")
    para2 = Paragraph.from_text("P2.")
    para3 = Paragraph.from_text("P3.")

    ann_para1 = AnnotatedPara.from_para(para1, fn_prefix="a", fn_start=1)
    ann_para1.add_annotation(0, "Note A1")  # a1

    ann_para2 = AnnotatedPara.from_para(para2, fn_prefix="b", fn_start=1)
    ann_para2.add_annotation(0, "Note B1")  # b1

    ann_para3 = AnnotatedPara.from_para(
        para3, fn_prefix="a", fn_start=2
    )  # Start at 2 to avoid conflict
    ann_para3.add_annotation(0, "Note A2")  # a2

    ann_doc = AnnotatedDoc.consolidate_annotations([ann_para1, ann_para2, ann_para3])
    output = ann_doc.as_markdown_with_footnotes()

    # Extract footnote IDs in output order
    lines = [line.strip() for line in output.split("\n") if line.startswith("[^")]
    ids_in_output = [line.split(":")[0][2:-1] for line in lines]  # remove "[^" and "]"

    assert ids_in_output == ["a1", "b1", "a2"], ids_in_output
