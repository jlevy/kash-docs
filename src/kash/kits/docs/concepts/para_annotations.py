from __future__ import annotations

from dataclasses import dataclass

from chopdiff.docs.text_doc import Paragraph

from kash.utils.common.testing import enable_if


@dataclass
class AnnotatedParagraph:
    """
    A paragraph with annotations that can be rendered as markdown footnotes.

    Wraps a `Paragraph` from chopdiff and adds annotation functionality.
    Annotations are stored as a mapping from sentence indices to lists of annotations.
    """

    paragraph: Paragraph
    annotations: dict[int, list[str]]
    fn_prefix: str = ""
    fn_start: int = 1

    @classmethod
    def from_paragraph(
        cls, paragraph: Paragraph, fn_prefix: str = "", fn_start: int = 1
    ) -> AnnotatedParagraph:
        """Create an AnnotatedParagraph from an existing Paragraph."""
        return cls(paragraph=paragraph, annotations={}, fn_prefix=fn_prefix, fn_start=fn_start)

    def add_annotation(self, sentence_index: int, annotation: str) -> None:
        """Add an annotation to a specific sentence."""
        if sentence_index not in self.annotations:
            self.annotations[sentence_index] = []
        self.annotations[sentence_index].append(annotation)

    def as_markdown_footnotes(self) -> str:
        """
        Reassemble the paragraph with annotations rendered as markdown footnotes.

        Each sentence with annotations gets footnote references appended,
        and footnotes are listed at the end of the paragraph.
        """
        if not self.annotations:
            return self.paragraph.reassemble()

        # Build footnote counter
        footnote_counter = self.fn_start
        footnote_refs: dict[int, list[int]] = {}  # sentence_index -> list of footnote numbers
        footnotes: list[str] = []  # list of footnote texts

        # Assign footnote numbers to each annotation
        for sentence_index in sorted(self.annotations.keys()):
            footnote_refs[sentence_index] = []
            for annotation in self.annotations[sentence_index]:
                footnote_refs[sentence_index].append(footnote_counter)
                footnotes.append(f"[^{self.fn_prefix}{footnote_counter}]: {annotation}")
                footnote_counter += 1

        # Build the paragraph with footnote references
        annotated_sentences: list[str] = []
        for i, sentence in enumerate(self.paragraph.sentences):
            sentence_text = sentence.text
            if i in footnote_refs:
                # Add footnote references to this sentence
                refs = "".join(f"[^{self.fn_prefix}{num}]" for num in footnote_refs[i])
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
        """Get all annotations for a specific sentence."""
        return self.annotations.get(sentence_index, [])

    def clear_annotations_for_sentence(self, sentence_index: int) -> None:
        """Remove all annotations for a specific sentence."""
        if sentence_index in self.annotations:
            del self.annotations[sentence_index]

    def next_footnote_number(self) -> int:
        """Get the next footnote number after all current annotations."""
        return self.fn_start + self.annotation_count()


def map_notes_with_embeddings(
    paragraph: Paragraph, notes: list[str], fn_prefix: str = "", fn_start: int = 1
) -> AnnotatedParagraph:
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
    from kash.kits.docs.concepts.similarity_cache import create_similarity_cache

    # Filter out empty notes and "(No results)" placeholder
    filtered_notes = [
        note.strip() for note in notes if note.strip() and note.strip() != "(No results)"
    ]

    annotated_para = AnnotatedParagraph.from_paragraph(
        paragraph, fn_prefix=fn_prefix, fn_start=fn_start
    )

    if not filtered_notes:
        return annotated_para

    # Get sentence texts from paragraph
    sentence_texts = [sent.text for sent in paragraph.sentences if sent.text.strip()]
    if not sentence_texts:
        return annotated_para

    # Create similarity cache with all sentences and notes
    sentence_keyvals = [(f"sent_{i}", text) for i, text in enumerate(sentence_texts)]
    note_keyvals = [(f"note_{i}", note) for i, note in enumerate(filtered_notes)]

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


def test_annotated_paragraph_basic() -> None:
    para = Paragraph.from_text("First sentence. Second sentence. Third sentence.")
    annotated = AnnotatedParagraph.from_paragraph(para)

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
    annotated = AnnotatedParagraph.from_paragraph(para)

    annotated.add_annotation(0, "First note")
    annotated.add_annotation(1, "Second note")
    annotated.add_annotation(1, "Third note")

    result = annotated.as_markdown_footnotes()

    # Should contain footnote references
    assert "[^1]" in result
    assert "[^2]" in result
    assert "[^3]" in result

    # Should contain footnote definitions
    assert "[^1]: First note" in result
    assert "[^2]: Second note" in result
    assert "[^3]: Third note" in result

    # Footnotes should be at the end
    lines = result.split("\n")
    footnote_lines = [line for line in lines if line.startswith("[^")]
    assert len(footnote_lines) == 3


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
