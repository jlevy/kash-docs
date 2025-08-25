from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from chopdiff.docs.text_doc import Paragraph, SentIndex, TextDoc

from kash.kits.docs.analysis.analysis_types import Footnote, FootnoteId, RefId, TextSpan
from kash.utils.common.testing import enable_if
from kash.utils.common.url import Url
from kash.utils.text_handling.markdown_footnotes import MarkdownFootnotes
from kash.utils.text_handling.markdown_utils import extract_urls
from kash.web_content.canon_url import canonicalize_url

# Valid footnote ID pattern: Unicode word characters (letters, digits, underscore), period, or hyphen
_FOOTNOTE_ID_PATTERN = re.compile(r"^[\w.-]+$")

# Pattern to match footnote references in text: [^id] where id contains no whitespace
# Matches: [^1], [^foo], [^note-1], [^ref.2]
# Does not match: [^ spaces], [^], [^foo bar]
_FOOTNOTE_REF_PATTERN = re.compile(r"\[\^([\w.-]+)\]")


def _normalize_footnote_id(footnote_id: str) -> FootnoteId:
    """
    Normalize a footnote ID to always include the ^ prefix.
    """
    return FootnoteId(footnote_id if footnote_id.startswith("^") else f"^{footnote_id}")


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


def increment_id(id_str: str) -> tuple[str, int]:
    """
    Increment the trailing number in an ID by 1.
    Only matches the trailing digits; everything before is treated as the base:
      "^res2" -> ("^res3", 3)
      "res.2"  -> ("res.3", 3)
      "^ref"  -> ("^ref1", 1)
      "ref"   -> ("ref1", 1)
    """
    match = re.match(r"^(.*?)(\d+)$", id_str)
    if match:
        base, num_str = match.groups()
        next_num = int(num_str) + 1
        return f"{base}{next_num}", next_num
    else:
        return f"{id_str}1", 1


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
    def unannotated(
        cls, paragraph: Paragraph, *, fn_prefix: str = "", fn_start: int = 1
    ) -> AnnotatedPara:
        """Create an AnnotatedParagraph from an existing Paragraph with no annotations."""
        return cls(paragraph=paragraph, annotations={}, fn_prefix=fn_prefix, fn_start=fn_start)

    @classmethod
    def from_para_with_footnotes(
        cls,
        paragraph: Paragraph,
        markdown_footnotes: MarkdownFootnotes | None,
        *,
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

                # Skip footnote definitions like "[^id]: content" (not references)
                if match.end() < len(sentence.text) and sentence.text[match.end()] == ":":
                    continue

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

    def get_urls(self) -> dict[Url, RefId | None]:
        """
        Get all  URLs in this paragraph, including links in the text as well as in footnotes.
        """
        # Get literal URLs from the paragraph.
        para_urls = set(canonicalize_url(url) for url in extract_urls(self.paragraph.reassemble()))
        url_map: dict[Url, RefId | None] = {url: None for url in para_urls}

        # Get footnotes and add them along with their
        for _sent_index, footnotes in self.annotations.items():
            for footnote in footnotes:
                for url in footnote.urls:
                    url_map[url] = FootnoteId(footnote.id)

        return url_map


@dataclass
class AnnotatedDoc:
    """
    A document with annotations that can be rendered with consolidated footnotes.

    Wraps a TextDoc and stores annotations indexed by SentIndex. Also preserves
    the original list of AnnotatedPara in full order so callers can choose to
    include or exclude footnote-definition paragraphs at usage time.

    Currently all annotations are footnotes but we may want to extend this to
    support links and highlights or other types of annotations.
    """

    text_doc: TextDoc
    """The original text doc, including footnote definitions."""

    ann_paras: list[AnnotatedPara]
    """Annotated paragraphs preserving the exact original order."""

    annotations: dict[SentIndex, list[FootnoteId]]
    """Mapping from sentence index to list of FootnoteIds for that sentence."""

    footnote_mapping: dict[FootnoteId, Footnote]
    """Mapping from footnote ID to Footnote (content and helpers)."""

    @classmethod
    def plain_doc(cls, text_doc: TextDoc) -> AnnotatedDoc:
        """
        Create an AnnotatedDoc from a TextDoc with no annotations.
        """
        ann_paras = [AnnotatedPara.unannotated(p) for p in text_doc.paragraphs]
        return cls(text_doc=text_doc, ann_paras=ann_paras, annotations={}, footnote_mapping={})

    @classmethod
    def from_doc_with_footnotes(cls, text_doc: TextDoc) -> AnnotatedDoc:
        """
        Build an AnnotatedDoc directly from a TextDoc that already contains
        Markdown footnote references and definitions, preserving original
        footnote IDs and content.
        """
        markdown_footnotes = MarkdownFootnotes.from_markdown(text_doc.reassemble())

        annotations: dict[SentIndex, list[FootnoteId]] = {}
        footnote_mapping: dict[FootnoteId, Footnote] = {}

        # Include all footnote definitions in document order
        for fid_str, info in markdown_footnotes.items():
            fid = check_fn_id(fid_str)
            footnote_mapping[fid] = Footnote(id=fid, content=info.content)

        # Preserve original paragraphs as AnnotatedPara list in original order
        ann_paras: list[AnnotatedPara] = [
            AnnotatedPara.from_para_with_footnotes(paragraph, markdown_footnotes)
            for paragraph in text_doc.paragraphs
        ]

        # Walk paragraphs (excluding footnote definition blocks) to collect references
        for para_index, ann_para in enumerate(ann_paras):
            if ann_para.paragraph.is_footnote_def():
                continue
            if not ann_para.has_annotations():
                continue

            for sentence_index, sentence_footnotes in ann_para.annotations.items():
                sent_index = SentIndex(para_index, sentence_index)
                for footnote in sentence_footnotes:
                    fid = check_fn_id(footnote.id)
                    if sent_index not in annotations:
                        annotations[sent_index] = []
                    annotations[sent_index].append(fid)

        return AnnotatedDoc(
            text_doc=text_doc,
            ann_paras=ann_paras,
            annotations=annotations,
            footnote_mapping=footnote_mapping,
        )

    @staticmethod
    def consolidate_annotations(ann_paras: list[AnnotatedPara]) -> AnnotatedDoc:
        """
        Consolidate a list of AnnotatedPara objects into an AnnotatedDoc.

        Handles footnote ID uniquing by tracking used IDs per prefix and
        renumbering as needed to ensure all footnote IDs are unique.
        """
        if not ann_paras:
            return AnnotatedDoc(
                text_doc=TextDoc([]),
                ann_paras=[],
                annotations={},
                footnote_mapping={},
            )

        footnote_mapping: dict[FootnoteId, Footnote] = {}
        annotations: dict[SentIndex, list[FootnoteId]] = {}

        # Build TextDoc from paragraphs
        paragraphs = [ann_para.paragraph for ann_para in ann_paras]
        text_doc = TextDoc(paragraphs)

        # Process annotations preserving order; ensure uniqueness across doc
        for para_index, ann_para in enumerate(ann_paras):
            if not ann_para.has_annotations():
                continue

            prefix = ann_para.fn_prefix

            # Process each sentence's annotations
            for sentence_index, sentence_annotations in ann_para.annotations.items():
                sent_index = SentIndex(para_index, sentence_index)

                for footnote in sentence_annotations:
                    # Build target footnote ID, applying prefix if provided
                    original_base_id = footnote.id.lstrip("^")
                    full_id_str = _normalize_footnote_id(
                        f"{prefix}{original_base_id}" if prefix else original_base_id
                    )

                    # Create validated footnote ID and ensure uniqueness
                    target_id = check_fn_id(full_id_str)
                    base_with_caret = full_id_str
                    counter = 1
                    while target_id in footnote_mapping:
                        target_id = check_fn_id(f"{base_with_caret}_{counter}")
                        counter += 1

                    # Store mapping with the final doc-level ID
                    footnote_mapping[target_id] = Footnote(id=target_id, content=footnote.content)

                    # Store the reference from this sentence to the footnote ID
                    if sent_index not in annotations:
                        annotations[sent_index] = []
                    annotations[sent_index].append(target_id)

        return AnnotatedDoc(
            text_doc=text_doc,
            ann_paras=list(ann_paras),
            annotations=annotations,
            footnote_mapping=footnote_mapping,
        )

    def non_footnote_paragraphs(self) -> Iterator[tuple[int, AnnotatedPara]]:
        """
        Iterate over (para_index, AnnotatedPara) for paragraphs that are not
        footnote-definition blocks, preserving original order.
        """
        for index, ann_para in enumerate(self.ann_paras):
            if not ann_para.paragraph.is_footnote_def():
                yield index, ann_para

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

        # Render each non-footnote paragraph with its annotations
        para_texts: list[str] = []
        for para_index, ann_para in self.non_footnote_paragraphs():
            paragraph = ann_para.paragraph

            # Build footnote references for this paragraph
            para_footnote_refs: dict[int, list[FootnoteId]] = {}

            # Collect footnote IDs for sentences in this paragraph
            for sentence_index, _sentence in enumerate(paragraph.sentences):
                sent_index = SentIndex(para_index, sentence_index)
                if sent_index in self.annotations:
                    para_footnote_refs[sentence_index] = list(self.annotations[sent_index])

            # Build the paragraph text with footnote references
            annotated_sentences: list[str] = []
            for sentence_index, sentence in enumerate(paragraph.sentences):
                sentence_text = sentence.text
                if sentence_index in para_footnote_refs:
                    # Add footnote references to this sentence (IDs already have ^)
                    # but avoid duplicating markers that already exist in the text
                    existing_text = sentence_text
                    missing_ids = [
                        footnote_id
                        for footnote_id in para_footnote_refs[sentence_index]
                        if f"[{footnote_id}]" not in existing_text
                    ]
                    if missing_ids:
                        refs = "".join(f"[{fid}]" for fid in missing_ids)
                        sentence_text = sentence_text.rstrip() + refs
                annotated_sentences.append(sentence_text)

            para_texts.append(" ".join(annotated_sentences))

        # Build output
        if self.footnote_mapping:
            # Preserve insertion order of footnotes (IDs already have ^)
            footnote_lines = [
                f"[{footnote_id}]: {self.footnote_mapping[footnote_id].content}"
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
        # Validate SentIndex is within bounds (against original ann_paras)
        if sent_index.para_index >= len(self.ann_paras):
            raise IndexError(f"Paragraph index {sent_index.para_index} out of range")
        para = self.ann_paras[sent_index.para_index].paragraph
        if sent_index.sent_index >= len(para.sentences):
            raise IndexError(
                f"Sentence index {sent_index.sent_index} out of range in paragraph {sent_index.para_index}"
            )

        # Start from the provided prefix
        # Build initial candidate and increment until unique
        candidate = f"^{fn_prefix}" if fn_prefix else "^1"
        if fn_prefix:
            candidate, _ = increment_id(candidate)

        footnote_id = check_fn_id(_normalize_footnote_id(candidate))
        while footnote_id in self.footnote_mapping:
            candidate, _ = increment_id(candidate)
            footnote_id = check_fn_id(_normalize_footnote_id(candidate))
        self.footnote_mapping[footnote_id] = Footnote(id=footnote_id, content=annotation)

        # Add reference from sentence to this new footnote ID
        if sent_index not in self.annotations:
            self.annotations[sent_index] = []
        self.annotations[sent_index].append(footnote_id)

    def get_sentence_annotations(self, sent_index: SentIndex) -> list[str]:
        """Get all annotations for a specific sentence."""
        footnote_ids = self.annotations.get(sent_index, [])
        return [self.footnote_mapping[fid].content for fid in footnote_ids]

    def clear_annotations_for_sentence(self, sent_index: SentIndex) -> None:
        """Remove all annotations for a specific sentence."""
        if sent_index in self.annotations:
            # Save the footnote IDs before deleting them from this sentence
            ids_to_consider = list(self.annotations[sent_index])
            del self.annotations[sent_index]

            # Remove footnote definitions that are no longer referenced anywhere else
            still_used: set[FootnoteId] = set()
            for refs in self.annotations.values():
                for fid in refs:
                    still_used.add(fid)

            for fid in ids_to_consider:
                if fid not in still_used and fid in self.footnote_mapping:
                    del self.footnote_mapping[fid]

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

    annotated_para = AnnotatedPara.unannotated(paragraph, fn_prefix=fn_prefix, fn_start=fn_start)

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
    annotated = AnnotatedPara.unannotated(para)

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
    annotated = AnnotatedPara.unannotated(para)

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
    ann_doc = AnnotatedDoc.plain_doc(text_doc)

    assert len(ann_doc.text_doc.paragraphs) == 2
    assert not ann_doc.has_annotations()
    assert ann_doc.total_annotation_count() == 0
    assert ann_doc.as_markdown_with_footnotes() == text_doc.reassemble()


def test_annotated_doc_add_annotation() -> None:
    """Test adding annotations to AnnotatedDoc."""
    text_doc = TextDoc.from_text("First paragraph.\n\nSecond paragraph.")
    ann_doc = AnnotatedDoc.plain_doc(text_doc)

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

    ann_para1 = AnnotatedPara.unannotated(para1)
    ann_para1.add_annotation(0, "Note 1")

    ann_para2 = AnnotatedPara.unannotated(para2)
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
    ann_para1 = AnnotatedPara.unannotated(para1, fn_prefix="a", fn_start=1)
    ann_para1.add_annotation(0, "Note A1")
    ann_para1.add_annotation(0, "Note A2")

    ann_para2 = AnnotatedPara.unannotated(para2, fn_prefix="b", fn_start=1)
    ann_para2.add_annotation(0, "Note B1")

    ann_para3 = AnnotatedPara.unannotated(para3, fn_prefix="a", fn_start=1)
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
    ann_para1 = AnnotatedPara.unannotated(para1, fn_prefix="", fn_start=1)
    ann_para1.add_annotation(0, "Note 1")
    ann_para1.add_annotation(0, "Note 2")

    ann_para2 = AnnotatedPara.unannotated(para2, fn_prefix="", fn_start=1)
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

    ann_para1 = AnnotatedPara.unannotated(para1)
    ann_para2 = AnnotatedPara.unannotated(para2)

    ann_doc = AnnotatedDoc.consolidate_annotations([ann_para1, ann_para2])

    assert len(ann_doc.text_doc.paragraphs) == 2
    assert ann_doc.total_annotation_count() == 0
    assert len(ann_doc.footnote_mapping) == 0
    assert not ann_doc.has_annotations()


def test_markdown_with_footnotes_consolidated() -> None:
    """Test markdown rendering with consolidated footnotes."""
    para1 = Paragraph.from_text("First paragraph.")
    para2 = Paragraph.from_text("Second paragraph.")

    ann_para1 = AnnotatedPara.unannotated(para1, fn_prefix="ref", fn_start=1)
    ann_para1.add_annotation(0, "Reference 1")

    ann_para2 = AnnotatedPara.unannotated(para2, fn_prefix="ref", fn_start=1)
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
    ann_doc = AnnotatedDoc.plain_doc(text_doc)

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
    annotated = AnnotatedPara.unannotated(para, fn_prefix="ref_", fn_start=1)
    footnote_id = annotated.footnote_id(1)
    assert footnote_id == FootnoteId("^ref_1")

    # Invalid prefix should raise error when creating footnote ID
    annotated_invalid = AnnotatedPara.unannotated(para, fn_prefix="invalid@prefix", fn_start=1)
    try:
        annotated_invalid.footnote_id(1)
        raise AssertionError("Expected ValueError for invalid footnote prefix")
    except ValueError:
        pass  # Expected


def test_markdown_with_footnotes_header() -> None:
    """Ensure footnote_header is inserted correctly above consolidated footnotes."""
    para = Paragraph.from_text("Some text.")
    ann_para = AnnotatedPara.unannotated(para, fn_prefix="ref", fn_start=1)
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

    ann_para1 = AnnotatedPara.unannotated(para1, fn_prefix="a", fn_start=1)
    ann_para1.add_annotation(0, "Note A1")  # a1

    ann_para2 = AnnotatedPara.unannotated(para2, fn_prefix="b", fn_start=1)
    ann_para2.add_annotation(0, "Note B1")  # b1

    ann_para3 = AnnotatedPara.unannotated(
        para3, fn_prefix="a", fn_start=2
    )  # Start at 2 to avoid conflict
    ann_para3.add_annotation(0, "Note A2")  # a2

    ann_doc = AnnotatedDoc.consolidate_annotations([ann_para1, ann_para2, ann_para3])
    output = ann_doc.as_markdown_with_footnotes()

    # Extract footnote IDs in output order
    lines = [line.strip() for line in output.split("\n") if line.startswith("[^")]
    ids_in_output = [line.split(":")[0][2:-1] for line in lines]  # remove "[^" and "]"

    assert ids_in_output == ["a1", "b1", "a2"], ids_in_output


def test_markdown_roundtrip_with_footnotes() -> None:
    """Roundtrip: parse markdown with footnotes and render back identically after canonicalization."""
    from textwrap import dedent

    from flowmark import flowmark_markdown, line_wrap_by_sentence

    def canonicalize(markdown: str) -> str:
        md = flowmark_markdown(line_wrap_by_sentence(is_markdown=True))

        return md.render(md.parse(markdown)).strip()

    raw_md = dedent(
        """
        # Sample Doc

        This is a sentence with a note.[^a1] Another sentence with a reference.[^b1]

        [^a1]: First footnote with a link <https://example.com/a>

        [^b1]: Second footnote with [link](https://example.com/b)
        """
    ).strip()

    canonical_input = canonicalize(raw_md)

    td = TextDoc.from_text(raw_md)
    ad = AnnotatedDoc.from_doc_with_footnotes(td)
    rendered = ad.as_markdown_with_footnotes().strip()

    canonical_output = canonicalize(rendered)

    assert canonical_output == canonical_input
    assert canonical_output == rendered


def test_from_doc_with_footnotes_preserves_order_and_filters_defs_at_usage() -> None:
    from textwrap import dedent

    raw_md = dedent(
        """
        Title.

        Body with a ref[^a1].

        [^a1]: Definition A1
        """
    ).strip()

    td = TextDoc.from_text(raw_md)
    ad = AnnotatedDoc.from_doc_with_footnotes(td)

    # Original doc still contains footnote definition blocks
    original = ad.text_doc.reassemble()
    assert "[^a1]:" in original

    # Non-footnote paragraphs iterator excludes definition blocks
    non_fn_text = "\n\n".join(
        " ".join(sent.text for sent in ap.paragraph.sentences)
        for _idx, ap in ad.non_footnote_paragraphs()
    )
    assert "[^a1]:" not in non_fn_text


def test_consolidate_annotations_iter_non_fn_matches_all_when_no_defs() -> None:
    para1 = Paragraph.from_text("One.")
    para2 = Paragraph.from_text("Two.")

    ap1 = AnnotatedPara.unannotated(para1, fn_prefix="r", fn_start=1)
    ap1.add_annotation(0, "Note 1")

    ap2 = AnnotatedPara.unannotated(para2, fn_prefix="r", fn_start=2)
    ap2.add_annotation(0, "Note 2")

    ad = AnnotatedDoc.consolidate_annotations([ap1, ap2])

    reconstructed = "\n\n".join(
        " ".join(sent.text for sent in ap.paragraph.sentences)
        for _idx, ap in ad.non_footnote_paragraphs()
    )
    assert reconstructed == ad.text_doc.reassemble()
