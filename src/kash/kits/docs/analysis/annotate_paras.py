from __future__ import annotations

from typing import Any, TypeVar

from chopdiff.docs import Paragraph, TextDoc, TextUnit
from strif import abbrev_list, abbrev_str

from kash.config.logger import get_logger
from kash.exec.llm_transforms import llm_transform_str
from kash.kits.docs.analysis.doc_annotations import (
    AnnotatedDoc,
    AnnotatedPara,
    map_notes_with_embeddings,
)
from kash.model import Format, Item, ItemType, LLMOptions
from kash.utils.api_utils.gather_limited import FuncTask
from kash.utils.api_utils.multitask_gather import multitask_gather
from kash.utils.errors import InvalidInput
from kash.utils.text_handling.markdown_utils import extract_bullet_points

log = get_logger(__name__)


T = TypeVar("T")


def research_paragraph(llm_options: LLMOptions, para: Paragraph) -> list[str] | None:
    """
    Research a single paragraph and return the parsed notes.
    Returns None if paragraph should be skipped.
    """
    if para.is_markup() or para.is_header() or para.size(TextUnit.words) <= 4:
        return None

    para_str = para.reassemble()

    # Call the sync function directly
    llm_response: str = llm_transform_str(llm_options, para_str)

    if llm_response.strip():
        parsed_notes = extract_bullet_points(llm_response)
        log.info("Parsed %d notes: %s", len(parsed_notes), abbrev_list(parsed_notes))
        return parsed_notes

    log.info("No notes found for paragraph")
    return []


def annotate_para(
    para: Paragraph, notes: list[str] | None, fn_prefix: str = "", fn_start: int = 1
) -> AnnotatedPara:
    """
    Apply footnotes to a paragraph and return the annotated paragraph.
    """
    para_str = para.reassemble()

    # TODO: Parse/handle previous footnotes in the doc
    ann_para = AnnotatedPara.unannotated(para, fn_prefix=fn_prefix, fn_start=fn_start)
    if notes is None:
        # Paragraph was skipped during research
        log.info("Skipping header or very short paragraph: %r", abbrev_str(para_str))
        return ann_para

    if notes:
        ann_para = map_notes_with_embeddings(para, notes, fn_prefix=fn_prefix, fn_start=fn_start)

        if ann_para.has_annotations():
            log.info(
                "Added %s annotations to paragraph: %r",
                ann_para.annotation_count(),
                abbrev_str(para_str),
            )
        else:
            log.info("No annotations found for paragraph")

    return ann_para


async def annotate_paras_async(
    llm_options: LLMOptions, item: Item, fn_prefix: str = "", fn_start: int = 1
) -> Item:
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")
    doc = TextDoc.from_text(item.body)
    paragraphs = [para for para in doc.paragraphs if para.size(TextUnit.words) > 0]

    log.message("Step 1: Researching %d paragraphs", len(paragraphs))
    research_tasks: list[FuncTask[list[str] | None]] = [
        FuncTask(research_paragraph, (llm_options, para)) for para in paragraphs
    ]

    def research_labeler(i: int, spec: Any) -> str:
        if isinstance(spec, FuncTask) and len(spec.args) >= 2:
            para = spec.args[1]  # Second arg is the paragraph
            if isinstance(para, Paragraph):
                nwords = para.size(TextUnit.words)
                para_text = abbrev_str(para.reassemble(), 30)
                return f"Research {i + 1}/{len(paragraphs)} ({nwords} words): {repr(para_text)}"
        return f"Research {i + 1}/{len(paragraphs)}"

    # Execute research in parallel with progress and default rate limits
    research_results = await multitask_gather(research_tasks, labeler=research_labeler)
    if len(research_results.successes) == 0:
        raise RuntimeError("No successful research tasks")

    # Preserve alignment with input paragraphs; treat failures as None
    paragraph_notes = research_results.successes_or_none

    log.message(
        "Step 2: Applying %d sets of footnotes (%s errors, %s total notes) to %d paragraphs",
        len(research_results.successes),
        len(research_results.errors),
        sum(len(notes or []) for notes in paragraph_notes if isinstance(notes, list)),
        len(paragraphs),
    )

    # Create annotation tasks
    annotation_tasks: list[FuncTask[AnnotatedPara]] = [
        FuncTask(annotate_para, (para, notes, fn_prefix, fn_start))
        for para, notes in zip(paragraphs, paragraph_notes, strict=False)
    ]

    def annotation_labeler(i: int, spec: Any) -> str:
        if isinstance(spec, FuncTask) and len(spec.args) >= 1:
            para = spec.args[0]  # First arg is the paragraph
            if isinstance(para, Paragraph):
                nwords = para.size(TextUnit.words)
                para_text = abbrev_str(para.reassemble(), 30)
                return f"Annotate {i + 1}/{len(paragraphs)} ({nwords} words): {repr(para_text)}"
        return f"Annotate {i + 1}/{len(paragraphs)}"

    # Execute annotations in parallel
    annotated_results = await multitask_gather(annotation_tasks, labeler=annotation_labeler)
    annotated_paras = annotated_results.successes

    # Consolidate all annotations into a single document with footnotes at the end
    log.message("Step 3: Consolidating footnotes at end of document")
    consolidated_doc = AnnotatedDoc.consolidate_annotations(annotated_paras)
    final_output = consolidated_doc.as_markdown_with_footnotes()

    # TODO: Remove near-duplicate footnotes.

    return item.derived_copy(type=ItemType.doc, body=final_output, format=Format.md_html)
