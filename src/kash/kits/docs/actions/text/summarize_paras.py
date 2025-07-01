from __future__ import annotations

import asyncio
from typing import Any

from chopdiff.divs import div
from chopdiff.docs import Paragraph, TextDoc, TextUnit
from strif import abbrev_str

from kash.config.logger import get_logger
from kash.config.settings import global_settings
from kash.exec import kash_action, kash_precondition
from kash.exec.llm_transforms import llm_transform_str
from kash.llm_utils import Message, MessageTemplate
from kash.model import Format, Item, ItemType, LLMOptions
from kash.shell.output.shell_output import multitask_status
from kash.utils.api_utils.gather_limited import FuncTask, Limit, gather_limited_sync
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


llm_options = LLMOptions(
    system_message=Message(
        """
        You are a careful and precise editor.
        You give exactly the results requested without additional commentary.
        """
    ),
    body_template=MessageTemplate(
        """
        Please describe what is said in the following one or two paragraphs, as a 
        summary for the content. Rules:

        - Mention only the most important points. Include all the key topics discussed.
        
        - Keep the summary short! Use 1-3 sentences, with a total of 10-40 words for
          one or two or more paragraphs.
          Your summary should be shorter than the input text.
        
        - Write in clean and and direct language.

        - Do NOT mention the text or the author. Simply state the points as presented.

        - DO NOT INCLUDE any other commentary.

        - If the input is very short or so unclear you can't summarize it, simply output
            "(No results)".

        - If the input is in a language other than English, output the summary in the same language.

        Sample input text:

        I think push ups are one of the most underrated exercises out there and they're also one of
        the exercises that is most frequently performed with poor technique.
        And I think this is because a lot of people think it's just an easy exercise and they adopt
        a form that allows them to achieve a rep count that they would expect from an easy exercise,
        but all that ends up happening is they they do a bunch of poor quality repetitions in order
        to get a high rep count. So I don't think push ups are particularly easy when they're done well
        and they're really effective for building just general fitness and muscle in the upper body
        if you do them properly. So here's how you get the most out of them.

        Sample output text:

        Push ups are an underrated exercise. People tend to adopt poor form as they pursue rep counts.
        They are not easy to do well and are effective for building general fitness and muscle in the
        upper body.

        Input text:

        {body}

        Output text:
        """
    ),
)


PARA = "para"
ANNOTATED_PARA = "annotated-para"
PARA_SUMMARY = "para-summary"


@kash_precondition
def has_annotated_paras(item: Item) -> bool:
    """
    Useful to check if an item has already been annotated with summarys.
    """
    return bool(item.body and item.body.find(f'<p class="{ANNOTATED_PARA}">') != -1)


def summarize_paragraph(llm_options: LLMOptions, para: Paragraph) -> str | None:
    """
    Summarize a single paragraph and return the summary.
    Returns None if paragraph should be skipped.
    """
    if para.is_markup() or para.is_header() or para.size(TextUnit.words) <= 40:
        return None

    para_str = para.reassemble()
    log.message(
        "Summarizeing paragraph (%s words): %r", para.size(TextUnit.words), abbrev_str(para_str)
    )

    llm_response: str = llm_transform_str(llm_options, para_str)
    log.message("Generated summary: %r", abbrev_str(llm_response))
    return llm_response


def apply_summary_to_paragraph(para: Paragraph, summary: str | None) -> str:
    """
    Apply summary to a paragraph and return the formatted paragraph text.
    """
    para_str = para.reassemble()

    if summary is None:
        # Paragraph was skipped during summarying
        log.message(
            "Skipping summarying very short paragraph (%s words)", para.size(TextUnit.words)
        )
        return para_str

    if summary:
        summary_div = div(PARA_SUMMARY, summary)
        new_div = div(ANNOTATED_PARA, summary_div, div(PARA, para_str))
        log.message("Added summary to paragraph: %r", abbrev_str(para_str))
        return new_div
    else:
        log.message("No summary generated for paragraph")
        return para_str


async def summarize_paras_async(item: Item) -> Item:
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    doc = TextDoc.from_text(item.body)
    paragraphs = [para for para in doc.paragraphs if para.size(TextUnit.words) > 0]

    log.message("Step 1: Summarizeing %d paragraphs", len(paragraphs))
    summary_tasks = [FuncTask(summarize_paragraph, (llm_options, para)) for para in paragraphs]

    def labeler(i: int, spec: Any) -> str:
        """Create descriptive labels for summary tasks using paragraph content."""
        if isinstance(spec, FuncTask) and len(spec.args) >= 2:
            para = spec.args[1]  # Second arg is the paragraph
            if isinstance(para, Paragraph):
                para_text = abbrev_str(para.reassemble())
                return f"Summarize {i + 1}/{len(paragraphs)}: {para_text}"
        return f"Summarize paragraph {i + 1}/{len(paragraphs)}"

    # Execute in parallel with rate limiting, retries, and progress tracking
    limit = Limit(rps=global_settings().limit_rps, concurrency=global_settings().limit_concurrency)
    async with multitask_status() as status:
        paragraph_summarys = await gather_limited_sync(
            *summary_tasks, limit=limit, status=status, labeler=labeler
        )

    log.message(
        "Step 2: Applying %d summarys to %d paragraphs",
        len(paragraph_summarys),
        len(paragraphs),
    )
    output: list[str] = []

    for para, summary in zip(paragraphs, paragraph_summarys, strict=False):
        para_text = apply_summary_to_paragraph(para, summary)
        output.append(para_text)

    final_output = "\n\n".join(output)
    return item.derived_copy(type=ItemType.doc, body=final_output, format=Format.md_html)


@kash_action(llm_options=llm_options, live_output=True)
def summarize_paras(item: Item) -> Item:
    """
    Summarize each paragraph in the text with a very short summary, wrapping the original
    and the summary in simple divs.
    """
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    return asyncio.run(summarize_paras_async(item))
