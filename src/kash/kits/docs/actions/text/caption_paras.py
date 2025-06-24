from __future__ import annotations

import asyncio
from typing import Any

from chopdiff.divs import div
from chopdiff.docs import Paragraph, TextDoc, TextUnit
from strif import abbrev_str

from kash.config.logger import get_logger
from kash.exec import kash_action, kash_precondition
from kash.exec.llm_transforms import llm_transform_str
from kash.llm_utils import Message, MessageTemplate
from kash.model import Format, Item, ItemType, LLMOptions
from kash.shell.output.shell_output import multitask_status
from kash.utils.api_utils.gather_limited import FuncTask, gather_limited_sync
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
        summary or caption for the content. Rules:

        - Mention only the most important points. Include all the key topics discussed.
        
        - Keep the caption short! Use ONE sentence or TWO SHORT sentences, with a total of 10-15
          words. It must be significantly shorter than the input text.
        
        - Write in clean and and direct language.

        - Do not mention the text or the author. Simply state the points as presented.

        - If the content contains other promotional material or only references information such as
            about what will be discussed later, ignore it.

        - DO NOT INCLUDE any other commentary.

        - If the input is very short or so unclear you can't summarize it, simply output
            "(No results)".

        - If the input is in a language other than English, output the caption in the same language.

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

        Push ups are an underrated exercise. They are not easy to do well.

        Input text:

        {body}

        Output text:
        """
    ),
)


PARA = "para"
ANNOTATED_PARA = "annotated-para"
PARA_CAPTION = "para-caption"


@kash_precondition
def has_annotated_paras(item: Item) -> bool:
    """
    Useful to check if an item has already been annotated with captions.
    """
    return bool(item.body and item.body.find(f'<p class="{ANNOTATED_PARA}">') != -1)


def caption_paragraph(llm_options: LLMOptions, para: Paragraph) -> str | None:
    """
    Caption a single paragraph and return the caption.
    Returns None if paragraph should be skipped.
    """
    if para.is_markup() or para.is_header() or para.size(TextUnit.words) <= 40:
        return None

    para_str = para.reassemble()
    log.message(
        "Captioning paragraph (%s words): %r", para.size(TextUnit.words), abbrev_str(para_str)
    )

    llm_response: str = llm_transform_str(llm_options, para_str)
    log.message("Generated caption: %r", abbrev_str(llm_response))
    return llm_response


def apply_caption_to_paragraph(para: Paragraph, caption: str | None) -> str:
    """
    Apply caption to a paragraph and return the formatted paragraph text.
    """
    para_str = para.reassemble()

    if caption is None:
        # Paragraph was skipped during captioning
        log.message(
            "Skipping captioning very short paragraph (%s words)", para.size(TextUnit.words)
        )
        return para_str

    if caption:
        caption_div = div(PARA_CAPTION, caption)
        new_div = div(ANNOTATED_PARA, caption_div, div(PARA, para_str))
        log.message("Added caption to paragraph: %r", abbrev_str(para_str))
        return new_div
    else:
        log.message("No caption generated for paragraph")
        return para_str


async def caption_paras_async(item: Item) -> Item:
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    doc = TextDoc.from_text(item.body)
    paragraphs = [para for para in doc.paragraphs if para.size(TextUnit.words) > 0]

    log.message("Step 1: Captioning %d paragraphs", len(paragraphs))
    caption_tasks = [FuncTask(caption_paragraph, (llm_options, para)) for para in paragraphs]

    def labeler(i: int, spec: Any) -> str:
        """Create descriptive labels for caption tasks using paragraph content."""
        if isinstance(spec, FuncTask) and len(spec.args) >= 2:
            para = spec.args[1]  # Second arg is the paragraph
            if isinstance(para, Paragraph):
                para_text = abbrev_str(para.reassemble())
                return f"Caption {i + 1}/{len(paragraphs)}: {para_text}"
        return f"Caption paragraph {i + 1}/{len(paragraphs)}"

    # Execute in parallel with rate limiting, retries, and progress tracking
    async with multitask_status() as status:
        paragraph_captions = await gather_limited_sync(
            *caption_tasks, status=status, labeler=labeler
        )

    log.message(
        "Step 2: Applying %d captions to %d paragraphs",
        len(paragraph_captions),
        len(paragraphs),
    )
    output: list[str] = []

    for para, caption in zip(paragraphs, paragraph_captions, strict=False):
        para_text = apply_caption_to_paragraph(para, caption)
        output.append(para_text)

    final_output = "\n\n".join(output)
    return item.derived_copy(type=ItemType.doc, body=final_output, format=Format.md_html)


@kash_action(llm_options=llm_options, live_output=True)
def caption_paras(item: Item) -> Item:
    """
    Caption each paragraph in the text with a very short summary, wrapping the original
    and the caption in simple divs.
    """
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    return asyncio.run(caption_paras_async(item))
