from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from chopdiff.docs import Paragraph, TextDoc, TextUnit
from strif import abbrev_list, abbrev_str

from kash.config.logger import get_logger
from kash.config.settings import global_settings
from kash.exec import kash_action
from kash.exec.llm_transforms import llm_transform_str
from kash.kits.docs.concepts.doc_annotations import (
    AnnotatedDoc,
    AnnotatedPara,
    map_notes_with_embeddings,
)
from kash.llm_utils import Message, MessageTemplate
from kash.model import Format, Item, ItemType, LLMOptions
from kash.shell.output.shell_output import multitask_status
from kash.utils.api_utils.gather_limited import FuncTask, Limit, gather_limited_sync
from kash.utils.errors import InvalidInput
from kash.utils.text_handling.markdown_utils import extract_bullet_points

log = get_logger(__name__)

FN_PREFIX = "RES"
T = TypeVar("T")


llm_options = LLMOptions(
    system_message=Message(
        """
        You are a careful and precise editor.
        You give exactly the results requested without additional commentary.
        """
    ),
    body_template=MessageTemplate(
        """

      You are a fact checker and researcher for a top-tier publication with high standards
      for accuracy and rigor.

      You will be given one or more paragraphs, likely an excerpt from a longer text.

      Your task it to research all facts and entities mentioned in the paragraphs and check
      for correctness or give essential context.
      This will provide key context for the author to revise the writing and may help the
      readers who wish for context and relevant citations.

      It is VERY important that writing and editorial notes are accurate and notable and
      always backed by reputable sources. NEVER give speculative information.

      The output should be a list of bulleted items in Markdown format
      (use - for bullets and DO NOT use • or other bullet characters).
      
      Each item should be ONE SENTENCE long and should cover one of the following:

      1. background facts about people, places, things, software, and other entities mentioned
         in the text that a reader may not already be familiar with

      2. closely related concepts or notable work that is closely related to the text

      3. corroborating or contradicting sources for any factual statements or assertions


      Guidelines for the items: Each bullet should be one of the following:

      1. A brief link to places, things, software, entities, etc, include a brief summary and
         link to the most relevant 1 or 2 pages on that entity, such as a Wikipedia page for a
         person or company, a GitHub link for an open source project, or a website for a
         product.

      2. For closely related work found during research, write a sentence with a link to the
         work, mentioning the author or source.

      3. For fact assertions, if there are corroborating or contradicting sources, write a
         sentence with a link to the source and briefly state what the source says.

      Use web searches to find the most relevant and authoritative sources.

      IMPORTANT:

      - Do fact check any factual assertion that is not obviously true.

      - Only include HELPFUL annotations for concepts. The more specific, obscure, or rare the concept,
        the more likely it is to need an annotation.
        DO NOT include obvious or common concepts or entities like "America" or
        "biology" or "elephants" or "software".
        Only mention concepts where additional detail would help the reader and contain
        important details they likely may not already know, like "HTTP" or "Nuremberg"
        or "Alan Kay".

      - Only define or link UNAMBIGUOUSLY RELEVANT concepts.
        DO NOT make comments if the concept reference is ambiguous in the given context 
        and could refer to more than one thing.
        For example, as in the examples below, the input text says "four zero six community issue"
        but it's unclear what it means. This should then NOT be included in the output.

      - Do NOT mention or include language *about* the text or the author of the text.
        The reader can see the text and know the context is the paragraph.
        Do not say "The text refers to..." or "X refers to ...". Simply say "X is ...".
      
      Additional guidelines:

      - Write in clean and and direct language. Keep it brief. Be concise as the links
        will have relevant detail.

      - If the input is very short doesn't have any obvious facts or entities
        referenced, simply output "(No results)".

      - If you can't find a link for a concept, simply omit it.

      - Do not mention the text or the author. Simply state the editorial note.

      - DO NOT INCLUDE any other commentary.

      - If the input is in a language other than English, output the caption in the same language.

      Here are three examples to be very clear on our standards of quality and style.

      Example input text #1:

        She said she had just come back from Tashkent.

      Example output text #1:

        - [Tashkent](https://en.wikipedia.org/wiki/Tashkent) is the capital of
          [Uzbekistan](https://en.wikipedia.org/wiki/Uzbekistan).

      Example input text #2:

        As Einstein said, 'The only thing more dangerous than ignorance is arrogance.'
        But even companies that do not show arrogance are susceptible to Conway's Law.

      Sample output text #2:

        - This quote is often misattributed to
          [Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) but appears in the
          2003 book *[Still Life with
          Crows](https://www.google.com/books/edition/Still_Life_with_Crows/yE7SYGtSOgMC?hl=en&gbpv=1&bsq=%22the%20only%20thing%20more%20dangerous%20than%20ignorance%20is%20arrogance%22)*.
          ([Snopes](https://www.snopes.com/fact-check/einstein-dangerous-ignorance-arrogance/))

        - [Conway's Law](https://en.wikipedia.org/wiki/Conway%27s_law) describes the link
          between communication structure of organizations and the systems they design

      Example input text #3:

      In Germany, in 1440, goldsmith Johannes Gutenberg invented the movable-type printing
      press. His work led to an information revolution and the unprecedented mass-spread of
      literature throughout Europe.
      Modelled on the design of the existing screw presses, a single Renaissance movable-type
      printing press could produce up to 3,600 pages per workday.
      Within a single generation, printers in Mainz, Venice, Nuremberg, and Paris were turning
      out incunabula—early printed books—on grammar, law, medicine, and, crucially, the Bible
      in both Latin and the emerging vernaculars.

      Sample output text #3:

      - [Johannes Gutenberg](https://en.wikipedia.org/wiki/Johannes_Gutenberg), a goldsmith
        from Mainz, Germany, introduced movable-type printing around 1440, igniting Europe's
        "printing revolution" ([Wikipedia](https://en.wikipedia.org/wiki/Printing_press)).

      - Modelled on screw wine-presses, a single Renaissance hand-press could deliver roughly
        3,600 pages in a work-day—nearly 100 times faster than manual copying
        ([Wikipedia](https://en.wikipedia.org/wiki/Printing_press)).

      - By 1500, presses were active in more than 200 cities across twenty European territories
        ([Wikipedia](https://en.wikipedia.org/wiki/Printing_press),
        [History of Information](https://www.historyofinformation.com/detail.php?id=27)).

      - Early centres such as Mainz, Venice, Nuremberg, and Paris established presses within
        two decades of the Gutenberg Bible and became prolific producers of early books
        ([Wikipedia](https://en.wikipedia.org/wiki/Incunable),
        [Britannica](https://www.britannica.com/topic/publishing/The-age-of-early-printing-1450-1550)).

      - [Incunabula](https://en.wikipedia.org/wiki/Incunable) are works printed in Europe
        before 1501; about 30,000 editions survive, with Germany and Italy accounting for the
        majority ([Britannica](https://www.britannica.com/topic/incunabula),
        [Christie's](https://www.christies.com/en/stories/collecting-guide-to-incunabula-earliest-printed-books-77562cac88854d00826550bfe969d900)).

      - Famous incunabula include the *Gutenberg Bible* (Mainz 1455), the *Nuremberg
        Chronicle* (Nuremberg 1493), and *Hypnerotomachia Poliphili* (Venice 1499)
        ([Britannica](https://www.britannica.com/biography/Johannes-Gutenberg),
        [Wikipedia](https://en.wikipedia.org/wiki/Incunable)).

      - Elizabeth L. Eisenstein's classic study *The Printing Press as an Agent of Change*
        (1979-80) explores how Gutenberg's technology reshaped Renaissance, Reformation, and
        scientific discourse ([Cambridge University
        Press](https://www.cambridge.org/core/books/printing-press-as-an-agent-of-change/7DC19878AB937940DE13075FE839BDBA)).

      - Historians estimate European presses had issued over 20 million volumes by 1500,
        demonstrating the explosive growth of print
        ([Wikipedia](https://en.wikipedia.org/wiki/Printing_press)).

      Example input text #4:

      Push ups are an underrated exercise.
      They are not easy to do well so are often done poorly.

      Sample output text #4:

      - [Push-up](https://en.wikipedia.org/wiki/Push-up) is a classic body-weight calisthenics
        exercise that raises and lowers the torso with the arms, engaging the chest, anterior
        deltoids, triceps and core ([Wikipedia](https://en.wikipedia.org/wiki/Push-up)).

      - Trainers call push-ups "underrated" because they combine upper-body strength work with
        core stabilization and require no gear, making them an efficient full-body move
        ([Simon Jarvis Movement
        Specialist](https://www.simonjarvismovementspecialist.com/new-blog-4/2024/4/3/the-power-of-push-ups-why-theyre-underrated-and-how-to-maximize-their-benefits),
        [The Guardian](https://www.theguardian.com/lifeandstyle/2025/mar/09/push-up-power-the-exercise-you-need-for-a-healthy-happy-life-and-eight-ways-to-make-it-easier)).

      - Form faults such as flared elbows, sagging hips, or shallow depth are so common that
        multiple guides list them among the top push-up mistakes to fix ([Nerd
        Fitness](https://www.nerdfitness.com/blog/5-common-push-up-mistakes-to-avoid/),
        [Livestrong](https://www.livestrong.com/article/13721448-push-up-mistakes/)).

      - Electromyography studies find push-ups elicit muscle-activation patterns comparable to
        the bench press, supporting their use as a strength-building substitute when free
        weights are unavailable ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6728153/)).

      - Beginner progressions—wall, knee, or incline push-ups—are widely recommended to master
        technique and reduce injury risk before advancing to the full movement ([The
        Guardian](https://www.theguardian.com/lifeandstyle/2025/mar/09/push-up-power-the-exercise-you-need-for-a-healthy-happy-life-and-eight-ways-to-make-it-easier)).

      Example input text #5:

      AI is radically changing the way we build and use software.
      On top of that, it's changing who is building software, since more people are able to
      build more things AI-powered developer tools.

      In technology, some things change fast—and some don't change at all.
      The hard part is in knowing which parts are which.

      Technology is the solution to human problems.
      And human problems tend to be complex.

      There is a saying that "Simple should be simple and complex should be possible".

      Sample output text #5:

      - [Alan Kay](https://en.wikipedia.org/wiki/Alan_Kay) introduced the maxim "*Simple
        things should be simple, complex things should be possible*" during the early-1970s
        Smalltalk project at Xerox PARC, framing it as a core goal for user-friendly yet
        powerful programming systems.

      - [Tim O'Reilly](https://en.wikipedia.org/wiki/Tim_O%27Reilly) states in his book
        *WTF?: What's the Future and Why It's Up to Us* that "technology is the solution to
        human problems, and we won't run out of work till we run out of problems," urging
        technologists to aim innovations at real societal needs
        ([weblogtheworld.com](https://weblogtheworld.com/formats/photos/tim-oreilly-next-economy)).

      - A February 2025 McKinsey study concluded that generative-AI tooling is "fundamentally
        transforming" software-product lifecycles by accelerating coding, testing, and release
        speed while improving overall quality
        ([McKinsey](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/how-an-ai-enabled-software-product-development-life-cycle-will-fuel-innovation)).

      - *Business Insider* notes that AI coding assistants such as Cursor, Devin, GitHub
        Copilot, Replit, and Bolt.new are "lowering the barrier" for non-programmers to create
        software—an emerging trend investors dub "vibe coding."
        ([Business
        Insider](https://www.businessinsider.com/tech-memo-ai-winners-startups-vibe-coding-2025-6)).

      - [GitHub Copilot](https://github.com/features/copilot), first previewed in 2021 and now
        integrated into major IDEs, uses LLMs to suggest edits to code, exemplifying AI's
        widening role in everyday software development
        ([Wikipedia](https://en.wikipedia.org/wiki/GitHub_Copilot)).

      - Amazon founder Jeff Bezos advises strategists to focus on "what's **not** going to
        change in the next 10 years," arguing that stable customer needs matter more than
        chasing fast-moving tech trends
        ([Inc.](https://www.inc.com/jeff-haden/20-years-ago-jeff-bezos-said-this-1-thing-separates-people-who-achieve-lasting-success-from-those-who-dont.html)).

      Example input text #6:

      We'll do a case study on the four zero six community issue. We'll look a little bit at
      the anatomy of the model spec, and then we'll talk about communicating
      intent to models, via a technique known as deliberative alignment. 

      Sample output text #6:

      - The [Model spec](https://cdn.openai.com/spec/model-spec-2024-05-08.pdf) is OpenAI's
        specification document that outlines desired model behavior and safety guidelines.
        ([OpenAI](https://openai.com/index/introducing-the-model-spec/)).

      - [Deliberative alignment](https://arxiv.org/abs/2406.11976) is a research approach
        where AI models are trained to reason through ethical considerations and
        stakeholder perspectives before making decisions ([arXiv](https://arxiv.org/abs/2406.11976)).

      Input text:

      {body}

      Output text:
        """
    ),
)


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


def annotate_paragraph(para: Paragraph, notes: list[str] | None) -> AnnotatedPara:
    """
    Apply footnotes to a paragraph and return the annotated paragraph.
    """
    para_str = para.reassemble()

    ann_para = AnnotatedPara.from_para(para, fn_prefix=FN_PREFIX, fn_start=1)
    if notes is None:
        # Paragraph was skipped during research
        log.info("Skipping header or very short paragraph: %r", abbrev_str(para_str))
        return ann_para

    if notes:
        ann_para = map_notes_with_embeddings(para, notes, fn_prefix=FN_PREFIX, fn_start=1)

        if ann_para.has_annotations():
            log.info(
                "Added %s annotations to paragraph: %r",
                ann_para.annotation_count(),
                abbrev_str(para_str),
            )
        else:
            log.info("No annotations found for paragraph")

    return ann_para


async def research_paras_async(item: Item) -> Item:
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")
    doc = TextDoc.from_text(item.body)
    paragraphs = [para for para in doc.paragraphs if para.size(TextUnit.words) > 0]

    log.message("Step 1: Researching %d paragraphs", len(paragraphs))
    research_tasks = [FuncTask(research_paragraph, (llm_options, para)) for para in paragraphs]

    def research_labeler(i: int, spec: Any) -> str:
        if isinstance(spec, FuncTask) and len(spec.args) >= 2:
            para = spec.args[1]  # Second arg is the paragraph
            if isinstance(para, Paragraph):
                nwords = para.size(TextUnit.words)
                para_text = abbrev_str(para.reassemble(), 30)
                return f"Research {i + 1}/{len(paragraphs)} ({nwords} words): {repr(para_text)}"
        return f"Research {i + 1}/{len(paragraphs)}"

    # Execute research in parallel with rate limiting, retries, and progress tracking
    limit = Limit(rps=global_settings().limit_rps, concurrency=global_settings().limit_concurrency)
    async with multitask_status() as status:
        paragraph_notes = await gather_limited_sync(
            *research_tasks, limit=limit, status=status, labeler=research_labeler
        )

    log.message(
        "Step 2: Applying %d sets of footnotes (%s total) to %d paragraphs",
        len(paragraph_notes),
        sum(len(notes or []) for notes in paragraph_notes),
        len(paragraphs),
    )

    # Create annotation tasks
    annotation_tasks = [
        FuncTask(annotate_paragraph, (para, notes))
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

    # Execute annotations in parallel (no rate limiting needed for local processing)
    async with multitask_status() as status:
        annotated_paras = await gather_limited_sync(
            *annotation_tasks,
            limit=limit,
            status=status,
            labeler=annotation_labeler,
        )

    # Consolidate all annotations into a single document with footnotes at the end
    log.message("Step 3: Consolidating footnotes at end of document")
    consolidated_doc = AnnotatedDoc.consolidate_annotations(annotated_paras)
    final_output = consolidated_doc.as_markdown_with_footnotes()

    # TODO: Remove near-duplicate footnotes.

    return item.derived_copy(type=ItemType.doc, body=final_output, format=Format.md_html)


@kash_action(llm_options=llm_options, live_output=True)
def research_paras(item: Item) -> Item:
    """
    Fact checks and researches each paragraph of a text.
    """
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    return asyncio.run(research_paras_async(item))
