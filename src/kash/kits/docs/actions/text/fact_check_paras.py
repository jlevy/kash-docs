from __future__ import annotations

import asyncio

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.kits.docs.analysis.annotate_paras import annotate_paras_async
from kash.llm_utils import Message, MessageTemplate
from kash.model import Item, LLMOptions
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

      You are a fact checker for a top-tier publication with high standards
      for accuracy and rigor.

      You will be given one or more paragraphs, likely an excerpt from a longer text.

      Your task it to research all assertions or facts mentioned in the paragraphs and check
      for correctness.
      
      You should also verify identities of people are correct and that concepts or technical
      terms are used correctly and accurately.

      This will provide key context both for the author to revise the writing and for
      readers who wish for context and confidence in the accuracy of the material.

      Use web searches to find the most relevant and authoritative sources for all key
      entities and assertions.

      It is VERY important that writing and editorial notes are accurate and notable and
      always backed by reputable sources. NEVER give speculative information.

      The output should be a list of bulleted items in Markdown format
      (use - for bullets and DO NOT use • or other bullet characters).
      
      Each item should be ONE SENTENCE long and should cover whether you found sources
      that SUPPORT, REFUTE, or GIVE CONTEXT for the assertion. If you can't find a source,
      note it as UNCONFIRMED.


      1. Check if all links are valid and whether or not they support assertions made.

      2. Check all stated facts about people, places, things, software, and other entities
         are correct.

      3. Check all assertions made, especially about numbers, dates, costs, financial data,
         etc., as well as technical or scientific assertions that can be checked.

      Guidelines for the items: Each bullet should be one of the following:

      1. ONE sentence describing ONE resource and the stance it takes, e.g. "The New York
         Times confirmed the 2024 revenue for X was $Y" and then give the link to the
         confirmed source.

      2. If the source only weakly supports or refutes the assertion, note that it is
         weakly supporting or may shed doubt on the assertion.

      3. If you uncover an assertion that is unclear, unconfirmed, or controversial,
         note that as well. "X is a private company and we could not confirm that the
         revenue in 2024 was $Y."
         
      4. If you uncover controversy, note that: "As noted in this Substack post by A, people
         disagree on whether Palantir has significant room for revenue growth."


      IMPORTANT:

      - FETCH ALL LINKS. VERIFY ALL SOURCES.

      - USE WEB SEARCHES TO FIND ADDITIONAL SOURCES. 

      - Do fact check any factual assertion that is not obviously true.

      - Add AT MOST 10 annotations. Pick only the MOST IMPORTANT and RELEVANT fact checks.
      
      - DO NOT fact check obvious or common facts like that Ottowa is the capital
        of Canada.

      - Do NOT mention or include language *about* the text or the author of the text.
        The reader can see the text and know the context is the paragraph.
        Do not say "The text refers to..." or "X refers to ...". Simply say "X is ...".

      - Use ONLY ONE LINK per item. If there are multiple sources that both support
        and refute the assertion, link to each, as separate items.
      
      Additional guidelines:

      - Write in clean and and direct language. Keep it brief. Be concise as the links
        will have relevant detail.

      - Prefer sources that are likely to be considered authoritative and reliable.
        But if only obscure sources are available, link and note this:
        "An unconfirmed post on Reddit in does describe X happening" (and insert the link).

      - If the input is very short doesn't have any obvious facts or entities
        referenced, simply output "(No results)".

      - If you can't find a link for a concept, simply omit it.

      - Do not mention the text or the author. Simply state the editorial note.

      - DO NOT INCLUDE any other commentary.

      - If the input is in a language other than English, output the caption in the same language.

      Here are three examples to be very clear on our standards of quality and style.

      Example input text #1:

        As Einstein said, 'The only thing more dangerous than ignorance is arrogance.'
        But even companies that do not show arrogance are susceptible to Conway's Law.

      Sample output text #1:

        - This quote is often misattributed to Albert Einstein but appears in the
          2003 book *Still Life with Crows*
          ([Snopes](https://www.snopes.com/fact-check/einstein-dangerous-ignorance-arrogance/)).

      Example input text #2:

      Push ups are an underrated exercise.
      They are not easy to do well so are often done poorly.

      Sample output text #2:

      - Trainers call push-ups "underrated" because they combine upper-body strength work with
        core stabilization and require no gear, making them an efficient full-body move
        ([The Guardian](https://www.theguardian.com/lifeandstyle/2025/mar/09/push-up-power-the-exercise-you-need-for-a-healthy-happy-life-and-eight-ways-to-make-it-easier)).

      - Form faults such as flared elbows, sagging hips, or shallow depth are so common that
        multiple guides list them among the top push-up mistakes to fix ([Nerd
        Fitness](https://www.nerdfitness.com/blog/5-common-push-up-mistakes-to-avoid/)).

      - Beginner progressions—wall, knee, or incline push-ups—are widely recommended to master
        technique and reduce injury risk before advancing to the full movement ([The
        Guardian](https://www.theguardian.com/lifeandstyle/2025/mar/09/push-up-power-the-exercise-you-need-for-a-healthy-happy-life-and-eight-ways-to-make-it-easier)).

      Input text:

      {body}

      Output text:
        """
    ),
)

FN_PREFIX = "fc"


@kash_action(llm_options=llm_options, live_output=True, mcp_tool=True)
def fact_check_paras(item: Item) -> Item:
    """
    Fact check each paragraph of a text.
    """
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    return asyncio.run(annotate_paras_async(llm_options, item, fn_prefix=FN_PREFIX))
