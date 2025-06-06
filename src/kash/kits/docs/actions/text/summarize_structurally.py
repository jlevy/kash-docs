from kash.exec import kash_action, llm_transform_item
from kash.llm_utils import LLM, LLMName, Message, MessageTemplate
from kash.model import Item, LLMOptions, common_params

llm_options = LLMOptions(
    system_message=Message(
        """
        You are a careful and precise editor.
        You give exactly the results requested without additional commentary.
        """
    ),
    body_template=MessageTemplate(
        """
        Summarize the following text into a hierarchical structure of nested bullet points.

        - Format your response as a list of nested bullet points in Markdown format.

        - **Formatting rules:**

          - ONLY use standard Markdown bullet points (`-` or `*` or `+`).
          
          - Do NOT use Unicode bullets (like `•` or other special formatting characters).

          - Do NOT use headings (e.g. `#`, `##`, etc).

          - Do NOT use other formatting such enumerated lists.

          - You may use standard Markdown links (`[link text](https://example.com)`) or
            italics (`*like this*`) or boldface (`**like this**`) if appropriate.

        - **Use of nested bullet points:**

          - If there is a stand-alone fact, list it as a top-level bullet, as a
            sentence (like the first point above). 

          - If there are several related facts, list them as a nested Markdown bullet
            point (like this one).

          - Use nesting strictly and logically to reflect the structure and organization
            of the original content as closely as possible.
            
          - If the topic naturally breaks into sections, use outer bullets label each
            section.
            
          - If these are larger, you may **boldface** the outermost bulleted items.
            Just be sure to do it consistently. (As in these instructions.)

          - For short documents, typically you will nest 2 levels (bullets and sub-bullets).
            For longer documents or talks, you may wish to use 3 levels.
            Use the structure that best reflects the original content.

          - Again, do NOT use headers or any other formatting.
        
        - **What to include:**

          - Include all key numbers or facts, without omitting any claims or important details.

          - Do NOT give any additional response at the beginning, such as "Here are the concise
            bullet points". Simply give the summary.
          
          - It is very important you do not add any details that are not directly stated in the
            original text. Do not change any numbers or alter its meaning in any way.
      
        - **Style guidelines:**

          - Use simple and precise language.

          - Simply state the facts or claims in the text. Do NOT reference the text or the author.
            
            - Example: If the text is about cheese being nutritious, say "Cheese is nutritious."
              But do NOT say "The author says cheese is nutritious" or "According to the text,
              cheese is nutritious."

        - If the input is very short or so unclear you can't summarize it, simply output
          "(No results)".

        - If the input is in a language other than English, output the summary in the same
          language.

        Input text:

        {body}

        Hierarchical bullet points:
        """
    ),
)


@kash_action(llm_options=llm_options, params=common_params("model"), mcp_tool=True)
def summarize_structurally(item: Item, model: LLMName = LLM.default_standard) -> Item:
    """
    Summarize text as a hierarchical structure of nested bullet points.
    """
    return llm_transform_item(item, model=model)
