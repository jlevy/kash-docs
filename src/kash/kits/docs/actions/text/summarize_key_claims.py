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
        You are to carefully summarize the most important key claims in the text below.

        The goal is to summarize key claims in a single sentence. These should be the
        *most important* overall claims or arguments made by the document.

        To help you with this, you should read the whole document but there is also
        an executive summary that is structured as bullet points, above the rest of the
        text.

        The goal should be to have **3 to 10** key claims, organized as bullet points, one sentence each.

        These should be very clear written with the same level of technical detail as the
        original text.

        Keep them short! Make them concise but do include all relevant terms and details.

        If the document makes many claims, pick the most important or key ones an expert
        analyst who understands the material would pick as most relevant.

        Input text:

        {body}

        Key claims:
        """
    ),
)


@kash_action(llm_options=llm_options, params=common_params("model"), mcp_tool=True)
def summarize_key_claims(item: Item, model: LLMName = LLM.default_standard) -> Item:
    """
    Summarize the key claims in the document.
    """
    return llm_transform_item(item, model=model)
