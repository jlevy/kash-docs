from kash.exec import kash_action
from kash.exec.preconditions import has_simple_text_body
from kash.kits.docs.analysis.claim_extraction import extract_granular_claims_text
from kash.llm_utils import LLM, LLMName
from kash.model import Format, Item, ItemType, common_params
from kash.utils.errors import InvalidInput


@kash_action(
    precondition=has_simple_text_body,
    output_format=Format.markdown,
    params=common_params("model"),
)
def extract_granular_claims(item: Item, model: LLMName = LLM.default_standard) -> Item:
    """
    Extract specific granular claims in a piece of text (typically one or more paragraphs).
    Returns just the Markdown text of the claims.
    """
    if not item.body:
        raise InvalidInput(f"Item must have a body: {item}")

    claims = extract_granular_claims_text(item.body)
    return item.derived_copy(type=ItemType.doc, format=Format.markdown, body=claims.text)
