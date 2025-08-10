from chopdiff.html import div_wrapper

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_simple_text_body
from kash.kits.docs.actions.text.summarize_key_claims import summarize_key_claims
from kash.llm_utils import LLM, LLMName
from kash.model import Format, Item, ItemType, common_params
from kash.utils.text_handling.markdown_utils import extract_bullet_points

log = get_logger(__name__)


KEY_CLAIMS = "key-claims"
"""Class name for the key claims."""

CLAIM = "claim"
"""Class name for individual claims."""


@kash_action(
    precondition=has_simple_text_body,
    params=common_params("model"),
)
def add_summary_key_claims(item: Item, model: LLMName = LLM.default_standard) -> Item:
    """
    Add a summary of the key claims in the document.
    """
    summary_item = summarize_key_claims(item, model=model)
    assert summary_item.body

    # Extract bullet points from the summary
    bullet_points = extract_bullet_points(summary_item.body)

    # Wrap each bullet point in a claim div
    claim_divs = []
    for bullet in bullet_points:
        claim_div = div_wrapper(class_name=CLAIM)(bullet)
        claim_divs.append(claim_div)

    # Join all claim divs
    claims_content = "\n\n".join(claim_divs)

    # Wrap all claims in the key-claims div
    summary_div = div_wrapper(class_name=KEY_CLAIMS)(claims_content)

    assert item.body
    combined_body = summary_div + "\n\n" + item.body

    combined_item = item.derived_copy(type=ItemType.doc, format=Format.md_html, body=combined_body)

    return combined_item
