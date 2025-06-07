from chopdiff.html import Attrs, div_wrapper, tag_with_attrs

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import has_simple_text_body
from kash.kits.docs.actions.text.summarize_structurally import summarize_structurally
from kash.llm_utils import LLM, LLMName
from kash.model import Format, Item, ItemType, common_params

log = get_logger(__name__)


SUMMARY = "summary"
"""Class name for the summary."""


ORIGINAL = "original"
"""Class name for the original content."""


def details_tag(summary: str, details: str) -> str:
    summary_tag = tag_with_attrs("summary", summary)
    details_tag = tag_with_attrs("details", summary_tag + "\n\n" + details, safe=True)
    return details_tag


@kash_action(
    precondition=has_simple_text_body,
    params=common_params("model", "model_list"),
)
def add_summary_bullets(
    item: Item, model: LLMName = LLM.default_standard, model_list: str | None = None
) -> Item:
    """
    Add a summary of the content (from `summarize_structurally`) above the full
    text of the item, with each wrapped in <details>/<summary> tags.

    Typically you'd use a single model, but for convenience, you may specify
    multiple models to use, so you can compare the different summaries.
    """
    models: list[LLMName] = []
    if model_list:
        models = [LLMName(model.strip()) for model in model_list.split(",")]
    else:
        models = [model]

    one_summary = len(models) == 1
    details_tags = []
    for model in models:
        summary_item = summarize_structurally(item, model=model)
        assert summary_item.body

        # If there's only one summary, make the title simply "Summary" and expand it by default.
        details_attrs: Attrs = {}
        if one_summary:
            details_attrs = {"open": True}
            summary_title = "Summary"
        else:
            summary_title = f"Summary ({model})"

        details_div = div_wrapper(class_name=SUMMARY, attrs={"data-model": model})(
            summary_item.body
        )
        summary_tag = tag_with_attrs("summary", summary_title)
        details_tag = tag_with_attrs(
            "details", summary_tag + "\n\n" + details_div, attrs=details_attrs, safe=True
        )
        details_tags.append(details_tag)
    summary_html = "\n\n".join(details_tags)

    assert item.body
    combined_body = summary_html + "\n\n" + div_wrapper(class_name=ORIGINAL)(item.body)

    combined_item = item.derived_copy(
        type=ItemType.doc,
        format=Format.md_html,
        body=combined_body,
    )

    return combined_item
