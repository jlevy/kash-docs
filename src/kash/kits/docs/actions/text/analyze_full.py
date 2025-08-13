from __future__ import annotations

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import (
    has_html_body,
    has_simple_text_body,
    is_docx_resource,
    is_pdf_resource,
    is_url_resource,
)
from kash.kits.docs.actions.text.analyze_key_claims import analyze_key_claims
from kash.kits.docs.actions.text.markdownify_doc import markdownify_doc
from kash.kits.docs.actions.text.research_paras import research_paras
from kash.llm_utils import LLM, LLMName
from kash.model import Item, Param, common_param

log = get_logger(__name__)


@kash_action(
    precondition=is_url_resource
    | is_docx_resource
    | is_pdf_resource
    | has_html_body
    | has_simple_text_body,
    params=(
        common_param("model"),
        Param(
            "include_debug",
            description="Include extra debugging details in the output doc (as divs with a debug class)",
            type=bool,
        ),
    ),
    mcp_tool=True,
)
def analyze_full(
    item: Item, model: LLMName = LLM.default_standard, include_debug: bool = False
) -> Item:
    """
    Do a full analysis of a document with research, key claims, and evidence.
    """
    as_markdown = markdownify_doc(item)

    with_research = research_paras(as_markdown)

    with_claims_analysis = analyze_key_claims(with_research)

    return with_claims_analysis
