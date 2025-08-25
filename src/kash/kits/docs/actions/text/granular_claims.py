from __future__ import annotations

from chopdiff.docs import TextDoc
from frontmatter_format import to_yaml_string

from kash.config.logger import get_logger
from kash.exec import kash_action
from kash.exec.preconditions import (
    has_div_chunks,
    has_html_body,
    has_simple_text_body,
    is_docx_resource,
    is_pdf_resource,
    is_url_resource,
)
from kash.kits.docs.actions.text.markdownify_doc import markdownify_doc
from kash.kits.docs.analysis.analysis_model import ClaimAnalysis, DocAnalysis
from kash.kits.docs.analysis.claim_mapping import extract_mapped_claims
from kash.kits.docs.analysis.doc_chunking import ChunkedDoc
from kash.llm_utils import LLM, LLMName
from kash.model import Format, Item, ItemType, common_param
from kash.utils.errors import InvalidInput

log = get_logger(__name__)


@kash_action(
    precondition=(
        is_url_resource | is_docx_resource | is_pdf_resource | has_html_body | has_simple_text_body
    )
    & ~has_div_chunks,
    params=(common_param("model"),),
)
def granular_claims(
    item: Item,
    model: LLMName = LLM.default_standard,
) -> Item:
    """
    A partial doc analysis, good for debugging granular claims.
    Returns a YAML data item listing granular claims and their mapped chunk IDs and source URLs.
    """
    # Convert to markdown if needed.
    as_markdown = markdownify_doc(item)
    if not as_markdown.body:
        raise InvalidInput(f"Item must have a body: {item}")

    # Chunk the document.
    log.message("Chunking document...")
    text_doc = TextDoc.from_text(as_markdown.body)
    chunked_doc = ChunkedDoc.from_text_doc(text_doc, min_size=1)

    # Compute mapped claims with only granular claims enabled.
    log.message("Extracting mapped granular claims...")
    mapped = extract_mapped_claims(
        chunked_doc,
        include_key_claims=False,
        include_granular_claims=True,
    )

    granular_analyses: list[ClaimAnalysis] = []
    for mc in mapped.granular_claims:
        granular_analyses.append(
            ClaimAnalysis(
                claim=mc.claim,
                chunk_ids=[cs.chunk_id for cs in mc.related_chunks],
                source_urls=mc.source_urls,
                chunk_similarity=[cs.similarity for cs in mc.related_chunks],
                rigor_analysis=None,
                claim_support=[],
                labels=[],
            )
        )

    doc_analysis = DocAnalysis(
        key_claims=[], granular_claims=granular_analyses, footnotes=chunked_doc.footnote_mapping
    )
    return item.derived_copy(
        type=ItemType.data, format=Format.yaml, body=to_yaml_string(doc_analysis.model_dump())
    )
