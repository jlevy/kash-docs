from dataclasses import dataclass
from functools import cached_property
from typing import Self

from chopdiff.divs import CHUNK, chunk_paras, div
from chopdiff.docs import Paragraph, TextDoc, TextUnit, first_wordtok, is_tag

from kash.config.logger import get_logger
from kash.kits.docs.analysis.analysis_model import FootnoteDetail, SourceUrl
from kash.kits.docs.analysis.analysis_types import ChunkId, FootnoteId, RefId, chunk_id_str
from kash.kits.docs.analysis.doc_annotations import AnnotatedDoc, AnnotatedPara
from kash.kits.docs.links.links_model import LinkResults
from kash.utils.common.url import Url
from kash.utils.text_handling.markdown_footnotes import MarkdownFootnotes

log = get_logger(__name__)


@dataclass(frozen=True)
class ChunkedDoc:
    """
    A TextDoc with its paragraphs grouped into chunks and mapped by chunk ID.
    Each chunk contains one or more paragraphs grouped together based on
    the min_size constraint. Chunks are numbered sequentially.
    """

    doc: TextDoc

    chunks: dict[ChunkId, list[Paragraph]]
    """
    Mapping from chunk ID to list of paragraphs in the chunk.
    This is the complete set of chunks covering all paragraphs in the document.
    """

    @classmethod
    def from_text_doc(cls, doc: TextDoc, min_size: int) -> Self:
        """
        Chunk a TextDoc's paragraphs into groups and return a ChunkedTextDoc.

        Paragraphs are grouped together to meet the minimum size requirement
        (measured in number of paragraphs). Each chunk is numbered sequentially
        (chunk-0, chunk-1, etc).
        """

        # Use chunk_paras to group paragraphs based on size constraints
        # TODO: Have a min_sentences and add paragraphs until chunk is big enough.
        # TODO: Also handle section headers intelligently.
        doc_chunks = list(chunk_paras(doc, min_size, TextUnit.paragraphs))

        chunks: dict[ChunkId, list[Paragraph]] = {}
        for i, chunk_doc in enumerate(doc_chunks):
            chunks[chunk_id_str(i)] = chunk_doc.paragraphs

        return cls(doc=doc, chunks=chunks)

    # TODO: Add static parse method to read from Markdown with HTML chunk divs.

    @cached_property
    def annotated_doc(self) -> AnnotatedDoc:
        """
        Get this doc as an AnnotatedDoc, to allow access to parsed footnotes.
        """
        return AnnotatedDoc.from_doc_with_footnotes(self.doc)

    @property
    def footnote_mapping(self) -> dict[FootnoteId, FootnoteDetail]:
        """
        Get the footnotes from the annotated doc.
        """
        footnotes: dict[FootnoteId, FootnoteDetail] = {}
        for (
            footnote_id,
            footnote_info,
        ) in self.annotated_doc.footnote_mapping.items():
            footnotes[FootnoteId(footnote_id)] = FootnoteDetail(
                content=footnote_info.content,
                urls=list(footnote_info.urls),
            )
        return footnotes

    @cached_property
    def markdown_footnotes(self) -> MarkdownFootnotes:
        md_footnotes = MarkdownFootnotes.from_markdown(self.doc.reassemble())
        log.message(
            "Found %d markdown footnotes on doc, %s",
            len(md_footnotes.footnotes),
            self.doc.size_summary(),
        )
        return md_footnotes

    def annotated_chunk(self, chunk_id: ChunkId) -> list[AnnotatedPara]:
        """
        Annotate paragraphs for a given chunk.
        """
        return [
            AnnotatedPara.from_para_with_footnotes(para, self.markdown_footnotes)
            for para in self.chunks[chunk_id]
        ]

    def _get_urls_for_chunk(self, chunk_id: ChunkId) -> dict[Url, RefId]:
        """
        Get unique URLs in a chunk, mapped to either the chunk ID or a footnote ID, as appropriate.
        Footnote ids take precedence over chunk IDs.
        """
        url_map: dict[Url, RefId] = {}
        for ann_para in self.annotated_chunk(chunk_id):
            for url, ref_id in ann_para.get_urls().items():
                # Prefer the first match, assigning to footnote id if relevant, or else the chunk id.
                if url not in url_map:
                    url_map[url] = ref_id or chunk_id

        return url_map

    def get_source_urls(
        self, chunk_ids: list[ChunkId], source_links: LinkResults | None
    ) -> list[SourceUrl]:
        """
        Get source URLs for the given chunk IDs.
        Merges in info about sources, status etc if available.
        """

        source_urls: list[SourceUrl] = []
        for chunk_id in chunk_ids:
            for url, ref_id in self._get_urls_for_chunk(chunk_id).items():
                status = None
                status_code = None
                content_md_path = None
                if source_links:
                    link = source_links.get_link(url)
                    if link:
                        status = link.status
                        status_code = link.status_code
                        content_md_path = link.content_md_path
                source_urls.append(
                    SourceUrl(
                        ref_id=ref_id,
                        url=url,
                        status=status,
                        status_code=status_code,
                        content_md_path=content_md_path,
                        doc_info=None,
                    )
                )

        return source_urls

    def is_content_chunk(self, cid: ChunkId) -> bool:
        """
        XXX Heuristic to verify a chunk is content and not a header or markup like a div.
        """
        return all(
            not is_tag(first_wordtok(p.reassemble())) and not p.is_header()
            for p in self.chunks[cid]
        )

    def reassemble(self, class_name: str = CHUNK) -> str:
        """
        Reassemble the chunked document as HTML divs with chunk IDs,
        skipping any headers or markup chunks like divs.

        Each chunk becomes a div with its chunk ID, containing the
        reassembled paragraphs from that chunk.
        """
        result_divs = []
        for cid, paragraphs in self.chunks.items():
            # Reassemble all paragraphs in this chunk
            para_strs = [para.reassemble() for para in paragraphs]
            chunk_str = "\n\n".join(para_strs)

            if self.is_content_chunk(cid):
                result_divs.append(div(class_name, chunk_str, attrs={"id": cid}))
            else:
                result_divs.append(chunk_str)

        return "\n\n".join(result_divs)
