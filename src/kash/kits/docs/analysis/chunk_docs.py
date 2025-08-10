from dataclasses import dataclass

from chopdiff.divs import CHUNK, chunk_paras, div
from chopdiff.docs import Paragraph, TextDoc, TextUnit

from kash.kits.docs.analysis.analysis_model import chunk_id


@dataclass
class ChunkedTextDoc:
    """
    A TextDoc with its paragraphs grouped into chunks and mapped by chunk ID.

    Each chunk contains one or more paragraphs grouped together based on
    the min_size constraint. Chunks are numbered sequentially.
    """

    doc: TextDoc
    chunks: dict[str, list[Paragraph]]

    def reassemble(self, class_name: str = CHUNK) -> str:
        """
        Reassemble the chunked document as HTML divs with chunk IDs.

        Each chunk becomes a div with its chunk ID, containing the
        reassembled paragraphs from that chunk.
        """
        result_divs = []
        for cid, paragraphs in self.chunks.items():
            # Reassemble all paragraphs in this chunk
            chunk_str = "\n\n".join(para.reassemble() for para in paragraphs)
            result_divs.append(div(class_name, chunk_str, attrs={"id": cid}))

        return "\n\n".join(result_divs)


def chunk_paragraphs(doc: TextDoc, min_size: int) -> ChunkedTextDoc:
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

    chunks = {}
    for i, chunk_doc in enumerate(doc_chunks):
        cid = chunk_id(i)
        chunks[cid] = chunk_doc.paragraphs

    return ChunkedTextDoc(doc=doc, chunks=chunks)
