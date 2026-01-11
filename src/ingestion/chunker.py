from typing import List

from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.schema import TextNode, Document


def chunk_text(text: str) -> List[TextNode]:
    """
    Split raw text into overlapping chunks (TextNodes).
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")

    # Wrap raw text into a Document (required by LlamaIndex)
    document = Document(text=text)

    splitter = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    nodes = splitter.get_nodes_from_documents([document])

    if not nodes:
        raise RuntimeError("Text splitting produced no chunks")

    return nodes
