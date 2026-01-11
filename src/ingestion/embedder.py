from typing import List

from llama_index.core.schema import TextNode

from src.services.ollama import ollama_client


def embed_nodes(nodes: List[TextNode]) -> List[TextNode]:
    """
    Generate embeddings for each TextNode using Ollama.
    The embedding is attached to the node.
    """
    if not nodes:
        raise ValueError("No nodes provided for embedding")

    embedded_nodes: List[TextNode] = []

    for node in nodes:
        if not node.text or not node.text.strip():
            continue

        vector = ollama_client.embed(node.text)

        # Attach embedding to node
        node.embedding = vector

        embedded_nodes.append(node)

    if not embedded_nodes:
        raise RuntimeError("Embedding produced no valid nodes")

    return embedded_nodes
