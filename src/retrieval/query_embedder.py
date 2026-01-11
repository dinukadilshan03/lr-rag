from typing import List

from src.services.ollama import ollama_client


def embed_query(query: str) -> List[float]:
    """
    Embed a user query into a vector for similarity search.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    vector = ollama_client.embed(query)

    if not vector:
        raise RuntimeError("Failed to generate query embedding")

    return vector
