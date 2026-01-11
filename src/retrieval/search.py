from typing import List, Dict, Any

from src.services.qdrant import qdrant_service
from src.config.settings import settings


def search_similar(
    query_vector: List[float],
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Search Qdrant for vectors similar to the query vector.
    Returns retrieved text chunks with metadata.
    """
    if not query_vector:
        raise ValueError("Query vector is empty")

    results = qdrant_service.search(
        query_vector=query_vector,
        limit=top_k or settings.TOP_K,
    )

    retrieved = []

    for point in results:
        if point.score < 0.5:
            continue
        payload = point.payload or {}

        retrieved.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "score": point.score,
            }
        )

    if not retrieved:
        raise RuntimeError("No similar documents found")

    return retrieved
