from typing import List, Dict, Any
import logging

from src.services.qdrant import qdrant_service
from src.config.settings import settings

logger = logging.getLogger(__name__)


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

    logger.info(f"Qdrant returned {len(results)} raw results")

    retrieved = []

    for point in results:
        if point.score < 0.5:
            logger.debug(f"Filtered out chunk with score {point.score:.3f}")
            continue
        payload = point.payload or {}

        chunk = {
            "text": payload.get("text", ""),
            "source": payload.get("source", ""),
            "score": point.score,
        }
        retrieved.append(chunk)
        logger.debug(f"Retrieved chunk (score {point.score:.3f}): {chunk['text'][:100]}...")

    if not retrieved:
        raise RuntimeError("No similar documents found")

    logger.info(f"Returning {len(retrieved)} chunks after filtering (score >= 0.5)")
    return retrieved
