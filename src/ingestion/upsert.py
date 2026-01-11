import hashlib
from typing import List

from llama_index.core.schema import TextNode

from src.services.qdrant import qdrant_service


import uuid


def _deterministic_uuid(source_id: str, chunk_text: str) -> uuid.UUID:
    """
    Generate a deterministic UUID (UUIDv5) from source + chunk text.
    """
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    name = f"{source_id}:{chunk_text}"
    return uuid.uuid5(namespace, name)


def upsert_nodes(
    nodes: List[TextNode],
    source_id: str,
) -> None:
    if not nodes:
        raise ValueError("No nodes provided for upsert")

    for node in nodes:
        if node.embedding is None:
            raise RuntimeError("Node has no embedding")

        point_id = _deterministic_uuid(source_id, node.text)

        payload = {
            "text": node.text,
            "source": source_id,
        }

        qdrant_service.upsert(
            point_id=point_id,
            vector=node.embedding,
            payload=payload,
        )
