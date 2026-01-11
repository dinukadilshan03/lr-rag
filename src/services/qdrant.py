from typing import List, Dict, Any
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from src.config.settings import settings


class QdrantService:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.COLLECTION_NAME
        self.vector_size = settings.VECTOR_DIMENSION

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Create the collection if it does not exist.
        """
        collections = self.client.get_collections().collections
        existing_names = {c.name for c in collections}

        if self.collection_name in existing_names:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert(
        self,
        point_id: UUID,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """
        Insert or update a single vector point.
        """
        point = PointStruct(
            id=str(point_id),
            vector=vector,
            payload=payload,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        
    def search(
        self,
        query_vector: List[float],
        limit: int | None = None,
    ):
        """
        Perform a similarity search.
        """
        limit = limit or settings.TOP_K

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )

        return result.points


# Singleton-style access
qdrant_service = QdrantService()
