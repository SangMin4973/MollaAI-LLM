from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient, models


logger = logging.getLogger("molla.llm.qdrant")


class UserMemoryStore:
    def __init__(
        self,
        *,
        url: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "user_memory")
        self.vector_size = vector_size or int(os.getenv("QDRANT_VECTOR_SIZE", "768"))
        self.client = QdrantClient(url=self.url)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            logger.info(
                "qdrant_collection_exists collection=%s url=%s",
                self.collection_name,
                self.url,
            )
            self._ensure_payload_indexes()
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
        )
        logger.info(
            "qdrant_collection_created collection=%s vector_size=%s url=%s",
            self.collection_name,
            self.vector_size,
            self.url,
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        indexes = (
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("text", models.PayloadSchemaType.TEXT),
            ("importance", models.PayloadSchemaType.FLOAT),
        )
        for field_name, field_schema in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                logger.exception(
                    "qdrant_payload_index_failed collection=%s field=%s",
                    self.collection_name,
                    field_name,
                )
