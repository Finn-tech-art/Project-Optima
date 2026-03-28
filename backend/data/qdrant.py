from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from backend.core.exceptions import ConfigurationError, PersistenceError
from backend.data.models import MemoryDocument


# This block defines the connection and collection defaults for Qdrant.
# It takes: the Qdrant endpoint, API key if needed, and vector configuration.
# It gives: one normalized config object for the vector-memory adapter.
@dataclass(slots=True, frozen=True)
class QdrantConfig:
    url: str
    api_key: str | None = None
    prefer_grpc: bool = False
    default_vector_size: int = 1536
    default_distance: qdrant_models.Distance = qdrant_models.Distance.COSINE


# This block stores one search hit returned from Qdrant.
# It takes: the raw scored point plus the decoded memory document.
# It gives: a structured retrieval result for downstream agent logic.
@dataclass(slots=True, frozen=True)
class MemorySearchResult:
    score: float
    document: MemoryDocument
    payload: dict[str, Any]


# This block is the main Qdrant adapter for semantic memory storage and retrieval.
# It takes: a validated QdrantConfig.
# It gives: collection management, upsert, delete, and similarity search methods.
class QdrantMemoryStore:
    def __init__(self, config: QdrantConfig) -> None:
        self.config = config
        self._validate_config()

        # This block creates the Qdrant client connection.
        # It takes: the configured URL, API key, and transport preference.
        # It gives: a live client object for vector operations.
        self.client = QdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            prefer_grpc=self.config.prefer_grpc,
        )

    # This block validates the adapter configuration before any network calls.
    # It takes: the config supplied by the caller.
    # It gives: early failure if required connection or vector settings are invalid.
    def _validate_config(self) -> None:
        if not self.config.url.strip():
            raise ConfigurationError("Qdrant URL is required.")

        if self.config.default_vector_size <= 0:
            raise ConfigurationError("Qdrant vector size must be greater than 0.")

    # This block ensures a target collection exists with the expected vector settings.
    # It takes: a collection name and optional vector overrides.
    # It gives: a ready collection for semantic memory operations.
    def ensure_collection(
        self,
        *,
        collection: str,
        vector_size: int | None = None,
        distance: qdrant_models.Distance | None = None,
    ) -> None:
        if not collection.strip():
            raise ConfigurationError("Qdrant collection name is required.")

        size = vector_size or self.config.default_vector_size
        metric = distance or self.config.default_distance

        try:
            exists = self.client.collection_exists(collection_name=collection)
            if exists:
                return

            self.client.create_collection(
                collection_name=collection,
                vectors_config=qdrant_models.VectorParams(
                    size=size,
                    distance=metric,
                ),
            )
        except Exception as error:  # noqa: BLE001
            raise PersistenceError(
                "Failed to ensure Qdrant collection.",
                context={
                    "collection": collection,
                    "vector_size": size,
                    "distance": str(metric),
                    "error": str(error),
                },
            ) from error

    # This block upserts one memory document into a Qdrant collection.
    # It takes: a MemoryDocument with a populated embedding vector.
    # It gives: the stored point id for later retrieval or deletion.
    def upsert_document(self, document: MemoryDocument) -> str:
        if document.vector is None or not document.vector:
            raise ConfigurationError("MemoryDocument.vector is required for Qdrant upsert.")

        self.ensure_collection(collection=document.collection, vector_size=len(document.vector))

        point_id = document.id or str(uuid4())
        payload = document.to_qdrant_payload()

        try:
            self.client.upsert(
                collection_name=document.collection,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=document.vector,
                        payload=payload,
                    )
                ],
                wait=True,
            )
        except Exception as error:  # noqa: BLE001
            raise PersistenceError(
                "Failed to upsert memory document into Qdrant.",
                context={
                    "collection": document.collection,
                    "document_id": point_id,
                    "error": str(error),
                },
            ) from error

        return point_id

    # This block upserts multiple memory documents in one request.
    # It takes: a list of MemoryDocument objects with populated vectors.
    # It gives: the list of stored point ids.
    def upsert_documents(self, documents: list[MemoryDocument]) -> list[str]:
        if not documents:
            return []

        collection = documents[0].collection
        for document in documents:
            if document.collection != collection:
                raise ConfigurationError("All documents in a batch must target the same collection.")
            if document.vector is None or not document.vector:
                raise ConfigurationError("Every MemoryDocument in a batch must include a vector.")

        vector_size = len(documents[0].vector)
        self.ensure_collection(collection=collection, vector_size=vector_size)

        points: list[qdrant_models.PointStruct] = []
        ids: list[str] = []

        # This block converts each typed memory document into a Qdrant point.
        # It takes: the batch of shared MemoryDocument models.
        # It gives: the point structs required by the Qdrant client.
        for document in documents:
            point_id = document.id or str(uuid4())
            ids.append(point_id)
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=document.vector,
                    payload=document.to_qdrant_payload(),
                )
            )

        try:
            self.client.upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )
        except Exception as error:  # noqa: BLE001
            raise PersistenceError(
                "Failed to batch upsert memory documents into Qdrant.",
                context={
                    "collection": collection,
                    "count": len(documents),
                    "error": str(error),
                },
            ) from error

        return ids

    # This block searches a collection using a query embedding.
    # It takes: a collection name, query vector, result limit, and optional metadata filter.
    # It gives: scored semantic matches decoded back into MemoryDocument objects.
    def search(
        self,
        *,
        collection: str,
        query_vector: list[float],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemorySearchResult]:
        if not collection.strip():
            raise ConfigurationError("Qdrant collection name is required.")
        if not query_vector:
            raise ConfigurationError("Qdrant query vector is required.")
        if limit <= 0:
            raise ConfigurationError("Qdrant search limit must be greater than 0.")

        query_filter = self._build_filter(metadata_filter)

        try:
            points = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as error:  # noqa: BLE001
            raise PersistenceError(
                "Failed to search Qdrant collection.",
                context={
                    "collection": collection,
                    "limit": limit,
                    "error": str(error),
                },
            ) from error

        results: list[MemorySearchResult] = []

        # This block converts raw Qdrant hits into typed memory-search results.
        # It takes: the scored points returned by Qdrant.
        # It gives: typed search results for later retrieval-aware agent logic.
        for point in points:
            payload = dict(point.payload or {})
            document = MemoryDocument.model_validate(
                {
                    "id": str(payload.get("id", point.id)),
                    "collection": payload.get("collection", collection),
                    "text": payload.get("text", ""),
                    "symbol": payload.get("symbol"),
                    "trace_id": payload.get("trace_id"),
                    "tags": payload.get("tags", []),
                    "metadata": payload.get("metadata", {}),
                    "source": payload.get("source", "system"),
                    "created_at": payload.get("created_at"),
                }
            )
            results.append(
                MemorySearchResult(
                    score=float(point.score),
                    document=document,
                    payload=payload,
                )
            )

        return results

    # This block deletes a stored memory document by point id.
    # It takes: the collection name and document id.
    # It gives: removal of the matching point from Qdrant.
    def delete_document(self, *, collection: str, document_id: str) -> None:
        if not collection.strip():
            raise ConfigurationError("Qdrant collection name is required.")
        if not document_id.strip():
            raise ConfigurationError("Qdrant document id is required.")

        try:
            self.client.delete(
                collection_name=collection,
                points_selector=qdrant_models.PointIdsList(
                    points=[document_id],
                ),
                wait=True,
            )
        except Exception as error:  # noqa: BLE001
            raise PersistenceError(
                "Failed to delete memory document from Qdrant.",
                context={
                    "collection": collection,
                    "document_id": document_id,
                    "error": str(error),
                },
            ) from error

    # This block converts a simple metadata dict into a Qdrant filter.
    # It takes: exact-match metadata fields from the caller.
    # It gives: a Qdrant filter object or None if no filter was requested.
    def _build_filter(
        self,
        metadata_filter: dict[str, Any] | None,
    ) -> qdrant_models.Filter | None:
        if not metadata_filter:
            return None

        conditions: list[qdrant_models.FieldCondition] = []

        for key, value in metadata_filter.items():
            conditions.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchValue(value=value),
                )
            )

        return qdrant_models.Filter(must=conditions)
