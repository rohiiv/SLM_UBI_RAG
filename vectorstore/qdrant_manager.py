"""
Banking RAG Qdrant Vector Store Manager.

Provides collection setup, batch payload upsert, vector similarity search, and payload metadata filtering.
"""

import threading
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from banking_rag.chunking.chunker import TextChunk
from banking_rag.config import QdrantConfig, get_config
from banking_rag.exceptions import VectorDBError
from banking_rag.utils.logger import get_logger

logger = get_logger("vectorstore.qdrant_manager")


@dataclass
class ScoredChunk:
    """Wrapper for a retrieved chunk accompanied by vector similarity score."""
    chunk: TextChunk
    score: float


class QdrantVectorStoreManager:
    """Wrapper around Qdrant client for collection management and vector search.

    The underlying client connection is a process-wide singleton: a class-level cache keyed by
    the effective connection target ensures the connection is established only once per process,
    no matter how many QdrantVectorStoreManager instances are constructed.
    """

    # Shared across ALL instances of this class. Maps connection-key -> (client, is_memory).
    _client_cache: Dict[Tuple[Any, ...], Tuple[Any, bool]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: Optional[QdrantConfig] = None):
        """Initializes Qdrant client connection.

        Args:
            config: Optional QdrantConfig instance.
        """
        self.config = config or get_config().qdrant
        self.collection_name = self.config.collection_name
        self.vector_size = self.config.vector_size
        self._client = None
        self._is_memory = False

    def _get_client(self):
        """Lazy initialization of Qdrant client, backed by a shared cache."""
        if self._client is not None:
            return self._client

        cache_key = (self.config.url, self.config.host, self.config.port, self.config.api_key)

        with QdrantVectorStoreManager._cache_lock:
            cached = QdrantVectorStoreManager._client_cache.get(cache_key)
            if cached is not None:
                self._client, self._is_memory = cached
                logger.info("Reusing already-established Qdrant connection from shared cache.")
                return self._client

            if not self.config.api_key:
                # AppConfig.__post_init__ already hard-fails outside dev environments; this
                # is a second, local reminder for whoever is running the dev/local path so
                # the gap doesn't get forgotten on the way to a real deployment.
                logger.warning(
                    "Connecting to Qdrant with NO API KEY configured. This is only acceptable "
                    "for local development against a Qdrant instance that isn't reachable from "
                    "outside this machine. Set QDRANT_API_KEY before pointing this at anything "
                    "containing real banking/compliance data."
                )

            try:
                from qdrant_client import QdrantClient
                if self.config.url:
                    logger.info(f"Connecting to Qdrant Cloud/URL: {self.config.url}")
                    client = QdrantClient(url=self.config.url, api_key=self.config.api_key)
                    is_memory = False
                elif self.config.host:
                    logger.info(f"Connecting to Qdrant server at {self.config.host}:{self.config.port}")
                    client = QdrantClient(host=self.config.host, port=self.config.port, api_key=self.config.api_key)
                    is_memory = False
                else:
                    logger.info("Initializing in-memory Qdrant instance.")
                    client = QdrantClient(location=":memory:")
                    is_memory = True

            except ImportError:
                logger.warning("qdrant-client not installed. Falling back to local in-memory mock client.")
                client = "MOCK"
                is_memory = True
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {str(e)}")
                raise VectorDBError(f"Qdrant connection failed: {str(e)}")

            QdrantVectorStoreManager._client_cache[cache_key] = (client, is_memory)
            self._client = client
            self._is_memory = is_memory
            return self._client

    def preload(self) -> None:
        """Forces the Qdrant client connection to be established immediately instead of lazily.

        Intended to be called once at application startup so the connection happens
        deterministically before the first query is served.
        """
        self._get_client()

    def create_collection(self, force_recreate: bool = False) -> bool:
        """Creates the Qdrant vector collection if it does not exist.

        Args:
            force_recreate: If True, deletes existing collection first.

        Returns:
            True if collection exists or was created successfully.
        """
        client = self._get_client()
        if client == "MOCK":
            logger.info("Mock Qdrant collection created.")
            return True

        try:
            from qdrant_client.http import models

            collections = client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if exists and force_recreate:
                logger.warning(f"Recreating existing Qdrant collection: {self.collection_name}")
                client.delete_collection(collection_name=self.collection_name)
                exists = False

            if not exists:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with vector size {self.vector_size}")
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            return True

        except Exception as e:
            logger.error(f"Error creating collection {self.collection_name}: {str(e)}")
            raise VectorDBError(f"Collection creation error: {str(e)}")

    def upsert_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> bool:
        """Upserts a batch of text chunks and vector embeddings into Qdrant.

        Args:
            chunks: List of TextChunk objects.
            embeddings: Corresponding vector embeddings list.

        Returns:
            True if upsert succeeded.

        Raises:
            VectorDBError: If sizes mismatch or DB operation fails.
        """
        if len(chunks) != len(embeddings):
            raise VectorDBError("Mismatch between number of chunks and embeddings.")

        if not chunks:
            return True

        client = self._get_client()
        self.create_collection(force_recreate=False)

        if client == "MOCK":
            logger.info(f"Mock upserted {len(chunks)} chunks into vector store.")
            return True

        try:
            from qdrant_client.http import models

            points = []
            for chunk, vector in zip(chunks, embeddings):
                # Ensure valid UUID string for Qdrant point ID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                
                payload = dict(chunk.metadata)
                payload["content"] = chunk.content
                payload["parent_doc_id"] = chunk.parent_doc_id

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            upsert_batch_size = get_config().ingestion.upsert_batch_size
            total_points = len(points)
            for i in range(0, total_points, upsert_batch_size):
                batch_points = points[i : i + upsert_batch_size]
                client.upsert(collection_name=self.collection_name, points=batch_points)

            logger.info(f"Successfully upserted {total_points} chunks into Qdrant collection '{self.collection_name}' in batches of {upsert_batch_size}")
            return True

        except Exception as e:
            logger.error(f"Error upserting vectors into Qdrant: {str(e)}")
            raise VectorDBError(f"Vector upsert failed: {str(e)}")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        query_filter: Optional[Any] = None,
    ) -> List[ScoredChunk]:
        """Performs vector similarity search in Qdrant with optional payload metadata filtering.

        Args:
            query_vector: Float list embedding of query.
            top_k: Number of nearest neighbors to retrieve.
            query_filter: Optional qdrant_client Filter object.

        Returns:
            List of ScoredChunk instances sorted by similarity score descending.
        """
        client = self._get_client()
        if client == "MOCK":
            logger.info("Mock search executed. Returning empty list.")
            return []

        try:
            # qdrant-client >=1.10 removed QdrantClient.search() in favor of query_points().
            # Support both so this works across whatever version ends up installed.
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                )
                results = response.points
            else:
                results = client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                )

            scored_chunks = []
            for hit in results:
                payload = hit.payload or {}
                content = payload.pop("content", "")
                parent_id = payload.pop("parent_doc_id", "")
                chunk_id = payload.get("chunk_id", str(hit.id))

                chunk = TextChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=payload,
                    parent_doc_id=parent_id,
                )
                scored_chunks.append(ScoredChunk(chunk=chunk, score=hit.score))

            logger.info(f"Retrieved {len(scored_chunks)} candidate chunks from Qdrant.")
            return scored_chunks

        except Exception as e:
            logger.error(f"Error executing similarity search in Qdrant: {str(e)}")
            raise VectorDBError(f"Similarity search failed: {str(e)}")
