"""
Banking RAG Dense Vector Retriever module.

Implements BaseRetriever interface executing vector similarity retrieval against Qdrant.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from banking_rag.config import RetrievalConfig, get_config
from banking_rag.embeddings.embedding_generator import BaseEmbeddingGenerator, BGEEmbeddingGenerator
from banking_rag.exceptions import RetrievalError
from banking_rag.retrieval.filters import MetadataFilterBuilder
from banking_rag.utils.logger import get_logger
from banking_rag.vectorstore.qdrant_manager import QdrantVectorStoreManager, ScoredChunk

logger = get_logger("retrieval.retriever")


class BaseRetriever(ABC):
    """Abstract Interface for Document Retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredChunk]:
        """Retrieves scored candidate chunks for a query.

        Args:
            query: User search query string.
            top_k: Optional override for top K candidate count.
            filters: Optional metadata filtering criteria dictionary.

        Returns:
            List of ScoredChunk objects sorted by relevance score descending.
        """
        pass


class DenseRetriever(BaseRetriever):
    """Dense vector retriever using BGE-M3 embeddings and Qdrant similarity search."""

    def __init__(
        self,
        embedding_generator: Optional[BaseEmbeddingGenerator] = None,
        vector_store: Optional[QdrantVectorStoreManager] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        """Initializes DenseRetriever with injected embedding generator and vector store.

        Args:
            embedding_generator: Strategy for converting text queries into dense vectors.
            vector_store: Manager for querying vector store.
            config: Retrieval settings.
        """
        self.embedding_generator = embedding_generator or BGEEmbeddingGenerator()
        self.vector_store = vector_store or QdrantVectorStoreManager()
        self.config = config or get_config().retrieval
        self.default_top_k = self.config.top_k_dense

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredChunk]:
        """Executes dense vector search.

        Args:
            query: Query string.
            top_k: Override count.
            filters: Metadata filter dictionary.

        Returns:
            List of ScoredChunk objects.

        Raises:
            RetrievalError: If search execution fails.
        """
        if not query.strip():
            logger.warning("Empty query passed to DenseRetriever.")
            return []

        k = top_k if top_k is not None else self.default_top_k
        logger.info(f"Executing dense retrieval for query: '{query}' (top_k={k})")

        try:
            # 1. Generate query embedding
            query_vector = self.embedding_generator.generate_query_embedding(query)

            # 2. Build metadata filter
            qdrant_filter = MetadataFilterBuilder.build_filter(filters)

            # 3. Query Qdrant vector store
            scored_chunks = self.vector_store.search(
                query_vector=query_vector,
                top_k=k,
                query_filter=qdrant_filter,
            )

            logger.info(f"DenseRetriever retrieved {len(scored_chunks)} candidate chunks.")
            return scored_chunks

        except Exception as e:
            logger.error(f"Dense retrieval failed for query '{query}': {str(e)}")
            raise RetrievalError(f"Dense vector retrieval error: {str(e)}")
