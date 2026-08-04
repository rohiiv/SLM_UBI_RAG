"""
Banking RAG Cross-Encoder Reranker module.

Implements BaseReranker interface to score query-chunk pairs using cross-encoder models (e.g., bge-reranker-large).
"""

import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Any

from banking_rag.config import ModelConfig, RetrievalConfig, get_config
from banking_rag.exceptions import RerankingError
from banking_rag.utils.logger import get_logger
from banking_rag.vectorstore.qdrant_manager import ScoredChunk

logger = get_logger("retrieval.reranker")


class BaseReranker(ABC):
    """Abstract Interface for Document Rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: Optional[int] = None,
    ) -> List[ScoredChunk]:
        """Reranks candidate chunks based on joint query-content relevance scoring.

        Args:
            query: User query string.
            candidates: Candidate ScoredChunk list from initial retrieval.
            top_k: Top K results to return.

        Returns:
            Reranked list of ScoredChunk objects sorted by cross-encoder score descending.
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder Reranker using SentenceTransformers CrossEncoder or mock fallback.

    Backed by a class-level cache keyed by (model_name, device) so the cross-encoder weights
    are loaded from disk only once per process, no matter how many CrossEncoderReranker
    instances get constructed.
    """

    # Shared across ALL instances of this class. Maps (model_name, device) -> model.
    _model_cache: Dict[Tuple[str, str], Any] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
    ):
        """Initializes CrossEncoderReranker.

        Args:
            model_config: Model configuration settings.
            retrieval_config: Retrieval configuration settings.
        """
        self.model_config = model_config or get_config().model
        self.retrieval_config = retrieval_config or get_config().retrieval
        
        self.model_name = self.model_config.reranker_model_name
        self.model_revision = self.model_config.reranker_model_revision
        self.device = self.model_config.device
        self.default_top_k = self.retrieval_config.top_k_reranked
        self._model = None

    def _load_model(self) -> None:
        """Lazy model loader for CrossEncoder, backed by a shared cache."""
        if self._model is not None:
            return

        cache_key = (self.model_name, self.device)

        with CrossEncoderReranker._cache_lock:
            cached = CrossEncoderReranker._model_cache.get(cache_key)
            if cached is not None:
                self._model = cached
                logger.info(f"Reusing already-loaded CrossEncoder model '{self.model_name}' from shared cache.")
                return

            if not self.model_revision:
                logger.warning(
                    f"Loading reranker model '{self.model_name}' with NO pinned revision. Set "
                    f"RERANKER_MODEL_REVISION to a specific commit SHA before real data is loaded."
                )
            logger.info(f"Loading CrossEncoder model '{self.model_name}' (revision={self.model_revision or 'UNPINNED'}) on device '{self.device}'...")
            try:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(self.model_name, device=self.device, revision=self.model_revision or None)
                logger.info("CrossEncoder reranker model loaded successfully.")
            except ImportError:
                logger.warning("sentence_transformers not installed. Using mock score passthrough for reranker.")
                model = "MOCK"
            except Exception as e:
                logger.error(f"Failed to load reranker model {self.model_name}: {str(e)}")
                raise RerankingError(f"Reranker loading error: {str(e)}")

            CrossEncoderReranker._model_cache[cache_key] = model
            self._model = model

    def preload(self) -> None:
        """Forces the CrossEncoder model to load immediately instead of lazily on first use.

        Intended to be called once at application startup so model loading happens
        deterministically before the first query is served.
        """
        self._load_model()

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: Optional[int] = None,
    ) -> List[ScoredChunk]:
        """Reranks candidate chunks.

        Args:
            query: Natural language query string.
            candidates: Initial list of ScoredChunk candidates.
            top_k: Maximum top K chunks to return.

        Returns:
            Reranked list of ScoredChunk objects.

        Raises:
            RerankingError: If cross-encoder scoring fails.
        """
        if not candidates:
            return []

        if not query.strip():
            return candidates[:top_k] if top_k else candidates

        k = top_k if top_k is not None else self.default_top_k
        logger.info(f"Reranking {len(candidates)} candidates for query '{query}' (top_k={k})")

        self._load_model()

        try:
            if self._model == "MOCK":
                logger.debug("Executing mock cross-encoder scoring.")
                return candidates[:k]

            # Prepare query-text pairs
            pairs = [(query, item.chunk.content) for item in candidates]
            scores = self._model.predict(pairs)

            reranked = []
            for item, score in zip(candidates, scores):
                reranked.append(ScoredChunk(chunk=item.chunk, score=float(score)))

            # Sort descending by cross-encoder score
            reranked.sort(key=lambda x: x.score, reverse=True)
            result = reranked[:k]

            logger.info(f"CrossEncoderReranker finished reranking. Returning top {len(result)} chunks.")
            return result

        except Exception as e:
            logger.error(f"Error during cross-encoder reranking: {str(e)}")
            raise RerankingError(f"Reranking failed: {str(e)}")
