"""
Banking RAG Embedding Generator module.

Provides abstract interface and BGE-M3 embedding generator supporting dense vector generation.
"""

import os
import threading
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Tuple, Any
import numpy as np

from banking_rag.config import ModelConfig, get_config
from banking_rag.constants import DEFAULT_DENSE_VECTOR_DIM
from banking_rag.exceptions import EmbeddingError
from banking_rag.utils.logger import get_logger

logger = get_logger("embeddings.embedding_generator")


class BaseEmbeddingGenerator(ABC):
    """Abstract Interface for Embedding Generators."""

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of input texts.

        Args:
            texts: List of text strings.

        Returns:
            List of float vector embeddings.
        """
        pass

    @abstractmethod
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generates a dense vector embedding for a single query string.

        Args:
            query: Query text string.

        Returns:
            Dense float vector.
        """
        pass

    @property
    @abstractmethod
    def vector_dimension(self) -> int:
        """Returns the dimension of output embeddings."""
        pass


class BGEEmbeddingGenerator(BaseEmbeddingGenerator):
    """BGE-M3 Embedding Generator supporting high-quality dense vector representations.

    The underlying model weights are process-wide singletons: a class-level cache keyed by
    (model_name, device) ensures that regardless of how many BGEEmbeddingGenerator instances
    are constructed, the actual model is loaded from disk only once per process.
    """

    # Shared across ALL instances of this class. Maps (model_name, device) -> (model, dim).
    _model_cache: Dict[Tuple[str, str], Tuple[Any, int]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initializes the BGE-M3 embedding model.

        Args:
            config: Optional ModelConfig object.
        """
        self.config = config or get_config().model
        self.model_name = self.config.embedding_model_name
        self.model_revision = self.config.embedding_model_revision
        self.device = self.config.device
        self._dim = DEFAULT_DENSE_VECTOR_DIM
        self._model = None

    def _load_model(self) -> None:
        """Lazy loader for sentence_transformers / HuggingFace model, backed by a shared cache."""
        if self._model is not None:
            return

        cache_key = (self.model_name, self.device)

        with BGEEmbeddingGenerator._cache_lock:
            # Another instance may have already loaded this exact model.
            cached = BGEEmbeddingGenerator._model_cache.get(cache_key)
            if cached is not None:
                self._model, self._dim = cached
                logger.info(f"Reusing already-loaded embedding model '{self.model_name}' from shared cache.")
                return

            if not self.model_revision:
                logger.warning(
                    f"Loading embedding model '{self.model_name}' with NO pinned revision - it "
                    f"will resolve to whatever 'main' currently points to on Hugging Face. Set "
                    f"EMBEDDING_MODEL_REVISION to a specific commit SHA before real data is loaded, "
                    f"to close the supply-chain risk of an upstream model update changing weights "
                    f"underneath you unnoticed."
                )
            logger.info(f"Loading embedding model '{self.model_name}' (revision={self.model_revision or 'UNPINNED'}) on device '{self.device}'...")
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(
                    self.model_name, device=self.device,
                    revision=self.model_revision or None,
                )

                # bge-m3 natively supports sequences up to 8192 tokens. Left uncapped, a batch
                # gets padded to that full length, which on some backends (notably Apple MPS,
                # which lacks an efficient attention kernel for very long sequences) tries to
                # allocate a full-size attention buffer and fails ("Invalid buffer size").
                # Our chunks are ~DEFAULT_CHUNK_SIZE words (see constants.py), so 1024 tokens
                # comfortably covers chunk content plus contextual headers.
                model.max_seq_length = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "1024"))

                dim = model.get_sentence_embedding_dimension() or DEFAULT_DENSE_VECTOR_DIM
                logger.info(f"Successfully loaded embedding model. Dimension: {dim}")
            except ImportError:
                logger.warning("sentence_transformers not installed. Using fallback random vector generator for mock/testing.")
                model = "MOCK"
                dim = self._dim
            except Exception as e:
                logger.error(f"Error loading embedding model {self.model_name}: {str(e)}")
                raise EmbeddingError(f"Failed to load embedding model: {str(e)}")

            BGEEmbeddingGenerator._model_cache[cache_key] = (model, dim)
            self._model = model
            self._dim = dim

    def preload(self) -> None:
        """Forces the embedding model to load immediately instead of lazily on first use.

        Intended to be called once at application startup so model loading happens
        deterministically before the first query is served.
        """
        self._load_model()

    @property
    def vector_dimension(self) -> int:
        """Returns vector dimension."""
        return self._dim

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of text strings.

        Args:
            texts: List of text strings.

        Returns:
            List of float vector lists.

        Raises:
            EmbeddingError: If vector generation fails.
        """
        if not texts:
            return []

        self._load_model()

        try:
            if self._model == "MOCK":
                # Mock fallback for test environment without model weights
                logger.debug("Generating mock embeddings.")
                np.random.seed(42)
                return [np.random.randn(self._dim).astype(float).tolist() for _ in texts]

            if len(texts) > 512:
                all_embeddings = []
                sub_batch_size = 512
                for i in range(0, len(texts), sub_batch_size):
                    batch_texts = texts[i : i + sub_batch_size]
                    batch_emb = self._model.encode(batch_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
                    all_embeddings.extend(batch_emb.tolist())
                return all_embeddings

            embeddings = self._model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch of size {len(texts)}: {str(e)}")
            raise EmbeddingError(f"Error generating embeddings: {str(e)}")

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generates embedding for a query string.

        Args:
            query: Input query.

        Returns:
            Float vector list.
        """
        if not query.strip():
            raise EmbeddingError("Query text cannot be empty.")

        results = self.generate_embeddings([query])
        return results[0]
