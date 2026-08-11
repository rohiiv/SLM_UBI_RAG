"""
Banking RAG Embedding Generator module.

Provides abstract interface and BGE-M3 embedding generator supporting dense vector generation.

# ── MPS Cold-Cache Note ───────────────────────────────────────────────────────
# PyTorch MPS compiles Apple Metal GPU shaders on FIRST USE for every unique
# tensor operation graph. For BGE-M3 (~570M params, 24 transformer layers) this
# compilation happens silently inside SentenceTransformer.__init__ and can take
# 5-30 minutes on a cold cache (e.g. after a macOS update or on a new machine).
# Subsequent runs read the pre-compiled shaders from:
#   ~/Library/Caches/com.apple.metal/
# and complete in ~10-15 seconds. PYTORCH_ENABLE_MPS_FALLBACK=1 is set below
# so that any op without a Metal kernel silently falls back to CPU instead of
# crashing, which also prevents compilation loops on edge-case ops.
# ─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import threading
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Tuple, Any
import numpy as np

# Set before any torch/sentence_transformers import so MPS ops without a Metal
# kernel fall back to CPU automatically instead of raising an error.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from banking_rag.config import ModelConfig, get_config
from banking_rag.constants import DEFAULT_DENSE_VECTOR_DIM
from banking_rag.exceptions import EmbeddingError
from banking_rag.utils.logger import get_logger

logger = get_logger("embeddings.embedding_generator")

# ── Module-level import of SentenceTransformer ───────────────────────────────
# Imported HERE (outside _load_model's threading lock) to avoid any interaction
# with Python 3.13's import machinery while a threading.Lock is held.
# The import is guarded so test environments without sentence-transformers installed
# can still import this module without crashing.
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _SentenceTransformer = None  # type: ignore[assignment,misc]
    _ST_AVAILABLE = False


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
        """Lazy loader for the BGE-M3 SentenceTransformer model, backed by a shared process-wide cache.

        MPS cold-cache note
        -------------------
        On first use on a new machine (or after a macOS update that invalidates the Metal shader
        cache), PyTorch MPS must compile Apple Metal GPU shaders for every unique tensor operation
        graph in BGE-M3. This compilation runs inside SentenceTransformer.__init__ without any
        progress indicator and can legitimately take 5-30 minutes. Subsequent runs read the
        pre-compiled shaders from ~/Library/Caches/com.apple.metal/ and complete in ~10-15s.
        PYTORCH_ENABLE_MPS_FALLBACK=1 (set at module import) lets edge-case ops fall back to CPU.
        """
        if self._model is not None:
            return

        cache_key = (self.model_name, self.device)

        with BGEEmbeddingGenerator._cache_lock:
            # Another thread may have finished loading while we waited for the lock.
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

            _load_start = time.perf_counter()

            try:
                if not _ST_AVAILABLE:
                    raise ImportError("sentence_transformers is not installed.")

                # ── STEP 1: Resolve effective device ────────────────────────────────────────
                effective_device = self.device
                logger.info(
                    f"[INIT STEP 1/5] Resolving device. Requested: '{effective_device}'. "
                    f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset')}"
                )

                # ── STEP 2: Validate MPS availability ───────────────────────────────────────
                if effective_device == "mps":
                    try:
                        import torch
                        if not torch.backends.mps.is_available():
                            logger.warning("[INIT STEP 2/5] MPS requested but not available. Falling back to CPU.")
                            effective_device = "cpu"
                        else:
                            # Quick MPS sanity check — if this hangs, MPS Metal driver is broken.
                            _t = time.perf_counter()
                            _test = torch.zeros(1, device="mps")
                            _ = (_test + 1).item()  # Force synchronous MPS dispatch
                            del _test
                            logger.info(f"[INIT STEP 2/5] MPS sanity check passed in {time.perf_counter()-_t:.3f}s.")
                    except Exception as mps_err:
                        logger.warning(f"[INIT STEP 2/5] MPS sanity check failed ({mps_err}). Falling back to CPU.")
                        effective_device = "cpu"
                else:
                    logger.info(f"[INIT STEP 2/5] Device is '{effective_device}', skipping MPS check.")

                # ── STEP 3: Construct SentenceTransformer ────────────────────────────────────
                # NOTE: On first MPS use (cold Metal shader cache), this step can take 5-30
                # minutes while Apple's Metal compiler builds GPU kernels for BGE-M3's
                # attention layers. This is NORMAL. Subsequent runs complete in ~10-15s.
                if effective_device == "mps":
                    logger.info(
                        "[INIT STEP 3/5] Constructing SentenceTransformer on MPS. "
                        "If the Metal shader cache is COLD (first run / after macOS update), "
                        "this step can take 5-30 minutes silently. This is expected behaviour — "
                        "Apple's Metal compiler is building GPU kernels. Subsequent runs will be fast."
                    )
                else:
                    logger.info(f"[INIT STEP 3/5] Constructing SentenceTransformer on device='{effective_device}'.")

                _t = time.perf_counter()
                model = _SentenceTransformer(
                    self.model_name,
                    device=effective_device,
                    revision=self.model_revision or None,
                )
                logger.info(f"[INIT STEP 3/5] SentenceTransformer constructed in {time.perf_counter()-_t:.2f}s.")

                # ── STEP 4: Cap sequence length ──────────────────────────────────────────────
                # bge-m3 natively supports up to 8192 tokens. Left uncapped, a batch gets padded
                # to that full length, which on MPS tries to allocate a full-size attention buffer
                # and fails ("Invalid buffer size"). 512 tokens covers all our chunked content.
                logger.info("[INIT STEP 4/5] Setting max_seq_length.")
                _t = time.perf_counter()
                model.max_seq_length = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512"))
                logger.info(f"[INIT STEP 4/5] max_seq_length={model.max_seq_length} set in {time.perf_counter()-_t:.4f}s.")

                # ── STEP 5: Get embedding dimension ─────────────────────────────────────────
                logger.info("[INIT STEP 5/5] Retrieving embedding dimension.")
                _t = time.perf_counter()
                # Use the new API name (ST 5.x) with graceful fallback for older versions.
                if hasattr(model, "get_embedding_dimension"):
                    dim = model.get_embedding_dimension() or DEFAULT_DENSE_VECTOR_DIM
                else:
                    dim = model.get_sentence_embedding_dimension() or DEFAULT_DENSE_VECTOR_DIM  # type: ignore[attr-defined]
                logger.info(
                    f"[INIT STEP 5/5] Embedding dimension: {dim} (retrieved in {time.perf_counter()-_t:.4f}s). "
                    f"Total init time: {time.perf_counter()-_load_start:.2f}s. "
                    f"Effective device: '{effective_device}'."
                )

            except ImportError:
                logger.warning("sentence_transformers not installed. Using mock random vector generator for testing.")
                model = "MOCK"
                dim = self._dim
            except Exception as e:
                logger.error(f"Error loading embedding model '{self.model_name}': {str(e)}")
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
                    batch_emb = self._model.encode(batch_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                    all_embeddings.extend(batch_emb.tolist())
                return all_embeddings

            embeddings = self._model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
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
