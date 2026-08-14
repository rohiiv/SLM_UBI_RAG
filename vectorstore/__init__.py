"""
Banking RAG Vector Store package.

Exports:
  - QdrantVectorStoreManager: Qdrant-backed vector store (server or embedded mode).
  - FaissVectorStoreManager:  FAISS-backed vector store (local IndexFlatIP + SQLite sidecar).
  - ScoredChunk:              Shared data class returned by both managers' search().
  - get_vector_store_manager: Factory that picks the right backend from VECTOR_DB_MODE config.
"""

from banking_rag.vectorstore.qdrant_manager import QdrantVectorStoreManager, ScoredChunk
from banking_rag.vectorstore.faiss_manager import FaissVectorStoreManager


def get_vector_store_manager(config=None):
    """Returns the appropriate vector store manager based on the VECTOR_DB_MODE config.

    Switching backends is a pure config change — set ``VECTOR_DB_MODE=faiss`` in your
    ``.env`` file to use the FAISS backend, or ``VECTOR_DB_MODE=server``/``embedded``
    to use Qdrant.  No code changes are required at call sites.

    Args:
        config: Optional ``AppConfig`` or ``QdrantConfig`` instance.  When an
                ``AppConfig`` is passed its ``.qdrant`` sub-config is used.  Defaults
                to the process-wide config via ``get_config()``.

    Returns:
        Either a ``QdrantVectorStoreManager`` or a ``FaissVectorStoreManager`` instance,
        both sharing the same public interface.
    """
    from banking_rag.config import get_config

    if config is None:
        qdrant_cfg = get_config().qdrant
    elif hasattr(config, "qdrant"):
        # Full AppConfig passed.
        qdrant_cfg = config.qdrant
    else:
        # QdrantConfig passed directly.
        qdrant_cfg = config

    mode = (qdrant_cfg.mode or "").lower()
    if mode == "faiss":
        return FaissVectorStoreManager(config=qdrant_cfg)
    else:
        return QdrantVectorStoreManager(config=qdrant_cfg)


__all__ = [
    "QdrantVectorStoreManager",
    "FaissVectorStoreManager",
    "ScoredChunk",
    "get_vector_store_manager",
]
