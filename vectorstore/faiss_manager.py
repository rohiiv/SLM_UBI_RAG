"""
Banking RAG FAISS Vector Store Manager.

Drop-in replacement for QdrantVectorStoreManager backed by a local FAISS IndexFlatIP index
and a SQLite sidecar metadata store (FaissMetadataStore).

IndexFlatIP performs exact inner-product search.  Because BGEEmbeddingGenerator produces
L2-normalised embeddings, inner-product == cosine similarity, so scores are comparable
with the cosine-distance results returned by Qdrant.

Filtering limitation:
  FAISS does not support server-side payload filtering.  When the caller passes a
  ``query_filter`` (a plain dict of field->value constraints), this manager over-fetches
  ``top_k * FILTER_OVERSAMPLE_FACTOR`` raw candidates from the FAISS index, applies the
  filter as a Python-side predicate against the sidecar metadata, and then truncates to
  ``top_k``.  This is an approximation: if fewer than ``top_k`` chunks survive the filter
  even after over-fetching, the result set will be smaller than requested.  A warning is
  logged every time this path is taken.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from banking_rag.chunking.chunker import TextChunk
from banking_rag.config import QdrantConfig, get_config
from banking_rag.exceptions import VectorDBError
from banking_rag.utils.logger import get_logger
from banking_rag.vectorstore.faiss_metadata_store import FaissMetadataStore
from banking_rag.vectorstore.qdrant_manager import ScoredChunk

logger = get_logger("vectorstore.faiss_manager")

# When a metadata filter is supplied, fetch this many extra raw FAISS candidates before
# filtering, to reduce the chance of returning fewer results than top_k.
_FILTER_OVERSAMPLE_FACTOR: int = 5

# Default filenames written inside the index directory.
_FAISS_INDEX_FILENAME = "faiss.index"
_SQLITE_DB_FILENAME = "faiss_metadata.sqlite"


class FaissVectorStoreManager:
    """FAISS-backed vector store manager with a SQLite sidecar for chunk payloads.

    Public interface is intentionally identical to QdrantVectorStoreManager so either
    class can be injected wherever a vector store is needed without changing call sites.

    The FAISS index and metadata store are lazily initialised on first use.  The index
    starts empty when the manager is constructed without loading an existing index from
    disk; use ``load()`` to restore a previously persisted index, or ``persist()`` to
    save it after ingestion.

    Thread safety: the FAISS index itself is not thread-safe for concurrent writes, so
    all index mutations (``upsert_chunks``) are serialised through a class-level lock
    shared across all instances that point at the same index path.  Read operations
    (``search``) acquire a separate per-instance read lock so concurrent queries don't
    block each other.
    """

    # Class-level write lock shared by all instances — same singleton pattern used by
    # QdrantVectorStoreManager and the model caches.
    _write_lock = threading.Lock()

    def __init__(self, config: Optional[QdrantConfig] = None) -> None:
        """Initialises the FAISS manager.

        Args:
            config: Optional QdrantConfig.  When ``config.mode == 'faiss'``,
                    ``config.faiss_index_path`` is used as the persistence directory.
                    Defaults to the process-wide config.
        """
        self.config = config or get_config().qdrant
        # faiss_index_path is injected by the updated QdrantConfig when mode='faiss'.
        self._index_dir: str = getattr(self.config, "faiss_index_path", "./faiss_data")
        self.vector_size: int = self.config.vector_size

        self._index = None           # faiss.IndexFlatIP; created lazily
        self._next_id: int = 0       # monotonically increasing FAISS row counter
        self._meta_store: Optional[FaissMetadataStore] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_index(self):
        """Lazily initialises the FAISS index and metadata store (not thread-safe alone)."""
        if self._index is not None:
            return self._index

        try:
            import faiss  # type: ignore
        except ImportError:
            raise VectorDBError(
                "faiss-cpu is not installed.  Add 'faiss-cpu' to requirements.txt and "
                "run pip install -r requirements.txt."
            )

        self._index = faiss.IndexFlatIP(self.vector_size)
        logger.info(
            f"Initialised FAISS IndexFlatIP (dim={self.vector_size}) "
            f"with persistence directory '{self._index_dir}'."
        )

        db_path = str(Path(self._index_dir) / _SQLITE_DB_FILENAME)
        self._meta_store = FaissMetadataStore(db_path=db_path)
        self._next_id = self._meta_store.count()

        return self._index

    def _apply_dict_filter(
        self,
        metadata_list: List[Optional[Dict[str, Any]]],
        query_filter: Dict[str, Any],
    ) -> List[int]:
        """Returns the positions (within metadata_list) that satisfy all filter conditions.

        Handles both scalar exact-match and list (match-any) values, mirroring the logic
        in MetadataFilterBuilder.build_filter() for consistency with the Qdrant path.
        """
        passing_indices = []
        for pos, meta in enumerate(metadata_list):
            if meta is None:
                continue
            match = True
            for key, expected in query_filter.items():
                if expected is None or expected == "":
                    continue
                actual = meta.get(key)
                if isinstance(expected, list):
                    if actual not in expected:
                        match = False
                        break
                else:
                    if actual != expected:
                        match = False
                        break
            if match:
                passing_indices.append(pos)
        return passing_indices

    # ------------------------------------------------------------------
    # Public interface (mirrors QdrantVectorStoreManager)
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """Forces the FAISS index and metadata store to initialise immediately.

        Mirrors QdrantVectorStoreManager.preload().  Call once at startup to make the
        first query deterministic rather than paying the init cost on demand.
        """
        with FaissVectorStoreManager._write_lock:
            self._get_index()

    def create_collection(self, force_recreate: bool = False) -> bool:
        """No-op compatibility shim — FAISS has no named collections.

        Mirrors QdrantVectorStoreManager.create_collection() so pipeline code that calls
        this method before upserting doesn't need to branch on backend type.

        Args:
            force_recreate: If True, resets the in-memory index and clears the metadata
                            store (destructive — all previously indexed vectors are lost).

        Returns:
            Always True.
        """
        with FaissVectorStoreManager._write_lock:
            if force_recreate and self._index is not None:
                logger.warning(
                    "FaissVectorStoreManager.create_collection(force_recreate=True): "
                    "resetting in-memory FAISS index and clearing metadata store."
                )
                self._index = None
                self._meta_store = None
                self._next_id = 0
            self._get_index()
        return True

    def upsert_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> bool:
        """Adds a batch of text chunks and their embeddings to the FAISS index.

        Mirrors QdrantVectorStoreManager.upsert_chunks() exactly:
        - Same parameter names and types.
        - Same return type (bool).
        - Same exception type on failure (VectorDBError).

        Args:
            chunks: List of TextChunk objects.
            embeddings: Corresponding list of L2-normalised dense embedding vectors.

        Returns:
            True if upsert succeeded.

        Raises:
            VectorDBError: If sizes mismatch or the FAISS add() call fails.
        """
        if len(chunks) != len(embeddings):
            raise VectorDBError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
            )

        if not chunks:
            return True

        try:
            vectors_np = np.array(embeddings, dtype=np.float32)
        except Exception as e:
            raise VectorDBError(f"Failed to convert embeddings to numpy array: {e}")

        if vectors_np.ndim != 2 or vectors_np.shape[1] != self.vector_size:
            raise VectorDBError(
                f"Embedding dimension mismatch: expected {self.vector_size}, "
                f"got {vectors_np.shape[1] if vectors_np.ndim == 2 else '?'}."
            )

        with FaissVectorStoreManager._write_lock:
            index = self._get_index()

            try:
                index.add(vectors_np)  # type: ignore[attr-defined]
            except Exception as e:
                raise VectorDBError(f"FAISS index.add() failed: {e}")

            # Build batch metadata records and insert in one transaction.
            batch_records = []
            for i, chunk in enumerate(chunks):
                faiss_id = self._next_id + i
                payload = dict(chunk.metadata)
                payload["chunk_id"] = chunk.chunk_id
                payload["content"] = chunk.content
                payload["parent_doc_id"] = chunk.parent_doc_id
                payload["faiss_index_id"] = faiss_id
                batch_records.append(payload)

            self._meta_store.add_batch(batch_records)
            self._next_id += len(chunks)

        logger.info(
            f"FaissVectorStoreManager: upserted {len(chunks)} chunks "
            f"(total indexed: {self._next_id})."
        )
        return True

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        query_filter: Optional[Any] = None,
    ) -> List[ScoredChunk]:
        """Performs approximate nearest-neighbour search using FAISS IndexFlatIP.

        Mirrors QdrantVectorStoreManager.search() exactly:
        - Same parameter names.
        - Returns List[ScoredChunk] in descending score order.
        - Same exception type on failure (VectorDBError).

        Filtering:
            FAISS does not support native server-side metadata filtering.  When
            ``query_filter`` is provided (as either a plain dict or a Qdrant Filter
            object — the latter is parsed back to a dict for the post-filter step),
            this method over-fetches ``top_k * _FILTER_OVERSAMPLE_FACTOR`` raw
            candidates and applies the filter as a Python predicate.  A warning is
            logged every time this approximation is used.

        Args:
            query_vector: L2-normalised float embedding of the query.
            top_k: Maximum number of chunks to return.
            query_filter: Optional filter.  Accepts a plain ``dict`` of field->value
                          constraints OR a Qdrant ``Filter`` object (ignored with a
                          warning — pass a plain dict when using FAISS).

        Returns:
            List of ScoredChunk objects sorted by similarity score descending.

        Raises:
            VectorDBError: If the FAISS search call or metadata retrieval fails.
        """
        # Normalise filter to a plain dict (or None) for the post-filter step.
        filter_dict: Optional[Dict[str, Any]] = None
        if query_filter is not None:
            if isinstance(query_filter, dict):
                filter_dict = query_filter
            else:
                # Qdrant Filter object passed — we can't easily introspect it.
                logger.warning(
                    "FaissVectorStoreManager.search(): received a non-dict query_filter "
                    "(likely a Qdrant Filter object).  FAISS cannot use this directly.  "
                    "Pass a plain dict of field->value constraints when using the FAISS "
                    "backend.  The filter will be IGNORED for this query."
                )
                filter_dict = None

        # Determine how many raw candidates to fetch from FAISS.
        if filter_dict:
            logger.warning(
                "FaissVectorStoreManager: metadata filter requested — performing "
                f"post-filter over-fetch (top_k * {_FILTER_OVERSAMPLE_FACTOR} = "
                f"{top_k * _FILTER_OVERSAMPLE_FACTOR} raw candidates).  "
                "This is an approximation; filtered results may be fewer than top_k."
            )
            fetch_k = min(top_k * _FILTER_OVERSAMPLE_FACTOR, self._next_id if self._next_id > 0 else top_k * _FILTER_OVERSAMPLE_FACTOR)
        else:
            fetch_k = top_k

        try:
            vec_np = np.array([query_vector], dtype=np.float32)
        except Exception as e:
            raise VectorDBError(f"Failed to convert query vector to numpy array: {e}")

        try:
            index = self._get_index()
            if index.ntotal == 0:  # type: ignore[attr-defined]
                logger.warning("FaissVectorStoreManager: index is empty, returning no results.")
                return []

            distances, indices = index.search(vec_np, min(fetch_k, index.ntotal))  # type: ignore[attr-defined]
        except Exception as e:
            raise VectorDBError(f"FAISS index.search() failed: {e}")

        raw_ids: List[int] = [int(i) for i in indices[0] if i >= 0]
        raw_scores: List[float] = [float(distances[0][pos]) for pos, i in enumerate(indices[0]) if i >= 0]

        if not raw_ids:
            return []

        # Fetch payloads from the sidecar store.
        try:
            meta_list = self._meta_store.get_batch(raw_ids)
        except Exception as e:
            raise VectorDBError(f"FaissMetadataStore.get_batch() failed: {e}")

        # Apply post-filter if needed.
        if filter_dict:
            passing_positions = self._apply_dict_filter(meta_list, filter_dict)
        else:
            passing_positions = list(range(len(meta_list)))

        # Build ScoredChunk list, truncating to top_k.
        scored_chunks: List[ScoredChunk] = []
        for pos in passing_positions:
            if len(scored_chunks) >= top_k:
                break
            meta = meta_list[pos]
            if meta is None:
                continue

            content = meta.pop("content", "")
            parent_doc_id = meta.pop("parent_doc_id", "")
            chunk_id = meta.pop("chunk_id", "")
            # Remove the internal faiss_index_id field — callers don't need it.
            meta.pop("faiss_index_id", None)

            chunk = TextChunk(
                chunk_id=chunk_id,
                content=content,
                metadata=meta,
                parent_doc_id=parent_doc_id,
            )
            scored_chunks.append(ScoredChunk(chunk=chunk, score=raw_scores[pos]))

        logger.info(
            f"FaissVectorStoreManager: retrieved {len(scored_chunks)} chunks "
            f"(fetch_k={fetch_k}, filter_applied={filter_dict is not None})."
        )
        return scored_chunks

    def persist(self, path: Optional[str] = None) -> None:
        """Saves the FAISS index and SQLite metadata store to disk.

        Args:
            path: Directory to write ``faiss.index`` and ``faiss_metadata.sqlite``.
                  Defaults to ``self._index_dir`` (from config or constructor).
        """
        try:
            import faiss  # type: ignore
        except ImportError:
            raise VectorDBError("faiss-cpu is not installed; cannot persist index.")

        save_dir = Path(path or self._index_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / _FAISS_INDEX_FILENAME
        db_path = save_dir / _SQLITE_DB_FILENAME

        with FaissVectorStoreManager._write_lock:
            if self._index is None:
                logger.warning("FaissVectorStoreManager.persist() called on an empty (uninitialised) index. Nothing written.")
                return
            faiss.write_index(self._index, str(index_path))
            self._meta_store.persist(str(db_path))

        logger.info(
            f"FaissVectorStoreManager: persisted index to '{index_path}' "
            f"and metadata to '{db_path}'."
        )

    def load(self, path: Optional[str] = None) -> None:
        """Loads a previously persisted FAISS index and SQLite metadata store from disk.

        Args:
            path: Directory containing ``faiss.index`` and ``faiss_metadata.sqlite``.
                  Defaults to ``self._index_dir``.
        """
        try:
            import faiss  # type: ignore
        except ImportError:
            raise VectorDBError("faiss-cpu is not installed; cannot load index.")

        load_dir = Path(path or self._index_dir)
        index_path = load_dir / _FAISS_INDEX_FILENAME
        db_path = load_dir / _SQLITE_DB_FILENAME

        if not index_path.exists():
            raise VectorDBError(f"FAISS index file not found at '{index_path}'.")
        if not db_path.exists():
            raise VectorDBError(f"FAISS metadata SQLite file not found at '{db_path}'.")

        with FaissVectorStoreManager._write_lock:
            self._index = faiss.read_index(str(index_path))
            self._meta_store = FaissMetadataStore(db_path=str(db_path))
            self._next_id = self._index.ntotal  # type: ignore[attr-defined]

        logger.info(
            f"FaissVectorStoreManager: loaded index ({self._next_id} vectors) from '{index_path}' "
            f"and metadata from '{db_path}'."
        )
