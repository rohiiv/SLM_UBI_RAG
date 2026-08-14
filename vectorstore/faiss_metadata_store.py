"""
Banking RAG FAISS Sidecar Metadata Store.

FAISS only stores dense vectors; this SQLite-backed sidecar maps each FAISS integer index
position to the full chunk payload (content, chunk_id, and all metadata fields) that
Qdrant would normally hold inside the point payload.
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from banking_rag.utils.logger import get_logger

logger = get_logger("vectorstore.faiss_metadata_store")

# SQLite schema: one row per FAISS vector (faiss_id = row insertion order, 0-indexed).
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunk_metadata (
    faiss_id   INTEGER PRIMARY KEY,
    chunk_id   TEXT,
    content    TEXT,
    parent_doc_id TEXT,
    metadata_json TEXT
)
"""
_INSERT_SQL = """
INSERT OR REPLACE INTO chunk_metadata
    (faiss_id, chunk_id, content, parent_doc_id, metadata_json)
VALUES (?, ?, ?, ?, ?)
"""
_SELECT_ONE_SQL = "SELECT chunk_id, content, parent_doc_id, metadata_json FROM chunk_metadata WHERE faiss_id = ?"
_SELECT_BATCH_SQL = "SELECT faiss_id, chunk_id, content, parent_doc_id, metadata_json FROM chunk_metadata WHERE faiss_id IN ({})"
_COUNT_SQL = "SELECT COUNT(*) FROM chunk_metadata"


class FaissMetadataStore:
    """SQLite-backed sidecar that maps FAISS integer index positions to chunk payloads.

    Thread-safe: all read/write operations are serialised through a single lock because
    SQLite's default isolation level is sufficient for the write-once, read-many
    access pattern used during ingestion and query.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialises the metadata store.

        The underlying SQLite database is created in-memory if ``db_path`` is None
        (useful for unit tests) or at the specified filesystem path.  The table is
        created automatically on first use.

        Args:
            db_path: Filesystem path for the SQLite database file.
                     Pass None (default) for an ephemeral in-memory database.
        """
        self._db_path: str = db_path if db_path is not None else ":memory:"
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._ensure_connected()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Opens the SQLite connection and creates the schema if absent."""
        if self._conn is None:
            # check_same_thread=False: we guard all access with self._lock ourselves.
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.commit()
            logger.info(f"FaissMetadataStore connected to SQLite database at '{self._db_path}'.")

    def _row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """Converts a SELECT result row into a unified metadata dictionary."""
        chunk_id, content, parent_doc_id, metadata_json = row
        meta = json.loads(metadata_json) if metadata_json else {}
        return {
            "chunk_id": chunk_id,
            "content": content,
            "parent_doc_id": parent_doc_id,
            **meta,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, faiss_index_id: int, metadata: Dict[str, Any]) -> None:
        """Inserts (or replaces) a metadata record keyed by FAISS integer index position.

        Args:
            faiss_index_id: 0-indexed integer matching the row position in the FAISS index.
            metadata: Full payload dictionary.  ``chunk_id``, ``content``, and
                      ``parent_doc_id`` are extracted into dedicated columns; all other
                      fields are stored as a JSON blob in ``metadata_json``.
        """
        chunk_id = metadata.get("chunk_id", "")
        content = metadata.get("content", "")
        parent_doc_id = metadata.get("parent_doc_id", "")
        # Store remaining fields as JSON — avoids needing a column per metadata key.
        extra = {k: v for k, v in metadata.items() if k not in {"chunk_id", "content", "parent_doc_id"}}

        with self._lock:
            self._conn.execute(
                _INSERT_SQL,
                (faiss_index_id, chunk_id, content, parent_doc_id, json.dumps(extra)),
            )
            self._conn.commit()

    def add_batch(self, records: List[Dict[str, Any]]) -> None:
        """Inserts a batch of (faiss_index_id, metadata) pairs in a single transaction.

        Args:
            records: List of dicts each containing a ``"faiss_index_id"`` key plus all
                     payload fields.  This is more efficient than calling ``add()`` in a
                     loop for large batches.
        """
        rows = []
        for rec in records:
            fid = rec["faiss_index_id"]
            chunk_id = rec.get("chunk_id", "")
            content = rec.get("content", "")
            parent_doc_id = rec.get("parent_doc_id", "")
            extra = {k: v for k, v in rec.items() if k not in {"faiss_index_id", "chunk_id", "content", "parent_doc_id"}}
            rows.append((fid, chunk_id, content, parent_doc_id, json.dumps(extra)))

        with self._lock:
            self._conn.executemany(_INSERT_SQL, rows)
            self._conn.commit()

        logger.debug(f"FaissMetadataStore: batch-inserted {len(rows)} records.")

    def get(self, faiss_index_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves the metadata dictionary for a single FAISS index position.

        Args:
            faiss_index_id: Integer index into the FAISS index.

        Returns:
            Metadata dictionary, or None if the id is not found.
        """
        with self._lock:
            cursor = self._conn.execute(_SELECT_ONE_SQL, (faiss_index_id,))
            row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def get_batch(self, faiss_index_ids: List[int]) -> List[Optional[Dict[str, Any]]]:
        """Retrieves metadata for a list of FAISS index positions in a single query.

        The returned list preserves the order of ``faiss_index_ids``: if a given id is
        not found in the store, ``None`` appears at that position.

        Args:
            faiss_index_ids: List of integer FAISS index positions.

        Returns:
            Ordered list of metadata dicts (or None for any missing id).
        """
        if not faiss_index_ids:
            return []

        placeholders = ",".join("?" * len(faiss_index_ids))
        sql = _SELECT_BATCH_SQL.format(placeholders)

        with self._lock:
            cursor = self._conn.execute(sql, faiss_index_ids)
            rows = cursor.fetchall()

        # Build a lookup keyed by faiss_id so we can preserve caller's order.
        by_id: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            fid = row[0]
            by_id[fid] = self._row_to_dict(row[1:])  # skip faiss_id column

        return [by_id.get(fid) for fid in faiss_index_ids]

    def count(self) -> int:
        """Returns the total number of stored metadata records."""
        with self._lock:
            cursor = self._conn.execute(_COUNT_SQL)
            return cursor.fetchone()[0]

    def persist(self, path: str) -> None:
        """Copies the in-memory (or current) SQLite database to ``path``.

        When the store was initialised with a real file path this is a no-op (the
        connection is already writing directly to that file).  For in-memory stores this
        creates a snapshot on disk.

        Args:
            path: Destination file path for the SQLite database.
        """
        dest_path = Path(path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if self._db_path == ":memory:":
            # Backup the in-memory DB to disk.
            dest_conn = sqlite3.connect(str(dest_path))
            with self._lock:
                self._conn.backup(dest_conn)
            dest_conn.close()
            logger.info(f"FaissMetadataStore: persisted in-memory SQLite to '{dest_path}'.")
        else:
            # Already on disk at self._db_path; just confirm path matches.
            if str(dest_path.resolve()) != str(Path(self._db_path).resolve()):
                logger.warning(
                    f"FaissMetadataStore.persist() called with path '{path}' but store "
                    f"is already backed by '{self._db_path}'. No copy made."
                )
            else:
                logger.info(f"FaissMetadataStore: already persisted at '{self._db_path}'.")

    def load(self, path: str) -> None:
        """Re-opens the store from a previously persisted SQLite file.

        Closes the current connection (discarding any in-memory data) and replaces it
        with a connection to the file at ``path``.

        Args:
            path: Path to the SQLite file written by ``persist()``.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"FaissMetadataStore: database file not found at '{load_path}'.")

        with self._lock:
            if self._conn is not None:
                self._conn.close()
            self._db_path = str(load_path)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")

        logger.info(f"FaissMetadataStore: loaded database from '{load_path}' ({self.count()} records).")
