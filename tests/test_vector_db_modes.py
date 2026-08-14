"""
Unit tests for Qdrant Dual-Mode (Server & Embedded) Configuration & Vector Store Manager.
"""

import os
import shutil
import pytest
from banking_rag.config import QdrantConfig
from banking_rag.exceptions import ConfigurationError
from banking_rag.vectorstore.qdrant_manager import QdrantVectorStoreManager, ScoredChunk
from banking_rag.chunking.chunker import TextChunk


def test_qdrant_config_server_mode_defaults(monkeypatch):
    """Test server mode configuration defaults."""
    monkeypatch.delenv("VECTOR_DB_MODE", raising=False)
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("QDRANT_PORT", "6333")

    config = QdrantConfig()
    assert config.vector_db_mode == "server"
    assert config.qdrant_host == "localhost"
    assert config.qdrant_port == 6333


def test_qdrant_config_embedded_mode(monkeypatch, tmp_path):
    """Test embedded mode configuration."""
    db_path = str(tmp_path / "custom_qdrant_data")
    monkeypatch.setenv("VECTOR_DB_MODE", "embedded")
    monkeypatch.setenv("QDRANT_PATH", db_path)

    config = QdrantConfig()
    assert config.vector_db_mode == "embedded"
    assert config.qdrant_path == db_path


def test_qdrant_config_validation_errors(monkeypatch):
    """Test validation errors for invalid modes and missing parameters."""
    # Invalid mode
    monkeypatch.setenv("VECTOR_DB_MODE", "invalid_mode")
    with pytest.raises(ConfigurationError) as exc_info:
        QdrantConfig()
    assert "VECTOR_DB_MODE must be 'server', 'embedded', or 'faiss'." in str(exc_info.value)

    # Embedded mode missing path
    monkeypatch.setenv("VECTOR_DB_MODE", "embedded")
    monkeypatch.setenv("QDRANT_PATH", "")
    with pytest.raises(ConfigurationError) as exc_info:
        QdrantConfig()
    assert "When VECTOR_DB_MODE=embedded, QDRANT_PATH is required." in str(exc_info.value)

    # Server mode missing host
    monkeypatch.setenv("VECTOR_DB_MODE", "server")
    monkeypatch.setenv("QDRANT_HOST", "")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(ConfigurationError) as exc_info:
        QdrantConfig()
    assert "When VECTOR_DB_MODE=server, QDRANT_HOST is required." in str(exc_info.value)


def test_qdrant_manager_embedded_mode_crud(tmp_path):
    """Test QdrantVectorStoreManager local disk CRUD operations without Docker."""
    db_path = str(tmp_path / "test_embedded_qdrant")
    config = QdrantConfig(
        mode="embedded",
        path=db_path,
        collection_name="test_embedded_collection",
        vector_size=4,
    )

    manager = QdrantVectorStoreManager(config=config)
    assert manager.create_collection(force_recreate=True) is True

    chunk = TextChunk(
        chunk_id="chunk_1",
        content="Test content for embedded Qdrant mode",
        metadata={"regulator": "RBI", "domain": "KYC"},
    )
    embeddings = [[0.1, 0.2, 0.3, 0.4]]

    assert manager.upsert_chunks([chunk], embeddings) is True

    # Perform similarity search
    results = manager.search(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.content == "Test content for embedded Qdrant mode"
    assert results[0].chunk.metadata["regulator"] == "RBI"
