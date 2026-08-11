"""
Unit and integration tests for FastAPI Banking RAG Server (banking_rag/api.py).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from banking_rag.api import app
from banking_rag.pipeline.rag_pipeline import RAGResponse


@pytest.fixture
def mock_rag_pipeline():
    """Mocks OnlineRAGPipeline for API testing."""
    mock_pipeline = MagicMock()
    
    # Mock query response
    mock_pipeline.query.return_value = RAGResponse(
        query="What are CDD rules?",
        answer="Customer Due Diligence rules require verifying user identity.",
        citations=["[Source: RBI_KYC | Regulator: RBI | Section: Sec 12 | Page: 4]"],
        retrieved_chunks=[],
        cached=False,
        metadata={"filters_applied": {}},
    )

    # Mock components for /health check
    mock_pipeline.retriever.dense_retriever.embedding_generator._model = "MOCK_EMBED"
    mock_pipeline.retriever.dense_retriever.vector_store._get_client.return_value = "MOCK"
    mock_pipeline.reranker._model = "MOCK_RERANK"
    mock_pipeline.llm_generator._model = "MOCK_LLM"

    return mock_pipeline


def test_api_health_endpoint_healthy(mock_rag_pipeline):
    """Tests GET /health returns 200 when pipeline components are ready."""
    with TestClient(app) as client:
        app.state.rag = mock_rag_pipeline
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["embedding_generator"] is True
        assert data["components"]["reranker"] is True
        assert data["components"]["llm_generator"] is True
        assert data["components"]["qdrant_client"] is True


def test_api_query_endpoint(mock_rag_pipeline):
    """Tests POST /query returns answer, citations, and cache state."""
    with TestClient(app) as client:
        app.state.rag = mock_rag_pipeline
        response = client.post("/query", json={"question": "What are CDD rules?", "top_k": 3})
        assert response.status_code == 200
        data = response.json()
        assert "Customer Due Diligence" in data["answer"]
        assert len(data["citations"]) == 1
        assert data["cached"] is False


def test_api_ingest_endpoint(tmp_path):
    """Tests POST /ingest invokes OfflineIngestionPipeline."""
    sample_file = tmp_path / "test_doc.txt"
    sample_file.write_text("Banking compliance rules test.", encoding="utf-8")

    with patch("banking_rag.api.OfflineIngestionPipeline") as mock_ingest_class:
        mock_instance = MagicMock()
        mock_instance.ingest_file.return_value = {
            "status": "success",
            "file_name": "test_doc.txt",
            "pages": 1,
            "chunks_ingested": 1,
        }
        mock_ingest_class.return_value = mock_instance

        with TestClient(app) as client:
            response = client.post("/ingest", json={"path": str(sample_file)})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["file_name"] == "test_doc.txt"
            assert data["chunks_ingested"] == 1


def test_api_docs_endpoint():
    """Tests that OpenAPI docs endpoint /docs is accessible."""
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200
