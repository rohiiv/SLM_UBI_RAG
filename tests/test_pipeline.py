"""
Integration tests for end-to-end offline ingestion and online RAG pipelines.
"""

import pytest
from pathlib import Path

from banking_rag.pipeline.ingest import OfflineIngestionPipeline
from banking_rag.pipeline.rag_pipeline import OnlineRAGPipeline, RAGResponse


def test_offline_ingestion_and_online_rag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end test verifying document ingestion and subsequent query execution."""
    monkeypatch.setenv("INGESTION_ENFORCE_ALLOWLIST", "false")
    # 1. Create a dummy banking text file
    sample_file = tmp_path / "RBI_KYC_Directive.txt"
    sample_file.write_text(
        "RESERVE BANK OF INDIA - MASTER DIRECTION ON KYC\n"
        "Section 15: All banks including Union Bank of India must maintain transaction records for at least 5 years.\n"
        "Section 16: Suspicious Transaction Reports (STR) shall be furnished to FIU-IND within seven working days.",
        encoding="utf-8",
    )

    # 2. Run offline ingestion
    ingest_pipeline = OfflineIngestionPipeline()
    summary = ingest_pipeline.ingest_file(sample_file)
    assert summary["status"] == "success"
    assert summary["chunks_ingested"] > 0

    # 3. Execute online RAG query
    rag_pipeline = OnlineRAGPipeline()
    response = rag_pipeline.query("How many years must Union Bank of India maintain transaction records?", top_k=2)

    assert isinstance(response, RAGResponse)
    assert response.query == "How many years must Union Bank of India maintain transaction records?"
    assert response.answer is not None
    assert len(response.answer) > 0
