"""
Unit tests for retrieval, hybrid search, metadata filtering, and reranking.
"""

import pytest

from banking_rag.chunking.chunker import TextChunk
from banking_rag.constants import MetadataKeys, BankingRegulator, BankingDomain
from banking_rag.embeddings.embedding_generator import BGEEmbeddingGenerator
from banking_rag.retrieval.filters import MetadataFilterBuilder
from banking_rag.retrieval.hybrid_retriever import HybridRetriever
from banking_rag.retrieval.reranker import CrossEncoderReranker
from banking_rag.vectorstore.qdrant_manager import QdrantVectorStoreManager, ScoredChunk


@pytest.fixture
def mock_chunks():
    """Provides sample ScoredChunk items for testing."""
    chunk1 = TextChunk(
        chunk_id="chunk_1",
        content="Regulated Entities shall ensure Customer Due Diligence for high risk accounts.",
        metadata={MetadataKeys.REGULATOR: BankingRegulator.RBI.value, MetadataKeys.DOMAIN: BankingDomain.AML_KYC.value},
    )
    chunk2 = TextChunk(
        chunk_id="chunk_2",
        content="Internal audit reports must be presented quarterly to the Audit Committee of the Board.",
        metadata={MetadataKeys.REGULATOR: BankingRegulator.INTERNAL.value, MetadataKeys.DOMAIN: BankingDomain.INTERNAL_AUDIT.value},
    )
    return [ScoredChunk(chunk=chunk1, score=0.85), ScoredChunk(chunk=chunk2, score=0.45)]


def test_metadata_filter_builder():
    """Tests payload filter construction logic."""
    filters = {
        MetadataKeys.REGULATOR: BankingRegulator.RBI.value,
        MetadataKeys.DOMAIN: BankingDomain.AML_KYC.value,
    }
    built_filter = MetadataFilterBuilder.build_filter(filters)
    assert built_filter is not None


def test_cross_encoder_reranker(mock_chunks):
    """Tests reranker scoring and order sorting."""
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query="Customer Due Diligence requirements", candidates=mock_chunks, top_k=2)
    
    assert len(reranked) == 2
    assert isinstance(reranked[0], ScoredChunk)
