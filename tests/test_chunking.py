"""
Unit tests for document text chunking and contextual header injection.
"""

import pytest

from banking_rag.chunking.chunker import RecursiveTextChunker, TextChunk
from banking_rag.chunking.contextual_chunker import ContextualChunker
from banking_rag.config import ChunkingConfig
from banking_rag.constants import MetadataKeys, BankingDomain, BankingRegulator
from banking_rag.loaders.base_loader import Document


@pytest.fixture
def sample_document():
    """Provides a sample RBI Master Direction document."""
    content = (
        "CHAPTER I - PRELIMINARY\n\n"
        "1. Short Title and Commencement.\n"
        "These Directions shall be called the Reserve Bank of India (Know Your Customer (KYC)) Directions, 2016.\n\n"
        "CHAPTER II - APPLICABILITY\n\n"
        "2. The provisions of these Directions shall apply to every Regulated Entity (RE) carried on by Union Bank of India. "
        "Every RE shall frame a Customer Acceptance Policy."
    )
    metadata = {
        MetadataKeys.DOC_NAME: "RBI_Master_Direction_KYC.pdf",
        MetadataKeys.REGULATOR: BankingRegulator.RBI.value,
        MetadataKeys.DOMAIN: BankingDomain.AML_KYC.value,
        MetadataKeys.SECTION: "Section 1",
    }
    return Document(content=content, metadata=metadata, doc_id="doc_test_101")


def test_recursive_chunker_basic(sample_document):
    """Tests basic chunking operation and metadata copy."""
    config = ChunkingConfig(chunk_size=150, chunk_overlap=20)
    chunker = RecursiveTextChunker(config=config)

    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) > 0
    assert isinstance(chunks[0], TextChunk)
    assert chunks[0].parent_doc_id == "doc_test_101"
    assert chunks[0].metadata[MetadataKeys.DOC_NAME] == "RBI_Master_Direction_KYC.pdf"


def test_contextual_chunker(sample_document):
    """Tests prepending of context headers."""
    base_chunker = RecursiveTextChunker(config=ChunkingConfig(chunk_size=200, chunk_overlap=20))
    contextual_chunker = ContextualChunker(base_chunker=base_chunker)

    context_chunks = contextual_chunker.process(sample_document)
    assert len(context_chunks) > 0
    
    first_chunk = context_chunks[0]
    assert "[Context:" in first_chunk.content
    assert "Regulator: Reserve Bank of India (RBI)" in first_chunk.content
    assert MetadataKeys.CONTEXT_HEADER in first_chunk.metadata
