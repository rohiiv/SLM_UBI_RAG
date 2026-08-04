"""
Banking RAG Exception Hierarchy.

Custom exceptions for domain-specific, informative error reporting across
all modules of the Union Bank of India RAG pipeline.
"""

from typing import Optional, Dict, Any


class BankingRAGException(Exception):
    """Base exception for all Banking RAG system errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(BankingRAGException):
    """Raised when application configuration is missing or invalid."""
    pass


class DocumentLoadError(BankingRAGException):
    """Raised when document loading fails (unsupported format, corrupt file, file missing)."""
    pass


class DocumentParsingError(BankingRAGException):
    """Raised when text parsing, cleaning, or structural extraction fails."""
    pass


class ChunkingError(BankingRAGException):
    """Raised when text chunking or contextual header injection fails."""
    pass


class MetadataExtractionError(BankingRAGException):
    """Raised when metadata extraction logic encounters invalid content or rules."""
    pass


class EmbeddingError(BankingRAGException):
    """Raised when vector embedding generation fails."""
    pass


class VectorDBError(BankingRAGException):
    """Raised when Qdrant database connection, collection management, or vector operations fail."""
    pass


class RetrievalError(BankingRAGException):
    """Raised during dense, sparse, or hybrid search execution."""
    pass


class RerankingError(BankingRAGException):
    """Raised when cross-encoder reranking encounters invalid inputs or model errors."""
    pass


class PromptBuilderError(BankingRAGException):
    """Raised when prompt construction or template rendering fails."""
    pass


class LLMGenerationError(BankingRAGException):
    """Raised when the fine-tuned Qwen SLM generation fails or times out."""
    pass


class CacheError(BankingRAGException):
    """Raised during retrieval or embedding cache operations."""
    pass


class PipelineError(BankingRAGException):
    """Raised when high-level offline ingestion or online RAG pipeline fails."""
    pass


class SecurityError(BankingRAGException):
    """Raised when a security control rejects an operation (ingestion allowlist violation,
    blocked retrieval content, missing Qdrant auth outside dev, canary token leakage, etc.)."""
    pass
