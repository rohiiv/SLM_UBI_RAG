"""
Banking RAG System Constants module.

This module defines system-wide enums, constants, metadata schema keys,
regulators, document types, and default configuration parameters used across
all components of the Union Bank of India RAG system.
"""

from enum import Enum
from typing import Final, List


class BankingDomain(str, Enum):
    """Supported Banking Operational and Governance Domains."""
    COMPLIANCE = "Compliance"
    RISK = "Risk"
    INTERNAL_AUDIT = "Internal Audit"
    AML_KYC = "AML/KYC"
    BOARD_SECRETARIAT = "Board Secretariat"


class BankingRegulator(str, Enum):
    """Regulatory and Legislative Authorities."""
    RBI = "Reserve Bank of India (RBI)"
    SEBI = "Securities and Exchange Board of India (SEBI)"
    FIU_IND = "Financial Intelligence Unit - India (FIU-IND)"
    MHA = "Ministry of Home Affairs (PMLA)"
    MCA = "Ministry of Corporate Affairs (Companies Act)"
    INTERNAL = "Union Bank Internal Policy"
    OTHER = "Other Regulatory Body"


class DocumentType(str, Enum):
    """Banking Document Types."""
    MASTER_DIRECTION = "Master Direction"
    CIRCULAR = "Circular"
    ACT = "Act / Statute"
    REGULATION = "Regulation / Guideline"
    POLICY = "Internal Policy"
    AUDIT_REPORT = "Audit Report / Observation"
    BOARD_MINUTES = "Board Minutes / Agenda"
    STANDARD_OPERATING_PROCEDURE = "SOP"


class MetadataKeys:
    """Standardized metadata field names for document chunks and vector payloads."""
    DOC_ID: Final[str] = "doc_id"
    DOC_NAME: Final[str] = "doc_name"
    SOURCE: Final[str] = "source"
    DOMAIN: Final[str] = "domain"
    REGULATOR: Final[str] = "regulator"
    DOC_TYPE: Final[str] = "doc_type"
    PAGE_NUMBER: Final[str] = "page_number"
    CHAPTER: Final[str] = "chapter"
    SECTION: Final[str] = "section"
    CLAUSE: Final[str] = "clause"
    VERSION: Final[str] = "version"
    DATE: Final[str] = "date"
    CHUNK_ID: Final[str] = "chunk_id"
    PARENT_ID: Final[str] = "parent_id"
    CONTEXT_HEADER: Final[str] = "context_header"
    CREATED_AT: Final[str] = "created_at"


# Supported File Extensions
SUPPORTED_FILE_EXTENSIONS: Final[List[str]] = [".pdf", ".docx", ".txt", ".jsonl"]

# Default Model Specifications
DEFAULT_EMBEDDING_MODEL: Final[str] = "BAAI/bge-m3"
DEFAULT_SLM_MODEL: Final[str] = "Qwen/Qwen2.5-3B-Instruct"  # Qwen3-4B banking SLM fallback target
DEFAULT_RERANKER_MODEL: Final[str] = "BAAI/bge-reranker-large"

# Default Vector DB Specs
DEFAULT_QDRANT_COLLECTION_NAME: Final[str] = "union_bank_knowledge_base"
DEFAULT_DENSE_VECTOR_DIM: Final[int] = 1024  # BGE-M3 dense dimension

# Citation Format
CITATION_TEMPLATE: Final[str] = "[Source: {doc_name} | Regulator: {regulator} | Section: {section} | Page: {page_number}]"

DEFAULT_TOP_K_DENSE: Final[int] = 20
DEFAULT_TOP_K_SPARSE: Final[int] = 20
DEFAULT_TOP_K_RERANKED: Final[int] = 5
DEFAULT_CHUNK_SIZE: Final[int] = 512
DEFAULT_CHUNK_OVERLAP: Final[int] = 64

# ---------------------------------------------------------------------------
# Security / Phase 1 hardening constants
# ---------------------------------------------------------------------------

# Model revisions (commit SHAs). Left as "" by default which means "unpinned" -
# every loader logs a WARNING when a revision isn't pinned. Set these via env vars
# (EMBEDDING_MODEL_REVISION / RERANKER_MODEL_REVISION / SLM_MODEL_REVISION) once you've
# looked up the exact commit hash you want to trust on the model's Hugging Face "Files and
# versions" tab, e.g. https://huggingface.co/BAAI/bge-m3/commits/main
DEFAULT_EMBEDDING_MODEL_REVISION: Final[str] = ""
DEFAULT_RERANKER_MODEL_REVISION: Final[str] = ""
DEFAULT_SLM_MODEL_REVISION: Final[str] = ""

# Metadata keys that are safe to expose in a citation / API response to an end user.
# Anything NOT in this allowlist (internal classification tags, ingestion source paths,
# raw file system paths, internal doc IDs, etc.) is stripped before a chunk's metadata
# leaves the system in a citation or API response.
PUBLIC_CITATION_METADATA_KEYS: Final[List[str]] = [
    MetadataKeys.DOC_NAME,
    MetadataKeys.REGULATOR,
    MetadataKeys.DOC_TYPE,
    MetadataKeys.SECTION,
    MetadataKeys.CHAPTER,
    MetadataKeys.CLAUSE,
    MetadataKeys.PAGE_NUMBER,
    MetadataKeys.DATE,
    MetadataKeys.VERSION,
]

# Canary token prefix embedded in the system prompt (with a random suffix generated per
# request) so leakage of the system prompt / retrieved context into a visible answer can be
# detected. Never log or expose the actual per-request token value anywhere the model output
# is displayed unfiltered.
CANARY_TOKEN_PREFIX: Final[str] = "UBI-RAG-CANARY"

# Heuristic patterns used by the retrieval-layer content guard to flag retrieved chunks that
# look like they are trying to inject instructions into the model rather than provide
# reference content. This is a heuristic first line of defense, not a substitute for a proper
# classifier (see retrieval/content_guard.py docstring).
INJECTION_HEURISTIC_PATTERNS: Final[List[str]] = [
    r"ignore (all|any|the) (previous|prior|above)",
    r"disregard (all|any|the) (previous|prior|above)",
    r"you are now",
    r"new instructions?:",
    r"system\s*prompt",
    r"act as (if|though)",
    r"do not (follow|obey|comply with) (the|your) (system|previous)",
    r"reveal (the|your) (system prompt|instructions)",
    r"print (the|your) (system prompt|instructions)",
    r"</?(system|instructions?|admin)>",
    r"\bAI:\s*I will\b",
    r"override (the|your|all) (rules|guardrails|instructions)",
]

# Zero-width / invisible Unicode characters commonly used to hide instructions inside
# otherwise-normal-looking document text.
ZERO_WIDTH_UNICODE_PATTERN: Final[str] = (
    "[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\ufeff]"
)
