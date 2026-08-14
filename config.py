"""
Banking RAG Configuration module.

Provides a centralized, immutable, and type-safe configuration container for
the Union Bank of India RAG system, loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from banking_rag.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DENSE_VECTOR_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    DEFAULT_QDRANT_COLLECTION_NAME,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_MODEL_REVISION,
    DEFAULT_SLM_MODEL,
    DEFAULT_SLM_MODEL_REVISION,
    DEFAULT_TOP_K_DENSE,
    DEFAULT_TOP_K_RERANKED,
    DEFAULT_TOP_K_SPARSE,
)


@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant Vector Database Connection Settings.

    Supports three backend modes controlled by VECTOR_DB_MODE:
      server   - Connect to a remote Qdrant server (requires QDRANT_HOST or QDRANT_URL).
      embedded - Run Qdrant in-process on a local directory (requires QDRANT_PATH).
      faiss    - Use a local FAISS IndexFlatIP index with a SQLite metadata sidecar
                 (requires FAISS_INDEX_PATH; does NOT require any Qdrant settings).
    """
    mode: str = field(default_factory=lambda: os.getenv("VECTOR_DB_MODE", "server").lower())
    path: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_PATH", "./qdrant_data"))
    host: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost") or None)
    port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    grpc_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_GRPC_PORT", "6334")))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", None) or None)
    url: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_URL", None))
    prefer_grpc: bool = field(default_factory=lambda: os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true")
    collection_name: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION") or os.getenv("QDRANT_COLLECTION_NAME") or DEFAULT_QDRANT_COLLECTION_NAME)
    vector_size: int = field(default_factory=lambda: int(os.getenv("QDRANT_VECTOR_SIZE", str(DEFAULT_DENSE_VECTOR_DIM))))
    timeout: float = field(default_factory=lambda: float(os.getenv("QDRANT_TIMEOUT", "60.0")))
    # FAISS-mode only: directory where faiss.index and faiss_metadata.sqlite are stored.
    faiss_index_path: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./faiss_data"))

    @property
    def vector_db_mode(self) -> str:
        return self.mode

    @property
    def qdrant_path(self) -> Optional[str]:
        return self.path

    @property
    def qdrant_host(self) -> Optional[str]:
        return self.host

    @property
    def qdrant_port(self) -> int:
        return self.port

    @property
    def qdrant_api_key(self) -> Optional[str]:
        return self.api_key

    def __post_init__(self) -> None:
        from banking_rag.exceptions import ConfigurationError
        mode_val = (self.mode or "").lower()
        if mode_val not in ("server", "embedded", "faiss"):
            raise ConfigurationError("VECTOR_DB_MODE must be 'server', 'embedded', or 'faiss'.")
        if mode_val == "embedded":
            if not self.path:
                raise ConfigurationError("When VECTOR_DB_MODE=embedded, QDRANT_PATH is required.")
        elif mode_val == "server":
            if not self.host and not self.url:
                raise ConfigurationError("When VECTOR_DB_MODE=server, QDRANT_HOST is required.")
        elif mode_val == "faiss":
            if not self.faiss_index_path:
                raise ConfigurationError("When VECTOR_DB_MODE=faiss, FAISS_INDEX_PATH is required.")



def _detect_device() -> str:
    env_device = os.getenv("MODEL_DEVICE")
    if env_device:
        return env_device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass(frozen=True)
class ModelConfig:
    """ML & SLM Model Weights & Execution Settings."""
    embedding_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL))
    slm_model_name: str = field(default_factory=lambda: os.getenv("SLM_MODEL_NAME", DEFAULT_SLM_MODEL))
    reranker_model_name: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL))
    device: str = field(default_factory=_detect_device)
    use_quantization: bool = field(default_factory=lambda: os.getenv("USE_QUANTIZATION", "false").lower() == "true")
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "1024")))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    top_p: float = field(default_factory=lambda: float(os.getenv("LLM_TOP_P", "0.9")))
    # Supply-chain hardening: pin every model to an exact HF commit SHA rather than "main"/
    # "latest". Empty string means unpinned; loaders log a WARNING (not an error) so dev
    # workflow isn't blocked, but this should be non-empty before real banking data is loaded.
    embedding_model_revision: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_REVISION", DEFAULT_EMBEDDING_MODEL_REVISION))
    reranker_model_revision: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL_REVISION", DEFAULT_RERANKER_MODEL_REVISION))
    slm_model_revision: str = field(default_factory=lambda: os.getenv("SLM_MODEL_REVISION", DEFAULT_SLM_MODEL_REVISION))


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking Strategy Hyperparameters."""
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))))
    enable_contextual_chunking: bool = field(default_factory=lambda: os.getenv("ENABLE_CONTEXTUAL_CHUNKING", "true").lower() == "true")


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval & Reranking Parameters."""
    top_k_dense: int = field(default_factory=lambda: int(os.getenv("TOP_K_DENSE", str(DEFAULT_TOP_K_DENSE))))
    top_k_sparse: int = field(default_factory=lambda: int(os.getenv("TOP_K_SPARSE", str(DEFAULT_TOP_K_SPARSE))))
    top_k_reranked: int = field(default_factory=lambda: int(os.getenv("TOP_K_RERANKED", str(DEFAULT_TOP_K_RERANKED))))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    pre_rrf_k: int = field(default_factory=lambda: int(os.getenv("PRE_RRF_K", "150")))
    min_similarity_score: float = field(default_factory=lambda: float(os.getenv("MIN_SIMILARITY_SCORE", "0.3")))
    abstention_reranker_threshold: float = field(default_factory=lambda: float(os.getenv("ABSTENTION_RERANKER_THRESHOLD", "0.0")))



@dataclass(frozen=True)
class IngestionConfig:
    """Ingestion Source Allowlisting.

    For banking/compliance data, ingestion must be restricted to a controlled, versioned set
    of source locations (e.g. an internal policy repo checkout, an RBI-circulars mirror) -
    never an arbitrary path an operator points the CLI at. `allowed_source_roots` is a list of
    directories; any file passed to `ingest_file` / `ingest_directory` must resolve to a path
    under one of these roots or ingestion is refused.

    Default is the project's own `./data` directory so the dummy-dataset workflow keeps
    working out of the box; set INGESTION_ALLOWED_ROOTS explicitly once real source
    directories exist.
    """
    allowed_source_roots: List[str] = field(
        default_factory=lambda: [
            p.strip() for p in os.getenv("INGESTION_ALLOWED_ROOTS", "./data,./banking_rag/data").split(",") if p.strip()
        ]
    )
    enforce_allowlist: bool = field(default_factory=lambda: os.getenv("INGESTION_ENFORCE_ALLOWLIST", "true").lower() == "true")
    batch_size: int = field(default_factory=lambda: int(os.getenv("INGESTION_BATCH_SIZE", "1000")))
    upsert_batch_size: int = field(default_factory=lambda: int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "500")))


@dataclass(frozen=True)
class SecurityConfig:
    """Retrieval-layer and output-layer security controls."""
    enable_retrieval_content_guard: bool = field(default_factory=lambda: os.getenv("ENABLE_RETRIEVAL_CONTENT_GUARD", "true").lower() == "true")
    # "flag": log + attach a warning flag to the chunk but still let it through (safe default
    # while you're tuning the heuristics against real data). "block": drop flagged chunks
    # before they reach the prompt.
    content_guard_action: str = field(default_factory=lambda: os.getenv("CONTENT_GUARD_ACTION", "flag"))
    enable_canary_tokens: bool = field(default_factory=lambda: os.getenv("ENABLE_CANARY_TOKENS", "true").lower() == "true")
    require_qdrant_auth_outside_dev: bool = field(default_factory=lambda: os.getenv("REQUIRE_QDRANT_AUTH_OUTSIDE_DEV", "true").lower() == "true")


@dataclass(frozen=True)
class CacheConfig:
    """Cache Settings."""
    enable_retrieval_cache: bool = field(default_factory=lambda: os.getenv("ENABLE_RETRIEVAL_CACHE", "true").lower() == "true")
    ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))
    max_cache_size: int = field(default_factory=lambda: int(os.getenv("MAX_CACHE_SIZE", "1000")))


@dataclass(frozen=True)
class AppConfig:
    """Root Application Configuration."""
    app_name: str = "Union Bank RAG System"
    environment: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "./logs")))
    
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __post_init__(self) -> None:
        """Fail fast on configuration that would silently leave production data exposed."""
        is_dev = self.environment.lower() in ("development", "dev", "local", "test")
        if (
            self.qdrant.mode == "server"
            and self.security.require_qdrant_auth_outside_dev
            and not is_dev
            and not self.qdrant.api_key
            and not (self.qdrant.url and self.qdrant.url.startswith("https"))
        ):
            from banking_rag.exceptions import ConfigurationError
            raise ConfigurationError(
                "Refusing to start: APP_ENV is not a dev/local/test environment, but "
                "QDRANT_API_KEY is not set (and QDRANT_URL is not an https:// endpoint with "
                "its own transport security). Set QDRANT_API_KEY, or explicitly set "
                "REQUIRE_QDRANT_AUTH_OUTSIDE_DEV=false if you understand the risk."
            )


def get_config() -> AppConfig:
    """Singleton getter for application configuration."""
    return AppConfig()
