# Banking RAG Pipeline — Architecture Overview

> **Primary Architecture Documentation:** See [architecture.md](file:///Users/rohinivemula/Desktop/EUCLID/RAG_2/architecture.md) for the complete engineering deep-dive, security architecture, data schemas, and pipeline specifications.

---

## High-Level Architecture Summary

```mermaid
graph TD
    subgraph "Interfaces"
        CLI["CLI (main.py)"]
        API["FastAPI Server (api.py)"]
    end

    subgraph "Offline Pipeline (ingest.py)"
        ALLOW["Source Path Allowlist"]
        LOAD["Document Loaders (JSONL, PDF, DOCX, TXT)"]
        PARSE["Document Parser (Sanitization)"]
        META["Metadata Extractor"]
        CHUNK["Contextual Chunker"]
        EMBED_I["BGE-M3 Embedding Generator"]
        UPSERT["Qdrant / FAISS Vector Store"]
    end

    subgraph "Online Pipeline (rag_pipeline.py)"
        CACHE_CHK["Cache Check"]
        HYBRID["Hybrid Retriever (Dense + BM25 RRF)"]
        RERANK["Cross-Encoder Reranker"]
        ABSTAIN["Confidence Abstention Guard"]
        GUARD["Retrieval Content Guard"]
        PROMPT["Prompt Builder (Canary Token)"]
        LLM["Qwen Banking SLM Generator"]
        CITE_VAL["Citation & Faithfulness Verifier"]
        CANARY["Canary Token Leak Detector"]
        CACHE_SET["Cache Store"]
    end

    subgraph "Infrastructure"
        QDRANT[("Qdrant / FAISS Vector DB")]
        CONFIG["AppConfig (Frozen Dataclass)"]
        CACHE[("Retrieval Cache (LRU + TTL)")]
    end

    CLI --> LOAD
    CLI --> CACHE_CHK
    API --> CACHE_CHK
    API --> LOAD

    LOAD --> PARSE --> META --> CHUNK --> EMBED_I --> UPSERT --> QDRANT

    CACHE_CHK -->|miss| HYBRID
    CACHE_CHK -->|hit| CACHE
    HYBRID --> RERANK --> ABSTAIN --> GUARD --> PROMPT --> LLM --> CITE_VAL --> CANARY --> CACHE_SET --> CACHE
    HYBRID -->|dense search| QDRANT
```

---

## System Subsystems

1. **Ingestion & Data Normalization**: Robust multi-format parsing with invisible Unicode stripping and metadata tagging (Regulator, Domain, Section, Date).
2. **Hybrid Search Engine**: BGE-M3 dense embeddings fused with BM25 term frequency scores via Reciprocal Rank Fusion ($k=60$).
3. **Cross-Encoder Reranker**: `BAAI/bge-reranker-large` re-scoring candidates for precision context selection.
4. **Security & Abstention Rails**: Content guard heuristic injection screening, per-request canary tokens, citation grounding verification, and metadata exposure scrubbing.
5. **Generative SLM Engine**: Fine-tuned `Qwen2.5-3B-Instruct` model generating objective, citation-backed compliance responses.
