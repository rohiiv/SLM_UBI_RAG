"""
Union Bank of India Banking RAG System - FastAPI Server Endpoint Module.

Provides a persistent HTTP API wrapper around OnlineRAGPipeline and OfflineIngestionPipeline.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure root directory is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root.parent) not in sys.path:
    sys.path.insert(0, str(project_root.parent))

from banking_rag.config import get_config
from banking_rag.main import build_rag_pipeline
from banking_rag.pipeline.ingest import OfflineIngestionPipeline
from banking_rag.utils.logger import setup_logger, get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager ensuring single-time eager startup initialization."""
    # Reuse setup_logger() exactly as main.py calls it
    config = get_config()
    log_file = project_root / "logs" / "banking_rag.log"
    setup_logger(log_level=config.log_level, log_file=log_file, console_level="WARNING")
    logger.info(f"API Server initializing RAG pipeline. Logging to file: {log_file}")

    # Build and cache RAG pipeline once per server lifetime
    app.state.rag = build_rag_pipeline()
    logger.info("RAG Pipeline successfully initialized and bound to app.state.rag.")
    yield
    logger.info("API Server shutting down.")


app = FastAPI(
    title="Union Bank of India Enterprise Banking RAG API",
    description="Persistent HTTP API wrapper for OnlineRAGPipeline and OfflineIngestionPipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., description="Banking compliance or regulatory question string")
    top_k: int = Field(default=3, description="Number of top reranked chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    cached: bool
    metadata: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory path string to ingest")


class IngestResponse(BaseModel):
    status: str
    file_name: Optional[str] = None
    pages: Optional[int] = None
    chunks_ingested: Optional[int] = None


# Endpoints
@app.post("/query", response_model=QueryResponse)
async def query_rag(payload: QueryRequest):
    """Executes a compliance query using the pre-loaded OnlineRAGPipeline instance."""
    # TODO: Add API authentication / API key validation (Out of scope for this phase)
    # TODO: Add rate limiting (Out of scope for this phase)
    # TODO: Add concurrency control / request queue for LLM generation (Out of scope for this phase)
    if not hasattr(app.state, "rag") or app.state.rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized.",
        )

    try:
        res = app.state.rag.query(query_text=payload.question, top_k=payload.top_k)
        return QueryResponse(
            answer=res.answer,
            citations=res.citations,
            cached=res.cached,
            metadata=res.metadata,
        )
    except Exception as e:
        logger.error(f"Error serving /query request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@app.post("/ingest")
async def ingest_document(payload: IngestRequest):
    """Ingests a banking document file or directory into vector store."""
    # TODO: Add role-based authentication / authorization for document ingestion (Out of scope for this phase)
    target_path = Path(payload.path)
    if not target_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path '{payload.path}' does not exist.",
        )

    try:
        pipeline = OfflineIngestionPipeline()
        if target_path.is_file():
            res = pipeline.ingest_file(target_path)
            return res
        elif target_path.is_dir():
            results = pipeline.ingest_directory(target_path)
            return {"status": "success", "processed_files": len(results), "details": results}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid path type for '{payload.path}'.",
            )
    except Exception as e:
        logger.error(f"Error serving /ingest request for path '{payload.path}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@app.get("/health")
async def health_check():
    """Confirms ready status of all 4 pipeline components and Qdrant DB connectivity."""
    if not hasattr(app.state, "rag") or app.state.rag is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "reason": "RAG pipeline not initialized"},
        )

    rag = app.state.rag
    components_status = {
        "embedding_generator": False,
        "reranker": False,
        "llm_generator": False,
        "qdrant_client": False,
    }

    # 1. Embedding generator loaded check
    try:
        embedder = rag.retriever.dense_retriever.embedding_generator
        if embedder is not None and getattr(embedder, "_model", None) is not None:
            components_status["embedding_generator"] = True
    except Exception:
        pass

    # 2. Reranker loaded check
    try:
        reranker = rag.reranker
        if reranker is not None and getattr(reranker, "_model", None) is not None:
            components_status["reranker"] = True
    except Exception:
        pass

    # 3. LLM generator loaded check
    try:
        llm = rag.llm_generator
        if llm is not None and getattr(llm, "_model", None) is not None:
            components_status["llm_generator"] = True
    except Exception:
        pass

    # 4. Qdrant client connectivity check
    try:
        vstore = rag.retriever.dense_retriever.vector_store
        client = vstore._get_client()
        if client == "MOCK":
            components_status["qdrant_client"] = True
        elif client is not None:
            client.get_collections()
            components_status["qdrant_client"] = True
    except Exception:
        pass

    all_healthy = all(components_status.values())
    health_content = {
        "status": "healthy" if all_healthy else "unhealthy",
        "components": components_status,
    }

    if not all_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_content,
        )

    return health_content
