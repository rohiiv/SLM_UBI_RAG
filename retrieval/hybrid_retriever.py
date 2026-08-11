"""
Banking RAG Hybrid Retriever module.

Combines dense vector retrieval with sparse BM25 keyword matching using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Optional

from banking_rag.config import RetrievalConfig, get_config
from banking_rag.exceptions import RetrievalError
from banking_rag.retrieval.retriever import BaseRetriever, DenseRetriever
from banking_rag.utils.logger import get_logger
from banking_rag.vectorstore.qdrant_manager import ScoredChunk

logger = get_logger("retrieval.hybrid_retriever")


class HybridRetriever(BaseRetriever):
    """Hybrid Retriever fusing dense vector search and sparse BM25 scores via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        """Initializes HybridRetriever.

        Args:
            dense_retriever: Dense retriever instance.
            config: Retrieval settings.
        """
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.config = config or get_config().retrieval
        self.rrf_k = self.config.rrf_k
        self.default_top_k = self.config.top_k_dense

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredChunk]:
        """Executes hybrid retrieval using Reciprocal Rank Fusion.

        Args:
            query: Natural language search query.
            top_k: Number of combined results to return.
            filters: Optional metadata filters.

        Returns:
            Fused list of ScoredChunk objects sorted by RRF score descending.

        Raises:
            RetrievalError: If hybrid fusion fails.
        """
        if not query.strip():
            return []

        k = top_k if top_k is not None else self.default_top_k
        logger.info(f"Executing hybrid retrieval with RRF for query: '{query}'")

        try:
            # 1. Fetch dense candidates (using pre_rrf_k pool size)
            dense_results = self.dense_retriever.retrieve(query=query, top_k=self.config.pre_rrf_k, filters=filters)

            # 2. Simulate/execute sparse BM25 candidates
            sparse_results = self._bm25_sparse_retrieve(query=query, dense_candidates=dense_results)

            # 3. Fuse candidate rankings via Reciprocal Rank Fusion (RRF)
            fused_chunks = self._reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=k,
            )

            logger.info(f"HybridRetriever successfully fused {len(fused_chunks)} results.")
            return fused_chunks

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            raise RetrievalError(f"Error during hybrid retrieval: {str(e)}")

    def _bm25_sparse_retrieve(self, query: str, dense_candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        """Simple in-memory BM25 term frequency scoring over candidate chunks.

        Args:
            query: Query string.
            dense_candidates: List of candidate chunks.

        Returns:
            Reranked list of candidate chunks based on term matching.
        """
        query_terms = set(query.lower().split())
        scored = []

        for item in dense_candidates:
            content_lower = item.chunk.content.lower()
            # Calculate term match frequency
            matches = sum(1 for term in query_terms if term in content_lower)
            bm25_score = matches / (len(query_terms) + 1e-5)
            scored.append(ScoredChunk(chunk=item.chunk, score=bm25_score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[ScoredChunk],
        sparse_results: List[ScoredChunk],
        top_k: int,
    ) -> List[ScoredChunk]:
        """Calculates Reciprocal Rank Fusion (RRF) scores across dense and sparse ranking lists.

        Formula: RRF_score(doc) = sum(1 / (rrf_k + rank(doc)))
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, ScoredChunk] = {}

        # Process dense ranks
        for rank, item in enumerate(dense_results):
            cid = item.chunk.chunk_id
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Process sparse ranks
        for rank, item in enumerate(sparse_results):
            cid = item.chunk.chunk_id
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sort by fused RRF score descending
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        fused = []
        for cid, rrf_score in sorted_ids:
            original_chunk = chunk_map[cid].chunk
            fused.append(ScoredChunk(chunk=original_chunk, score=rrf_score))

        return fused
