"""
Banking RAG Online RAG Inference Pipeline.

Orchestrates: User Query -> Cache Check -> Hybrid Retrieval -> Cross-Encoder Rerank -> Prompt Construction -> Fine-tuned Qwen SLM -> Citation Answer.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from banking_rag.cache.retrieval_cache import RetrievalCacheManager
from banking_rag.config import SecurityConfig, get_config
from banking_rag.exceptions import PipelineError, SecurityError
from banking_rag.llm.generator import BaseLLMGenerator, QwenBankingSLMGenerator
from banking_rag.prompts.prompt_builder import PromptBuilder, generate_canary_token
from banking_rag.retrieval.content_guard import RetrievalContentGuard
from banking_rag.retrieval.hybrid_retriever import HybridRetriever
from banking_rag.retrieval.reranker import BaseReranker, CrossEncoderReranker
from banking_rag.retrieval.retriever import BaseRetriever
from banking_rag.utils.logger import get_logger
from banking_rag.utils.text_utils import format_citation
from banking_rag.vectorstore.qdrant_manager import ScoredChunk

logger = get_logger("pipeline.rag_pipeline")


@dataclass
class RAGResponse:
    """Encapsulates the complete RAG output payload."""
    query: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[ScoredChunk]
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class OnlineRAGPipeline:
    """End-to-end Online RAG Pipeline for Union Bank of India."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        reranker: Optional[BaseReranker] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        llm_generator: Optional[BaseLLMGenerator] = None,
        cache_manager: Optional[RetrievalCacheManager] = None,
        content_guard: Optional[RetrievalContentGuard] = None,
        security_config: Optional[SecurityConfig] = None,
    ):
        """Initializes OnlineRAGPipeline with injected components.

        Args:
            retriever: Search & retrieval strategy (default HybridRetriever).
            reranker: Cross-encoder reranking strategy (default CrossEncoderReranker).
            prompt_builder: Prompt construction module.
            llm_generator: SLM text generation engine.
            cache_manager: Query caching manager.
            content_guard: Retrieval-layer content guard screening chunks before they reach
                the prompt (default RetrievalContentGuard).
            security_config: Security settings; defaults to the process-wide config.
        """
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_generator = llm_generator or QwenBankingSLMGenerator()
        self.cache_manager = cache_manager or RetrievalCacheManager()
        self.content_guard = content_guard or RetrievalContentGuard()
        self.security_config = security_config or get_config().security

    def query(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> RAGResponse:
        """Processes user query online through the complete RAG pipeline.

        Args:
            query_text: Natural language user question.
            filters: Optional metadata filtering criteria.
            top_k: Number of final reranked chunks to feed to LLM context.

        Returns:
            RAGResponse instance containing answer, citations, and retrieved source chunks.

        Raises:
            PipelineError: If query execution fails.
        """
        if not query_text or not query_text.strip():
            raise PipelineError("Query string cannot be empty.")

        query_str = query_text.strip()
        logger.info(f"Processing online RAG query: '{query_str}' (filters={filters}, top_k={top_k})")

        try:
            # 1. Check Retrieval & Answer Cache
            cache_key = self.cache_manager.generate_cache_key(query=query_str, filters=filters, top_k=top_k)
            cached_response = self.cache_manager.get(cache_key)
            if cached_response is not None:
                logger.info(f"Returning cached RAG response for query key: {cache_key[:12]}...")
                cached_response.cached = True
                return cached_response

            # 2. Hybrid Retrieval (Dense + BM25)
            retrieved_candidates = self.retriever.retrieve(query=query_str, top_k=top_k * 4, filters=filters)

            # 3. Cross-Encoder Reranking
            reranked_chunks = self.reranker.rerank(query=query_str, candidates=retrieved_candidates, top_k=top_k)

            # 3b. Retrieval-layer content guard: screen every chunk AFTER retrieval/rerank and
            # BEFORE it is allowed into the prompt. This is what catches indirect prompt
            # injection hiding inside an ingested document - input-side filtering alone never
            # sees this content, since it never came through the user's query box.
            guarded_chunks, flagged_chunks = self.content_guard.screen(chunks=reranked_chunks, query=query_str)
            if flagged_chunks and self.security_config.content_guard_action == "block" and not guarded_chunks:
                logger.warning(
                    f"All {len(flagged_chunks)} retrieved chunks were blocked by the content "
                    f"guard for query '{query_str}'; proceeding with empty context."
                )

            # 4. Construct Prompt (with a per-request canary token embedded for leak detection)
            canary_token = generate_canary_token() if self.security_config.enable_canary_tokens else None
            prompt_payload = self.prompt_builder.build_prompt(
                query=query_str, retrieved_chunks=guarded_chunks, canary_token=canary_token,
            )

            # 5. Execute Fine-tuned Qwen SLM Generation
            answer = self.llm_generator.generate(prompt_payload)

            # 5b. Canary-token leakage check: if the token embedded in the system prompt shows
            # up in the visible answer, the model was tricked into echoing internal state
            # (either its own system prompt, or something it inferred about it). Strip it and
            # log loudly rather than ever showing it to the user.
            canary_leak_detected = bool(canary_token and canary_token in answer)
            if canary_leak_detected:
                logger.error(
                    f"CANARY TOKEN LEAK DETECTED for query '{query_str}': the model echoed its "
                    f"internal tracking token in the answer. This indicates system-prompt "
                    f"leakage and should be investigated as a possible successful injection."
                )
                answer = answer.replace(canary_token, "[REDACTED]")

            # 6. Extract unique source citations (format_citation only ever reads the
            # allowlisted public metadata fields - see utils/text_utils.scrub_metadata_for_exposure)
            citations = list(set([format_citation(item.chunk.metadata) for item in guarded_chunks]))

            response = RAGResponse(
                query=query_str,
                answer=answer,
                citations=citations,
                retrieved_chunks=guarded_chunks,
                cached=False,
                metadata={
                    "total_retrieved": len(retrieved_candidates),
                    "total_reranked": len(reranked_chunks),
                    "content_guard_flagged": len(flagged_chunks),
                    "canary_leak_detected": canary_leak_detected,
                    "filters_applied": filters or {},
                },
            )

            # 7. Store in Cache
            self.cache_manager.set(cache_key, response)

            logger.info("Online RAG pipeline query execution completed successfully.")
            return response

        except Exception as e:
            logger.error(f"Error executing online RAG pipeline for query '{query_str}': {str(e)}")
            raise PipelineError(f"RAG execution failure: {str(e)}")
