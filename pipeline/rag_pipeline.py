"""
Banking RAG Online RAG Inference Pipeline.

Orchestrates: User Query -> Cache Check -> Hybrid Retrieval -> Cross-Encoder Rerank -> Prompt Construction -> Fine-tuned Qwen SLM -> Citation Answer.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

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

    def _validate_citations(
        self,
        answer: str,
        guarded_chunks: List[ScoredChunk],
    ) -> Tuple[str, List[str]]:
        """Validates inline citations against retrieved context document names.

        Strips citations whose document names do not match any retrieved chunk source metadata.
        """
        if not answer:
            return answer, []

        valid_doc_names = set()
        for item in guarded_chunks:
            meta = item.chunk.metadata or {}
            d_name = (meta.get("doc_name") or meta.get("source") or "").strip().lower()
            if d_name:
                valid_doc_names.add(d_name)

        citation_pattern = re.compile(r"\[Source:\s*(?P<doc>[^\|\]]+)(?:\|[^\]]*)?\]")
        dropped: List[str] = []

        def _replacer(match: re.Match) -> str:
            full_citation = match.group(0)
            cited_doc = match.group("doc").strip().lower()

            is_valid = (
                any(cited_doc in v_name or v_name in cited_doc for v_name in valid_doc_names)
                if valid_doc_names
                else False
            )

            if is_valid:
                return full_citation
            else:
                logger.warning(
                    f"Citation validation: dropping hallucinated citation '{full_citation}' "
                    f"with doc name '{match.group('doc').strip()}' - not found in retrieved chunks."
                )
                dropped.append(full_citation)
                return ""

        cleaned_answer = citation_pattern.sub(_replacer, answer).strip()
        cleaned_answer = re.sub(r"[ \t]+", " ", cleaned_answer)
        return cleaned_answer, dropped

    def _verify_faithfulness(
        self,
        answer: str,
        guarded_chunks: List[ScoredChunk],
        faithfulness_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Verifies factual grounding of generated answer sentences against retrieved context."""
        if not answer or not answer.strip():
            return {
                "sentences_checked": 0,
                "sentences_grounded": 0,
                "faithfulness_score": 1.0,
            }

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        corpus = " ".join(item.chunk.content.lower() for item in guarded_chunks)

        grounded_count = 0
        for sentence in sentences:
            clean_sentence = re.sub(r"\[Source:[^\]]+\]", "", sentence).lower()
            words = [re.sub(r"[^\w]", "", w) for w in clean_sentence.split()]
            content_words = [w for w in words if len(w) > 4]

            if not content_words:
                grounded_count += 1
                continue

            matched_count = sum(1 for w in content_words if w in corpus)
            fraction = matched_count / len(content_words)

            if fraction >= faithfulness_threshold:
                grounded_count += 1
            else:
                logger.warning(
                    f"Faithfulness verification: sentence may not be grounded "
                    f"(match_fraction={fraction:.2f} < {faithfulness_threshold}): '{sentence[:80]}...'"
                )

        total = len(sentences)
        score = (grounded_count / total) if total > 0 else 1.0

        return {
            "sentences_checked": total,
            "sentences_grounded": grounded_count,
            "faithfulness_score": round(score, 3),
        }

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
            print("🔍 Searching vector database & retrieving candidates...")
            retrieved_candidates = self.retriever.retrieve(query=query_str, top_k=top_k, filters=filters)

            # 3. Cross-Encoder Reranking
            print("🎯 Reranking retrieved candidates...")
            reranked_chunks = self.reranker.rerank(query=query_str, candidates=retrieved_candidates, top_k=top_k)

            # 3a. Abstention Guard - Reranker Score Threshold
            retrieval_config = get_config().retrieval
            threshold = retrieval_config.abstention_reranker_threshold

            if threshold > 0:
                confident_chunks = [c for c in reranked_chunks if c.score >= threshold]
                if not confident_chunks:
                    logger.warning(
                        f"Abstaining for query '{query_str}': no reranked chunks met "
                        f"confidence threshold ({threshold}). Returning INSUFFICIENT_EVIDENCE."
                    )
                    return RAGResponse(
                        query=query_str,
                        answer="INSUFFICIENT_EVIDENCE: The retrieved context does not contain enough relevant information to answer this question reliably.",
                        citations=[],
                        retrieved_chunks=[],
                        cached=False,
                        metadata={
                            "abstained": True,
                            "abstention_reason": "no_confident_chunks",
                            "total_retrieved": len(retrieved_candidates),
                            "total_reranked": len(reranked_chunks),
                            "filters_applied": filters or {},
                        },
                    )
            else:
                confident_chunks = reranked_chunks

            # 3b. Content Guard & Empty Context Abstention
            guarded_chunks, flagged_chunks = self.content_guard.screen(chunks=confident_chunks, query=query_str)
            if not guarded_chunks:
                logger.warning(
                    f"Abstaining for query '{query_str}': all retrieved chunks were "
                    f"filtered by the content guard."
                )
                return RAGResponse(
                    query=query_str,
                    answer="INSUFFICIENT_EVIDENCE: All retrieved context was filtered by the content security guard. Unable to generate a safe answer.",
                    citations=[],
                    retrieved_chunks=[],
                    cached=False,
                    metadata={
                        "abstained": True,
                        "abstention_reason": "all_chunks_blocked",
                        "total_retrieved": len(retrieved_candidates),
                        "total_reranked": len(reranked_chunks),
                        "content_guard_flagged": len(flagged_chunks),
                        "filters_applied": filters or {},
                    },
                )

            # 4. Construct Prompt
            canary_token = generate_canary_token() if self.security_config.enable_canary_tokens else None
            prompt_payload = self.prompt_builder.build_prompt(
                query=query_str, retrieved_chunks=guarded_chunks, canary_token=canary_token,
            )

            # 5. Execute Fine-tuned Qwen SLM Generation
            print("🧠 Generating response with Qwen SLM...")
            answer = self.llm_generator.generate(prompt_payload)

            # 5a. Citation validation
            answer, dropped_citations = self._validate_citations(answer, guarded_chunks)

            # 5b. Canary-token leakage check
            canary_leak_detected = bool(canary_token and canary_token in answer)
            if canary_leak_detected:
                logger.error(
                    f"CANARY TOKEN LEAK DETECTED for query '{query_str}': the model echoed its "
                    f"internal tracking token in the answer. This indicates system-prompt "
                    f"leakage and should be investigated as a possible successful injection."
                )
                answer = answer.replace(canary_token, "[REDACTED]")

            # 5c. Faithfulness verification
            faithfulness_result = self._verify_faithfulness(answer, guarded_chunks)

            # 6. Extract unique source citations
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
                    "dropped_hallucinated_citations": dropped_citations,
                    "faithfulness": faithfulness_result,
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

