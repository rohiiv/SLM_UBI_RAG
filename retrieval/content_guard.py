"""
Banking RAG Retrieval-Layer Content Guard.

Implements the Layer 2 "retrieval rails" control: every retrieved chunk is screened AFTER it
comes back from Qdrant and BEFORE it is allowed into the LLM prompt. This is the layer most
RAG implementations skip entirely - input-side filtering only ever sees the user's query, so
it can't catch an instruction hidden inside a document that gets pulled in by retrieval
(indirect / poisoned-document prompt injection).

Design notes (deliberately conservative for this stage):

- This is a HEURISTIC classifier (regex pattern matching + a couple of structural signals),
  not a fine-tuned model. Heuristic-only filtering is known to be an incomplete defense
  against injection payloads phrased as normal narrative text rather than explicit
  instructions - meaning this WILL miss covert attacks. Treat this as a first, cheap layer,
  not a complete solution. The natural next step, once you have real traffic, is adding a
  second, semantic layer such as ProtectAI's deberta-v3-base-prompt-injection-v2 classifier
  or a store of known-attack embeddings (Rebuff's approach), run alongside this one rather
  than instead of it - single-classifier setups have a documented failure mode where
  over-filtering benign procedural text is just as damaging as under-filtering.
- Two response modes, controlled by SecurityConfig.content_guard_action:
    "flag"  - chunk is kept but annotated (chunk.metadata['content_guard_flagged']=True) and
              logged loudly. Use this first, while you're tuning the pattern list against your
              real corpus, so you can see false-positive rate before you start dropping content.
    "block" - flagged chunks are removed from the candidate list entirely before reranking/
              prompting. Switch to this once you trust the pattern list isn't eating good
              regulatory text (which is written in fairly formal, sometimes imperative,
              language and can trip naive filters - hence starting in "flag" mode).
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from banking_rag.config import SecurityConfig, get_config
from banking_rag.constants import INJECTION_HEURISTIC_PATTERNS
from banking_rag.utils.logger import get_logger
from banking_rag.vectorstore.qdrant_manager import ScoredChunk

logger = get_logger("retrieval.content_guard")

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_HEURISTIC_PATTERNS]

# A chunk with an implausibly high density of imperative/instructional sentences relative to
# its length is itself a weak structural signal of an injected instruction block, independent
# of whether it matches a specific phrase pattern.
_IMPERATIVE_OPENERS = re.compile(
    r"(?:^|\n)\s*(ignore|disregard|override|forget|pretend|act as|respond only|always say|never mention)\b",
    re.IGNORECASE,
)


@dataclass
class GuardVerdict:
    """Result of screening a single chunk."""
    flagged: bool
    reasons: List[str]


def screen_chunk_text(text: str) -> GuardVerdict:
    """Screens a single chunk's text content for injection-like patterns.

    Args:
        text: The chunk's content string.

    Returns:
        GuardVerdict indicating whether the chunk was flagged and why.
    """
    reasons: List[str] = []

    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            reasons.append(f"matched pattern: {pattern.pattern!r}")

    imperative_hits = len(_IMPERATIVE_OPENERS.findall(text))
    if imperative_hits >= 2:
        reasons.append(f"{imperative_hits} imperative-instruction-style sentence openers")

    return GuardVerdict(flagged=bool(reasons), reasons=reasons)


class RetrievalContentGuard:
    """Screens a list of retrieved/reranked chunks before they enter the prompt."""

    def __init__(self, security_config: SecurityConfig = None):
        self.config = security_config or get_config().security

    def screen(self, chunks: List[ScoredChunk], query: str) -> Tuple[List[ScoredChunk], List[ScoredChunk]]:
        """Screens candidate chunks, returning (passed_chunks, flagged_chunks).

        In "flag" mode, passed_chunks == all input chunks (flagged ones are annotated but not
        removed) and flagged_chunks is a duplicate list for logging/telemetry. In "block" mode,
        flagged chunks are excluded from passed_chunks entirely.

        Args:
            chunks: Candidate ScoredChunk list (typically post-rerank, pre-prompt).
            query: The user's query string, included only for logging context.

        Returns:
            Tuple of (chunks allowed into the prompt, chunks that were flagged).
        """
        if not self.config.enable_retrieval_content_guard or not chunks:
            return chunks, []

        passed, flagged = [], []
        for item in chunks:
            verdict = screen_chunk_text(item.chunk.content)
            if verdict.flagged:
                item.chunk.metadata = dict(item.chunk.metadata or {})
                item.chunk.metadata["content_guard_flagged"] = True
                item.chunk.metadata["content_guard_reasons"] = verdict.reasons
                flagged.append(item)
                logger.warning(
                    f"Content guard flagged chunk '{item.chunk.chunk_id}' for query "
                    f"'{query}': {verdict.reasons}. Action={self.config.content_guard_action}"
                )
                if self.config.content_guard_action == "block":
                    continue  # drop from passed list
            passed.append(item)

        if flagged:
            logger.warning(
                f"Content guard flagged {len(flagged)}/{len(chunks)} retrieved chunks "
                f"(action={self.config.content_guard_action})."
            )

        return passed, flagged
