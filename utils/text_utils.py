"""
Banking RAG Text Utility module.

Provides text cleaning, normalization, section extraction, and citation formatting helper functions.
"""

import re
from typing import Dict, Any, List
from banking_rag.constants import (
    CITATION_TEMPLATE,
    MetadataKeys,
    PUBLIC_CITATION_METADATA_KEYS,
    ZERO_WIDTH_UNICODE_PATTERN,
)

# HTML comments (<!-- ... -->) and zero-width/invisible Unicode characters are both known
# vectors for hiding instructions inside otherwise normal-looking ingested documents -
# see Layer 1 sanitization guidance. Compiled once at module load.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_ZERO_WIDTH_RE = re.compile(ZERO_WIDTH_UNICODE_PATTERN)


def clean_text(text: str) -> str:
    """Normalizes raw document text by removing control characters, hidden/invisible
    Unicode, HTML comments, and redundant whitespace.

    This is the ingestion-time sanitization pass: anything that lets an attacker hide an
    instruction inside a document that otherwise looks like normal regulatory text should be
    stripped here, before the text is ever chunked or embedded.

    Args:
        text: Raw text string.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    # Remove non-printable control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Remove zero-width / invisible Unicode characters (e.g. U+200B ZERO WIDTH SPACE,
    # U+FEFF BOM/ZERO WIDTH NO-BREAK SPACE, bidi override characters) that can be used to
    # hide text from a human reviewer while still being tokenized by the model.
    text = _ZERO_WIDTH_RE.sub("", text)

    # Strip HTML comments outright - never let their content reach the model. Note this runs
    # BEFORE whitespace collapsing so multi-line comments are removed cleanly.
    text = _HTML_COMMENT_RE.sub("", text)

    # Normalize carriage returns and line feeds
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace multiple spaces/tabs with single space (preserve single newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def scrub_metadata_for_exposure(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Filters a chunk's raw metadata dict down to the allowlisted, user-safe fields.

    Raw chunk metadata can carry internal-only fields (ingestion file paths, internal
    classification tags, doc/chunk IDs used only for retrieval bookkeeping, content-guard
    flags, etc.) that should never round-trip into a citation or API response shown to an
    end user. Call this at the boundary where a chunk's metadata is about to be exposed
    (citations, API responses, logs shown outside the ops team).

    Args:
        metadata: Raw chunk metadata dictionary.

    Returns:
        A new dictionary containing only PUBLIC_CITATION_METADATA_KEYS entries that were
        actually present in the input.
    """
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if k in PUBLIC_CITATION_METADATA_KEYS}


def estimate_token_count(text: str) -> int:
    """Estimates token count of a given text (approx. 4 characters or 0.75 words per token).

    Args:
        text: Input text string.

    Returns:
        Estimated integer token count.
    """
    if not text:
        return 0
    words = text.split()
    return int(len(words) * 1.3)


def format_citation(metadata: Dict[str, Any]) -> str:
    """Constructs a clean, human-readable citation string from chunk metadata.

    Args:
        metadata: Metadata dictionary containing document fields.

    Returns:
        Formatted citation string.
    """
    # Only ever read citation fields from the scrubbed, allowlisted view of metadata - this
    # guarantees that even if a caller passes raw chunk.metadata (which may include internal
    # fields), nothing beyond PUBLIC_CITATION_METADATA_KEYS can leak into a citation string.
    safe_metadata = scrub_metadata_for_exposure(metadata)

    doc_name = safe_metadata.get(MetadataKeys.DOC_NAME, "Unknown Document")
    regulator = safe_metadata.get(MetadataKeys.REGULATOR, "N/A")
    section = safe_metadata.get(MetadataKeys.SECTION, "N/A")
    page_number = safe_metadata.get(MetadataKeys.PAGE_NUMBER, "N/A")

    return CITATION_TEMPLATE.format(
        doc_name=doc_name,
        regulator=regulator,
        section=section,
        page_number=page_number,
    )


def extract_sections(text: str) -> List[Dict[str, str]]:
    """Simple regex parser to segment text by section headers (e.g., 'Section 3', 'Chapter II').

    Args:
        text: Full document text.

    Returns:
        List of dictionaries with 'title' and 'content'.
    """
    pattern = r"((?:Chapter|Section|Clause|Part)\s+[IVXLCDM\d\.\-]+[^\n]*)"
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    sections = []
    if len(parts) <= 1:
        return [{"title": "General", "content": text.strip()}]

    # Handle intro text before first heading
    if parts[0].strip():
        sections.append({"title": "Preamble", "content": parts[0].strip()})

    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append({"title": header, "content": body})

    return sections
