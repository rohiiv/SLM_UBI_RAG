"""
Banking RAG Contextual Chunker.

Prepends document-level and section-level metadata context headers to every text chunk,
improving retriever relevance for isolated regulatory text fragments.
"""

from typing import List, Optional

from banking_rag.chunking.chunker import RecursiveTextChunker, TextChunk
from banking_rag.constants import MetadataKeys
from banking_rag.loaders.base_loader import Document
from banking_rag.utils.logger import get_logger

logger = get_logger("chunking.contextual_chunker")


class ContextualChunker:
    """Wraps RecursiveTextChunker to produce context-aware chunks."""

    def __init__(self, base_chunker: Optional[RecursiveTextChunker] = None):
        """Initializes Contextual Chunker.

        Args:
            base_chunker: Optional RecursiveTextChunker instance.
        """
        self.chunker = base_chunker or RecursiveTextChunker()

    def process(self, document: Document) -> List[TextChunk]:
        """Generates contextual chunks for a single document.

        Args:
            document: Document object to chunk.

        Returns:
            List of TextChunk instances with prepended context headers.
        """
        # Extract document-level context header elements
        doc_name = document.metadata.get(MetadataKeys.DOC_NAME, "Document")
        regulator = document.metadata.get(MetadataKeys.REGULATOR, "")
        domain = document.metadata.get(MetadataKeys.DOMAIN, "")
        section = document.metadata.get(MetadataKeys.SECTION, "")

        header_prefix_parts = []
        if regulator:
            header_prefix_parts.append(f"Regulator: {regulator}")
        if domain:
            header_prefix_parts.append(f"Domain: {domain}")
        header_prefix_parts.append(f"Document: {doc_name}")
        if section:
            header_prefix_parts.append(f"Section: {section}")

        doc_header = " | ".join(header_prefix_parts)

        # Handle pre-chunked documents (e.g., JSONL records) without re-splitting
        if document.metadata.get("is_prechunked", False):
            context_header = f"[Context: {doc_header}]\n"
            contextual_content = context_header + document.content

            chunk_metadata = dict(document.metadata)
            chunk_metadata[MetadataKeys.CONTEXT_HEADER] = context_header.strip()
            chunk_metadata[MetadataKeys.CHUNK_ID] = document.doc_id
            chunk_metadata[MetadataKeys.PARENT_ID] = document.metadata.get("document_id", document.doc_id)
            chunk_metadata["chunk_index"] = document.metadata.get("chunk_index", 1)
            chunk_metadata["total_chunks"] = 1

            chunk_obj = TextChunk(
                chunk_id=document.doc_id,
                content=contextual_content,
                metadata=chunk_metadata,
                parent_doc_id=document.metadata.get("document_id", document.doc_id),
                start_char_idx=0,
                end_char_idx=len(document.content),
            )
            return [chunk_obj]

        raw_chunks = self.chunker.chunk_document(document)
        contextual_chunks: List[TextChunk] = []

        for chunk in raw_chunks:
            # Build contextual block header
            context_header = f"[Context: {doc_header}]\n"
            contextual_content = context_header + chunk.content

            # Update chunk object metadata
            chunk.metadata[MetadataKeys.CONTEXT_HEADER] = context_header.strip()
            
            updated_chunk = TextChunk(
                chunk_id=chunk.chunk_id,
                content=contextual_content,
                metadata=chunk.metadata,
                parent_doc_id=chunk.parent_doc_id,
                start_char_idx=chunk.start_char_idx,
                end_char_idx=chunk.end_char_idx,
            )
            contextual_chunks.append(updated_chunk)

        logger.info(f"Added contextual headers to {len(contextual_chunks)} chunks for {doc_name}")
        return contextual_chunks

    def process_batch(self, documents: List[Document]) -> List[TextChunk]:
        """Processes a batch of documents into contextual chunks.

        Args:
            documents: List of Document objects.

        Returns:
            Flattened list of contextual TextChunk objects.
        """
        all_chunks: List[TextChunk] = []
        for doc in documents:
            all_chunks.extend(self.process(doc))
        return all_chunks
