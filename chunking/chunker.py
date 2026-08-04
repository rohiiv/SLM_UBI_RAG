"""
Banking RAG Recursive Text Chunker.

Splits parsed documents recursively using hierarchical separators (\n\n, \n, ., space) while preserving boundaries and overlap.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from banking_rag.config import ChunkingConfig, get_config
from banking_rag.constants import MetadataKeys
from banking_rag.exceptions import ChunkingError
from banking_rag.loaders.base_loader import Document
from banking_rag.utils.logger import get_logger

logger = get_logger("chunking.chunker")


@dataclass
class TextChunk:
    """Represents a text chunk ready for vector embedding and indexing."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_doc_id: str = field(default="")
    start_char_idx: int = field(default=0)
    end_char_idx: int = field(default=0)


class RecursiveTextChunker:
    """Splits documents into overlapping chunks using recursive separators."""

    SEPARATORS = ["\n\n", "\n", ". ", "; ", " ", ""]

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initializes chunker with configuration settings.

        Args:
            config: Optional ChunkingConfig object.
        """
        self.config = config or get_config().chunking
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ChunkingError("Chunk overlap cannot be greater than or equal to chunk size.")

    def chunk_document(self, document: Document) -> List[TextChunk]:
        """Chunks a single Document into a list of TextChunk objects.

        Args:
            document: Parsed Document instance.

        Returns:
            List of TextChunk objects with populated metadata.

        Raises:
            ChunkingError: If chunking fails.
        """
        if not document or not document.content.strip():
            logger.warning(f"Skipping empty document: {document.doc_id}")
            return []

        try:
            raw_text = document.content
            raw_chunks = self._split_text_recursively(raw_text, self.SEPARATORS)

            chunks: List[TextChunk] = []
            current_pos = 0

            for idx, text_block in enumerate(raw_chunks):
                chunk_id = f"{document.doc_id}_c{idx + 1}"
                
                # Copy base metadata from document
                chunk_metadata = dict(document.metadata)
                chunk_metadata.update({
                    MetadataKeys.CHUNK_ID: chunk_id,
                    MetadataKeys.PARENT_ID: document.doc_id,
                    "chunk_index": idx + 1,
                    "total_chunks": len(raw_chunks),
                })

                chunk_obj = TextChunk(
                    chunk_id=chunk_id,
                    content=text_block,
                    metadata=chunk_metadata,
                    parent_doc_id=document.doc_id,
                    start_char_idx=current_pos,
                    end_char_idx=current_pos + len(text_block),
                )
                chunks.append(chunk_obj)
                current_pos += max(1, len(text_block) - self.chunk_overlap)

            logger.info(f"Chunked document {document.doc_id} into {len(chunks)} text chunks.")
            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk document {document.doc_id}: {str(e)}")
            raise ChunkingError(f"Error during recursive text chunking: {str(e)}")

    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """Chunks a list of documents.

        Args:
            documents: List of Document objects.

        Returns:
            Flattened list of TextChunk objects across all input documents.
        """
        all_chunks: List[TextChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _split_text_recursively(self, text: str, separators: List[str]) -> List[str]:
        """Recursive helper method to break text into blocks under chunk_size."""
        final_chunks = []
        
        # Base case
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Find best separator
        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break

        # Split text by separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks respecting max chunk_size and overlap
        current_chunk = []
        current_length = 0

        for s in splits:
            item = s if separator == "" else s + separator
            item_len = len(item)

            if current_length + item_len > self.chunk_size and current_chunk:
                merged_text = "".join(current_chunk).strip()
                if merged_text:
                    final_chunks.append(merged_text)
                
                # Overlap step: keep last items up to chunk_overlap
                overlap_items = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) <= self.chunk_overlap:
                        overlap_items.insert(0, prev)
                        overlap_len += len(prev)
                    else:
                        break
                
                current_chunk = overlap_items
                current_length = overlap_len

            current_chunk.append(item)
            current_length += item_len

        if current_chunk:
            merged_text = "".join(current_chunk).strip()
            if merged_text:
                final_chunks.append(merged_text)

        return final_chunks
