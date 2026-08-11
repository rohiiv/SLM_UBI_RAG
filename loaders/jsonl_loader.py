"""
Banking RAG JSONL Document Loader.

Reads JSON Lines (.jsonl) banking documents line-by-line, extracting text content and metadata fields.
"""

import json
from pathlib import Path
from typing import List, Union, Dict, Any

from banking_rag.constants import MetadataKeys
from banking_rag.exceptions import DocumentLoadError
from banking_rag.loaders.base_loader import BaseDocumentLoader, Document
from banking_rag.utils.file_utils import compute_file_hash, get_file_extension, validate_file_exists
from banking_rag.utils.logger import get_logger

logger = get_logger("loaders.jsonl_loader")


class JSONLLoader(BaseDocumentLoader):
    """Loader for JSON Lines (.jsonl) document datasets."""

    TEXT_KEYS = ["content", "text", "passage", "body", "document", "context", "chunk", "answer", "question"]

    def supports(self, file_path: Union[str, Path]) -> bool:
        """Checks if file extension is .jsonl."""
        return get_file_extension(file_path) == ".jsonl"

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """Loads a .jsonl file line by line into Document objects.

        Args:
            file_path: Path to target .jsonl file.

        Returns:
            List of Document objects created from JSON lines.

        Raises:
            DocumentLoadError: If file loading or JSON parsing fails.
        """
        return list(self.load_iter(file_path))

    def load_iter(self, file_path: Union[str, Path]):
        """Lazily yields Document objects from a .jsonl file line by line.

        Args:
            file_path: Path to target .jsonl file.

        Yields:
            Document objects created from JSON lines.

        Raises:
            DocumentLoadError: If file loading fails.
        """
        path = validate_file_exists(file_path)
        if not self.supports(path):
            raise DocumentLoadError(f"Unsupported file format for JSONLLoader: {path}")

        logger.info(f"Streaming JSONL dataset line-by-line: {path.name}")
        file_hash = compute_file_hash(path)
        count = 0

        import os
        filter_relevance = os.getenv("JSONL_FILTER_RELEVANCE", "true").lower() == "true"

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_idx, raw_line in enumerate(f):
                    line_str = raw_line.strip()
                    if not line_str:
                        continue

                    try:
                        record = json.loads(line_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {line_idx + 1} in {path.name}: {str(e)}")
                        continue

                    if not isinstance(record, dict):
                        continue

                    # Filter out non-relevant entries if configured
                    relevance_val = str(record.get("relevance", "")).strip().upper()
                    if filter_relevance and relevance_val == "NOT_RELEVANT":
                        logger.debug(f"Skipping line {line_idx + 1} tagged NOT_RELEVANT in {path.name}")
                        continue

                    # Extract text content from key candidates
                    content = self._extract_content(record)
                    if not content:
                        logger.warning(f"No text content key found in JSON line {line_idx + 1} of {path.name}")
                        continue

                    # Extract metadata fields
                    metadata = self._extract_metadata(record, path, file_hash, line_idx + 1)

                    doc_id = (
                        record.get("chunk_id")
                        or record.get("doc_id")
                        or record.get("document_id")
                        or record.get("id")
                        or f"{file_hash[:12]}_line{line_idx + 1}"
                    )

                    page_num = record.get(MetadataKeys.PAGE_NUMBER) or record.get("page_number") or 1

                    doc = Document(
                        content=content,
                        file_path=path,
                        page_number=page_num,
                        doc_id=doc_id,
                        metadata=metadata,
                    )
                    count += 1
                    yield doc

            logger.info(f"Successfully streamed {count} document entries from JSONL file: {path.name}")

        except Exception as e:
            logger.error(f"Error streaming JSONL file {path}: {str(e)}")
            raise DocumentLoadError(f"Failed to load JSONL content from {path.name}: {str(e)}")

    def _extract_content(self, record: Dict[str, Any]) -> str:
        """Finds text content from known dictionary key candidates."""
        for key in self.TEXT_KEYS:
            if key in record and record[key] and isinstance(record[key], str):
                return record[key].strip()
        
        # Fallback: if 'question' and 'answer' are both present, combine them
        if "question" in record and "answer" in record:
            return f"Question: {record['question']}\nAnswer: {record['answer']}"

        return ""

    def _extract_metadata(self, record: Dict[str, Any], file_path: Path, file_hash: str, line_num: int) -> Dict[str, Any]:
        """Builds document metadata dictionary from JSON record fields."""
        doc_name = (
            record.get(MetadataKeys.DOC_NAME)
            or record.get("source_file")
            or record.get("source")
            or record.get("title")
            or file_path.name
        )
        regulator = (
            record.get(MetadataKeys.REGULATOR)
            or record.get("regulatory_body")
            or ""
        )
        domain = (
            record.get(MetadataKeys.DOMAIN)
            or record.get("department")
            or ""
        )
        doc_type = (
            record.get(MetadataKeys.DOC_TYPE)
            or record.get("document_type")
            or ""
        )
        section = (
            record.get(MetadataKeys.SECTION)
            or record.get("section_heading")
            or ""
        )
        date_val = (
            record.get(MetadataKeys.DATE)
            or record.get("publication_date")
            or ""
        )
        page_num = record.get(MetadataKeys.PAGE_NUMBER) or record.get("page_number") or 1

        metadata = {
            MetadataKeys.DOC_NAME: str(doc_name),
            MetadataKeys.SOURCE: str(file_path),
            MetadataKeys.PAGE_NUMBER: page_num,
            MetadataKeys.REGULATOR: str(regulator) if regulator else "",
            MetadataKeys.DOMAIN: str(domain) if domain else "",
            MetadataKeys.DOC_TYPE: str(doc_type) if doc_type else "",
            MetadataKeys.SECTION: str(section) if section else "",
            MetadataKeys.CHAPTER: record.get(MetadataKeys.CHAPTER, ""),
            MetadataKeys.CLAUSE: record.get(MetadataKeys.CLAUSE, ""),
            MetadataKeys.DATE: str(date_val) if date_val else "",
            MetadataKeys.VERSION: record.get(MetadataKeys.VERSION, ""),
            "file_hash": file_hash,
            "jsonl_line": line_num,
            "is_prechunked": True,
        }

        # Include any extra non-standard metadata keys from record (including lists like keywords & entities)
        for k, v in record.items():
            if k not in self.TEXT_KEYS and k not in metadata and isinstance(v, (str, int, float, bool, list)):
                metadata[k] = v

        return metadata
