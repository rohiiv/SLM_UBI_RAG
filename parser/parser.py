"""
Banking RAG Document Parser.

Normalizes raw document text, strips invalid control sequences, extracts chapter/section headings,
and preserves banking document structure.
"""

import re
from typing import List, Dict, Any, Tuple

from banking_rag.exceptions import DocumentParsingError
from banking_rag.loaders.base_loader import Document
from banking_rag.utils.logger import get_logger
from banking_rag.utils.text_utils import clean_text

logger = get_logger("parser.parser")


class DocumentParser:
    """Parser responsible for cleaning raw text and extracting hierarchy metadata."""

    # Heading detection patterns common in banking regulations and Master Directions
    HEADING_PATTERNS = [
        r"^(CHAPTER\s+[IVXLCDM\d]+[^\n]*)",
        r"^(SECTION\s+[\d\.]+[^\n]*)",
        r"^(PART\s+[IVXLCDM\d]+[^\n]*)",
        r"^(\d+\.\s+[A-Z][^\n]+)",
        r"^([A-Z\s]{4,}:)",
    ]

    def parse(self, document: Document) -> Document:
        """Parses and cleans a single Document instance.

        Args:
            document: Raw Document input.

        Returns:
            Parsed Document instance with cleaned content and populated hierarchy headers.

        Raises:
            DocumentParsingError: If parsing fails.
        """
        if not document or not document.content:
            logger.warning("Empty document passed to DocumentParser.")
            return document

        try:
            # 1. Clean raw text
            cleaned_content = clean_text(document.content)

            # 2. Extract structural headings
            detected_headings = self._extract_headings(cleaned_content)
            
            # 3. Update document metadata
            updated_metadata = dict(document.metadata)
            if detected_headings:
                updated_metadata["detected_headings"] = detected_headings
                # Main section header if present
                updated_metadata["primary_header"] = detected_headings[0]

            return Document(
                content=cleaned_content,
                metadata=updated_metadata,
                doc_id=document.doc_id,
                file_path=document.file_path,
                page_number=document.page_number,
            )

        except Exception as e:
            logger.error(f"Error parsing document {document.doc_id}: {str(e)}")
            raise DocumentParsingError(f"Failed to parse document content: {str(e)}")

    def parse_batch(self, documents: List[Document]) -> List[Document]:
        """Parses a batch of Document objects.

        Args:
            documents: List of Document instances.

        Returns:
            List of parsed Document instances.
        """
        logger.info(f"Batch parsing {len(documents)} document objects.")
        return [self.parse(doc) for doc in documents]

    def _extract_headings(self, text: str) -> List[str]:
        """Scans text for structural section headers.

        Args:
            text: Cleaned text string.

        Returns:
            List of matched heading strings.
        """
        headings = []
        lines = text.split("\n")
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            for pattern in self.HEADING_PATTERNS:
                match = re.match(pattern, line_str, re.IGNORECASE)
                if match:
                    headings.append(match.group(1).strip())
                    break

        return headings
