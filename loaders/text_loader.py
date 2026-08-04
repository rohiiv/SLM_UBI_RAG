"""
Banking RAG Text Document Loader.

Reads plain text (.txt) banking documents with UTF-8 / fallback encodings.
"""

from pathlib import Path
from typing import List, Union

from banking_rag.constants import MetadataKeys
from banking_rag.exceptions import DocumentLoadError
from banking_rag.loaders.base_loader import BaseDocumentLoader, Document
from banking_rag.utils.file_utils import compute_file_hash, get_file_extension, validate_file_exists
from banking_rag.utils.logger import get_logger

logger = get_logger("loaders.text_loader")


class TextLoader(BaseDocumentLoader):
    """Loader for plain text (.txt) files."""

    def supports(self, file_path: Union[str, Path]) -> bool:
        """Checks if file extension is .txt."""
        return get_file_extension(file_path) == ".txt"

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """Loads a plain text document.

        Args:
            file_path: Path to target text file.

        Returns:
            List containing single Document object.

        Raises:
            DocumentLoadError: If file reading fails.
        """
        path = validate_file_exists(file_path)
        if not self.supports(path):
            raise DocumentLoadError(f"Unsupported file format for TextLoader: {path}")

        logger.info(f"Loading TXT document: {path.name}")
        file_hash = compute_file_hash(path)

        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        content = None

        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"Successfully read {path.name} using encoding {encoding}")
                break
            except (UnicodeDecodeError, Exception):
                continue

        if content is None:
            logger.error(f"Failed to decode text file {path.name} with all attempted encodings.")
            raise DocumentLoadError(f"Could not decode text file {path.name}")

        document = Document(
            content=content,
            file_path=path,
            page_number=1,
            doc_id=f"{file_hash[:12]}_txt",
            metadata={
                MetadataKeys.DOC_NAME: path.name,
                MetadataKeys.SOURCE: str(path),
                MetadataKeys.PAGE_NUMBER: 1,
                "file_hash": file_hash,
            },
        )

        logger.info(f"Successfully loaded TXT file: {path.name}")
        return [document]
