"""
Banking RAG PDF Document Loader.

Reads PDF banking documents page-by-page, attaching initial structural metadata.
"""

from pathlib import Path
from typing import List, Union

from banking_rag.constants import MetadataKeys
from banking_rag.exceptions import DocumentLoadError
from banking_rag.loaders.base_loader import BaseDocumentLoader, Document
from banking_rag.utils.file_utils import compute_file_hash, get_file_extension, validate_file_exists
from banking_rag.utils.logger import get_logger

logger = get_logger("loaders.pdf_loader")


class PDFLoader(BaseDocumentLoader):
    """Loader for PDF documents (RBI Directions, Acts, Board Minutes)."""

    def supports(self, file_path: Union[str, Path]) -> bool:
        """Checks if file extension is .pdf."""
        return get_file_extension(file_path) == ".pdf"

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """Loads a PDF document page by page.

        Args:
            file_path: Path to target PDF file.

        Returns:
            List of Document objects representing extracted pages.

        Raises:
            DocumentLoadError: If PDF reading or text extraction fails.
        """
        path = validate_file_exists(file_path)
        if not self.supports(path):
            raise DocumentLoadError(f"Unsupported file format for PDFLoader: {path}")

        logger.info(f"Loading PDF document: {path.name}")
        file_hash = compute_file_hash(path)
        documents: List[Document] = []

        try:
            # Try pypdf first
            import pypdf
            reader = pypdf.PdfReader(str(path))
            num_pages = len(reader.pages)
            
            for page_idx in range(num_pages):
                page = reader.pages[page_idx]
                page_text = page.extract_text() or ""
                
                doc = Document(
                    content=page_text,
                    file_path=path,
                    page_number=page_idx + 1,
                    doc_id=f"{file_hash[:12]}_p{page_idx + 1}",
                    metadata={
                        MetadataKeys.DOC_NAME: path.name,
                        MetadataKeys.SOURCE: str(path),
                        MetadataKeys.PAGE_NUMBER: page_idx + 1,
                        "total_pages": num_pages,
                        "file_hash": file_hash,
                    },
                )
                documents.append(doc)
                
            logger.info(f"Successfully loaded {len(documents)} pages from PDF: {path.name}")
            return documents

        except ImportError:
            logger.warning("pypdf package not found. Falling back to basic text loader strategy.")
            # Basic raw binary stream fallback or error out gracefully
            raise DocumentLoadError("PDF extraction dependency 'pypdf' is not installed.")
        except Exception as e:
            logger.error(f"Error parsing PDF file {path}: {str(e)}")
            raise DocumentLoadError(f"Failed to extract PDF content from {path.name}: {str(e)}")
