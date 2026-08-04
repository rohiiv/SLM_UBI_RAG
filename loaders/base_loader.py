"""
Banking RAG Base Document Loader Interface.

Defines the core Document abstraction and BaseDocumentLoader abstract interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Union, Optional


@dataclass
class Document:
    """Core Document abstraction holding raw/parsed text and associated metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default="")
    file_path: Optional[Path] = None
    page_number: Optional[int] = None


class BaseDocumentLoader(ABC):
    """Abstract base class for all document format loaders (PDF, DOCX, TXT)."""

    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """Loads a single file into one or more Document instances.

        Args:
            file_path: Path string or Path object pointing to target document file.

        Returns:
            List of loaded Document objects (e.g., one per page or single doc).

        Raises:
            DocumentLoadError: If loading fails or format is corrupted.
        """
        pass

    def load_iter(self, file_path: Union[str, Path]):
        """Lazily yields Document instances from target file.

        Default implementation yields from self.load(file_path). Specialized loaders
        (e.g., JSONLLoader) override this for true memory-efficient line-by-line streaming.

        Args:
            file_path: Path string or Path object.

        Yields:
            Document instances.
        """
        for doc in self.load(file_path):
            yield doc

    @abstractmethod
    def supports(self, file_path: Union[str, Path]) -> bool:
        """Determines whether this loader supports the given file extension.

        Args:
            file_path: Path to inspect.

        Returns:
            True if supported, False otherwise.
        """
        pass
