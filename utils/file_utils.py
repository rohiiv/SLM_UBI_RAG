"""
Banking RAG File Utility module.

Provides safe file operations, extension validation, path resolution, and document hashing.
"""

import hashlib
from pathlib import Path
from typing import List, Union

from banking_rag.constants import SUPPORTED_FILE_EXTENSIONS
from banking_rag.exceptions import DocumentLoadError
from banking_rag.utils.logger import get_logger

logger = get_logger("utils.file_utils")


def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """Validates that a given file exists and is a regular file.

    Args:
        file_path: Path string or Path object.

    Returns:
        Resolved Path object.

    Raises:
        DocumentLoadError: If file does not exist or is a directory.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise DocumentLoadError(f"Target file does not exist: {path}", details={"path": str(path)})
    if not path.is_file():
        logger.error(f"Path is not a file: {path}")
        raise DocumentLoadError(f"Target path is not a file: {path}", details={"path": str(path)})
    return path


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Returns the lowercased extension of a file.

    Args:
        file_path: File path or string.

    Returns:
        Extension string including leading dot (e.g., '.pdf').
    """
    return Path(file_path).suffix.lower()


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """Checks whether the file has a supported banking document extension.

    Args:
        file_path: File path or string.

    Returns:
        True if supported, False otherwise.
    """
    ext = get_file_extension(file_path)
    return ext in SUPPORTED_FILE_EXTENSIONS


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file for change detection and deduplication.

    Args:
        file_path: File path or string.

    Returns:
        Hexadecimal SHA-256 checksum string.
    """
    path = validate_file_exists(file_path)
    sha256_hash = hashlib.sha256()
    
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()


def find_documents(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Finds all supported banking documents in a directory.

    Args:
        directory: Directory path to scan.
        recursive: Whether to scan subdirectories recursively.

    Returns:
        List of resolved Path objects.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning(f"Invalid scan directory: {dir_path}")
        return []

    pattern = "**/*" if recursive else "*"
    found_files = []
    
    for item in dir_path.glob(pattern):
        if item.is_file() and is_supported_file(item):
            found_files.append(item)
            
    logger.info(f"Found {len(found_files)} supported banking documents in {dir_path}")
    return found_files
