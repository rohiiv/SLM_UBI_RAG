"""
Banking RAG Base Cache Manager Interface.

Defines the contract for caching query embeddings, retrieval results, and LLM responses.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseCacheManager(ABC):
    """Abstract Interface for Cache Managers."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieves a value from the cache by key.

        Args:
            key: Unique cache key string.

        Returns:
            Cached value if found and valid, otherwise None.
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stores a value in the cache with an optional time-to-live.

        Args:
            key: Unique cache key string.
            value: Value object to cache.
            ttl: Time-to-live in seconds (overrides default TTL if specified).
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears all cached items."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Deletes a specific key from the cache.

        Args:
            key: Cache key to remove.

        Returns:
            True if key existed and was deleted, False otherwise.
        """
        pass
