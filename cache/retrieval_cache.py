"""
Banking RAG Retrieval Cache Implementation.

Thread-safe, in-memory TTL LRU Cache for caching retrieval outputs and query embeddings.
"""

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional, Dict

from banking_rag.cache.cache_manager import BaseCacheManager
from banking_rag.config import CacheConfig, get_config
from banking_rag.utils.logger import get_logger

logger = get_logger("cache.retrieval_cache")


class RetrievalCacheManager(BaseCacheManager):
    """In-memory TTL LRU Cache implementation."""

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initializes the retrieval cache manager.

        Args:
            config: Optional CacheConfig instance. If None, loaded from global config.
        """
        self.config = config or get_config().cache
        self.enabled = self.config.enable_retrieval_cache
        self.max_size = self.config.max_cache_size
        self.default_ttl = self.config.ttl_seconds
        
        # Internal cache map: key -> (value, expiry_timestamp)
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def generate_cache_key(query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> str:
        """Helper to generate a deterministic SHA-256 cache key for a query and filters.

        Args:
            query: Natural language query string.
            filters: Optional metadata filters.
            top_k: Top K retrieved documents target.

        Returns:
            Hex string cache key.
        """
        raw_key = f"{query.strip().lower()}|{str(filters)}|{top_k}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieves an unexpired cached item.

        Args:
            key: Cache key.

        Returns:
            Cached item or None if missing/expired.
        """
        if not self.enabled:
            return None

        with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]
            
            # Check expiration
            if time.time() > expiry:
                logger.debug(f"Cache key expired: {key}")
                del self._cache[key]
                return None

            # Move to end (LRU update)
            self._cache.move_to_end(key)
            logger.debug(f"Cache hit for key: {key[:12]}...")
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stores a key-value pair in cache.

        Args:
            key: Cache key.
            value: Item to store.
            ttl: Optional TTL override in seconds.
        """
        if not self.enabled:
            return

        effective_ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + effective_ttl

        with self._lock:
            # Enforce max size (LRU eviction)
            if key not in self._cache and len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Cache full. Evicted LRU key: {oldest_key[:12]}...")

            self._cache[key] = (value, expiry)
            self._cache.move_to_end(key)
            logger.debug(f"Cached item key: {key[:12]}... TTL: {effective_ttl}s")

    def delete(self, key: str) -> bool:
        """Deletes a key from cache.

        Args:
            key: Cache key.

        Returns:
            True if key was deleted, False if missing.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clears all entries from the cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Retrieval cache cleared successfully.")
