"""Redis-based query response caching for performance optimization."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from assistant.core.logging import get_logger, log_service_method
from assistant.infrastructure.cache.redis_client import RedisClient


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    data: Any
    timestamp: float
    ttl: int
    hit_count: int = 0
    last_accessed: float = None


class QueryCache:
    """Redis-based query response caching system."""

    def __init__(
        self,
        redis_client: RedisClient,
        default_ttl: int = 300,  # 5 minutes
        key_prefix: str = "query_cache",
    ):
        """
        Initialize query cache.

        Args:
            redis_client: Redis client instance
            default_ttl: Default TTL in seconds
            key_prefix: Prefix for cache keys
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.logger = get_logger(self.__class__.__name__)

    def _generate_cache_key(self, operation: str, **kwargs) -> str:
        """
        Generate unique cache key from operation and parameters.

        Args:
            operation: Operation name
            **kwargs: Operation parameters

        Returns:
            Unique cache key
        """
        # Create a deterministic hash of the operation and parameters
        key_data = {"operation": operation, "params": kwargs}

        # Sort dict for consistent hashing
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]

        return f"{self.key_prefix}:{operation}:{key_hash}"

    @log_service_method()
    async def get(self, operation: str, **kwargs) -> Optional[Any]:
        """
        Get cached result for operation.

        Args:
            operation: Operation name
            **kwargs: Operation parameters

        Returns:
            Cached result or None if not found
        """
        try:
            cache_key = self._generate_cache_key(operation, **kwargs)
            cached_data = await self.redis.get(cache_key)

            if cached_data is None:
                return None

            # Parse cached entry
            try:
                entry_data = json.loads(cached_data)
                entry = CacheEntry(**entry_data)

                # Update access tracking
                entry.hit_count += 1
                entry.last_accessed = time.time()

                # Store updated entry back to Redis
                await self.redis.set(cache_key, json.dumps(asdict(entry)), expire_seconds=entry.ttl)

                self.logger.debug(
                    f"Cache hit for {operation}: key={cache_key[:16]}..., "
                    f"hits={entry.hit_count}"
                )

                return entry.data

            except (json.JSONDecodeError, TypeError) as e:
                self.logger.warning(f"Invalid cache entry format: {str(e)}")
                # Remove invalid entry
                await self.redis.delete(cache_key)
                return None

        except Exception as e:
            self.logger.error(f"Cache get failed for {operation}: {str(e)}")
            return None

    @log_service_method()
    async def set(self, operation: str, data: Any, ttl: Optional[int] = None, **kwargs) -> bool:
        """
        Cache result for operation.

        Args:
            operation: Operation name
            data: Data to cache
            ttl: Time to live in seconds
            **kwargs: Operation parameters

        Returns:
            True if cached successfully
        """
        try:
            cache_key = self._generate_cache_key(operation, **kwargs)
            ttl = ttl or self.default_ttl

            entry = CacheEntry(
                data=data, timestamp=time.time(), ttl=ttl, hit_count=0, last_accessed=None
            )

            success = await self.redis.set(
                cache_key, json.dumps(asdict(entry), default=str), expire_seconds=ttl
            )

            if success:
                self.logger.debug(f"Cache set for {operation}: key={cache_key[:16]}..., ttl={ttl}s")

            return success

        except Exception as e:
            self.logger.error(f"Cache set failed for {operation}: {str(e)}")
            return False

    @log_service_method()
    async def invalidate(self, operation: str, **kwargs) -> bool:
        """
        Invalidate cached result for operation.

        Args:
            operation: Operation name
            **kwargs: Operation parameters

        Returns:
            True if invalidated
        """
        try:
            cache_key = self._generate_cache_key(operation, **kwargs)
            success = await self.redis.delete(cache_key)

            if success:
                self.logger.info(f"Cache invalidated for {operation}: key={cache_key[:16]}...")

            return success

        except Exception as e:
            self.logger.error(f"Cache invalidation failed for {operation}: {str(e)}")
            return False

    @log_service_method()
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching pattern.

        Args:
            pattern: Pattern to match (e.g., "search_*")

        Returns:
            Number of entries invalidated
        """
        try:
            full_pattern = f"{self.key_prefix}:{pattern}"
            count = 0

            # Use Redis SCAN to find matching keys
            async for key in self.redis.redis.scan_iter(match=full_pattern):
                await self.redis.delete(key)
                count += 1

            self.logger.info(f"Invalidated {count} cache entries matching pattern: {pattern}")
            return count

        except Exception as e:
            self.logger.error(f"Pattern invalidation failed for {pattern}: {str(e)}")
            return 0

    @log_service_method()
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            pattern = f"{self.key_prefix}:*"
            total_keys = 0
            total_size = 0
            entries_by_operation = {}

            async for key in self.redis.redis.scan_iter(match=pattern):
                total_keys += 1

                # Get entry size
                try:
                    data = await self.redis.get(key)
                    if data:
                        total_size += len(data.encode("utf-8"))

                        # Parse operation from key
                        key_parts = key.split(":")
                        if len(key_parts) >= 3:
                            operation = key_parts[2]
                            entries_by_operation[operation] = (
                                entries_by_operation.get(operation, 0) + 1
                            )

                except Exception:
                    continue

            return {
                "total_entries": total_keys,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "entries_by_operation": entries_by_operation,
                "default_ttl": self.default_ttl,
                "key_prefix": self.key_prefix,
            }

        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {str(e)}")
            return {"error": str(e), "total_entries": 0, "total_size_bytes": 0}


class CachedQueryMixin:
    """Mixin to add caching capabilities to services."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._query_cache: Optional[QueryCache] = None

    async def _get_query_cache(self) -> Optional[QueryCache]:
        """Get or create query cache instance."""
        if self._query_cache is None:
            try:
                from assistant.infrastructure.cache.redis_client import get_redis_client

                redis_client = await get_redis_client()
                self._query_cache = QueryCache(redis_client)
            except Exception as e:
                self.logger.warning(f"Failed to initialize query cache: {str(e)}")
                return None

        return self._query_cache

    async def cached_operation(
        self, operation_name: str, operation_func, ttl: int = 300, **operation_kwargs
    ) -> Any:
        """
        Execute operation with caching.

        Args:
            operation_name: Name of the operation for cache key
            operation_func: Async function to execute
            ttl: Cache TTL in seconds
            **operation_kwargs: Arguments for the operation

        Returns:
            Operation result (from cache or fresh execution)
        """
        cache = await self._get_query_cache()
        if cache is None:
            # No cache available, execute directly
            return await operation_func(**operation_kwargs)

        # Try to get from cache first
        cached_result = await cache.get(operation_name, **operation_kwargs)
        if cached_result is not None:
            return cached_result

        # Execute operation and cache result
        try:
            result = await operation_func(**operation_kwargs)
            await cache.set(operation_name, result, ttl=ttl, **operation_kwargs)
            return result
        except Exception as e:
            # Log error but don't fail the operation
            self.logger.error(f"Operation {operation_name} failed: {str(e)}")
            raise

    async def invalidate_cache(self, operation_name: str, **operation_kwargs):
        """Invalidate cache for specific operation."""
        cache = await self._get_query_cache()
        if cache:
            await cache.invalidate(operation_name, **operation_kwargs)

    async def invalidate_cache_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        cache = await self._get_query_cache()
        if cache:
            return await cache.invalidate_pattern(pattern)
        return 0
