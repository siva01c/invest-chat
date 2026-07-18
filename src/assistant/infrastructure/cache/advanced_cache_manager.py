"""Advanced multi-tier caching system with intelligent cache management."""

import asyncio
import functools
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from assistant.config import get_settings
from assistant.core.logging import get_logger
from assistant.infrastructure.cache.redis_client import get_redis_client

T = TypeVar("T")


class CacheStrategy(Enum):
    """Cache strategy enumeration."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"


class CacheLevel(Enum):
    """Cache level enumeration for multi-tier caching."""

    L1_MEMORY = "l1_memory"  # In-memory cache (fastest)
    L2_REDIS = "l2_redis"  # Redis cache (fast, persistent)
    L3_DATABASE = "l3_database"  # Database cache (slowest, authoritative)


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    size_bytes: int = 0
    cache_level: CacheLevel = CacheLevel.L1_MEMORY
    compression_enabled: bool = False
    serialization_format: str = "json"


@dataclass
class CacheStats:
    """Comprehensive cache statistics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    average_response_time_ms: float = 0.0
    hit_rate: float = 0.0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    last_reset: datetime = field(default_factory=datetime.utcnow)


class AdvancedCacheManager(Generic[T]):
    """
    Advanced multi-tier cache manager with intelligent caching strategies.

    Features:
    - Multi-tier caching (L1: Memory, L2: Redis, L3: Database)
    - Multiple eviction strategies (LRU, LFU, TTL)
    - Cache warming and preloading
    - Intelligent cache invalidation
    - Compression and serialization
    - Cache tagging and bulk operations
    - Performance monitoring and metrics
    """

    def __init__(
        self,
        name: str,
        max_memory_size_mb: int = 100,
        default_ttl_seconds: int = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
        enable_compression: bool = False,
        enable_metrics: bool = True,
    ):
        """
        Initialize the advanced cache manager.

        Args:
            name: Cache instance name
            max_memory_size_mb: Maximum memory cache size in MB
            default_ttl_seconds: Default TTL for cache entries
            strategy: Cache eviction strategy
            enable_compression: Enable data compression
            enable_metrics: Enable performance metrics
        """
        self.settings = get_settings()
        self.logger = get_logger(f"{self.__class__.__name__}[{name}]")

        self.name = name
        self.max_memory_size_bytes = max_memory_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.strategy = strategy
        self.enable_compression = enable_compression
        self.enable_metrics = enable_metrics

        # Multi-tier cache storage
        self._l1_cache: Dict[str, CacheEntry] = {}  # In-memory cache
        self._l2_redis_client = None  # Redis client (lazy loaded)

        # Cache management
        self._cache_lock = asyncio.Lock()
        self._stats = CacheStats()

        # Eviction tracking
        self._access_order: List[str] = []  # For LRU
        self._access_frequency: Dict[str, int] = defaultdict(int)  # For LFU

        # Cache warming
        self._warming_tasks: Dict[str, asyncio.Task] = {}

        # Invalidation tracking
        self._tag_to_keys: Dict[str, set] = defaultdict(set)
        self._key_to_tags: Dict[str, set] = defaultdict(set)

    async def _get_redis_client(self):
        """Get Redis client with lazy initialization."""
        if self._l2_redis_client is None:
            try:
                self._l2_redis_client = await get_redis_client()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis client: {str(e)}")
        return self._l2_redis_client

    def _generate_cache_key(self, key: Union[str, Dict[str, Any]]) -> str:
        """Generate a standardized cache key."""
        if isinstance(key, str):
            return f"{self.name}:{key}"
        elif isinstance(key, dict):
            # Create deterministic key from dict
            sorted_items = sorted(key.items())
            key_str = json.dumps(sorted_items, sort_keys=True)
            key_hash = hashlib.sha256(key_str.encode()).hexdigest()
            return f"{self.name}:{key_hash}"
        else:
            key_str = str(key)
            return f"{self.name}:{key_str}"

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value).encode("utf-8"))
            elif isinstance(value, bytes):
                return len(value)
            else:
                return len(str(value).encode("utf-8"))
        except Exception:
            return 100  # Default estimate

    def _compress_data(self, data: Any) -> bytes:
        """Compress data if compression is enabled."""
        if not self.enable_compression:
            return json.dumps(data).encode("utf-8")

        try:
            import gzip

            json_data = json.dumps(data).encode("utf-8")
            return gzip.compress(json_data)
        except Exception as e:
            self.logger.warning(f"Compression failed: {str(e)}")
            return json.dumps(data).encode("utf-8")

    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data if compression was used."""
        if not self.enable_compression:
            return json.loads(data.decode("utf-8"))

        try:
            import gzip

            decompressed = gzip.decompress(data)
            return json.loads(decompressed.decode("utf-8"))
        except Exception as e:
            self.logger.warning(f"Decompression failed, trying direct decode: {str(e)}")
            return json.loads(data.decode("utf-8"))

    def _should_evict_memory_cache(self) -> bool:
        """Check if memory cache needs eviction."""
        current_size = sum(entry.size_bytes for entry in self._l1_cache.values())
        return current_size > self.max_memory_size_bytes

    async def _evict_from_memory_cache(self):
        """Evict entries from memory cache based on strategy."""
        if not self._l1_cache:
            return

        key_to_evict = None

        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self._access_order:
                key_to_evict = self._access_order[0]
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_freq = min(self._access_frequency.values()) if self._access_frequency else 0
            for key, freq in self._access_frequency.items():
                if freq == min_freq and key in self._l1_cache:
                    key_to_evict = key
                    break
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first
            now = datetime.utcnow()
            for key, entry in self._l1_cache.items():
                if (
                    entry.ttl_seconds
                    and (now - entry.created_at).total_seconds() > entry.ttl_seconds
                ):
                    key_to_evict = key
                    break

        if key_to_evict:
            await self._remove_from_l1_cache(key_to_evict)
            self._stats.evictions += 1

    async def _remove_from_l1_cache(self, key: str):
        """Remove entry from L1 cache and update tracking."""
        if key in self._l1_cache:
            entry = self._l1_cache[key]

            # Remove from tracking structures
            if key in self._access_order:
                self._access_order.remove(key)
            if key in self._access_frequency:
                del self._access_frequency[key]

            # Remove tags
            for tag in entry.tags:
                self._tag_to_keys[tag].discard(key)
                if not self._tag_to_keys[tag]:
                    del self._tag_to_keys[tag]

            if key in self._key_to_tags:
                del self._key_to_tags[key]

            # Remove from cache
            del self._l1_cache[key]

    def _update_access_tracking(self, key: str):
        """Update access tracking for eviction strategies."""
        # Update LRU tracking
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        # Update LFU tracking
        self._access_frequency[key] += 1

    async def get(
        self,
        key: Union[str, Dict[str, Any]],
        default: Optional[T] = None,
        cache_levels: List[CacheLevel] = None,
    ) -> Optional[T]:
        """
        Get value from cache with multi-tier lookup.

        Args:
            key: Cache key
            default: Default value if not found
            cache_levels: Cache levels to search (defaults to all)

        Returns:
            Cached value or default
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

        cache_key = self._generate_cache_key(key)
        start_time = time.time()

        try:
            async with self._cache_lock:
                self._stats.total_requests += 1

                # Try L1 memory cache first
                if CacheLevel.L1_MEMORY in cache_levels and cache_key in self._l1_cache:
                    entry = self._l1_cache[cache_key]

                    # Check TTL
                    if entry.ttl_seconds:
                        age_seconds = (datetime.utcnow() - entry.created_at).total_seconds()
                        if age_seconds > entry.ttl_seconds:
                            await self._remove_from_l1_cache(cache_key)
                        else:
                            # Valid L1 hit
                            entry.last_accessed = datetime.utcnow()
                            entry.access_count += 1
                            self._update_access_tracking(cache_key)

                            self._stats.cache_hits += 1
                            self._stats.l1_hits += 1

                            response_time = (time.time() - start_time) * 1000
                            self._update_response_time(response_time)

                            return entry.value

                # Try L2 Redis cache
                if CacheLevel.L2_REDIS in cache_levels:
                    redis_client = await self._get_redis_client()
                    if redis_client:
                        try:
                            redis_key = f"cache:{cache_key}"
                            cached_data = await redis_client.get(redis_key)

                            if cached_data:
                                # L2 hit - deserialize and promote to L1
                                value = self._decompress_data(cached_data)

                                # Promote to L1 cache
                                await self.set(key, value, promote_only=True)

                                self._stats.cache_hits += 1
                                self._stats.l2_hits += 1

                                response_time = (time.time() - start_time) * 1000
                                self._update_response_time(response_time)

                                return value

                        except Exception as e:
                            self.logger.warning(f"L2 cache lookup failed: {str(e)}")

                # Cache miss
                self._stats.cache_misses += 1
                return default

        except Exception as e:
            self.logger.error(f"Cache get operation failed: {str(e)}")
            return default

    async def set(
        self,
        key: Union[str, Dict[str, Any]],
        value: T,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        cache_levels: List[CacheLevel] = None,
        promote_only: bool = False,
    ) -> bool:
        """
        Set value in cache with multi-tier storage.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            tags: Tags for cache invalidation
            cache_levels: Cache levels to store in
            promote_only: Only promote to L1, don't store in L2

        Returns:
            True if successful
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

        cache_key = self._generate_cache_key(key)
        ttl = ttl_seconds or self.default_ttl_seconds
        tags = tags or []

        try:
            async with self._cache_lock:
                # Store in L1 memory cache
                if CacheLevel.L1_MEMORY in cache_levels:
                    # Check if eviction is needed
                    while self._should_evict_memory_cache():
                        await self._evict_from_memory_cache()

                    # Create cache entry
                    entry = CacheEntry(
                        key=cache_key,
                        value=value,
                        created_at=datetime.utcnow(),
                        last_accessed=datetime.utcnow(),
                        ttl_seconds=ttl,
                        tags=tags,
                        size_bytes=self._calculate_size(value),
                        cache_level=CacheLevel.L1_MEMORY,
                        compression_enabled=self.enable_compression,
                    )

                    # Remove existing entry if present
                    if cache_key in self._l1_cache:
                        await self._remove_from_l1_cache(cache_key)

                    # Store new entry
                    self._l1_cache[cache_key] = entry
                    self._update_access_tracking(cache_key)

                    # Update tag tracking
                    for tag in tags:
                        self._tag_to_keys[tag].add(cache_key)
                        self._key_to_tags[cache_key].add(tag)

                # Store in L2 Redis cache (if not promotion only)
                if CacheLevel.L2_REDIS in cache_levels and not promote_only:
                    redis_client = await self._get_redis_client()
                    if redis_client:
                        try:
                            redis_key = f"cache:{cache_key}"
                            compressed_data = self._compress_data(value)

                            await redis_client.setex(redis_key, ttl, compressed_data)

                            # Store tags in Redis for invalidation
                            if tags:
                                for tag in tags:
                                    tag_key = f"tag:{tag}"
                                    await redis_client.sadd(tag_key, cache_key)
                                    await redis_client.expire(tag_key, ttl)

                        except Exception as e:
                            self.logger.warning(f"L2 cache storage failed: {str(e)}")

                return True

        except Exception as e:
            self.logger.error(f"Cache set operation failed: {str(e)}")
            return False

    async def delete(self, key: Union[str, Dict[str, Any]]) -> bool:
        """Delete entry from all cache levels."""
        cache_key = self._generate_cache_key(key)

        try:
            async with self._cache_lock:
                # Remove from L1
                if cache_key in self._l1_cache:
                    await self._remove_from_l1_cache(cache_key)

                # Remove from L2
                redis_client = await self._get_redis_client()
                if redis_client:
                    redis_key = f"cache:{cache_key}"
                    await redis_client.delete(redis_key)

                return True

        except Exception as e:
            self.logger.error(f"Cache delete operation failed: {str(e)}")
            return False

    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """
        Invalidate cache entries by tags.

        Args:
            tags: List of tags to invalidate

        Returns:
            Number of entries invalidated
        """
        invalidated_count = 0

        try:
            async with self._cache_lock:
                keys_to_invalidate = set()

                # Collect keys from L1 cache
                for tag in tags:
                    if tag in self._tag_to_keys:
                        keys_to_invalidate.update(self._tag_to_keys[tag])

                # Remove from L1
                for cache_key in keys_to_invalidate:
                    if cache_key in self._l1_cache:
                        await self._remove_from_l1_cache(cache_key)
                        invalidated_count += 1

                # Remove from L2 Redis
                redis_client = await self._get_redis_client()
                if redis_client:
                    for tag in tags:
                        tag_key = f"tag:{tag}"
                        redis_keys = await redis_client.smembers(tag_key)

                        if redis_keys:
                            # Delete the cached entries
                            cache_keys = [f"cache:{key}" for key in redis_keys]
                            await redis_client.delete(*cache_keys)

                            # Delete the tag set
                            await redis_client.delete(tag_key)

                self.logger.info(f"Invalidated {invalidated_count} cache entries by tags: {tags}")
                return invalidated_count

        except Exception as e:
            self.logger.error(f"Tag-based invalidation failed: {str(e)}")
            return 0

    async def warm_cache(
        self, warm_function: Callable, keys: List[Union[str, Dict[str, Any]]], batch_size: int = 10
    ):
        """
        Warm cache by preloading data.

        Args:
            warm_function: Function to call to get data for keys
            keys: Keys to warm
            batch_size: Number of keys to process in parallel
        """
        try:
            self.logger.info(f"Starting cache warming for {len(keys)} keys")

            # Process in batches
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i : i + batch_size]
                tasks = []

                for key in batch_keys:
                    task = asyncio.create_task(self._warm_single_key(warm_function, key))
                    tasks.append(task)

                # Wait for batch to complete
                await asyncio.gather(*tasks, return_exceptions=True)

            self.logger.info(f"Cache warming completed for {len(keys)} keys")

        except Exception as e:
            self.logger.error(f"Cache warming failed: {str(e)}")

    async def _warm_single_key(self, warm_function: Callable, key: Union[str, Dict[str, Any]]):
        """Warm a single cache key."""
        try:
            # Check if already cached
            cached_value = await self.get(key)
            if cached_value is not None:
                return

            # Call warm function to get data
            if asyncio.iscoroutinefunction(warm_function):
                value = await warm_function(key)
            else:
                value = warm_function(key)

            # Cache the value
            if value is not None:
                await self.set(key, value)

        except Exception as e:
            self.logger.warning(f"Failed to warm cache key {key}: {str(e)}")

    def _update_response_time(self, response_time_ms: float):
        """Update average response time metric."""
        if self._stats.total_requests > 1:
            self._stats.average_response_time_ms = (
                self._stats.average_response_time_ms * (self._stats.total_requests - 1)
                + response_time_ms
            ) / self._stats.total_requests
        else:
            self._stats.average_response_time_ms = response_time_ms

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            # Calculate hit rate
            total_requests = self._stats.cache_hits + self._stats.cache_misses
            hit_rate = self._stats.cache_hits / total_requests if total_requests > 0 else 0

            # Calculate memory usage
            memory_usage = sum(entry.size_bytes for entry in self._l1_cache.values())

            # Get Redis stats
            redis_stats = {}
            redis_client = await self._get_redis_client()
            if redis_client:
                try:
                    redis_info = await redis_client.get_info()
                    redis_stats = {
                        "memory_usage_mb": redis_info.get("used_memory", 0) / (1024 * 1024),
                        "connected_clients": redis_info.get("connected_clients", 0),
                        "total_commands": redis_info.get("total_commands_processed", 0),
                    }
                except Exception:
                    pass

            return {
                "cache_name": self.name,
                "strategy": self.strategy.value,
                "l1_cache_size": len(self._l1_cache),
                "l1_memory_usage_bytes": memory_usage,
                "l1_memory_usage_mb": memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_size_bytes / (1024 * 1024),
                "total_requests": total_requests,
                "cache_hits": self._stats.cache_hits,
                "cache_misses": self._stats.cache_misses,
                "hit_rate": hit_rate,
                "l1_hits": self._stats.l1_hits,
                "l2_hits": self._stats.l2_hits,
                "evictions": self._stats.evictions,
                "average_response_time_ms": self._stats.average_response_time_ms,
                "compression_enabled": self.enable_compression,
                "redis_stats": redis_stats,
                "uptime_seconds": (datetime.utcnow() - self._stats.last_reset).total_seconds(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {str(e)}")
            return {"error": str(e)}

    async def clear_cache(self, cache_levels: List[CacheLevel] = None) -> bool:
        """Clear cache at specified levels."""
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

        try:
            async with self._cache_lock:
                # Clear L1
                if CacheLevel.L1_MEMORY in cache_levels:
                    self._l1_cache.clear()
                    self._access_order.clear()
                    self._access_frequency.clear()
                    self._tag_to_keys.clear()
                    self._key_to_tags.clear()

                # Clear L2
                if CacheLevel.L2_REDIS in cache_levels:
                    redis_client = await self._get_redis_client()
                    if redis_client:
                        # Delete all keys with our cache prefix
                        pattern = f"cache:{self.name}:*"
                        keys = await redis_client.keys(pattern)
                        if keys:
                            await redis_client.delete(*keys)

                        # Delete all tag keys
                        tag_pattern = f"tag:*"
                        tag_keys = await redis_client.keys(tag_pattern)
                        if tag_keys:
                            await redis_client.delete(*tag_keys)

                # Reset stats
                self._stats = CacheStats()

                self.logger.info(
                    f"Cache cleared for levels: {[level.value for level in cache_levels]}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Cache clear operation failed: {str(e)}")
            return False


# Decorator for automatic caching
def cached(
    cache_name: str = "default",
    ttl_seconds: int = 3600,
    tags: Optional[List[str]] = None,
    key_generator: Optional[Callable] = None,
):
    """
    Decorator for automatic function result caching.

    Args:
        cache_name: Name of cache instance to use
        ttl_seconds: Time to live for cached results
        tags: Tags for cache invalidation
        key_generator: Custom function to generate cache keys
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create cache manager
            if not hasattr(wrapper, "_cache_manager"):
                wrapper._cache_manager = AdvancedCacheManager(cache_name)

            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_result = await wrapper._cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Cache the result
            await wrapper._cache_manager.set(
                cache_key, result, ttl_seconds=ttl_seconds, tags=tags or []
            )

            return result

        return wrapper

    return decorator


# Global cache managers
_cache_managers: Dict[str, AdvancedCacheManager] = {}


async def get_cache_manager(name: str, **kwargs) -> AdvancedCacheManager:
    """Get or create a cache manager instance."""
    if name not in _cache_managers:
        _cache_managers[name] = AdvancedCacheManager(name, **kwargs)
    return _cache_managers[name]


async def clear_all_caches() -> Dict[str, bool]:
    """Clear all cache manager instances."""
    results = {}
    for name, manager in _cache_managers.items():
        results[name] = await manager.clear_cache()
    return results
