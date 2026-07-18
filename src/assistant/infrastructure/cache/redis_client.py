"""Redis client for caching and rate limiting operations."""

import asyncio
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from assistant.config import get_settings
from assistant.core.exceptions import CacheException, ErrorCode
from assistant.core.logging import get_logger


class RedisClient:
    """Async Redis client with connection pooling and error handling."""

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis client with connection pool.

        Args:
            redis_url: Redis connection URL. If None, uses settings.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.settings = get_settings()

        # Use provided URL or get from settings/environment
        self.redis_url = redis_url or getattr(
            self.settings, "redis_url", "redis://localhost:6379/0"
        )

        # Connection pool for efficient connection management
        self.pool: Optional[ConnectionPool] = None
        self.redis: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish Redis connection with pool."""
        async with self._lock:
            if self.redis is not None:
                return

            try:
                # Create connection pool
                self.pool = ConnectionPool.from_url(
                    self.redis_url, max_connections=20, retry_on_timeout=True, decode_responses=True
                )

                # Create Redis client
                self.redis = redis.Redis(connection_pool=self.pool)

                # Test connection
                await self.redis.ping()
                self.logger.info(f"Connected to Redis at {self.redis_url}")

            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {str(e)}")
                raise CacheException(
                    f"Redis connection failed: {str(e)}",
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    details={"redis_url": self.redis_url},
                    cause=e,
                )

    async def disconnect(self) -> None:
        """Close Redis connection."""
        async with self._lock:
            if self.redis:
                await self.redis.close()
                self.redis = None

            if self.pool:
                await self.pool.disconnect()
                self.pool = None

            self.logger.info("Disconnected from Redis")

    async def _ensure_connected(self) -> None:
        """Ensure Redis connection is established."""
        if self.redis is None:
            await self.connect()

    async def get(self, key: str) -> Optional[str]:
        """
        Get value by key.

        Args:
            key: Cache key

        Returns:
            Value or None if not found
        """
        try:
            await self._ensure_connected()
            value = await self.redis.get(key)
            return value
        except Exception as e:
            self.logger.error(f"Failed to get key '{key}': {str(e)}")
            raise CacheException(
                f"Cache get operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key},
                cause=e,
            )

    async def set(self, key: str, value: str, expire_seconds: Optional[int] = None) -> bool:
        """
        Set key-value pair with optional expiration.

        Args:
            key: Cache key
            value: Value to store
            expire_seconds: Optional expiration in seconds

        Returns:
            True if successful
        """
        try:
            await self._ensure_connected()
            result = await self.redis.set(key, value, ex=expire_seconds)
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to set key '{key}': {str(e)}")
            raise CacheException(
                f"Cache set operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "expire_seconds": expire_seconds},
                cause=e,
            )

    async def delete(self, key: str) -> bool:
        """
        Delete key.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        try:
            await self._ensure_connected()
            result = await self.redis.delete(key)
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to delete key '{key}': {str(e)}")
            raise CacheException(
                f"Cache delete operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key},
                cause=e,
            )

    async def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment key value.

        Args:
            key: Cache key
            amount: Increment amount

        Returns:
            New value after increment
        """
        try:
            await self._ensure_connected()
            result = await self.redis.incrby(key, amount)
            return result
        except Exception as e:
            self.logger.error(f"Failed to increment key '{key}': {str(e)}")
            raise CacheException(
                f"Cache increment operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "amount": amount},
                cause=e,
            )

    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration for key.

        Args:
            key: Cache key
            seconds: Expiration in seconds

        Returns:
            True if expiration was set
        """
        try:
            await self._ensure_connected()
            result = await self.redis.expire(key, seconds)
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to set expiration for key '{key}': {str(e)}")
            raise CacheException(
                f"Cache expire operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "seconds": seconds},
                cause=e,
            )

    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """
        Add members to sorted set.

        Args:
            key: Sorted set key
            mapping: Dict of member->score pairs

        Returns:
            Number of elements added
        """
        try:
            await self._ensure_connected()
            result = await self.redis.zadd(key, mapping)
            return result
        except Exception as e:
            self.logger.error(f"Failed to add to sorted set '{key}': {str(e)}")
            raise CacheException(
                f"Cache zadd operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "mapping_size": len(mapping)},
                cause=e,
            )

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """
        Remove members from sorted set by score range.

        Args:
            key: Sorted set key
            min_score: Minimum score
            max_score: Maximum score

        Returns:
            Number of elements removed
        """
        try:
            await self._ensure_connected()
            result = await self.redis.zremrangebyscore(key, min_score, max_score)
            return result
        except Exception as e:
            self.logger.error(f"Failed to remove from sorted set '{key}': {str(e)}")
            raise CacheException(
                f"Cache zremrangebyscore operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "min_score": min_score, "max_score": max_score},
                cause=e,
            )

    async def zcard(self, key: str) -> int:
        """
        Get sorted set size.

        Args:
            key: Sorted set key

        Returns:
            Number of elements in set
        """
        try:
            await self._ensure_connected()
            result = await self.redis.zcard(key)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get sorted set size '{key}': {str(e)}")
            raise CacheException(
                f"Cache zcard operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key},
                cause=e,
            )

    async def bzpopmax(self, key: str, timeout: float = 0) -> Optional[tuple]:
        """
        Blocking pop max element from sorted set.

        Args:
            key: Sorted set key
            timeout: Timeout in seconds (0 for no timeout)

        Returns:
            Tuple of (key, member, score) or None if timeout
        """
        try:
            await self._ensure_connected()
            result = await self.redis.bzpopmax(key, timeout=timeout)
            return result
        except Exception as e:
            self.logger.error(f"Failed to bzpopmax from sorted set '{key}': {str(e)}")
            raise CacheException(
                f"Cache bzpopmax operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "timeout": timeout},
                cause=e,
            )

    async def zrangebyscore(
        self, key: str, min_score: float, max_score: float, withscores: bool = False
    ) -> List:
        """
        Get members from sorted set by score range.

        Args:
            key: Sorted set key
            min_score: Minimum score
            max_score: Maximum score
            withscores: Whether to include scores in result

        Returns:
            List of members (and scores if withscores=True)
        """
        try:
            await self._ensure_connected()
            result = await self.redis.zrangebyscore(
                key, min_score, max_score, withscores=withscores
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to zrangebyscore from sorted set '{key}': {str(e)}")
            raise CacheException(
                f"Cache zrangebyscore operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "min_score": min_score, "max_score": max_score},
                cause=e,
            )

    async def zrem(self, key: str, *members) -> int:
        """
        Remove members from sorted set.

        Args:
            key: Sorted set key
            *members: Members to remove

        Returns:
            Number of members removed
        """
        try:
            await self._ensure_connected()
            result = await self.redis.zrem(key, *members)
            return result
        except Exception as e:
            self.logger.error(f"Failed to zrem from sorted set '{key}': {str(e)}")
            raise CacheException(
                f"Cache zrem operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "members_count": len(members)},
                cause=e,
            )

    async def hset(self, key: str, field: str, value: str) -> int:
        """
        Set field in hash.

        Args:
            key: Hash key
            field: Hash field
            value: Value to set

        Returns:
            Number of fields that were added
        """
        try:
            await self._ensure_connected()
            result = await self.redis.hset(key, field, value)
            return result
        except Exception as e:
            self.logger.error(f"Failed to hset in hash '{key}': {str(e)}")
            raise CacheException(
                f"Cache hset operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "field": field},
                cause=e,
            )

    async def hget(self, key: str, field: str) -> Optional[str]:
        """
        Get field from hash.

        Args:
            key: Hash key
            field: Hash field

        Returns:
            Field value or None if not found
        """
        try:
            await self._ensure_connected()
            result = await self.redis.hget(key, field)
            return result
        except Exception as e:
            self.logger.error(f"Failed to hget from hash '{key}': {str(e)}")
            raise CacheException(
                f"Cache hget operation failed: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"key": key, "field": field},
                cause=e,
            )

    async def pipeline(self):
        """Get Redis pipeline for batch operations."""
        await self._ensure_connected()
        return self.redis.pipeline()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform Redis health check.

        Returns:
            Health check results
        """
        try:
            await self._ensure_connected()

            # Test basic operations
            test_key = "health_check"
            await self.redis.set(test_key, "ok", ex=5)
            value = await self.redis.get(test_key)
            await self.redis.delete(test_key)

            # Get Redis info
            info = await self.redis.info()

            return {
                "status": "healthy",
                "ping_successful": True,
                "basic_operations": value == "ok",
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            self.logger.error(f"Redis health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e), "ping_successful": False}


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """
    Get global Redis client instance.

    Returns:
        Redis client instance
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.connect()

    return _redis_client


async def cleanup_redis_client() -> None:
    """Clean up global Redis client."""
    global _redis_client

    if _redis_client:
        await _redis_client.disconnect()
        _redis_client = None
