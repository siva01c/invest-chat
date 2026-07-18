"""Redis-based rate limiter with sliding window algorithm."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from assistant.core.exceptions import ErrorCode, RateLimitException
from assistant.core.logging import get_logger, log_service_method
from assistant.infrastructure.cache.redis_client import RedisClient


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_window: int
    window_seconds: int
    burst_requests: Optional[int] = None  # Allow burst beyond normal limit

    @property
    def burst_limit(self) -> int:
        """Get effective burst limit."""
        return self.burst_requests or self.requests_per_window


@dataclass
class RateLimitResult:
    """Rate limiting check result."""

    allowed: bool
    current_requests: int
    limit: int
    window_seconds: int
    reset_time: float
    retry_after: Optional[int] = None


class RedisRateLimiter:
    """Redis-based rate limiter using sliding window algorithm."""

    def __init__(self, redis_client: RedisClient, key_prefix: str = "rate_limit"):
        """
        Initialize rate limiter.

        Args:
            redis_client: Redis client instance
            key_prefix: Prefix for Redis keys
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.logger = get_logger(self.__class__.__name__)

    def _get_key(self, identifier: str, rate_type: str = "default") -> str:
        """
        Generate Redis key for rate limiting.

        Args:
            identifier: Client identifier (IP, user ID, etc.)
            rate_type: Type of rate limit (api, email, etc.)

        Returns:
            Redis key
        """
        return f"{self.key_prefix}:{rate_type}:{identifier}"

    @log_service_method()
    async def check_rate_limit(
        self, identifier: str, config: RateLimitConfig, rate_type: str = "default"
    ) -> RateLimitResult:
        """
        Check if request is within rate limit using sliding window.

        Args:
            identifier: Client identifier
            config: Rate limiting configuration
            rate_type: Type of rate limit

        Returns:
            Rate limit check result
        """
        try:
            current_time = time.time()
            window_start = current_time - config.window_seconds
            key = self._get_key(identifier, rate_type)

            # Use Redis pipeline for atomic operations
            pipeline = await self.redis.pipeline()

            # Remove expired entries (outside sliding window)
            pipeline.zremrangebyscore(key, 0, window_start)

            # Count current requests in window
            pipeline.zcard(key)

            # Add current request with timestamp as score
            request_id = f"{current_time}:{id(self)}"
            pipeline.zadd(key, {request_id: current_time})

            # Set expiration for cleanup
            pipeline.expire(key, config.window_seconds + 60)

            # Execute pipeline
            results = await pipeline.execute()

            # Get current request count (after cleanup, before adding new request)
            current_requests = results[1]  # zcard result

            # Determine if request is allowed
            allowed = current_requests < config.requests_per_window

            # Calculate reset time
            reset_time = current_time + config.window_seconds

            # Calculate retry_after if rate limited
            retry_after = None
            if not allowed:
                # Find oldest request in window to determine when limit resets
                try:
                    oldest_requests = await self.redis.redis.zrange(key, 0, 0, withscores=True)
                    if oldest_requests:
                        oldest_time = oldest_requests[0][1]
                        retry_after = int(oldest_time + config.window_seconds - current_time) + 1
                except Exception:
                    retry_after = config.window_seconds

            result = RateLimitResult(
                allowed=allowed,
                current_requests=current_requests + 1,  # Include current request
                limit=config.requests_per_window,
                window_seconds=config.window_seconds,
                reset_time=reset_time,
                retry_after=retry_after,
            )

            # Log rate limit status
            if not allowed:
                self.logger.warning(
                    f"Rate limit exceeded for {identifier} ({rate_type}): "
                    f"{current_requests + 1}/{config.requests_per_window}"
                )
            else:
                self.logger.debug(
                    f"Rate limit check passed for {identifier} ({rate_type}): "
                    f"{current_requests + 1}/{config.requests_per_window}"
                )

            return result

        except Exception as e:
            self.logger.error(f"Rate limit check failed for {identifier}: {str(e)}")

            # Fail open - allow request if Redis is unavailable
            return RateLimitResult(
                allowed=True,
                current_requests=0,
                limit=config.requests_per_window,
                window_seconds=config.window_seconds,
                reset_time=time.time() + config.window_seconds,
                retry_after=None,
            )

    @log_service_method()
    async def check_and_enforce_rate_limit(
        self, identifier: str, config: RateLimitConfig, rate_type: str = "default"
    ) -> RateLimitResult:
        """
        Check rate limit and raise exception if exceeded.

        Args:
            identifier: Client identifier
            config: Rate limiting configuration
            rate_type: Type of rate limit

        Returns:
            Rate limit check result

        Raises:
            RateLimitException: If rate limit is exceeded
        """
        result = await self.check_rate_limit(identifier, config, rate_type)

        if not result.allowed:
            raise RateLimitException(
                f"Rate limit exceeded: {result.current_requests}/{result.limit} "
                f"requests in {result.window_seconds} seconds",
                rate_type=rate_type,
                identifier=identifier,
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                details={
                    "current_requests": result.current_requests,
                    "limit": result.limit,
                    "window_seconds": result.window_seconds,
                    "retry_after": result.retry_after,
                },
            )

        return result

    @log_service_method()
    async def reset_rate_limit(self, identifier: str, rate_type: str = "default") -> bool:
        """
        Reset rate limit for identifier.

        Args:
            identifier: Client identifier
            rate_type: Type of rate limit

        Returns:
            True if reset successfully
        """
        try:
            key = self._get_key(identifier, rate_type)
            result = await self.redis.delete(key)

            self.logger.info(f"Rate limit reset for {identifier} ({rate_type})")
            return bool(result)

        except Exception as e:
            self.logger.error(f"Failed to reset rate limit for {identifier}: {str(e)}")
            return False

    @log_service_method()
    async def get_rate_limit_status(
        self, identifier: str, config: RateLimitConfig, rate_type: str = "default"
    ) -> Dict[str, Any]:
        """
        Get current rate limit status without affecting the limit.

        Args:
            identifier: Client identifier
            config: Rate limiting configuration
            rate_type: Type of rate limit

        Returns:
            Rate limit status information
        """
        try:
            current_time = time.time()
            window_start = current_time - config.window_seconds
            key = self._get_key(identifier, rate_type)

            # Clean expired entries and count current requests
            pipeline = await self.redis.pipeline()
            pipeline.zremrangebyscore(key, 0, window_start)
            pipeline.zcard(key)
            results = await pipeline.execute()

            current_requests = results[1]
            remaining_requests = max(0, config.requests_per_window - current_requests)

            return {
                "identifier": identifier,
                "rate_type": rate_type,
                "current_requests": current_requests,
                "limit": config.requests_per_window,
                "remaining_requests": remaining_requests,
                "window_seconds": config.window_seconds,
                "reset_time": current_time + config.window_seconds,
                "is_limited": current_requests >= config.requests_per_window,
            }

        except Exception as e:
            self.logger.error(f"Failed to get rate limit status for {identifier}: {str(e)}")
            return {
                "identifier": identifier,
                "rate_type": rate_type,
                "error": str(e),
                "current_requests": 0,
                "limit": config.requests_per_window,
                "remaining_requests": config.requests_per_window,
                "window_seconds": config.window_seconds,
                "reset_time": time.time() + config.window_seconds,
                "is_limited": False,
            }

    @log_service_method()
    async def cleanup_expired_entries(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up expired rate limit entries across all keys.

        Args:
            max_age_seconds: Maximum age of entries to keep

        Returns:
            Number of keys cleaned up
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - max_age_seconds

            # Get all rate limit keys
            pattern = f"{self.key_prefix}:*"
            keys = []

            # Use scan for memory-efficient key iteration
            async for key in self.redis.redis.scan_iter(match=pattern):
                keys.append(key)

            cleaned_count = 0
            for key in keys:
                try:
                    # Remove old entries from sorted set
                    removed = await self.redis.zremrangebyscore(key, 0, cutoff_time)
                    if removed > 0:
                        cleaned_count += 1

                    # Remove empty keys
                    size = await self.redis.zcard(key)
                    if size == 0:
                        await self.redis.delete(key)

                except Exception as e:
                    self.logger.warning(f"Failed to clean key {key}: {str(e)}")

            self.logger.info(f"Cleaned up {cleaned_count} rate limit keys")
            return cleaned_count

        except Exception as e:
            self.logger.error(f"Rate limit cleanup failed: {str(e)}")
            return 0


# Pre-configured rate limit configurations
class RateLimitConfigs:
    """Standard rate limiting configurations."""

    # API endpoints
    API_DEFAULT = RateLimitConfig(requests_per_window=60, window_seconds=60)  # 1 per second
    API_STRICT = RateLimitConfig(requests_per_window=30, window_seconds=60)  # 0.5 per second
    API_BURST = RateLimitConfig(requests_per_window=100, window_seconds=60, burst_requests=150)

    # Email rate limiting
    EMAIL_DEFAULT = RateLimitConfig(requests_per_window=5, window_seconds=300)  # 5 per 5 minutes
    EMAIL_STRICT = RateLimitConfig(requests_per_window=2, window_seconds=600)  # 2 per 10 minutes

    # Authentication
    AUTH_LOGIN = RateLimitConfig(requests_per_window=5, window_seconds=300)  # 5 per 5 minutes
    AUTH_RESET = RateLimitConfig(requests_per_window=3, window_seconds=3600)  # 3 per hour
