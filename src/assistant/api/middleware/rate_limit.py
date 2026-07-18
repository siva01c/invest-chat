"""Advanced rate limiting middleware with Redis backend."""

import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from assistant.config import get_settings
from assistant.core.logging import get_logger
from assistant.infrastructure.cache.redis_client import get_redis_client
from assistant.infrastructure.cache.redis_rate_limiter import (
    RateLimitConfig,
    RateLimitConfigs,
    RedisRateLimiter,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with Redis backend and configurable rules."""

    def __init__(self, app, enable_rate_limiting: bool = True):
        """
        Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            enable_rate_limiting: Whether to enable rate limiting
        """
        super().__init__(app)
        self.logger = get_logger(self.__class__.__name__)
        self.settings = get_settings()
        self.enable_rate_limiting = enable_rate_limiting
        self._rate_limiter: Optional[RedisRateLimiter] = None

        # Rate limiting rules for different endpoints
        self.rate_limit_rules = {
            # API endpoints
            "POST:/": RateLimitConfigs.API_DEFAULT,
            "GET:/health": RateLimitConfig(
                requests_per_window=120, window_seconds=60
            ),  # More lenient for health checks
            "POST:/chat": RateLimitConfigs.API_DEFAULT,
            # Auth endpoints (if any)
            "POST:/auth/login": RateLimitConfigs.AUTH_LOGIN,
            "POST:/auth/reset": RateLimitConfigs.AUTH_RESET,
            # Admin endpoints (stricter)
            "POST:/admin": RateLimitConfigs.API_STRICT,
            "PUT:/admin": RateLimitConfigs.API_STRICT,
            "DELETE:/admin": RateLimitConfigs.API_STRICT,
        }

    async def get_rate_limiter(self) -> Optional[RedisRateLimiter]:
        """Get or create rate limiter instance."""
        if not self.enable_rate_limiting:
            return None

        if self._rate_limiter is None:
            try:
                redis_client = await get_redis_client()
                self._rate_limiter = RedisRateLimiter(redis_client, "api_rate_limit")
                self.logger.info("Rate limiter initialized with Redis backend")
            except Exception as e:
                self.logger.error(f"Failed to initialize Redis rate limiter: {str(e)}")
                # Continue without rate limiting if Redis is unavailable
                return None

        return self._rate_limiter

    def get_client_identifier(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.

        Args:
            request: HTTP request

        Returns:
            Client identifier (IP address or user ID)
        """
        # Check for forwarded IP first (if behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Check for authorization header to use user-based limiting
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Could decode JWT here to get user ID for user-based limiting
            # For now, fall back to IP-based limiting
            pass

        # Fall back to direct connection IP
        return request.client.host if request.client else "unknown"

    def get_rate_limit_config(self, request: Request) -> Optional[RateLimitConfig]:
        """
        Get rate limit configuration for the request.

        Args:
            request: HTTP request

        Returns:
            Rate limit configuration or None if no limit applies
        """
        method = request.method
        path = request.url.path

        # Exact match first
        key = f"{method}:{path}"
        if key in self.rate_limit_rules:
            return self.rate_limit_rules[key]

        # Pattern matching for dynamic routes
        if path.startswith("/admin/"):
            return self.rate_limit_rules.get("POST:/admin", RateLimitConfigs.API_STRICT)

        # Default for API endpoints
        if method == "POST" and path in ["/", "/chat"]:
            return self.rate_limit_rules.get("POST:/")

        # No rate limit for other endpoints
        return None

    async def check_rate_limit(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Check rate limit for request.

        Args:
            request: HTTP request

        Returns:
            Rate limit result or None if rate limiting is disabled
        """
        rate_limiter = await self.get_rate_limiter()
        if not rate_limiter:
            return None

        config = self.get_rate_limit_config(request)
        if not config:
            return None

        client_id = self.get_client_identifier(request)
        rate_type = f"{request.method.lower()}_{request.url.path.replace('/', '_').strip('_')}"

        try:
            result = await rate_limiter.check_rate_limit(client_id, config, rate_type)

            return {
                "allowed": result.allowed,
                "current_requests": result.current_requests,
                "limit": result.limit,
                "window_seconds": result.window_seconds,
                "reset_time": result.reset_time,
                "retry_after": result.retry_after,
                "client_id": client_id,
                "rate_type": rate_type,
            }

        except Exception as e:
            self.logger.error(f"Rate limit check failed: {str(e)}")
            # Fail open - allow request if rate limiting fails
            return {"allowed": True, "error": str(e)}

    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting checks."""
        start_time = time.time()

        # Check rate limit
        rate_limit_result = await self.check_rate_limit(request)

        if rate_limit_result and not rate_limit_result["allowed"]:
            # Rate limit exceeded
            retry_after = rate_limit_result.get("retry_after", 60)

            self.logger.warning(
                f"Rate limit exceeded for {rate_limit_result['client_id']} "
                f"on {request.method} {request.url.path}: "
                f"{rate_limit_result['current_requests']}/{rate_limit_result['limit']}"
            )

            # Return rate limit error
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {rate_limit_result['limit']} requests per {rate_limit_result['window_seconds']} seconds",
                    "current_requests": rate_limit_result["current_requests"],
                    "limit": rate_limit_result["limit"],
                    "window_seconds": rate_limit_result["window_seconds"],
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(rate_limit_result["limit"]),
                    "X-RateLimit-Remaining": str(
                        max(0, rate_limit_result["limit"] - rate_limit_result["current_requests"])
                    ),
                    "X-RateLimit-Reset": str(int(rate_limit_result["reset_time"])),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limiting headers to successful responses
        if rate_limit_result:
            response.headers["X-RateLimit-Limit"] = str(rate_limit_result.get("limit", 0))
            response.headers["X-RateLimit-Remaining"] = str(
                max(
                    0,
                    rate_limit_result.get("limit", 0)
                    - rate_limit_result.get("current_requests", 0),
                )
            )
            response.headers["X-RateLimit-Reset"] = str(
                int(rate_limit_result.get("reset_time", time.time()))
            )

        # Log request timing
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        return response


class EmailRateLimitMixin:
    """Mixin for email rate limiting functionality."""

    async def check_email_rate_limit(self, session_id: str) -> bool:
        """
        Check email rate limit for session.

        Args:
            session_id: Session identifier

        Returns:
            True if email is allowed, False if rate limited
        """
        try:
            rate_limiter = await get_redis_client()
            if not rate_limiter:
                return True  # Fail open

            redis_rate_limiter = RedisRateLimiter(rate_limiter, "email_rate_limit")
            result = await redis_rate_limiter.check_rate_limit(
                session_id, RateLimitConfigs.EMAIL_DEFAULT, "email"
            )

            if not result.allowed:
                self.logger.warning(f"Email rate limit exceeded for session {session_id}")

            return result.allowed

        except Exception as e:
            self.logger.error(f"Email rate limit check failed: {str(e)}")
            return True  # Fail open
