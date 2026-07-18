"""Security middleware for input validation and sanitization."""

import html
import re
from typing import Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from assistant.config import get_settings
from assistant.core.logging import get_logger
from assistant.infrastructure.cache.redis_client import get_redis_client
from assistant.infrastructure.cache.redis_rate_limiter import RateLimitConfig, RedisRateLimiter

# Get settings instance
settings = get_settings()
logger = get_logger(__name__)

# Rate limiting configuration from settings
RATE_LIMIT_REQUESTS = settings.rate_limit_requests
RATE_LIMIT_WINDOW = settings.rate_limit_window
EMAIL_RATE_LIMIT = settings.email_rate_limit
MAX_MESSAGE_LENGTH = settings.max_message_length

# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


def sanitize_input(text: str) -> str:
    """Sanitize user input for security."""
    if not text:
        return ""

    # HTML escape and length limit using configuration
    sanitized = html.escape(text.strip())[:MAX_MESSAGE_LENGTH]

    # Remove potential script injections
    sanitized = re.sub(r"<script[^>]*>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL)

    # Remove potential SQL injection patterns (basic protection)
    sanitized = re.sub(
        r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )

    return sanitized


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    # Check for forwarded IP first (if behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct connection IP
    return request.client.host if request.client else "unknown"


async def get_rate_limiter() -> RedisRateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter

    if _rate_limiter is None:
        try:
            redis_client = await get_redis_client()
            _rate_limiter = RedisRateLimiter(redis_client)
        except Exception as e:
            logger.error(f"Failed to initialize Redis rate limiter: {str(e)}")
            # Fall back to a mock limiter that always allows requests
            _rate_limiter = None

    return _rate_limiter


async def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit using Redis."""
    try:
        rate_limiter = await get_rate_limiter()
        if rate_limiter is None:
            # Redis unavailable, allow request (fail open)
            logger.warning("Rate limiter unavailable, allowing request")
            return True

        config = RateLimitConfig(
            requests_per_window=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW
        )

        result = await rate_limiter.check_rate_limit(client_ip, config, "api")
        return result.allowed

    except Exception as e:
        logger.error(f"Rate limit check failed for {client_ip}: {str(e)}")
        # Fail open - allow request if rate limiting fails
        return True


async def check_email_rate_limit(session_id: str) -> bool:
    """Check if session has exceeded email rate limit using Redis."""
    try:
        rate_limiter = await get_rate_limiter()
        if rate_limiter is None:
            # Redis unavailable, allow request (fail open)
            logger.warning("Email rate limiter unavailable, allowing request")
            return True

        config = RateLimitConfig(requests_per_window=1, window_seconds=EMAIL_RATE_LIMIT)  # 1 email

        result = await rate_limiter.check_rate_limit(session_id, config, "email")
        return result.allowed

    except Exception as e:
        logger.error(f"Email rate limit check failed for {session_id}: {str(e)}")
        # Fail open - allow request if rate limiting fails
        return True


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for rate limiting and input validation."""

    async def dispatch(self, request: Request, call_next):
        """Process request through security checks."""
        # Rate limiting check for chat endpoints
        if request.url.path == "/" and request.method == "POST":
            client_ip = get_client_ip(request)
            if not await check_rate_limit(client_ip):
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds.",
                    headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
                )

        response = await call_next(request)
        return response
