"""Response caching middleware for FastAPI with intelligent caching strategies."""

import asyncio
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from assistant.config import get_settings
from assistant.core.logging import get_logger
from assistant.infrastructure.cache.advanced_cache_manager import (
    CacheStrategy,
    get_cache_manager,
)


class ResponseCacheMiddleware(BaseHTTPMiddleware):
    """
    Advanced response caching middleware for FastAPI.

    Features:
    - Intelligent caching based on HTTP methods and status codes
    - Configurable TTL per endpoint
    - Conditional caching based on request/response characteristics
    - Cache warming and invalidation
    - Performance metrics and monitoring
    """

    def __init__(
        self,
        app,
        default_ttl_seconds: int = 300,
        cache_strategy: CacheStrategy = CacheStrategy.LRU,
        max_cache_size_mb: int = 200,
        enable_compression: bool = True,
        cache_query_params: bool = True,
        cache_headers: List[str] = None,
    ):
        """
        Initialize response caching middleware.

        Args:
            app: FastAPI application
            default_ttl_seconds: Default cache TTL
            cache_strategy: Cache eviction strategy
            max_cache_size_mb: Maximum cache size in MB
            enable_compression: Enable response compression
            cache_query_params: Include query params in cache key
            cache_headers: Headers to include in cache key
        """
        super().__init__(app)
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)

        self.default_ttl_seconds = default_ttl_seconds
        self.cache_strategy = cache_strategy
        self.max_cache_size_mb = max_cache_size_mb
        self.enable_compression = enable_compression
        self.cache_query_params = cache_query_params
        self.cache_headers = cache_headers or ["Authorization", "Accept-Language"]

        # Cache configuration per endpoint
        self._endpoint_config: Dict[str, Dict[str, Any]] = {
            # Health endpoints - short TTL
            "/health": {"ttl": 30, "cacheable": True},
            "/health/detailed": {"ttl": 60, "cacheable": True},
            "/health/cache": {"ttl": 30, "cacheable": True},
            "/health/database": {"ttl": 60, "cacheable": True},
            "/health/performance": {"ttl": 30, "cacheable": True},
            # Chat endpoints - moderate TTL with user-specific caching
            "/chat": {"ttl": 300, "cacheable": True, "user_specific": True},
            # Knowledge endpoints - longer TTL
            "/knowledge/search": {"ttl": 600, "cacheable": True},
            "/knowledge/stats": {"ttl": 1800, "cacheable": True},
            # Admin endpoints - very short TTL
            "/admin": {"ttl": 10, "cacheable": True},
            "/admin/cache/clear": {"ttl": 0, "cacheable": False},
            "/admin/database/stats": {"ttl": 60, "cacheable": True},
            "/admin/performance/metrics": {"ttl": 30, "cacheable": True},
            # Static content - very long TTL
            "/static": {"ttl": 86400, "cacheable": True},  # 24 hours
        }

        # Methods that should never be cached
        self._non_cacheable_methods = {"POST", "PUT", "DELETE", "PATCH"}

        # Status codes that should be cached
        self._cacheable_status_codes = {200, 201, 300, 301, 302, 304, 404, 410}

        # Initialize cache manager (lazy loaded)
        self._cache_manager = None

    async def _get_cache_manager(self):
        """Get cache manager with lazy initialization."""
        if self._cache_manager is None:
            self._cache_manager = await get_cache_manager(
                "response_cache",
                max_memory_size_mb=self.max_cache_size_mb,
                strategy=self.cache_strategy,
                enable_compression=self.enable_compression,
            )
        return self._cache_manager

    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request."""
        key_parts = [request.method, request.url.path]

        # Include query parameters if enabled
        if self.cache_query_params and request.query_params:
            sorted_params = sorted(request.query_params.items())
            params_str = "&".join(f"{k}={v}" for k, v in sorted_params)
            key_parts.append(params_str)

        # Include specific headers in cache key
        for header_name in self.cache_headers:
            header_value = request.headers.get(header_name)
            if header_value:
                key_parts.append(f"{header_name}:{header_value}")

        # Create hash of the key for consistent length
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()

        return f"response:{key_hash}"

    def _get_endpoint_config(self, path: str) -> Dict[str, Any]:
        """Get caching configuration for endpoint."""
        # Try exact match first
        if path in self._endpoint_config:
            return self._endpoint_config[path]

        # Try prefix match
        for pattern, config in self._endpoint_config.items():
            if path.startswith(pattern):
                return config

        # Default configuration
        return {"ttl": self.default_ttl_seconds, "cacheable": True, "user_specific": False}

    def _is_cacheable_request(self, request: Request) -> bool:
        """Determine if request should be cached."""
        # Check HTTP method
        if request.method in self._non_cacheable_methods:
            return False

        # Check endpoint configuration
        endpoint_config = self._get_endpoint_config(request.url.path)
        if not endpoint_config.get("cacheable", True):
            return False

        # Check for cache control headers
        cache_control = request.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False

        return True

    def _is_cacheable_response(self, response: Response) -> bool:
        """Determine if response should be cached."""
        # Check status code
        if response.status_code not in self._cacheable_status_codes:
            return False

        # Check response headers
        cache_control = response.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control or "private" in cache_control:
            return False

        # Check for error responses in JSON
        if hasattr(response, "body") and response.headers.get("content-type", "").startswith(
            "application/json"
        ):
            try:
                if isinstance(response.body, bytes):
                    response_data = json.loads(response.body.decode())
                    if "error" in response_data or "detail" in response_data:
                        return False
            except (json.JSONDecodeError, AttributeError):
                pass

        return True

    def _get_cache_tags(self, request: Request, response: Response) -> List[str]:
        """Generate cache tags for invalidation."""
        tags = []

        # Add path-based tags
        path_parts = request.url.path.strip("/").split("/")
        for i in range(len(path_parts)):
            tag = "/".join(path_parts[: i + 1])
            if tag:
                tags.append(f"path:{tag}")

        # Add method tag
        tags.append(f"method:{request.method}")

        # Add status code tag
        tags.append(f"status:{response.status_code}")

        # Add user-specific tag if needed
        endpoint_config = self._get_endpoint_config(request.url.path)
        if endpoint_config.get("user_specific", False):
            user_id = request.headers.get("Authorization", "anonymous")
            if user_id != "anonymous":
                # Use hash of auth header for privacy
                user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:8]
                tags.append(f"user:{user_hash}")

        return tags

    async def _store_in_cache(
        self, cache_key: str, response_data: Dict[str, Any], ttl_seconds: int, tags: List[str]
    ):
        """Store response in cache."""
        try:
            cache_manager = await self._get_cache_manager()
            await cache_manager.set(cache_key, response_data, ttl_seconds=ttl_seconds, tags=tags)
        except Exception as e:
            self.logger.warning(f"Failed to store response in cache: {str(e)}")

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get response from cache."""
        try:
            cache_manager = await self._get_cache_manager()
            return await cache_manager.get(cache_key)
        except Exception as e:
            self.logger.warning(f"Failed to get response from cache: {str(e)}")
            return None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with caching logic."""
        # Check if request is cacheable
        if not self._is_cacheable_request(request):
            return await call_next(request)

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Try to get from cache
        start_time = time.time()
        cached_response = await self._get_from_cache(cache_key)

        if cached_response:
            # Cache hit - reconstruct response
            response_time = time.time() - start_time

            # Add cache headers
            headers = cached_response.get("headers", {})
            headers["X-Cache"] = "HIT"
            headers["X-Cache-Key"] = cache_key
            headers["X-Response-Time"] = f"{response_time * 1000:.2f}ms"

            self.logger.debug(
                f"Cache HIT: {request.method} {request.url.path} "
                f"(response_time={response_time * 1000:.2f}ms)"
            )

            return JSONResponse(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers=headers,
            )

        # Cache miss - call next middleware/endpoint
        response = await call_next(request)
        response_time = time.time() - start_time

        # Check if response should be cached
        if self._is_cacheable_response(response):
            try:
                # Get endpoint configuration
                endpoint_config = self._get_endpoint_config(request.url.path)
                ttl_seconds = endpoint_config.get("ttl", self.default_ttl_seconds)

                # Extract response data
                response_body = None
                if hasattr(response, "body"):
                    if isinstance(response.body, bytes):
                        response_body = response.body.decode()
                    else:
                        response_body = response.body

                # Parse JSON content if possible
                response_content = None
                if response_body:
                    try:
                        response_content = json.loads(response_body)
                    except json.JSONDecodeError:
                        response_content = response_body

                # Prepare cache data
                cache_data = {
                    "content": response_content,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "cached_at": time.time(),
                    "cache_key": cache_key,
                }

                # Generate cache tags
                tags = self._get_cache_tags(request, response)

                # Store in cache
                if ttl_seconds > 0:
                    await self._store_in_cache(cache_key, cache_data, ttl_seconds, tags)

                # Add cache headers to response
                response.headers["X-Cache"] = "MISS"
                response.headers["X-Cache-Key"] = cache_key
                response.headers["X-Cache-TTL"] = str(ttl_seconds)

                self.logger.debug(
                    f"Cache MISS: {request.method} {request.url.path} "
                    f"(response_time={response_time * 1000:.2f}ms, ttl={ttl_seconds}s)"
                )

            except Exception as e:
                self.logger.warning(f"Failed to process response for caching: {str(e)}")

        # Add response time header
        response.headers["X-Response-Time"] = f"{response_time * 1000:.2f}ms"

        return response

    async def invalidate_cache_by_path(self, path_pattern: str) -> int:
        """Invalidate cache entries by path pattern."""
        try:
            cache_manager = await self._get_cache_manager()
            tag = f"path:{path_pattern.strip('/')}"
            return await cache_manager.invalidate_by_tags([tag])
        except Exception as e:
            self.logger.error(f"Failed to invalidate cache by path: {str(e)}")
            return 0

    async def invalidate_cache_by_user(self, user_id: str) -> int:
        """Invalidate cache entries for specific user."""
        try:
            cache_manager = await self._get_cache_manager()
            user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:8]
            tag = f"user:{user_hash}"
            return await cache_manager.invalidate_by_tags([tag])
        except Exception as e:
            self.logger.error(f"Failed to invalidate cache by user: {str(e)}")
            return 0

    async def warm_cache(self, endpoints: List[str], base_url: str = "http://localhost:5000"):
        """Warm cache for specific endpoints."""
        try:
            import httpx

            self.logger.info(f"Starting cache warming for {len(endpoints)} endpoints")

            async with httpx.AsyncClient() as client:
                tasks = []
                for endpoint in endpoints:
                    url = f"{base_url}{endpoint}"
                    task = asyncio.create_task(client.get(url))
                    tasks.append(task)

                # Execute requests in parallel
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                success_count = sum(
                    1
                    for response in responses
                    if not isinstance(response, Exception) and response.status_code == 200
                )

                self.logger.info(
                    f"Cache warming completed: {success_count}/{len(endpoints)} successful"
                )

        except Exception as e:
            self.logger.error(f"Cache warming failed: {str(e)}")

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            cache_manager = await self._get_cache_manager()
            stats = await cache_manager.get_cache_stats()

            # Add middleware-specific stats
            stats.update(
                {
                    "middleware_name": "ResponseCacheMiddleware",
                    "default_ttl_seconds": self.default_ttl_seconds,
                    "cache_strategy": self.cache_strategy.value,
                    "compression_enabled": self.enable_compression,
                    "endpoint_configs": len(self._endpoint_config),
                    "cacheable_status_codes": list(self._cacheable_status_codes),
                    "non_cacheable_methods": list(self._non_cacheable_methods),
                }
            )

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get cache statistics: {str(e)}")
            return {"error": str(e)}

    async def clear_all_cache(self) -> bool:
        """Clear all cached responses."""
        try:
            cache_manager = await self._get_cache_manager()
            return await cache_manager.clear_cache()
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
            return False
