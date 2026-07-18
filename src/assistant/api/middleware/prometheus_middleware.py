"""Prometheus metrics middleware for FastAPI applications."""

import time
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match

from assistant.core.logging import get_logger
from assistant.infrastructure.observability import get_prometheus_metrics


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic Prometheus metrics collection.

    Features:
    - HTTP request/response metrics
    - Request duration tracking
    - Request/response size tracking
    - Status code distribution
    - Endpoint-specific metrics
    """

    def __init__(
        self,
        app,
        enable_request_size_metrics: bool = True,
        enable_response_size_metrics: bool = True,
        track_in_progress_requests: bool = True,
        exclude_paths: Optional[set] = None,
    ):
        """
        Initialize Prometheus middleware.

        Args:
            app: FastAPI application
            enable_request_size_metrics: Track request size metrics
            enable_response_size_metrics: Track response size metrics
            track_in_progress_requests: Track in-progress requests gauge
            exclude_paths: Set of paths to exclude from metrics
        """
        super().__init__(app)
        self.metrics = get_prometheus_metrics()
        self.logger = get_logger(self.__class__.__name__)

        self.enable_request_size_metrics = enable_request_size_metrics
        self.enable_response_size_metrics = enable_response_size_metrics
        self.track_in_progress_requests = track_in_progress_requests

        # Default excluded paths (health checks, metrics endpoint)
        self.exclude_paths = exclude_paths or {
            "/health",
            "/metrics",
            "/favicon.ico",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

    def _should_track_request(self, request: Request) -> bool:
        """Determine if request should be tracked."""
        path = request.url.path
        return path not in self.exclude_paths

    def _get_endpoint_path(self, request: Request) -> str:
        """Get the route template for the request, or path if no match."""
        try:
            # Try to get the route pattern from FastAPI
            for route in request.app.routes:
                match, _ = route.matches(
                    {"type": "http", "path": request.url.path, "method": request.method}
                )
                if match == Match.FULL:
                    return route.path

            # Fallback to actual path
            return request.url.path
        except Exception:
            return request.url.path

    def _get_content_length(self, headers) -> int:
        """Extract content length from headers."""
        try:
            content_length = headers.get("content-length")
            return int(content_length) if content_length else 0
        except (ValueError, TypeError):
            return 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        if not self._should_track_request(request):
            return await call_next(request)

        start_time = time.time()
        method = request.method
        endpoint = self._get_endpoint_path(request)
        status_code = 500  # Default to error if something goes wrong

        # Track request size
        request_size = 0
        if self.enable_request_size_metrics:
            request_size = self._get_content_length(request.headers)
            if request_size > 0:
                self.metrics.http_request_size_bytes.labels(
                    method=method, endpoint=endpoint
                ).observe(request_size)

        # Track in-progress requests
        if self.track_in_progress_requests:
            self.metrics.http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()

        try:
            # Process the request
            response = await call_next(request)
            status_code = response.status_code

            # Track response size
            if self.enable_response_size_metrics:
                response_size = self._get_content_length(response.headers)
                if response_size > 0:
                    self.metrics.http_response_size_bytes.labels(
                        method=method, endpoint=endpoint
                    ).observe(response_size)

            return response

        except Exception as e:
            # Track errors
            status_code = 500
            self.logger.error(f"Request processing error: {str(e)}")
            raise

        finally:
            # Always track request completion metrics
            duration = time.time() - start_time

            # Record metrics
            labels = {"method": method, "endpoint": endpoint, "status_code": str(status_code)}

            self.metrics.http_request_duration.labels(**labels).observe(duration)
            self.metrics.http_requests_total.labels(**labels).inc()

            # Decrement in-progress counter
            if self.track_in_progress_requests:
                self.metrics.http_requests_in_progress.labels(
                    method=method, endpoint=endpoint
                ).dec()

            # Log slow requests
            if duration > 1.0:  # Log requests slower than 1 second
                self.logger.warning(
                    f"Slow request: {method} {endpoint} took {duration:.3f}s (status: {status_code})"
                )


def create_prometheus_middleware(**kwargs) -> PrometheusMiddleware:
    """
    Factory function to create Prometheus middleware with configuration.

    Args:
        **kwargs: Configuration options for the middleware

    Returns:
        Configured PrometheusMiddleware instance
    """
    return lambda app: PrometheusMiddleware(app, **kwargs)
