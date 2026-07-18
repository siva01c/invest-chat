"""Correlation ID middleware for FastAPI applications."""

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from assistant.config import get_settings
from assistant.infrastructure.observability.correlation_context import CorrelationContext
from assistant.infrastructure.observability.enhanced_logging import get_enhanced_logger


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for correlation ID management and request tracing.

    Features:
    - Extract or generate correlation IDs for requests
    - Set up request context with metadata
    - Add correlation ID to response headers
    - Log request/response lifecycle with correlation context
    """

    def __init__(
        self,
        app,
        correlation_header_name: str = "X-Correlation-ID",
        response_header_name: str = "X-Correlation-ID",
        enable_request_logging: bool = True,
        log_request_body: bool = False,
        log_response_body: bool = False,
    ):
        """
        Initialize correlation middleware.

        Args:
            app: FastAPI application
            correlation_header_name: Header name for incoming correlation ID
            response_header_name: Header name for outgoing correlation ID
            enable_request_logging: Enable automatic request/response logging
            log_request_body: Include request body in logs (be careful with sensitive data)
            log_response_body: Include response body in logs (be careful with sensitive data)
        """
        super().__init__(app)
        self.settings = get_settings()
        self.correlation_header_name = correlation_header_name
        self.response_header_name = response_header_name
        self.enable_request_logging = enable_request_logging
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.logger = get_enhanced_logger(self.__class__.__name__)

    def _extract_request_metadata(self, request: Request) -> dict:
        """Extract metadata from the request."""
        metadata = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }

        # Add forwarded headers if present (for proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            metadata["forwarded_for"] = forwarded_for

        forwarded_proto = request.headers.get("x-forwarded-proto")
        if forwarded_proto:
            metadata["forwarded_proto"] = forwarded_proto

        # Add authentication headers if present (without sensitive data)
        auth_header = request.headers.get("authorization")
        if auth_header:
            # Just indicate presence and type, not the actual token
            auth_type = auth_header.split(" ")[0] if " " in auth_header else "unknown"
            metadata["auth_type"] = auth_type

        return metadata

    def _should_log_request(self, request: Request) -> bool:
        """Determine if request should be logged."""
        # Skip logging for certain paths
        skip_paths = {"/health", "/metrics", "/favicon.ico"}
        return self.enable_request_logging and request.url.path not in skip_paths

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and manage correlation context."""
        start_time = time.time()

        # Extract or generate correlation ID
        correlation_id = request.headers.get(self.correlation_header_name)
        if not correlation_id:
            correlation_id = CorrelationContext.generate_correlation_id()

        # Extract request metadata
        request_metadata = self._extract_request_metadata(request)

        # Set up correlation context
        with CorrelationContext.correlation_scope(
            correlation_id=correlation_id, context=request_metadata
        ):
            # Log request start
            if self._should_log_request(request):
                log_data = {"event_phase": "request_start", "request_metadata": request_metadata}

                if self.log_request_body and hasattr(request, "body"):
                    try:
                        # Read request body (this is tricky with FastAPI)
                        # We'll add this as a placeholder for now
                        log_data["request_body_logged"] = False
                    except Exception:
                        pass

                self.logger.info(
                    f"Request started: {request.method} {request.url.path}", extra=log_data
                )

            try:
                # Process the request
                response = await call_next(request)

                # Calculate request duration
                duration_ms = (time.time() - start_time) * 1000

                # Add correlation ID to response headers
                response.headers[self.response_header_name] = correlation_id

                # Log request completion
                if self._should_log_request(request):
                    response_metadata = {
                        "status_code": response.status_code,
                        "response_headers": dict(response.headers),
                        "duration_ms": duration_ms,
                    }

                    log_data = {
                        "event_phase": "request_complete",
                        "response_metadata": response_metadata,
                    }

                    # Log API call using enhanced logger
                    self.logger.log_api_call(
                        method=request.method,
                        url=request.url.path,
                        status_code=response.status_code,
                        duration_ms=duration_ms,
                        request_size=int(request.headers.get("content-length", 0)),
                        response_size=len(response.body) if hasattr(response, "body") else None,
                    )

                return response

            except Exception as e:
                # Calculate request duration for failed requests
                duration_ms = (time.time() - start_time) * 1000

                # Log request error
                if self._should_log_request(request):
                    error_metadata = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "duration_ms": duration_ms,
                    }

                    self.logger.log_api_call(
                        method=request.method,
                        url=request.url.path,
                        status_code=500,
                        duration_ms=duration_ms,
                        error_details=error_metadata,
                    )

                # Re-raise the exception
                raise


def create_correlation_middleware(**kwargs) -> CorrelationMiddleware:
    """
    Factory function to create correlation middleware with configuration.

    Args:
        **kwargs: Configuration options for the middleware

    Returns:
        Configured CorrelationMiddleware instance
    """
    return lambda app: CorrelationMiddleware(app, **kwargs)
