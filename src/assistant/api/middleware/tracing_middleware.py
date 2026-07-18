"""Distributed tracing middleware for FastAPI applications."""

from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from assistant.config import get_settings
from assistant.infrastructure.observability.correlation_context import CorrelationContext
from assistant.infrastructure.observability.distributed_tracing import (
    SpanKind,
    SpanStatus,
    get_tracer,
)
from assistant.infrastructure.observability.enhanced_logging import get_enhanced_logger


class TracingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for distributed tracing integration.

    Features:
    - Automatic trace creation for HTTP requests
    - Integration with correlation IDs
    - Request/response attribute collection
    - Error tracking in spans
    """

    def __init__(self, app, enable_tracing: bool = True, exclude_paths: set = None):
        """
        Initialize tracing middleware.

        Args:
            app: FastAPI application
            enable_tracing: Enable distributed tracing
            exclude_paths: Set of paths to exclude from tracing
        """
        super().__init__(app)
        self.settings = get_settings()
        self.enable_tracing = enable_tracing and self.settings.enable_distributed_tracing
        self.exclude_paths = exclude_paths or {"/health", "/metrics", "/favicon.ico"}
        self.tracer = get_tracer()
        self.logger = get_enhanced_logger(self.__class__.__name__)

    def _should_trace_request(self, request: Request) -> bool:
        """Determine if request should be traced."""
        if not self.enable_tracing:
            return False

        path = request.url.path
        return path not in self.exclude_paths

    def _extract_trace_headers(self, request: Request) -> dict:
        """Extract tracing headers from request."""
        trace_headers = {}

        # Check for common tracing headers
        trace_id_headers = ["x-trace-id", "traceid", "x-request-id", "x-correlation-id"]

        for header in trace_id_headers:
            value = request.headers.get(header)
            if value:
                trace_headers[header] = value
                break

        # Check for span ID headers
        span_id_headers = ["x-span-id", "spanid", "x-parent-span-id"]

        for header in span_id_headers:
            value = request.headers.get(header)
            if value:
                trace_headers[header] = value
                break

        return trace_headers

    def _collect_request_attributes(self, request: Request) -> dict:
        """Collect request attributes for the span."""
        attributes = {
            "http.method": request.method,
            "http.url": str(request.url),
            "http.scheme": request.url.scheme,
            "http.host": request.url.hostname,
            "http.target": request.url.path,
            "http.user_agent": request.headers.get("user-agent", ""),
            "http.client_ip": request.client.host if request.client else None,
        }

        # Add content-related attributes
        content_type = request.headers.get("content-type")
        if content_type:
            attributes["http.request_content_type"] = content_type

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                attributes["http.request_content_length"] = int(content_length)
            except ValueError:
                pass

        # Add query parameters count
        if request.query_params:
            attributes["http.query_params_count"] = len(request.query_params)

        # Add forwarded headers for proxy setups
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            attributes["http.x_forwarded_for"] = forwarded_for

        return {k: v for k, v in attributes.items() if v is not None}

    def _collect_response_attributes(self, response: Response) -> dict:
        """Collect response attributes for the span."""
        attributes = {
            "http.status_code": response.status_code,
        }

        # Add response content type
        content_type = response.headers.get("content-type")
        if content_type:
            attributes["http.response_content_type"] = content_type

        # Add response size if available
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                attributes["http.response_content_length"] = int(content_length)
            except ValueError:
                pass

        return attributes

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with distributed tracing."""
        if not self._should_trace_request(request):
            return await call_next(request)

        # Extract trace headers
        trace_headers = self._extract_trace_headers(request)

        # Create operation name
        operation_name = f"{request.method} {request.url.path}"

        # Start trace
        trace_id = trace_headers.get("x-trace-id") or CorrelationContext.get_correlation_id()
        trace = self.tracer.start_trace(operation_name, trace_id)

        if not trace:
            # Tracing disabled or not sampled
            return await call_next(request)

        # Get the root span
        span = self.tracer.get_current_span()
        if span:
            # Set span kind and attributes
            span.kind = SpanKind.SERVER

            # Add request attributes
            request_attributes = self._collect_request_attributes(request)
            for key, value in request_attributes.items():
                span.set_attribute(key, value)

            # Add correlation ID
            correlation_id = CorrelationContext.get_correlation_id()
            if correlation_id:
                span.set_attribute("correlation_id", correlation_id)

            # Add trace headers as tags
            for key, value in trace_headers.items():
                span.set_tag(f"header.{key}", value)

        try:
            # Add request start event
            if span:
                span.add_event(
                    "request.start", {"method": request.method, "path": request.url.path}
                )

            # Process the request
            response = await call_next(request)

            # Add response attributes
            if span:
                response_attributes = self._collect_response_attributes(response)
                for key, value in response_attributes.items():
                    span.set_attribute(key, value)

                # Add response event
                span.add_event("request.complete", {"status_code": response.status_code})

                # Set status based on response code
                if response.status_code >= 400:
                    span.set_status(SpanStatus.ERROR, f"HTTP {response.status_code}")
                else:
                    span.set_status(SpanStatus.OK)

                # Add trace ID to response headers
                response.headers["X-Trace-ID"] = span.trace_id

            return response

        except Exception as e:
            # Record exception in span
            if span:
                span.record_exception(e)
                span.add_event(
                    "request.error", {"error_type": type(e).__name__, "error_message": str(e)}
                )

            self.logger.error(
                f"Request failed with tracing: {request.method} {request.url.path}",
                exc_info=True,
                extra={
                    "trace_id": span.trace_id if span else None,
                    "span_id": span.span_id if span else None,
                },
            )

            raise

        finally:
            # Finish the trace
            self.tracer.finish_trace()


def create_tracing_middleware(**kwargs) -> TracingMiddleware:
    """
    Factory function to create tracing middleware with configuration.

    Args:
        **kwargs: Configuration options for the middleware

    Returns:
        Configured TracingMiddleware instance
    """
    return lambda app: TracingMiddleware(app, **kwargs)
