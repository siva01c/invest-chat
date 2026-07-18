"""Error handling middleware for FastAPI application."""

import time
import traceback
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from assistant.core.exceptions import (
    AssistantException,
    ConfigurationException,
    DatabaseException,
    ErrorCode,
    LLMException,
    RateLimitException,
    ServiceException,
)
from assistant.core.logging import get_logger, log_api_request, log_exception


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors and logging API requests."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("api.error_handler")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and handle any errors."""
        start_time = time.time()
        error_details: Optional[Dict[str, Any]] = None
        status_code = 200

        try:
            # Process the request
            response = await call_next(request)
            status_code = response.status_code

            return response

        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            status_code = e.status_code
            error_details = {"type": "HTTPException", "detail": e.detail}

            response_data = {
                "error": True,
                "error_code": "HTTP_ERROR",
                "message": e.detail or "HTTP error occurred",
                "status_code": e.status_code,
            }

            return JSONResponse(status_code=e.status_code, content=response_data)

        except AssistantException as e:
            # Handle custom application exceptions
            status_code = self._get_http_status_for_error(e.error_code)
            error_details = e.to_dict()

            # Log the exception
            log_exception(
                e,
                context={
                    "request_method": request.method,
                    "request_path": str(request.url.path),
                    "user_agent": request.headers.get("user-agent"),
                    "client_ip": self._get_client_ip(request),
                },
                logger_name="api.error_handler",
            )

            response_data = {
                "error": True,
                "error_code": e.error_code.value,
                "message": e.message,
                "details": e.details if e.details else {},
            }

            # Add development-only information
            if self._is_development_mode():
                response_data["debug"] = {
                    "exception_type": e.__class__.__name__,
                    "traceback": traceback.format_exc(),
                }

            return JSONResponse(status_code=status_code, content=response_data)

        except Exception as e:
            # Handle unexpected exceptions
            status_code = 500
            error_details = {"type": type(e).__name__, "message": str(e)}

            # Log the unexpected exception
            log_exception(
                e,
                context={
                    "request_method": request.method,
                    "request_path": str(request.url.path),
                    "user_agent": request.headers.get("user-agent"),
                    "client_ip": self._get_client_ip(request),
                },
                logger_name="api.error_handler",
            )

            response_data = {
                "error": True,
                "error_code": ErrorCode.UNKNOWN_ERROR.value,
                "message": "An unexpected error occurred",
            }

            # Add development-only information
            if self._is_development_mode():
                response_data["debug"] = {
                    "exception_type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }

            return JSONResponse(status_code=500, content=response_data)

        finally:
            # Log the API request
            duration_ms = (time.time() - start_time) * 1000
            log_api_request(
                method=request.method,
                path=str(request.url.path),
                status_code=status_code,
                duration_ms=duration_ms,
                user_id=self._extract_user_id(request),
                error_details=error_details,
            )

    def _get_http_status_for_error(self, error_code: ErrorCode) -> int:
        """Map error codes to HTTP status codes."""
        status_mapping = {
            ErrorCode.INVALID_INPUT: 400,
            ErrorCode.AUTHENTICATION_FAILED: 401,
            ErrorCode.TOKEN_EXPIRED: 401,
            ErrorCode.INSUFFICIENT_PERMISSIONS: 403,
            ErrorCode.FILE_NOT_FOUND: 404,
            ErrorCode.CONFIG_FILE_NOT_FOUND: 404,
            ErrorCode.TRANSLATION_FILE_NOT_FOUND: 404,
            ErrorCode.RATE_LIMIT_EXCEEDED: 429,
            ErrorCode.SERVICE_UNAVAILABLE: 503,
            ErrorCode.DATABASE_CONNECTION_ERROR: 503,
            ErrorCode.VECTOR_STORE_ERROR: 503,
            ErrorCode.LLM_API_ERROR: 503,
            ErrorCode.LLM_TIMEOUT: 504,
            ErrorCode.EXTERNAL_API_TIMEOUT: 504,
            ErrorCode.EMAIL_SEND_ERROR: 502,
            ErrorCode.EXTERNAL_API_ERROR: 502,
        }

        return status_mapping.get(error_code, 500)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request if available."""
        # Try to get from JWT token or session
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # This would require JWT decoding logic
            # For now, return None
            pass

        # Try to get from session or other sources
        # This depends on your authentication implementation

        return None

    def _is_development_mode(self) -> bool:
        """Check if application is in development mode."""
        try:
            from assistant.config import get_settings

            settings = get_settings()
            return settings.debug if hasattr(settings, "debug") else False
        except Exception:
            return False


class ValidationErrorHandler:
    """Handler for Pydantic validation errors."""

    @staticmethod
    def format_validation_error(exc) -> Dict[str, Any]:
        """Format Pydantic validation error for API response."""
        errors = []

        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(
                {
                    "field": field_path,
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input"),
                }
            )

        return {
            "error": True,
            "error_code": ErrorCode.INVALID_INPUT.value,
            "message": "Validation error",
            "details": {"validation_errors": errors},
        }


# Global error handlers for specific exception types
def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    status_code: Optional[int] = None,
) -> JSONResponse:
    """Create a standardized error response."""
    if status_code is None:
        status_code = 500

        # Map common error codes to status codes
        status_mapping = {
            ErrorCode.INVALID_INPUT: 400,
            ErrorCode.AUTHENTICATION_FAILED: 401,
            ErrorCode.INSUFFICIENT_PERMISSIONS: 403,
            ErrorCode.FILE_NOT_FOUND: 404,
            ErrorCode.RATE_LIMIT_EXCEEDED: 429,
            ErrorCode.SERVICE_UNAVAILABLE: 503,
        }
        status_code = status_mapping.get(error_code, 500)

    response_data = {"error": True, "error_code": error_code.value, "message": message}

    if details:
        response_data["details"] = details

    return JSONResponse(status_code=status_code, content=response_data)


# Specific exception handlers
def handle_service_exception(exc: ServiceException) -> JSONResponse:
    """Handle service exceptions."""
    return create_error_response(
        error_code=exc.error_code, message=f"Service error: {exc.message}", details=exc.details
    )


def handle_llm_exception(exc: LLMException) -> JSONResponse:
    """Handle LLM exceptions."""
    return create_error_response(
        error_code=exc.error_code,
        message=f"AI processing error: {exc.message}",
        details=exc.details,
    )


def handle_database_exception(exc: DatabaseException) -> JSONResponse:
    """Handle database exceptions."""
    return create_error_response(
        error_code=exc.error_code, message=f"Database error: {exc.message}", details=exc.details
    )


def handle_configuration_exception(exc: ConfigurationException) -> JSONResponse:
    """Handle configuration exceptions."""
    return create_error_response(
        error_code=exc.error_code,
        message=f"Configuration error: {exc.message}",
        details=exc.details,
    )


def handle_rate_limit_exception(exc: RateLimitException) -> JSONResponse:
    """Handle rate limit exceptions."""
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)

    response = create_error_response(
        error_code=exc.error_code, message=exc.message, details=exc.details, status_code=429
    )

    # Add headers to response
    for key, value in headers.items():
        response.headers[key] = value

    return response
