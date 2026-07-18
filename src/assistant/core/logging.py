"""Structured logging system for the assistant application."""

import asyncio
import json
import logging
import logging.handlers
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import AssistantException


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }

        # Add structured exception data if it's an AssistantException
        if hasattr(record, "exception_data"):
            log_entry["error_details"] = record.exception_data

        # Add any extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
                "exception_data",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


class AssistantLogger:
    """Centralized logging utility for the assistant application."""

    _instance: Optional["AssistantLogger"] = None
    _initialized: bool = False

    def __new__(cls) -> "AssistantLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            self._initialized = True

    def _setup_logging(self):
        """Set up logging configuration."""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Console handler with structured format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = StructuredFormatter()
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "assistant.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)

        # Error file handler for errors only
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "assistant_errors.log", maxBytes=5 * 1024 * 1024, backupCount=3  # 5MB
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(console_formatter)
        root_logger.addHandler(error_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific module."""
        return logging.getLogger(name)

    def log_exception(
        self,
        logger: logging.Logger,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        level: int = logging.ERROR,
    ):
        """Log an exception with structured data."""
        # Prepare exception data
        exception_data = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
        }

        # Add structured data for AssistantException
        if isinstance(exception, AssistantException):
            exception_data.update(exception.to_dict())

        # Add context if provided
        if context:
            exception_data["context"] = context

        # Create log record with exception data
        logger.log(
            level,
            f"Exception occurred: {exception}",
            exc_info=True,
            extra={"exception_data": exception_data},
        )

    def log_service_operation(
        self,
        logger: logging.Logger,
        service_name: str,
        operation: str,
        success: bool,
        duration_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log a service operation with structured data."""
        log_data = {"service_name": service_name, "operation": operation, "success": success}

        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms

        if details:
            log_data["details"] = details

        level = logging.INFO if success else logging.WARNING
        message = (
            f"Service operation {'completed' if success else 'failed'}: {service_name}.{operation}"
        )

        logger.log(level, message, extra=log_data)

    def log_api_request(
        self,
        logger: logging.Logger,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ):
        """Log an API request with structured data."""
        log_data = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }

        if user_id:
            log_data["user_id"] = user_id

        if error_details:
            log_data["error_details"] = error_details

        level = logging.INFO if status_code < 400 else logging.ERROR
        message = f"API {method} {path} - {status_code} ({duration_ms:.2f}ms)"

        logger.log(level, message, extra=log_data)

    def log_llm_operation(
        self,
        logger: logging.Logger,
        model_name: str,
        operation: str,
        success: bool,
        token_count: Optional[int] = None,
        duration_ms: Optional[float] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ):
        """Log an LLM operation with structured data."""
        log_data = {"model_name": model_name, "operation": operation, "success": success}

        if token_count is not None:
            log_data["token_count"] = token_count

        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms

        if error_details:
            log_data["error_details"] = error_details

        level = logging.INFO if success else logging.ERROR
        message = f"LLM operation {'completed' if success else 'failed'}: {model_name}.{operation}"

        logger.log(level, message, extra=log_data)


# Global logger instance
_logger_instance = AssistantLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return _logger_instance.get_logger(name)


def log_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    logger_name: str = "assistant",
    level: int = logging.ERROR,
):
    """Convenience function to log an exception."""
    logger = get_logger(logger_name)
    _logger_instance.log_exception(logger, exception, context, level)


@contextmanager
def log_operation(service_name: str, operation: str, logger_name: Optional[str] = None):
    """Context manager for logging service operations with timing."""
    logger = get_logger(logger_name or service_name)
    start_time = datetime.now(UTC)

    try:
        logger.info(f"Starting operation: {service_name}.{operation}")
        yield logger

        # Calculate duration
        duration = (datetime.now(UTC) - start_time).total_seconds() * 1000
        _logger_instance.log_service_operation(logger, service_name, operation, True, duration)

    except Exception as e:
        # Calculate duration
        duration = (datetime.now(UTC) - start_time).total_seconds() * 1000
        _logger_instance.log_service_operation(logger, service_name, operation, False, duration)
        _logger_instance.log_exception(logger, e)
        raise


def log_service_method(service_name: Optional[str] = None):
    """Decorator for automatically logging service method calls."""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get service name from class or parameter
                method_service_name = service_name
                if not method_service_name and args:
                    instance = args[0]
                    if hasattr(instance, "get_service_name"):
                        method_service_name = instance.get_service_name()
                    else:
                        method_service_name = instance.__class__.__name__

                operation = func.__name__

                with log_operation(method_service_name or "unknown", operation):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get service name from class or parameter
                method_service_name = service_name
                if not method_service_name and args:
                    instance = args[0]
                    if hasattr(instance, "get_service_name"):
                        method_service_name = instance.get_service_name()
                    else:
                        method_service_name = instance.__class__.__name__

                operation = func.__name__

                with log_operation(method_service_name or "unknown", operation):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


# Convenience functions for specific log types
def log_service_operation(
    service_name: str,
    operation: str,
    success: bool,
    duration_ms: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    logger_name: Optional[str] = None,
):
    """Log a service operation."""
    logger = get_logger(logger_name or service_name)
    _logger_instance.log_service_operation(
        logger, service_name, operation, success, duration_ms, details
    )


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    error_details: Optional[Dict[str, Any]] = None,
    logger_name: str = "api",
):
    """Log an API request."""
    logger = get_logger(logger_name)
    _logger_instance.log_api_request(
        logger, method, path, status_code, duration_ms, user_id, error_details
    )


def log_llm_operation(
    model_name: str,
    operation: str,
    success: bool,
    token_count: Optional[int] = None,
    duration_ms: Optional[float] = None,
    error_details: Optional[Dict[str, Any]] = None,
    logger_name: str = "llm",
):
    """Log an LLM operation."""
    logger = get_logger(logger_name)
    _logger_instance.log_llm_operation(
        logger, model_name, operation, success, token_count, duration_ms, error_details
    )
