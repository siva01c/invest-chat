"""Enhanced structured logging with correlation ID support."""

import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from assistant.core.logging import StructuredFormatter, get_logger

from .correlation_context import CorrelationContext


class CorrelationAwareFormatter(StructuredFormatter):
    """Enhanced structured formatter that includes correlation IDs and request context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON with correlation data."""
        # Start with the base structured format
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        correlation_id = CorrelationContext.get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add request context if available
        request_context = CorrelationContext.get_request_context()
        if request_context:
            log_entry["request_context"] = request_context

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
            if key not in {
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
            }:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class EnhancedLogger:
    """
    Enhanced logger with correlation ID support and advanced structured logging.

    Features:
    - Automatic correlation ID inclusion
    - Request context tracking
    - Performance metrics logging
    - Security event logging
    - Business event logging
    """

    def __init__(self, name: str):
        """
        Initialize enhanced logger.

        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
        self.name = name

    def _add_correlation_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add correlation context to log extra data."""
        log_extra = extra or {}

        # Add correlation context
        context_data = CorrelationContext.get_context_data()
        log_extra.update(context_data)

        return log_extra

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with correlation context."""
        self.logger.debug(message, extra=self._add_correlation_context(extra))

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with correlation context."""
        self.logger.info(message, extra=self._add_correlation_context(extra))

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with correlation context."""
        self.logger.warning(message, extra=self._add_correlation_context(extra))

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message with correlation context."""
        self.logger.error(message, extra=self._add_correlation_context(extra), exc_info=exc_info)

    def critical(
        self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False
    ):
        """Log critical message with correlation context."""
        self.logger.critical(message, extra=self._add_correlation_context(extra), exc_info=exc_info)

    def log_performance_metric(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log performance metric with correlation context.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
            metadata: Additional metadata
        """
        log_data = {
            "event_type": "performance_metric",
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
        }

        if metadata:
            log_data["metadata"] = metadata

        level = logging.INFO if success else logging.WARNING
        message = f"Performance metric: {operation} took {duration_ms:.2f}ms"

        self.logger.log(level, message, extra=self._add_correlation_context(log_data))

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Log security event with correlation context.

        Args:
            event_type: Type of security event
            severity: Severity level (low, medium, high, critical)
            description: Event description
            source_ip: Source IP address
            user_agent: User agent string
            additional_data: Additional security context
        """
        log_data = {
            "event_type": "security_event",
            "security_event_type": event_type,
            "severity": severity,
            "description": description,
        }

        if source_ip:
            log_data["source_ip"] = source_ip

        if user_agent:
            log_data["user_agent"] = user_agent

        if additional_data:
            log_data["additional_data"] = additional_data

        # Map severity to log level
        level_map = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        level = level_map.get(severity, logging.WARNING)

        message = f"Security event: {event_type} - {description}"
        self.logger.log(level, message, extra=self._add_correlation_context(log_data))

    def log_business_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        action: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log business event with correlation context.

        Args:
            event_type: Type of business event
            entity_type: Type of entity affected
            entity_id: ID of the entity
            action: Action performed
            success: Whether action was successful
            metadata: Additional business context
        """
        log_data = {
            "event_type": "business_event",
            "business_event_type": event_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "success": success,
        }

        if metadata:
            log_data["metadata"] = metadata

        level = logging.INFO if success else logging.WARNING
        message = f"Business event: {action} on {entity_type}#{entity_id}"

        self.logger.log(level, message, extra=self._add_correlation_context(log_data))

    def log_api_call(
        self,
        method: str,
        url: str,
        status_code: int,
        duration_ms: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log API call with correlation context.

        Args:
            method: HTTP method
            url: Request URL
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            request_size: Request size in bytes
            response_size: Response size in bytes
            error_details: Error details if applicable
        """
        log_data = {
            "event_type": "api_call",
            "method": method,
            "url": url,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }

        if request_size is not None:
            log_data["request_size_bytes"] = request_size

        if response_size is not None:
            log_data["response_size_bytes"] = response_size

        if error_details:
            log_data["error_details"] = error_details

        level = logging.INFO if status_code < 400 else logging.ERROR
        message = f"API call: {method} {url} - {status_code} ({duration_ms:.2f}ms)"

        self.logger.log(level, message, extra=self._add_correlation_context(log_data))

    def log_database_operation(
        self,
        operation: str,
        collection: str,
        duration_ms: float,
        record_count: Optional[int] = None,
        success: bool = True,
        error_details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log database operation with correlation context.

        Args:
            operation: Database operation type
            collection: Collection/table name
            duration_ms: Operation duration in milliseconds
            record_count: Number of records affected
            success: Whether operation was successful
            error_details: Error details if applicable
        """
        log_data = {
            "event_type": "database_operation",
            "operation": operation,
            "collection": collection,
            "duration_ms": duration_ms,
            "success": success,
        }

        if record_count is not None:
            log_data["record_count"] = record_count

        if error_details:
            log_data["error_details"] = error_details

        level = logging.INFO if success else logging.ERROR
        message = f"Database operation: {operation} on {collection} ({duration_ms:.2f}ms)"

        self.logger.log(level, message, extra=self._add_correlation_context(log_data))

    def log_cache_operation(
        self,
        operation: str,
        cache_name: str,
        hit: Optional[bool] = None,
        duration_ms: Optional[float] = None,
        key_pattern: Optional[str] = None,
    ):
        """
        Log cache operation with correlation context.

        Args:
            operation: Cache operation type (get, set, delete, etc.)
            cache_name: Name of the cache
            hit: Whether it was a cache hit (for get operations)
            duration_ms: Operation duration in milliseconds
            key_pattern: Pattern of the cache key (for debugging)
        """
        log_data = {
            "event_type": "cache_operation",
            "operation": operation,
            "cache_name": cache_name,
        }

        if hit is not None:
            log_data["cache_hit"] = hit

        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms

        if key_pattern:
            log_data["key_pattern"] = key_pattern

        message = f"Cache operation: {operation} on {cache_name}"
        if hit is not None:
            message += f" ({'HIT' if hit else 'MISS'})"

        self.logger.info(message, extra=self._add_correlation_context(log_data))


def get_enhanced_logger(name: str) -> EnhancedLogger:
    """
    Get an enhanced logger instance with correlation ID support.

    Args:
        name: Logger name

    Returns:
        Enhanced logger instance
    """
    return EnhancedLogger(name)


# Apply correlation-aware formatter to existing loggers
def setup_correlation_aware_logging():
    """Set up correlation-aware logging for all existing handlers."""
    root_logger = logging.getLogger()
    formatter = CorrelationAwareFormatter()

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
