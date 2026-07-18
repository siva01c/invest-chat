"""Correlation ID context management for distributed tracing."""

import contextvars
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

from assistant.core.logging import get_logger

# Context variable for correlation ID
correlation_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Context variable for additional request context
request_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "request_context", default=None
)


class CorrelationContext:
    """
    Manages correlation IDs and request context for distributed tracing.

    Features:
    - Automatic correlation ID generation
    - Context propagation across async operations
    - Request-specific metadata tracking
    - Integration with structured logging
    """

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())

    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get the current correlation ID from context."""
        return correlation_id_context.get()

    @staticmethod
    def set_correlation_id(correlation_id: str):
        """Set the correlation ID in the current context."""
        correlation_id_context.set(correlation_id)

    @staticmethod
    def get_or_create_correlation_id() -> str:
        """Get existing correlation ID or create a new one."""
        correlation_id = CorrelationContext.get_correlation_id()
        if not correlation_id:
            correlation_id = CorrelationContext.generate_correlation_id()
            CorrelationContext.set_correlation_id(correlation_id)
        return correlation_id

    @staticmethod
    def get_request_context() -> Optional[Dict[str, Any]]:
        """Get the current request context."""
        return request_context.get()

    @staticmethod
    def set_request_context(context: Dict[str, Any]):
        """Set the request context."""
        request_context.set(context)

    @staticmethod
    def update_request_context(updates: Dict[str, Any]):
        """Update the request context with new values."""
        current_context = CorrelationContext.get_request_context() or {}
        current_context.update(updates)
        CorrelationContext.set_request_context(current_context)

    @staticmethod
    def clear_context():
        """Clear all context variables."""
        correlation_id_context.set(None)
        request_context.set(None)

    @staticmethod
    @contextmanager
    def correlation_scope(
        correlation_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for correlation ID scope.

        Args:
            correlation_id: Specific correlation ID to use, or None to generate one
            context: Additional request context to set
        """
        # Store previous values
        previous_correlation_id = CorrelationContext.get_correlation_id()
        previous_context = CorrelationContext.get_request_context()

        try:
            # Set new correlation ID
            if correlation_id:
                CorrelationContext.set_correlation_id(correlation_id)
            else:
                CorrelationContext.get_or_create_correlation_id()

            # Set request context
            if context:
                CorrelationContext.set_request_context(context)

            yield CorrelationContext.get_correlation_id()

        finally:
            # Restore previous values
            if previous_correlation_id:
                CorrelationContext.set_correlation_id(previous_correlation_id)
            else:
                correlation_id_context.set(None)

            if previous_context:
                CorrelationContext.set_request_context(previous_context)
            else:
                request_context.set(None)

    @staticmethod
    def get_context_data() -> Dict[str, Any]:
        """Get all context data for logging."""
        context_data = {}

        correlation_id = CorrelationContext.get_correlation_id()
        if correlation_id:
            context_data["correlation_id"] = correlation_id

        request_ctx = CorrelationContext.get_request_context()
        if request_ctx:
            context_data["request_context"] = request_ctx

        return context_data


class TraceableOperation:
    """
    Decorator class for making operations traceable with correlation IDs.
    """

    def __init__(
        self,
        operation_name: Optional[str] = None,
        include_args: bool = False,
        include_result: bool = False,
    ):
        """
        Initialize traceable operation decorator.

        Args:
            operation_name: Custom operation name, defaults to function name
            include_args: Include function arguments in context
            include_result: Include function result in context (be careful with sensitive data)
        """
        self.operation_name = operation_name
        self.include_args = include_args
        self.include_result = include_result
        self.logger = get_logger(self.__class__.__name__)

    def __call__(self, func):
        """Make function traceable."""
        from functools import wraps

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            operation_name = self.operation_name or func.__name__
            correlation_id = CorrelationContext.get_or_create_correlation_id()

            # Prepare operation context
            operation_context = {
                "operation": operation_name,
                "function": func.__name__,
                "module": func.__module__,
            }

            if self.include_args:
                operation_context["args"] = {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }

            # Update request context
            CorrelationContext.update_request_context(operation_context)

            self.logger.info(
                f"Starting traceable operation: {operation_name}",
                extra=CorrelationContext.get_context_data(),
            )

            try:
                result = await func(*args, **kwargs)

                if self.include_result and result is not None:
                    operation_context["result_type"] = type(result).__name__

                self.logger.info(
                    f"Completed traceable operation: {operation_name}",
                    extra=CorrelationContext.get_context_data(),
                )

                return result

            except Exception as e:
                operation_context["error"] = {"type": type(e).__name__, "message": str(e)}

                self.logger.error(
                    f"Failed traceable operation: {operation_name}",
                    exc_info=True,
                    extra=CorrelationContext.get_context_data(),
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            operation_name = self.operation_name or func.__name__
            correlation_id = CorrelationContext.get_or_create_correlation_id()

            # Prepare operation context
            operation_context = {
                "operation": operation_name,
                "function": func.__name__,
                "module": func.__module__,
            }

            if self.include_args:
                operation_context["args"] = {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }

            # Update request context
            CorrelationContext.update_request_context(operation_context)

            self.logger.info(
                f"Starting traceable operation: {operation_name}",
                extra=CorrelationContext.get_context_data(),
            )

            try:
                result = func(*args, **kwargs)

                if self.include_result and result is not None:
                    operation_context["result_type"] = type(result).__name__

                self.logger.info(
                    f"Completed traceable operation: {operation_name}",
                    extra=CorrelationContext.get_context_data(),
                )

                return result

            except Exception as e:
                operation_context["error"] = {"type": type(e).__name__, "message": str(e)}

                self.logger.error(
                    f"Failed traceable operation: {operation_name}",
                    exc_info=True,
                    extra=CorrelationContext.get_context_data(),
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# Convenience decorators
def traceable(
    operation_name: Optional[str] = None, include_args: bool = False, include_result: bool = False
):
    """
    Decorator to make a function traceable with correlation IDs.

    Args:
        operation_name: Custom operation name
        include_args: Include function arguments in context
        include_result: Include function result in context
    """
    return TraceableOperation(operation_name, include_args, include_result)


def trace_service_method(service_name: Optional[str] = None):
    """
    Decorator specifically for service methods with automatic service name detection.

    Args:
        service_name: Override service name detection
    """

    def decorator(func):
        from functools import wraps

        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Determine service name
            method_service_name = service_name
            if not method_service_name:
                if hasattr(self, "get_service_name"):
                    method_service_name = self.get_service_name()
                else:
                    method_service_name = self.__class__.__name__

            operation_name = f"{method_service_name}.{func.__name__}"

            # Use traceable operation
            traceable_op = TraceableOperation(operation_name)
            return await traceable_op(func)(self, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Determine service name
            method_service_name = service_name
            if not method_service_name:
                if hasattr(self, "get_service_name"):
                    method_service_name = self.get_service_name()
                else:
                    method_service_name = self.__class__.__name__

            operation_name = f"{method_service_name}.{func.__name__}"

            # Use traceable operation
            traceable_op = TraceableOperation(operation_name)
            return traceable_op(func)(self, *args, **kwargs)

        # Return appropriate wrapper
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
