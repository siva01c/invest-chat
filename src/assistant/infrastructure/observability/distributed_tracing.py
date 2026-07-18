"""Distributed tracing system for tracking requests across services."""

import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from assistant.config import get_settings
from assistant.infrastructure.observability.correlation_context import CorrelationContext
from assistant.infrastructure.observability.enhanced_logging import get_enhanced_logger


class SpanKind(Enum):
    """Types of spans in distributed tracing."""

    INTERNAL = "internal"  # Internal function call
    SERVER = "server"  # Server request handling
    CLIENT = "client"  # Client request making
    PRODUCER = "producer"  # Message producer
    CONSUMER = "consumer"  # Message consumer


class SpanStatus(Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """Event within a span."""

    timestamp: datetime
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    A span represents a single operation within a trace.
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    kind: SpanKind = SpanKind.INTERNAL
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    error_details: Optional[Dict[str, Any]] = None

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to this span."""
        event = SpanEvent(timestamp=datetime.utcnow(), name=name, attributes=attributes or {})
        self.events.append(event)

    def set_attribute(self, key: str, value: Any):
        """Set an attribute on this span."""
        self.attributes[key] = value

    def set_tag(self, key: str, value: str):
        """Set a tag on this span."""
        self.tags[key] = value

    def set_status(self, status: SpanStatus, description: Optional[str] = None):
        """Set the status of this span."""
        self.status = status
        if description:
            self.set_attribute("status.description", description)

    def record_exception(self, exception: Exception):
        """Record an exception in this span."""
        self.status = SpanStatus.ERROR
        self.error_details = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            "exception.stacktrace": None,  # Could add full stacktrace if needed
        }
        self.add_event("exception", self.error_details)

    def finish(self):
        """Finish this span."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "kind": self.kind.value,
            "attributes": self.attributes,
            "tags": self.tags,
            "events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "name": event.name,
                    "attributes": event.attributes,
                }
                for event in self.events
            ],
            "error_details": self.error_details,
        }


@dataclass
class Trace:
    """A trace represents a collection of spans that represent a single request."""

    trace_id: str
    root_span_id: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    span_count: int = 0
    error_count: int = 0
    spans: List[Span] = field(default_factory=list)

    def add_span(self, span: Span):
        """Add a span to this trace."""
        self.spans.append(span)
        self.span_count += 1
        if span.status == SpanStatus.ERROR:
            self.error_count += 1

        # Update trace timing
        if self.end_time is None or (span.end_time and span.end_time > self.end_time):
            self.end_time = span.end_time

        if self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "root_span_id": self.root_span_id,
            "service_name": self.service_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "span_count": self.span_count,
            "error_count": self.error_count,
            "spans": [span.to_dict() for span in self.spans],
        }


class TraceContext:
    """Context for managing the current trace and span."""

    def __init__(self):
        """Initialize trace context."""
        self.current_trace: Optional[Trace] = None
        self.current_span: Optional[Span] = None
        self.span_stack: List[Span] = []

    def start_trace(
        self,
        operation_name: str,
        service_name: str = "sales-assistant",
        trace_id: Optional[str] = None,
    ) -> Trace:
        """Start a new trace."""
        if not trace_id:
            trace_id = str(uuid.uuid4())

        # Create root span
        root_span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=None,
            operation_name=operation_name,
            service_name=service_name,
            start_time=datetime.utcnow(),
            kind=SpanKind.SERVER,
        )

        # Create trace
        trace = Trace(
            trace_id=trace_id,
            root_span_id=root_span.span_id,
            service_name=service_name,
            start_time=root_span.start_time,
        )

        trace.add_span(root_span)

        self.current_trace = trace
        self.current_span = root_span
        self.span_stack = [root_span]

        return trace

    def start_span(
        self,
        operation_name: str,
        service_name: Optional[str] = None,
        span_kind: SpanKind = SpanKind.INTERNAL,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a new span within the current trace."""
        if not self.current_trace:
            # Start a new trace if none exists
            self.start_trace(operation_name, service_name or "sales-assistant")
            return self.current_span

        parent = parent_span or self.current_span
        parent_span_id = parent.span_id if parent else None

        span = Span(
            trace_id=self.current_trace.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=service_name or self.current_trace.service_name,
            start_time=datetime.utcnow(),
            kind=span_kind,
        )

        self.current_trace.add_span(span)
        self.current_span = span
        self.span_stack.append(span)

        return span

    def finish_span(self, span: Optional[Span] = None):
        """Finish the current or specified span."""
        target_span = span or self.current_span
        if not target_span:
            return

        target_span.finish()

        # Remove from stack if it's the current span
        if target_span == self.current_span and self.span_stack:
            self.span_stack.pop()
            self.current_span = self.span_stack[-1] if self.span_stack else None

    def finish_trace(self):
        """Finish the current trace."""
        if not self.current_trace:
            return

        # Finish all open spans
        while self.span_stack:
            self.finish_span()

        self.current_trace = None
        self.current_span = None

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self.current_span

    def get_current_trace(self) -> Optional[Trace]:
        """Get the current active trace."""
        return self.current_trace


class DistributedTracer:
    """
    Main distributed tracing service.

    Features:
    - Lightweight span and trace management
    - Integration with correlation IDs
    - Automatic HTTP request tracing
    - Service-to-service call tracking
    - Performance metrics integration
    """

    def __init__(self, service_name: str = "sales-assistant"):
        """
        Initialize distributed tracer.

        Args:
            service_name: Name of this service
        """
        self.settings = get_settings()
        self.service_name = service_name
        self.logger = get_enhanced_logger(self.__class__.__name__)
        self.enabled = self.settings.enable_distributed_tracing
        self.sample_rate = self.settings.trace_sample_rate

        # Store completed traces for a short time (for debugging/export)
        self.completed_traces: List[Trace] = []
        self.max_completed_traces = 100

        # Context storage
        self._context = TraceContext()

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        if not self.enabled:
            return False

        import random

        return random.random() < self.sample_rate

    @contextmanager
    def trace(
        self,
        operation_name: str,
        span_kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Context manager for tracing an operation.

        Args:
            operation_name: Name of the operation
            span_kind: Kind of span
            attributes: Initial attributes
            tags: Initial tags
        """
        if not self._should_sample():
            # If not sampling, just yield without tracing
            yield None
            return

        span = self.start_span(operation_name, span_kind)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        if tags:
            for key, value in tags.items():
                span.set_tag(key, value)

        # Add correlation ID as an attribute
        correlation_id = CorrelationContext.get_correlation_id()
        if correlation_id:
            span.set_attribute("correlation_id", correlation_id)

        try:
            yield span
            span.set_status(SpanStatus.OK)

        except Exception as e:
            span.record_exception(e)
            raise

        finally:
            self.finish_span(span)

    def start_trace(self, operation_name: str, trace_id: Optional[str] = None) -> Optional[Trace]:
        """Start a new trace."""
        if not self._should_sample():
            return None

        # Use correlation ID as trace ID if available
        if not trace_id:
            trace_id = CorrelationContext.get_correlation_id()

        trace = self._context.start_trace(operation_name, self.service_name, trace_id)

        self.logger.info(
            f"Started trace: {operation_name}",
            extra={
                "event_type": "trace_started",
                "trace_id": trace.trace_id,
                "operation_name": operation_name,
            },
        )

        return trace

    def start_span(
        self,
        operation_name: str,
        span_kind: SpanKind = SpanKind.INTERNAL,
        parent_span: Optional[Span] = None,
    ) -> Optional[Span]:
        """Start a new span."""
        if not self.enabled:
            return None

        span = self._context.start_span(operation_name, self.service_name, span_kind, parent_span)

        self.logger.debug(
            f"Started span: {operation_name}",
            extra={
                "event_type": "span_started",
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "operation_name": operation_name,
            },
        )

        return span

    def finish_span(self, span: Optional[Span] = None):
        """Finish a span."""
        if not self.enabled:
            return

        self._context.finish_span(span)

        if span:
            self.logger.debug(
                f"Finished span: {span.operation_name}",
                extra={
                    "event_type": "span_finished",
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "duration_ms": span.duration_ms,
                    "status": span.status.value,
                },
            )

    def finish_trace(self):
        """Finish the current trace."""
        if not self.enabled:
            return

        trace = self._context.current_trace
        if trace:
            self._context.finish_trace()

            # Store completed trace
            self.completed_traces.append(trace)
            if len(self.completed_traces) > self.max_completed_traces:
                self.completed_traces.pop(0)

            self.logger.info(
                f"Finished trace: {trace.trace_id}",
                extra={
                    "event_type": "trace_finished",
                    "trace_id": trace.trace_id,
                    "duration_ms": trace.duration_ms,
                    "span_count": trace.span_count,
                    "error_count": trace.error_count,
                },
            )

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self._context.get_current_span()

    def get_current_trace(self) -> Optional[Trace]:
        """Get the current active trace."""
        return self._context.get_current_trace()

    def get_completed_traces(self, limit: int = 50) -> List[Trace]:
        """Get recently completed traces."""
        return self.completed_traces[-limit:]

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get a summary of tracing activity."""
        current_trace = self.get_current_trace()
        completed_count = len(self.completed_traces)

        return {
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "service_name": self.service_name,
            "current_trace_id": current_trace.trace_id if current_trace else None,
            "current_span_count": current_trace.span_count if current_trace else 0,
            "completed_traces_count": completed_count,
            "recent_traces": [
                {
                    "trace_id": trace.trace_id,
                    "duration_ms": trace.duration_ms,
                    "span_count": trace.span_count,
                    "error_count": trace.error_count,
                }
                for trace in self.completed_traces[-10:]
            ],
        }


# Global tracer instance
_distributed_tracer: Optional[DistributedTracer] = None


def get_tracer() -> DistributedTracer:
    """Get the global distributed tracer instance."""
    global _distributed_tracer

    if _distributed_tracer is None:
        _distributed_tracer = DistributedTracer()

    return _distributed_tracer


def trace_operation(
    operation_name: str,
    span_kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """
    Decorator for tracing operations.

    Args:
        operation_name: Name of the operation
        span_kind: Kind of span
        attributes: Initial attributes
        tags: Initial tags
    """

    def decorator(func):
        from functools import wraps

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.trace(operation_name, span_kind, attributes, tags) as span:
                if span:
                    span.set_attribute("function", func.__name__)
                    span.set_attribute("module", func.__module__)
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.trace(operation_name, span_kind, attributes, tags) as span:
                if span:
                    span.set_attribute("function", func.__name__)
                    span.set_attribute("module", func.__module__)
                return func(*args, **kwargs)

        # Return appropriate wrapper
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
