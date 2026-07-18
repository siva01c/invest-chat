"""Distributed tracing endpoints for observability."""

import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from assistant.config import get_settings
from assistant.infrastructure.observability.correlation_context import CorrelationContext
from assistant.infrastructure.observability.distributed_tracing import get_tracer
from assistant.infrastructure.observability.enhanced_logging import get_enhanced_logger

router = APIRouter(prefix="/tracing", tags=["tracing"])
logger = get_enhanced_logger(__name__)
settings = get_settings()


@router.get("/status")
async def get_tracing_status() -> Dict[str, Any]:
    """
    Get distributed tracing status and configuration.

    Returns:
        Tracing system status and configuration
    """
    try:
        tracer = get_tracer()
        summary = tracer.get_trace_summary()

        return {
            "service": "distributed_tracing",
            "timestamp": time.time(),
            "configuration": {
                "enabled": settings.enable_distributed_tracing,
                "sample_rate": settings.trace_sample_rate,
                "service_name": tracer.service_name,
            },
            "current_state": summary,
            "correlation_id": CorrelationContext.get_correlation_id(),
        }

    except Exception as e:
        logger.error(f"Failed to get tracing status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tracing status retrieval failed: {str(e)}")


@router.get("/traces")
async def get_recent_traces(
    limit: int = Query(default=10, le=100, description="Maximum number of traces to return"),
    include_spans: bool = Query(default=False, description="Include span details"),
) -> Dict[str, Any]:
    """
    Get recently completed traces.

    Args:
        limit: Maximum number of traces to return
        include_spans: Include detailed span information

    Returns:
        List of recent traces with optional span details
    """
    try:
        tracer = get_tracer()
        traces = tracer.get_completed_traces(limit)

        if include_spans:
            # Return full trace data including spans
            traces_data = [trace.to_dict() for trace in traces]
        else:
            # Return summary data only
            traces_data = [
                {
                    "trace_id": trace.trace_id,
                    "service_name": trace.service_name,
                    "start_time": trace.start_time.isoformat(),
                    "end_time": trace.end_time.isoformat() if trace.end_time else None,
                    "duration_ms": trace.duration_ms,
                    "span_count": trace.span_count,
                    "error_count": trace.error_count,
                    "root_span_id": trace.root_span_id,
                }
                for trace in traces
            ]

        return {
            "traces": traces_data,
            "total_count": len(traces),
            "include_spans": include_spans,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to get recent traces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trace retrieval failed: {str(e)}")


@router.get("/traces/{trace_id}")
async def get_trace_by_id(trace_id: str) -> Dict[str, Any]:
    """
    Get a specific trace by ID.

    Args:
        trace_id: Trace ID to retrieve

    Returns:
        Complete trace data including all spans
    """
    try:
        tracer = get_tracer()
        traces = tracer.get_completed_traces(100)  # Get more traces to search

        # Find the trace
        target_trace = None
        for trace in traces:
            if trace.trace_id == trace_id:
                target_trace = trace
                break

        if not target_trace:
            raise HTTPException(status_code=404, detail=f"Trace with ID {trace_id} not found")

        return {"trace": target_trace.to_dict(), "timestamp": time.time()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trace {trace_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trace retrieval failed: {str(e)}")


@router.get("/current")
async def get_current_trace_info() -> Dict[str, Any]:
    """
    Get information about the current active trace.

    Returns:
        Current trace and span information
    """
    try:
        tracer = get_tracer()
        current_trace = tracer.get_current_trace()
        current_span = tracer.get_current_span()
        correlation_id = CorrelationContext.get_correlation_id()

        if not current_trace:
            return {
                "active_trace": None,
                "active_span": None,
                "correlation_id": correlation_id,
                "message": "No active trace",
                "timestamp": time.time(),
            }

        trace_info = {
            "trace_id": current_trace.trace_id,
            "service_name": current_trace.service_name,
            "start_time": current_trace.start_time.isoformat(),
            "span_count": current_trace.span_count,
            "error_count": current_trace.error_count,
        }

        span_info = None
        if current_span:
            span_info = {
                "span_id": current_span.span_id,
                "parent_span_id": current_span.parent_span_id,
                "operation_name": current_span.operation_name,
                "start_time": current_span.start_time.isoformat(),
                "status": current_span.status.value,
                "kind": current_span.kind.value,
            }

        return {
            "active_trace": trace_info,
            "active_span": span_info,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to get current trace info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Current trace info retrieval failed: {str(e)}"
        )


@router.get("/analytics")
async def get_trace_analytics() -> Dict[str, Any]:
    """
    Get analytics about recent tracing activity.

    Returns:
        Trace analytics and performance metrics
    """
    try:
        tracer = get_tracer()
        traces = tracer.get_completed_traces(100)

        if not traces:
            return {
                "analytics": {
                    "total_traces": 0,
                    "average_duration_ms": 0,
                    "error_rate": 0,
                    "traces_by_service": {},
                    "slowest_operations": [],
                    "error_operations": [],
                },
                "timestamp": time.time(),
            }

        # Calculate analytics
        total_traces = len(traces)
        total_duration = sum(trace.duration_ms or 0 for trace in traces)
        total_errors = sum(trace.error_count for trace in traces)

        average_duration = total_duration / total_traces if total_traces > 0 else 0
        error_rate = (
            total_errors / sum(trace.span_count for trace in traces) if total_traces > 0 else 0
        )

        # Group by service
        traces_by_service = {}
        for trace in traces:
            service = trace.service_name
            if service not in traces_by_service:
                traces_by_service[service] = 0
            traces_by_service[service] += 1

        # Find slowest operations
        all_spans = []
        for trace in traces:
            all_spans.extend(trace.spans)

        slowest_spans = sorted(
            [span for span in all_spans if span.duration_ms],
            key=lambda s: s.duration_ms,
            reverse=True,
        )[:10]

        slowest_operations = [
            {
                "operation_name": span.operation_name,
                "duration_ms": span.duration_ms,
                "trace_id": span.trace_id,
                "span_id": span.span_id,
            }
            for span in slowest_spans
        ]

        # Find error operations
        error_spans = [span for span in all_spans if span.error_details]
        error_operations = [
            {
                "operation_name": span.operation_name,
                "error_type": span.error_details.get("exception.type"),
                "error_message": span.error_details.get("exception.message"),
                "trace_id": span.trace_id,
                "span_id": span.span_id,
            }
            for span in error_spans[:10]
        ]

        return {
            "analytics": {
                "total_traces": total_traces,
                "average_duration_ms": round(average_duration, 2),
                "error_rate": round(error_rate, 4),
                "traces_by_service": traces_by_service,
                "slowest_operations": slowest_operations,
                "error_operations": error_operations,
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to get trace analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Trace analytics retrieval failed: {str(e)}")


@router.post("/test")
async def create_test_trace() -> Dict[str, Any]:
    """
    Create a test trace for demonstration purposes.

    Returns:
        Information about the created test trace
    """
    try:
        tracer = get_tracer()

        # Create a test trace
        trace = tracer.start_trace("test_operation")
        if not trace:
            return {"message": "Tracing disabled or not sampled", "timestamp": time.time()}

        # Add some test spans
        with tracer.trace("database_query", attributes={"table": "users", "query_type": "select"}):
            time.sleep(0.001)  # Simulate work

        with tracer.trace(
            "external_api_call", attributes={"endpoint": "/api/test", "method": "GET"}
        ):
            time.sleep(0.002)  # Simulate work

        # Finish the trace
        tracer.finish_trace()

        return {
            "message": "Test trace created successfully",
            "trace_id": trace.trace_id,
            "span_count": trace.span_count,
            "duration_ms": trace.duration_ms,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to create test trace: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test trace creation failed: {str(e)}")
