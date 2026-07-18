"""Prometheus metrics endpoints for observability."""

import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST

from assistant.config import get_settings
from assistant.core.logging import get_logger
from assistant.infrastructure.cache.redis_client import get_redis_client
from assistant.infrastructure.database.connection_pool import get_connection_pool
from assistant.infrastructure.monitoring.performance_monitor import get_performance_monitor
from assistant.infrastructure.observability import get_prometheus_metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])
logger = get_logger(__name__)
settings = get_settings()


@router.get("", response_class=PlainTextResponse)
@router.get("/", response_class=PlainTextResponse)
async def get_prometheus_metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus formatted metrics for scraping
    """
    try:
        # Update system metrics before returning
        await _update_system_metrics()

        metrics = get_prometheus_metrics()
        metrics_text = metrics.get_metrics_text()

        return PlainTextResponse(content=metrics_text, media_type=CONTENT_TYPE_LATEST)

    except Exception as e:
        logger.error(f"Failed to generate Prometheus metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics generation failed: {str(e)}")


@router.get("/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a human-readable summary of current metrics.

    Returns:
        JSON summary of metrics status and health
    """
    try:
        metrics = get_prometheus_metrics()
        summary = metrics.get_metrics_summary()

        # Add performance monitoring data
        try:
            performance_monitor = await get_performance_monitor()
            performance_report = performance_monitor.get_performance_report()
            summary["performance_monitoring"] = performance_report
        except Exception as e:
            logger.warning(f"Failed to get performance report: {str(e)}")
            summary["performance_monitoring"] = {"error": str(e)}

        return {
            "timestamp": time.time(),
            "service": "sales-assistant",
            "metrics_summary": summary,
            "status": "healthy",
        }

    except Exception as e:
        logger.error(f"Failed to generate metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics summary generation failed: {str(e)}")


@router.post("/update")
async def force_metrics_update() -> Dict[str, Any]:
    """
    Force an update of all system metrics.

    Returns:
        Update status and current metrics
    """
    try:
        start_time = time.time()

        # Update all system metrics
        await _update_system_metrics()
        await _update_application_metrics()

        update_duration = time.time() - start_time

        return {
            "status": "success",
            "update_duration_seconds": update_duration,
            "timestamp": time.time(),
            "message": "All metrics updated successfully",
        }

    except Exception as e:
        logger.error(f"Failed to update metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics update failed: {str(e)}")


async def _update_system_metrics():
    """Update system-level metrics."""
    try:
        import psutil

        # Get system metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Get process metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss
        process_cpu_time = process.cpu_times().user + process.cpu_times().system

        metrics = get_prometheus_metrics()

        # Update system metrics
        metrics.update_system_metrics(
            cpu_percent=cpu_percent,
            memory_bytes=memory.used,
            memory_total=memory.total,
            disk_bytes=disk.used,
            disk_total=disk.total,
        )

        # Update process metrics
        metrics.update_process_metrics(memory_bytes=process_memory, cpu_time=process_cpu_time)

    except ImportError:
        logger.debug("psutil not available for system metrics")
    except Exception as e:
        logger.warning(f"Failed to update system metrics: {str(e)}")


async def _update_application_metrics():
    """Update application-specific metrics."""
    metrics = get_prometheus_metrics()

    # Update Redis metrics
    try:
        redis_client = await get_redis_client()
        if redis_client:
            redis_info = await redis_client.get_info()
            if redis_info:
                metrics.update_redis_metrics(
                    memory_bytes=redis_info.get("used_memory", 0),
                    connected_clients=redis_info.get("connected_clients", 0),
                    commands_processed=redis_info.get("total_commands_processed", 0),
                )
                # Update service health
                metrics.update_service_health("redis", True)
            else:
                metrics.update_service_health("redis", False)
    except Exception as e:
        logger.warning(f"Failed to update Redis metrics: {str(e)}")
        metrics.update_service_health("redis", False)

    # Update database metrics
    try:
        connection_pool = await get_connection_pool()
        if connection_pool:
            pool_stats = await connection_pool.get_pool_stats()
            if pool_stats and "current_stats" in pool_stats:
                current_stats = pool_stats["current_stats"]
                pool_config = pool_stats.get("pool_config", {})

                active_connections = current_stats.get("active_connections", 0)
                idle_connections = current_stats.get("idle_connections", 0)
                total_connections = current_stats.get("total_connections", 0)
                max_connections = pool_config.get("max_connections", 1)

                utilization = total_connections / max_connections if max_connections > 0 else 0

                metrics.update_database_connection_metrics(
                    active=active_connections, idle=idle_connections, utilization=utilization
                )
                # Update service health
                metrics.update_service_health("database", True)
            else:
                metrics.update_service_health("database", False)
    except Exception as e:
        logger.warning(f"Failed to update database metrics: {str(e)}")
        metrics.update_service_health("database", False)

    # Update overall application health
    metrics.update_service_health("application", True)


@router.get("/health")
async def metrics_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the metrics system.

    Returns:
        Health status of the metrics collection system
    """
    try:
        metrics = get_prometheus_metrics()
        summary = metrics.get_metrics_summary()

        # Check if metrics collection is healthy
        collection_errors = summary.get("collection_errors", {})
        total_errors = sum(collection_errors.values())

        health_status = "healthy" if total_errors < 10 else "degraded"

        return {
            "service": "prometheus_metrics",
            "status": health_status,
            "timestamp": time.time(),
            "collection_errors": collection_errors,
            "total_errors": total_errors,
            "collector_count": summary.get("registry_collector_count", 0),
        }

    except Exception as e:
        logger.error(f"Metrics health check failed: {str(e)}")
        return {
            "service": "prometheus_metrics",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time(),
        }
