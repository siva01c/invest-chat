"""Enhanced observability health check endpoints."""

import asyncio
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from assistant.config import get_settings
from assistant.infrastructure.cache.redis_client import get_redis_client
from assistant.infrastructure.database.connection_pool import get_connection_pool
from assistant.infrastructure.monitoring.performance_monitor import get_performance_monitor
from assistant.infrastructure.observability import (
    CorrelationContext,
    get_enhanced_logger,
    get_prometheus_metrics,
    get_tracer,
)

router = APIRouter(prefix="/observability", tags=["observability"])
logger = get_enhanced_logger(__name__)
settings = get_settings()


@router.get("/health")
async def comprehensive_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint with full observability status.

    Returns:
        Complete health status of all observability components
    """
    try:
        start_time = time.time()
        health_results = {
            "timestamp": time.time(),
            "service": "sales-assistant",
            "version": "1.0.0",
            "overall_status": "healthy",
            "components": {},
            "observability": {},
            "performance": {},
        }

        # Check all components in parallel for faster response
        component_checks = await asyncio.gather(
            _check_prometheus_metrics(),
            _check_structured_logging(),
            _check_distributed_tracing(),
            _check_performance_monitoring(),
            _check_redis_health(),
            _check_database_health(),
            return_exceptions=True,
        )

        # Process component check results
        component_names = [
            "prometheus_metrics",
            "structured_logging",
            "distributed_tracing",
            "performance_monitoring",
            "redis",
            "database",
        ]

        for i, result in enumerate(component_checks):
            component_name = component_names[i]

            if isinstance(result, Exception):
                health_results["components"][component_name] = {
                    "status": "unhealthy",
                    "error": str(result),
                }
                health_results["overall_status"] = "degraded"
            else:
                health_results["components"][component_name] = result
                if result.get("status") != "healthy":
                    health_results["overall_status"] = "degraded"

        # Add observability configuration
        health_results["observability"] = {
            "structured_logging_enabled": settings.enable_structured_logging,
            "prometheus_metrics_enabled": settings.enable_prometheus_metrics,
            "distributed_tracing_enabled": settings.enable_distributed_tracing,
            "correlation_tracking": bool(CorrelationContext.get_correlation_id()),
            "trace_sample_rate": (
                settings.trace_sample_rate if settings.enable_distributed_tracing else 0
            ),
        }

        # Add performance summary
        try:
            performance_monitor = await get_performance_monitor()
            performance_report = performance_monitor.get_performance_report()
            health_results["performance"] = {
                "monitoring_active": True,
                "system_health": performance_report.get("system_health", "unknown"),
                "recent_alerts_count": len(performance_report.get("recent_alerts", [])),
            }
        except Exception as e:
            health_results["performance"] = {"monitoring_active": False, "error": str(e)}

        # Add timing
        duration_ms = (time.time() - start_time) * 1000
        health_results["health_check_duration_ms"] = round(duration_ms, 2)

        # Log the health check
        logger.log_performance_metric(
            "health_check", duration_ms, health_results["overall_status"] == "healthy"
        )

        # Set appropriate HTTP status code
        status_code = 200 if health_results["overall_status"] == "healthy" else 503

        if status_code == 503:
            raise HTTPException(status_code=status_code, detail=health_results)

        return health_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "timestamp": time.time(),
                "overall_status": "unhealthy",
                "error": str(e),
                "service": "sales-assistant",
            },
        )


@router.get("/readiness")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint - indicates if the service is ready to handle requests.

    Returns:
        Service readiness status
    """
    try:
        readiness_checks = {
            "timestamp": time.time(),
            "service": "sales-assistant",
            "ready": True,
            "checks": {},
        }

        # Essential readiness checks
        checks = await asyncio.gather(
            _check_redis_connectivity(), _check_database_connectivity(), return_exceptions=True
        )

        check_names = ["redis", "database"]

        for i, result in enumerate(checks):
            check_name = check_names[i]

            if isinstance(result, Exception):
                readiness_checks["checks"][check_name] = {"ready": False, "error": str(result)}
                readiness_checks["ready"] = False
            else:
                readiness_checks["checks"][check_name] = result
                if not result.get("ready", False):
                    readiness_checks["ready"] = False

        status_code = 200 if readiness_checks["ready"] else 503

        if status_code == 503:
            raise HTTPException(status_code=status_code, detail=readiness_checks)

        return readiness_checks

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail={"timestamp": time.time(), "ready": False, "error": str(e)}
        )


@router.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check endpoint - indicates if the service is alive and not stuck.

    Returns:
        Service liveness status
    """
    try:
        return {
            "timestamp": time.time(),
            "service": "sales-assistant",
            "alive": True,
            "uptime_seconds": time.time() - _get_service_start_time(),
            "correlation_id": CorrelationContext.get_correlation_id(),
        }

    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail={"timestamp": time.time(), "alive": False, "error": str(e)}
        )


@router.get("/metrics-health")
async def metrics_health_check() -> Dict[str, Any]:
    """
    Health check specifically for metrics collection systems.

    Returns:
        Metrics system health status
    """
    try:
        metrics_health = {
            "timestamp": time.time(),
            "service": "metrics_collection",
            "status": "healthy",
            "systems": {},
        }

        # Check Prometheus metrics
        try:
            prometheus_metrics = get_prometheus_metrics()
            summary = prometheus_metrics.get_metrics_summary()

            metrics_health["systems"]["prometheus"] = {
                "status": "healthy",
                "collector_count": summary.get("registry_collector_count", 0),
                "collection_errors": summary.get("collection_errors", {}),
                "service_health": summary.get("service_health", {}),
            }
        except Exception as e:
            metrics_health["systems"]["prometheus"] = {"status": "unhealthy", "error": str(e)}
            metrics_health["status"] = "degraded"

        # Check performance monitoring
        try:
            performance_monitor = await get_performance_monitor()
            perf_report = performance_monitor.get_performance_report()

            metrics_health["systems"]["performance_monitoring"] = {
                "status": "healthy",
                "monitoring_config": perf_report.get("monitoring_config", {}),
                "system_health": perf_report.get("system_health", "unknown"),
            }
        except Exception as e:
            metrics_health["systems"]["performance_monitoring"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            metrics_health["status"] = "degraded"

        return metrics_health

    except Exception as e:
        logger.error(f"Metrics health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"timestamp": time.time(), "status": "unhealthy", "error": str(e)},
        )


@router.get("/dependency-status")
async def dependency_status_check() -> Dict[str, Any]:
    """
    Check the status of all external dependencies.

    Returns:
        Status of all external dependencies
    """
    try:
        dependencies = {
            "timestamp": time.time(),
            "service": "dependency_status",
            "overall_status": "healthy",
            "dependencies": {},
        }

        # Check all dependencies
        dependency_checks = await asyncio.gather(
            _check_redis_detailed(),
            _check_database_detailed(),
            _check_openai_connection(),
            return_exceptions=True,
        )

        dependency_names = ["redis", "database", "openai"]

        for i, result in enumerate(dependency_checks):
            dep_name = dependency_names[i]

            if isinstance(result, Exception):
                dependencies["dependencies"][dep_name] = {
                    "status": "unhealthy",
                    "error": str(result),
                    "critical": True,
                }
                dependencies["overall_status"] = "degraded"
            else:
                dependencies["dependencies"][dep_name] = result
                if result.get("status") != "healthy":
                    dependencies["overall_status"] = "degraded"

        return dependencies

    except Exception as e:
        logger.error(f"Dependency status check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"timestamp": time.time(), "overall_status": "unhealthy", "error": str(e)},
        )


# Helper functions for health checks


async def _check_prometheus_metrics() -> Dict[str, Any]:
    """Check Prometheus metrics health."""
    try:
        metrics = get_prometheus_metrics()
        summary = metrics.get_metrics_summary()

        total_errors = sum(summary.get("collection_errors", {}).values())
        status = "healthy" if total_errors < 10 else "degraded"

        return {
            "status": status,
            "collector_count": summary.get("registry_collector_count", 0),
            "total_collection_errors": total_errors,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def _check_structured_logging() -> Dict[str, Any]:
    """Check structured logging health."""
    try:
        # Test logger functionality
        test_logger = get_enhanced_logger("health_check")
        test_logger.debug("Health check test log")

        return {
            "status": "healthy",
            "correlation_tracking": bool(CorrelationContext.get_correlation_id()),
            "enhanced_logging": True,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def _check_distributed_tracing() -> Dict[str, Any]:
    """Check distributed tracing health."""
    try:
        tracer = get_tracer()
        summary = tracer.get_trace_summary()

        return {
            "status": "healthy",
            "enabled": summary.get("enabled", False),
            "sample_rate": summary.get("sample_rate", 0),
            "completed_traces": summary.get("completed_traces_count", 0),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def _check_performance_monitoring() -> Dict[str, Any]:
    """Check performance monitoring health."""
    try:
        monitor = await get_performance_monitor()
        report = monitor.get_performance_report()

        return {
            "status": "healthy",
            "system_health": report.get("system_health", "unknown"),
            "recent_alerts": len(report.get("recent_alerts", [])),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis health."""
    try:
        redis_client = await get_redis_client()
        health_result = await redis_client.health_check()

        return {
            "status": health_result.get("status", "unknown"),
            "connection_pool": health_result.get("connection_pool", {}),
            "server_info": health_result.get("server_info", {}),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def _check_database_health() -> Dict[str, Any]:
    """Check database health."""
    try:
        connection_pool = await get_connection_pool()
        pool_stats = await connection_pool.get_pool_stats()

        if pool_stats:
            current_stats = pool_stats.get("current_stats", {})
            return {
                "status": "healthy",
                "active_connections": current_stats.get("active_connections", 0),
                "total_connections": current_stats.get("total_connections", 0),
            }
        else:
            return {"status": "unhealthy", "error": "No pool stats available"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def _check_redis_connectivity() -> Dict[str, Any]:
    """Basic Redis connectivity check."""
    try:
        redis_client = await get_redis_client()
        await redis_client.ping()
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "error": str(e)}


async def _check_database_connectivity() -> Dict[str, Any]:
    """Basic database connectivity check."""
    try:
        connection_pool = await get_connection_pool()
        health = await connection_pool.health_check()
        return {"ready": health.get("status") == "healthy"}
    except Exception as e:
        return {"ready": False, "error": str(e)}


async def _check_redis_detailed() -> Dict[str, Any]:
    """Detailed Redis status check."""
    try:
        redis_client = await get_redis_client()
        info = await redis_client.get_info()

        return {
            "status": "healthy",
            "memory_usage_mb": info.get("used_memory", 0) / (1024 * 1024),
            "connected_clients": info.get("connected_clients", 0),
            "uptime_seconds": info.get("uptime_in_seconds", 0),
            "critical": True,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "critical": True}


async def _check_database_detailed() -> Dict[str, Any]:
    """Detailed database status check."""
    try:
        connection_pool = await get_connection_pool()
        stats = await connection_pool.get_pool_stats()

        if stats:
            current = stats.get("current_stats", {})
            config = stats.get("pool_config", {})

            utilization = 0
            if config.get("max_connections", 0) > 0:
                utilization = current.get("total_connections", 0) / config["max_connections"]

            return {
                "status": "healthy",
                "pool_utilization": round(utilization, 2),
                "active_connections": current.get("active_connections", 0),
                "critical": True,
            }
        else:
            return {"status": "unhealthy", "error": "No statistics available", "critical": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "critical": True}


async def _check_openai_connection() -> Dict[str, Any]:
    """Check OpenAI API connectivity."""
    try:
        # For now, just check if the API key is configured
        if settings.openai_api_key:
            return {
                "status": "healthy",
                "configured": True,
                "model": settings.openai_model,
                "critical": True,
            }
        else:
            return {
                "status": "unhealthy",
                "error": "OpenAI API key not configured",
                "critical": True,
            }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "critical": True}


def _get_service_start_time() -> float:
    """Get service start time (placeholder - would need to track this)."""
    # This would typically be tracked when the service starts
    # For now, return a reasonable estimate
    return time.time() - 3600  # Assume service has been running for 1 hour
