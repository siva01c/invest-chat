"""Health check endpoints."""

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from assistant.api.middleware.enhanced_security import EnhancedSecurityMiddleware
from assistant.core.models import HealthCheckResponse
from assistant.core.services.cached_knowledge_service import CachedKnowledgeService
from assistant.infrastructure.cache.advanced_cache_manager import get_cache_manager
from assistant.infrastructure.cache.cache_config_manager import get_cache_config_manager
from assistant.infrastructure.cache.cache_invalidation_service import get_invalidation_service
from assistant.infrastructure.cache.query_cache import QueryCache
from assistant.infrastructure.cache.redis_client import get_redis_client
from assistant.infrastructure.database.async_vector_store import AsyncVectorStore
from assistant.infrastructure.database.connection_pool import get_connection_pool
from assistant.infrastructure.database.optimized_vector_store import OptimizedVectorStore
from assistant.infrastructure.database.vector_store import VectorStore
from assistant.infrastructure.monitoring.performance_monitor import get_performance_monitor

router = APIRouter()


@router.get("/health", response_class=JSONResponse)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring

    Returns:
        Service health status
    """
    try:
        # Test database connection
        store = VectorStore()
        await store.get_all_records()

        response = HealthCheckResponse(
            status="healthy", service="assistant", timestamp=int(time.time()), version="1.0.0"
        )

        return response.model_dump()

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@router.get("/health/redis", response_class=JSONResponse)
async def redis_health_check() -> Dict[str, Any]:
    """
    Redis health check endpoint

    Returns:
        Redis service health status
    """
    try:
        redis_client = await get_redis_client()
        health_result = await redis_client.health_check()

        return {"service": "redis", "timestamp": int(time.time()), **health_result}

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "service": "redis",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time()),
            },
        )


@router.get("/health/detailed", response_class=JSONResponse)
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check endpoint for all services

    Returns:
        Comprehensive health status
    """
    timestamp = int(time.time())
    results = {"timestamp": timestamp, "overall_status": "healthy", "services": {}}

    # Check database
    try:
        store = VectorStore()
        await store.get_all_records()
        results["services"]["database"] = {
            "status": "healthy",
            "service": "chromadb",
            "message": "Database connection successful",
        }
    except Exception as e:
        results["services"]["database"] = {
            "status": "unhealthy",
            "service": "chromadb",
            "error": str(e),
        }
        results["overall_status"] = "degraded"

    # Check Redis
    try:
        redis_client = await get_redis_client()
        redis_health = await redis_client.health_check()
        results["services"]["cache"] = {"service": "redis", **redis_health}
        if redis_health["status"] != "healthy":
            results["overall_status"] = "degraded"
    except Exception as e:
        results["services"]["cache"] = {"status": "unhealthy", "service": "redis", "error": str(e)}
        results["overall_status"] = "degraded"

    # Set appropriate status code
    status_code = 200 if results["overall_status"] == "healthy" else 503

    if status_code == 503:
        raise HTTPException(status_code=status_code, detail=results)

    return results


@router.get("/health/performance", response_class=JSONResponse)
async def performance_health_check() -> Dict[str, Any]:
    """
    Performance-optimized health check endpoint

    Returns:
        Health status with performance metrics
    """
    try:
        timestamp = int(time.time())
        results = {"timestamp": timestamp, "overall_status": "healthy", "services": {}}

        # Check async vector store
        try:
            async_vector_store = AsyncVectorStore()
            vector_health = await async_vector_store.health_check()
            results["services"]["async_vector_store"] = vector_health
            if vector_health["status"] != "healthy":
                results["overall_status"] = "degraded"
        except Exception as e:
            results["services"]["async_vector_store"] = {"status": "unhealthy", "error": str(e)}
            results["overall_status"] = "degraded"

        # Check query cache
        try:
            redis_client = await get_redis_client()
            query_cache = QueryCache(redis_client)
            cache_stats = await query_cache.get_cache_stats()
            results["services"]["query_cache"] = {"status": "healthy", "stats": cache_stats}
        except Exception as e:
            results["services"]["query_cache"] = {"status": "unhealthy", "error": str(e)}
            results["overall_status"] = "degraded"

        # Check cached knowledge service
        try:
            knowledge_service = CachedKnowledgeService()
            service_health = await knowledge_service.get_service_health()
            results["services"]["cached_knowledge_service"] = service_health
            if service_health["status"] != "healthy":
                results["overall_status"] = "degraded"
        except Exception as e:
            results["services"]["cached_knowledge_service"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            results["overall_status"] = "degraded"

        # Set appropriate status code
        status_code = 200 if results["overall_status"] == "healthy" else 503

        if status_code == 503:
            raise HTTPException(status_code=status_code, detail=results)

        return results

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"timestamp": int(time.time()), "overall_status": "unhealthy", "error": str(e)},
        )


@router.get("/health/cache", response_class=JSONResponse)
async def cache_health_check() -> Dict[str, Any]:
    """
    Cache system health check endpoint

    Returns:
        Cache system health status
    """
    try:
        # Check Redis cache
        redis_client = await get_redis_client()
        redis_health = await redis_client.health_check()

        # Check query cache
        query_cache = QueryCache(redis_client)
        cache_stats = await query_cache.get_cache_stats()

        return {
            "service": "cache_system",
            "timestamp": int(time.time()),
            "redis": redis_health,
            "query_cache": {"status": "healthy", "statistics": cache_stats},
            "overall_status": "healthy" if redis_health["status"] == "healthy" else "degraded",
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "service": "cache_system",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time()),
            },
        )


@router.post("/admin/cache/clear", response_class=JSONResponse)
async def clear_all_caches() -> Dict[str, Any]:
    """
    Clear all system caches (admin endpoint)

    Returns:
        Cache clearing results
    """
    try:
        # Clear knowledge service caches
        knowledge_service = CachedKnowledgeService()
        clear_results = await knowledge_service.clear_all_caches()

        return {
            "action": "clear_all_caches",
            "timestamp": int(time.time()),
            "status": "completed",
            **clear_results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "action": "clear_all_caches",
                "status": "failed",
                "error": str(e),
                "timestamp": int(time.time()),
            },
        )


@router.get("/csrf-token", response_class=JSONResponse)
async def get_csrf_token() -> Dict[str, Any]:
    """
    Generate CSRF token for frontend applications

    Returns:
        CSRF token that can be used in subsequent requests
    """
    try:
        # Generate a new CSRF token
        csrf_token = EnhancedSecurityMiddleware.generate_csrf_token()

        return {
            "csrf_token": csrf_token,
            "timestamp": int(time.time()),
            "expires_in": 3600,  # 1 hour
            "usage": "Include this token in X-CSRF-Token header for state-changing requests",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate CSRF token",
                "message": str(e),
                "timestamp": int(time.time()),
            },
        )


@router.get("/health/database", response_class=JSONResponse)
async def database_health_check() -> Dict[str, Any]:
    """
    Database optimization health check endpoint

    Returns:
        Database health status with performance metrics
    """
    try:
        # Check optimized vector store
        optimized_store = OptimizedVectorStore()
        store_health = await optimized_store.health_check()

        # Check connection pool
        connection_pool = await get_connection_pool()
        pool_health = await connection_pool.health_check()

        return {
            "service": "database_optimization",
            "timestamp": int(time.time()),
            "optimized_vector_store": store_health,
            "connection_pool": pool_health,
            "overall_status": (
                "healthy"
                if (store_health["status"] == "healthy" and pool_health["status"] == "healthy")
                else "degraded"
            ),
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "service": "database_optimization",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time()),
            },
        )


@router.get("/health/performance", response_class=JSONResponse)
async def performance_monitoring_health() -> Dict[str, Any]:
    """
    Performance monitoring health check endpoint

    Returns:
        Performance monitoring status and metrics
    """
    try:
        # Get performance monitor
        monitor = await get_performance_monitor()
        performance_report = monitor.get_performance_report()

        # Get optimization suggestions
        suggestions = await monitor.get_optimization_suggestions()

        return {
            "service": "performance_monitoring",
            "timestamp": int(time.time()),
            "performance_report": performance_report,
            "optimization_suggestions": suggestions,
            "monitoring_active": True,
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "service": "performance_monitoring",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time()),
            },
        )


@router.get("/admin/performance/metrics/{metric_name}", response_class=JSONResponse)
async def get_metric_history(metric_name: str, hours: int = 1) -> Dict[str, Any]:
    """
    Get historical data for a specific performance metric

    Args:
        metric_name: Name of the metric to retrieve
        hours: Number of hours of history to return

    Returns:
        Metric history and summary statistics
    """
    try:
        monitor = await get_performance_monitor()

        # Get metric history
        history = monitor.get_metric_history(metric_name, hours)
        summary = monitor.get_metric_summary(metric_name, hours)

        return {
            "metric_name": metric_name,
            "time_range_hours": hours,
            "summary": summary,
            "data_points": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "metadata": point.metadata,
                }
                for point in history
            ],
            "retrieved_at": int(time.time()),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Failed to retrieve metric history: {str(e)}",
                "metric_name": metric_name,
                "timestamp": int(time.time()),
            },
        )


@router.get("/admin/database/stats", response_class=JSONResponse)
async def get_database_stats() -> Dict[str, Any]:
    """
    Get comprehensive database performance statistics

    Returns:
        Database performance metrics and connection pool statistics
    """
    try:
        # Get optimized vector store metrics
        optimized_store = OptimizedVectorStore()
        performance_metrics = await optimized_store.get_performance_metrics()

        # Get connection pool statistics
        connection_pool = await get_connection_pool()
        pool_stats = await connection_pool.get_pool_stats()

        return {
            "service": "database_statistics",
            "timestamp": int(time.time()),
            "vector_store_metrics": performance_metrics,
            "connection_pool_stats": pool_stats,
            "database_optimization_enabled": True,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Failed to get database statistics: {str(e)}",
                "timestamp": int(time.time()),
            },
        )


@router.get("/health/cache", response_class=JSONResponse)
async def advanced_cache_health() -> Dict[str, Any]:
    """
    Advanced caching layer health check endpoint

    Returns:
        Comprehensive cache system health status
    """
    try:
        # Get cache managers
        default_cache = await get_cache_manager("default")
        response_cache = await get_cache_manager("response_cache")

        # Get cache statistics
        default_stats = await default_cache.get_cache_stats()
        response_stats = await response_cache.get_cache_stats()

        # Get invalidation service stats
        invalidation_service = await get_invalidation_service()
        invalidation_stats = await invalidation_service.get_invalidation_stats()

        # Get configuration manager stats
        config_manager = await get_cache_config_manager()
        config_summary = await config_manager.get_config_summary()

        return {
            "service": "advanced_caching_layer",
            "timestamp": int(time.time()),
            "cache_managers": {"default_cache": default_stats, "response_cache": response_stats},
            "invalidation_service": invalidation_stats,
            "configuration": config_summary,
            "overall_status": "healthy",
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "service": "advanced_caching_layer",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time()),
            },
        )


@router.get("/admin/cache/stats", response_class=JSONResponse)
async def get_cache_statistics() -> Dict[str, Any]:
    """
    Get comprehensive cache statistics

    Returns:
        Detailed cache performance metrics
    """
    try:
        # Get all cache manager stats
        cache_managers = ["default", "response_cache", "query_cache", "knowledge_cache"]
        cache_stats = {}

        for cache_name in cache_managers:
            try:
                cache_manager = await get_cache_manager(cache_name)
                stats = await cache_manager.get_cache_stats()
                cache_stats[cache_name] = stats
            except Exception as e:
                cache_stats[cache_name] = {"error": str(e)}

        # Get invalidation stats
        invalidation_service = await get_invalidation_service()
        invalidation_stats = await invalidation_service.get_invalidation_stats()

        # Get recent invalidation events
        recent_events = await invalidation_service.get_recent_events(limit=20)

        return {
            "cache_managers": cache_stats,
            "invalidation_service": invalidation_stats,
            "recent_invalidation_events": recent_events,
            "total_cache_managers": len(cache_managers),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Failed to get cache statistics: {str(e)}",
                "timestamp": int(time.time()),
            },
        )


@router.post("/admin/cache/invalidate", response_class=JSONResponse)
async def invalidate_cache_by_pattern(
    pattern: str, cache_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Invalidate cache entries by pattern

    Args:
        pattern: Pattern to match cache keys (supports wildcards)
        cache_names: Specific cache names to invalidate (optional)

    Returns:
        Invalidation results
    """
    try:
        invalidation_service = await get_invalidation_service()
        result = await invalidation_service.invalidate_by_pattern(
            pattern=pattern, cache_names=cache_names
        )

        return {"action": "invalidate_by_pattern", "timestamp": int(time.time()), **result}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "action": "invalidate_by_pattern",
                "error": str(e),
                "pattern": pattern,
                "timestamp": int(time.time()),
            },
        )
