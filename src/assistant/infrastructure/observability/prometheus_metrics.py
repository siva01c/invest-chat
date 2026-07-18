"""Prometheus metrics collection service for comprehensive observability."""

import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, List, Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from assistant.config import get_settings
from assistant.core.logging import get_logger


class PrometheusMetrics:
    """
    Comprehensive Prometheus metrics collection service.

    Features:
    - HTTP request metrics with detailed labels
    - Database operation metrics
    - Cache performance metrics
    - System resource metrics
    - Business logic metrics
    - Custom application metrics
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus metrics service.

        Args:
            registry: Optional custom registry, uses default if None
        """
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self.registry = registry or CollectorRegistry()

        # Initialize all metrics
        self._init_http_metrics()
        self._init_database_metrics()
        self._init_cache_metrics()
        self._init_system_metrics()
        self._init_business_metrics()
        self._init_application_info()

        # Track metric collection health
        self._collection_errors = defaultdict(int)

    def _init_http_metrics(self):
        """Initialize HTTP-related metrics."""
        # HTTP request duration histogram
        self.http_request_duration = Histogram(
            name="http_request_duration_seconds",
            documentation="Time spent processing HTTP requests",
            labelnames=["method", "endpoint", "status_code"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )

        # HTTP request counter
        self.http_requests_total = Counter(
            name="http_requests_total",
            documentation="Total number of HTTP requests",
            labelnames=["method", "endpoint", "status_code"],
            registry=self.registry,
        )

        # HTTP request size histogram
        self.http_request_size_bytes = Histogram(
            name="http_request_size_bytes",
            documentation="Size of HTTP requests",
            labelnames=["method", "endpoint"],
            buckets=(64, 256, 1024, 4096, 16384, 65536, 262144, 1048576),
            registry=self.registry,
        )

        # HTTP response size histogram
        self.http_response_size_bytes = Histogram(
            name="http_response_size_bytes",
            documentation="Size of HTTP responses",
            labelnames=["method", "endpoint"],
            buckets=(64, 256, 1024, 4096, 16384, 65536, 262144, 1048576),
            registry=self.registry,
        )

        # Currently active requests
        self.http_requests_in_progress = Gauge(
            name="http_requests_in_progress",
            documentation="Number of HTTP requests currently being processed",
            labelnames=["method", "endpoint"],
            registry=self.registry,
        )

    def _init_database_metrics(self):
        """Initialize database-related metrics."""
        # Database query duration
        self.db_query_duration = Histogram(
            name="database_query_duration_seconds",
            documentation="Time spent executing database queries",
            labelnames=["operation", "collection", "status"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )

        # Database operations counter
        self.db_operations_total = Counter(
            name="database_operations_total",
            documentation="Total number of database operations",
            labelnames=["operation", "collection", "status"],
            registry=self.registry,
        )

        # Connection pool metrics
        self.db_connections_active = Gauge(
            name="database_connections_active",
            documentation="Number of active database connections",
            registry=self.registry,
        )

        self.db_connections_idle = Gauge(
            name="database_connections_idle",
            documentation="Number of idle database connections",
            registry=self.registry,
        )

        self.db_connection_pool_utilization = Gauge(
            name="database_connection_pool_utilization_ratio",
            documentation="Database connection pool utilization ratio",
            registry=self.registry,
        )

        # Vector search specific metrics
        self.vector_search_results = Histogram(
            name="vector_search_results_count",
            documentation="Number of results returned by vector search",
            buckets=(1, 3, 5, 10, 20, 50, 100),
            registry=self.registry,
        )

        self.vector_search_score = Histogram(
            name="vector_search_score",
            documentation="Relevance scores from vector search",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )

    def _init_cache_metrics(self):
        """Initialize cache-related metrics."""
        # Cache operations
        self.cache_operations_total = Counter(
            name="cache_operations_total",
            documentation="Total number of cache operations",
            labelnames=["operation", "cache_name", "result"],
            registry=self.registry,
        )

        # Cache hit rate
        self.cache_hit_rate = Gauge(
            name="cache_hit_rate_ratio",
            documentation="Cache hit rate ratio",
            labelnames=["cache_name"],
            registry=self.registry,
        )

        # Cache size metrics
        self.cache_size_bytes = Gauge(
            name="cache_size_bytes",
            documentation="Current cache size in bytes",
            labelnames=["cache_name", "level"],
            registry=self.registry,
        )

        self.cache_entries_count = Gauge(
            name="cache_entries_count",
            documentation="Number of entries in cache",
            labelnames=["cache_name", "level"],
            registry=self.registry,
        )

        # Cache operation duration
        self.cache_operation_duration = Histogram(
            name="cache_operation_duration_seconds",
            documentation="Time spent on cache operations",
            labelnames=["operation", "cache_name"],
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
            registry=self.registry,
        )

        # Cache evictions
        self.cache_evictions_total = Counter(
            name="cache_evictions_total",
            documentation="Total number of cache evictions",
            labelnames=["cache_name", "reason"],
            registry=self.registry,
        )

    def _init_system_metrics(self):
        """Initialize system resource metrics."""
        # CPU usage
        self.system_cpu_usage = Gauge(
            name="system_cpu_usage_percent",
            documentation="System CPU usage percentage",
            registry=self.registry,
        )

        # Memory usage
        self.system_memory_usage = Gauge(
            name="system_memory_usage_bytes",
            documentation="System memory usage in bytes",
            registry=self.registry,
        )

        self.system_memory_total = Gauge(
            name="system_memory_total_bytes",
            documentation="Total system memory in bytes",
            registry=self.registry,
        )

        # Disk usage
        self.system_disk_usage = Gauge(
            name="system_disk_usage_bytes",
            documentation="System disk usage in bytes",
            registry=self.registry,
        )

        self.system_disk_total = Gauge(
            name="system_disk_total_bytes",
            documentation="Total system disk space in bytes",
            registry=self.registry,
        )

        # Process metrics
        self.process_memory_usage = Gauge(
            name="process_memory_usage_bytes",
            documentation="Process memory usage in bytes",
            registry=self.registry,
        )

        self.process_cpu_time = Counter(
            name="process_cpu_time_seconds_total",
            documentation="Total process CPU time in seconds",
            registry=self.registry,
        )

        # Redis metrics
        self.redis_memory_usage = Gauge(
            name="redis_memory_usage_bytes",
            documentation="Redis memory usage in bytes",
            registry=self.registry,
        )

        self.redis_connected_clients = Gauge(
            name="redis_connected_clients",
            documentation="Number of Redis connected clients",
            registry=self.registry,
        )

        self.redis_commands_processed = Counter(
            name="redis_commands_processed_total",
            documentation="Total Redis commands processed",
            registry=self.registry,
        )

    def _init_business_metrics(self):
        """Initialize business logic specific metrics."""
        # Chat interactions
        self.chat_messages_total = Counter(
            name="chat_messages_total",
            documentation="Total number of chat messages processed",
            labelnames=["language", "category", "status"],
            registry=self.registry,
        )

        self.chat_response_time = Histogram(
            name="chat_response_time_seconds",
            documentation="Time to generate chat responses",
            labelnames=["language", "category"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry,
        )

        # Email operations
        self.emails_sent_total = Counter(
            name="emails_sent_total",
            documentation="Total number of emails sent",
            labelnames=["status"],
            registry=self.registry,
        )

        # Classification results
        self.message_classifications_total = Counter(
            name="message_classifications_total",
            documentation="Total number of message classifications",
            labelnames=["category", "confidence_level"],
            registry=self.registry,
        )

        # Knowledge base queries
        self.knowledge_queries_total = Counter(
            name="knowledge_queries_total",
            documentation="Total number of knowledge base queries",
            labelnames=["query_type", "has_results"],
            registry=self.registry,
        )

        self.knowledge_query_relevance = Histogram(
            name="knowledge_query_relevance_score",
            documentation="Relevance scores for knowledge base queries",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )

    def _init_application_info(self):
        """Initialize application information metrics."""
        self.application_info = Info(
            name="application", documentation="Application information", registry=self.registry
        )

        # Set application information
        self.application_info.info(
            {
                "version": "1.0.0",
                "environment": "development" if self.settings.debug else "production",
                "python_version": "3.11",
                "service_name": "sales-assistant",
            }
        )

        # Service uptime
        self.service_start_time = Gauge(
            name="service_start_time_seconds",
            documentation="Unix timestamp when the service started",
            registry=self.registry,
        )
        self.service_start_time.set_to_current_time()

        # Health status
        self.service_health = Gauge(
            name="service_health_status",
            documentation="Service health status (1=healthy, 0=unhealthy)",
            labelnames=["component"],
            registry=self.registry,
        )

    # Context managers and decorators for easy metric collection

    @contextmanager
    def track_http_request(self, method: str, endpoint: str, status_code: int = 200):
        """Context manager to track HTTP request metrics."""
        start_time = time.time()
        labels = {"method": method, "endpoint": endpoint}

        # Track request in progress
        self.http_requests_in_progress.labels(**labels).inc()

        try:
            yield
        finally:
            # Track request completion
            duration = time.time() - start_time
            labels_with_status = {**labels, "status_code": str(status_code)}

            self.http_request_duration.labels(**labels_with_status).observe(duration)
            self.http_requests_total.labels(**labels_with_status).inc()
            self.http_requests_in_progress.labels(**labels).dec()

    @contextmanager
    def track_database_operation(self, operation: str, collection: str = "default"):
        """Context manager to track database operation metrics."""
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception as e:
            status = "error"
            self.logger.warning(f"Database operation failed: {operation} on {collection}: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            labels = {"operation": operation, "collection": collection, "status": status}

            self.db_query_duration.labels(**labels).observe(duration)
            self.db_operations_total.labels(**labels).inc()

    @contextmanager
    def track_cache_operation(self, operation: str, cache_name: str):
        """Context manager to track cache operation metrics."""
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time
            labels = {"operation": operation, "cache_name": cache_name}
            self.cache_operation_duration.labels(**labels).observe(duration)

    def track_cache_hit(self, cache_name: str, hit: bool):
        """Track cache hit/miss."""
        result = "hit" if hit else "miss"
        self.cache_operations_total.labels(
            operation="get", cache_name=cache_name, result=result
        ).inc()

    def track_cache_eviction(self, cache_name: str, reason: str):
        """Track cache eviction."""
        self.cache_evictions_total.labels(cache_name=cache_name, reason=reason).inc()

    def track_chat_message(
        self, language: str, category: str, response_time: float, status: str = "success"
    ):
        """Track chat message processing."""
        labels = {"language": language, "category": category}

        self.chat_messages_total.labels(**labels, status=status).inc()
        if status == "success":
            self.chat_response_time.labels(**labels).observe(response_time)

    def track_email_sent(self, success: bool):
        """Track email sending."""
        status = "success" if success else "error"
        self.emails_sent_total.labels(status=status).inc()

    def track_message_classification(self, category: str, confidence: float):
        """Track message classification results."""
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        self.message_classifications_total.labels(
            category=category, confidence_level=confidence_level
        ).inc()

    def track_knowledge_query(
        self, query_type: str, has_results: bool, relevance_scores: List[float] = None
    ):
        """Track knowledge base query."""
        self.knowledge_queries_total.labels(
            query_type=query_type, has_results=str(has_results).lower()
        ).inc()

        if relevance_scores:
            for score in relevance_scores:
                self.knowledge_query_relevance.observe(score)

    def track_vector_search(self, result_count: int, scores: List[float]):
        """Track vector search metrics."""
        self.vector_search_results.observe(result_count)
        for score in scores:
            self.vector_search_score.observe(score)

    def update_system_metrics(
        self,
        cpu_percent: float,
        memory_bytes: int,
        memory_total: int,
        disk_bytes: int,
        disk_total: int,
    ):
        """Update system resource metrics."""
        self.system_cpu_usage.set(cpu_percent)
        self.system_memory_usage.set(memory_bytes)
        self.system_memory_total.set(memory_total)
        self.system_disk_usage.set(disk_bytes)
        self.system_disk_total.set(disk_total)

    def update_process_metrics(self, memory_bytes: int, cpu_time: float):
        """Update process-specific metrics."""
        self.process_memory_usage.set(memory_bytes)
        self.process_cpu_time.inc(cpu_time)

    def update_redis_metrics(
        self, memory_bytes: int, connected_clients: int, commands_processed: int
    ):
        """Update Redis metrics."""
        self.redis_memory_usage.set(memory_bytes)
        self.redis_connected_clients.set(connected_clients)
        self.redis_commands_processed.inc(commands_processed)

    def update_database_connection_metrics(self, active: int, idle: int, utilization: float):
        """Update database connection metrics."""
        self.db_connections_active.set(active)
        self.db_connections_idle.set(idle)
        self.db_connection_pool_utilization.set(utilization)

    def update_cache_metrics(
        self, cache_name: str, level: str, size_bytes: int, entries_count: int, hit_rate: float
    ):
        """Update cache metrics."""
        labels = {"cache_name": cache_name, "level": level}

        self.cache_size_bytes.labels(**labels).set(size_bytes)
        self.cache_entries_count.labels(**labels).set(entries_count)
        self.cache_hit_rate.labels(cache_name=cache_name).set(hit_rate)

    def update_service_health(self, component: str, healthy: bool):
        """Update service health status."""
        self.service_health.labels(component=component).set(1 if healthy else 0)

    def get_metrics_text(self, content_type: str = CONTENT_TYPE_LATEST) -> str:
        """
        Get metrics in Prometheus text format.

        Args:
            content_type: Content type for the response

        Returns:
            Metrics in Prometheus format
        """
        try:
            return generate_latest(self.registry).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Failed to generate metrics: {str(e)}")
            self._collection_errors["generate_latest"] += 1
            return f"# Error generating metrics: {str(e)}\\n"

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics for debugging."""
        return {
            "timestamp": time.time(),
            "collection_errors": dict(self._collection_errors),
            "registry_collector_count": len(list(self.registry._collector_to_names.keys())),
            "service_health": {
                "application": "healthy",
                "metrics_collection": (
                    "healthy" if sum(self._collection_errors.values()) < 10 else "degraded"
                ),
            },
        }


# Global Prometheus metrics instance
_prometheus_metrics: Optional[PrometheusMetrics] = None


def get_prometheus_metrics() -> PrometheusMetrics:
    """Get the global Prometheus metrics instance."""
    global _prometheus_metrics

    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()

    return _prometheus_metrics


def metrics_decorator(operation: str, collection: str = "default"):
    """
    Decorator to automatically track database operations.

    Args:
        operation: Type of operation (query, insert, update, delete)
        collection: Collection or table name
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_prometheus_metrics()
            with metrics.track_database_operation(operation, collection):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_prometheus_metrics()
            with metrics.track_database_operation(operation, collection):
                return func(*args, **kwargs)

        return (
            async_wrapper
            if hasattr(func, "__code__") and func.__code__.co_flags & 0x80
            else sync_wrapper
        )

    return decorator
