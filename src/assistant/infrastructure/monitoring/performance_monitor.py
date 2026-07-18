"""Performance monitoring service for database and application metrics."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from assistant.config import get_settings
from assistant.core.logging import get_logger
from assistant.infrastructure.cache.redis_client import get_redis_client
from assistant.infrastructure.database.connection_pool import get_connection_pool


@dataclass
class PerformanceMetric:
    """Individual performance metric with timestamp."""

    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert when thresholds are exceeded."""

    metric_name: str
    threshold_type: str  # "max", "min", "rate"
    threshold_value: float
    current_value: float
    timestamp: datetime
    severity: str  # "warning", "critical"
    message: str


class PerformanceMonitor:
    """
    Comprehensive performance monitoring service.

    Features:
    - Real-time metric collection
    - Threshold-based alerting
    - Historical data retention
    - Performance trend analysis
    - Automatic optimization suggestions
    """

    def __init__(
        self,
        retention_hours: int = 24,
        alert_cooldown_minutes: int = 5,
        collection_interval_seconds: int = 30,
    ):
        """
        Initialize the performance monitor.

        Args:
            retention_hours: How long to keep historical metrics
            alert_cooldown_minutes: Minimum time between similar alerts
            collection_interval_seconds: How often to collect metrics
        """
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)

        self.retention_hours = retention_hours
        self.alert_cooldown_minutes = alert_cooldown_minutes
        self.collection_interval = collection_interval_seconds

        # Metric storage (in-memory with configurable retention)
        self._metrics: Dict[str, deque] = {}
        self._alerts: List[PerformanceAlert] = []
        self._last_alert_time: Dict[str, datetime] = {}

        # Monitoring control
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Performance thresholds
        self._thresholds = {
            "database_query_time_ms": {"max": 1000, "warning": 500},
            "cache_hit_rate": {"min": 0.7, "warning": 0.8},
            "connection_pool_utilization": {"max": 0.9, "warning": 0.75},
            "memory_usage_mb": {"max": 1024, "warning": 512},
            "error_rate": {"max": 0.05, "warning": 0.02},
        }

    def _should_alert(self, metric_name: str) -> bool:
        """Check if enough time has passed since last alert for this metric."""
        if metric_name not in self._last_alert_time:
            return True

        time_since_last = datetime.utcnow() - self._last_alert_time[metric_name]
        return time_since_last >= timedelta(minutes=self.alert_cooldown_minutes)

    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds and generate alerts."""
        if metric.name not in self._thresholds:
            return

        thresholds = self._thresholds[metric.name]
        alert = None

        # Check maximum thresholds
        if "max" in thresholds and metric.value > thresholds["max"]:
            alert = PerformanceAlert(
                metric_name=metric.name,
                threshold_type="max",
                threshold_value=thresholds["max"],
                current_value=metric.value,
                timestamp=metric.timestamp,
                severity="critical",
                message=f"{metric.name} exceeded maximum threshold: {metric.value} > {thresholds['max']}",
            )
        elif "warning" in thresholds and metric.value > thresholds["warning"]:
            alert = PerformanceAlert(
                metric_name=metric.name,
                threshold_type="warning",
                threshold_value=thresholds["warning"],
                current_value=metric.value,
                timestamp=metric.timestamp,
                severity="warning",
                message=f"{metric.name} exceeded warning threshold: {metric.value} > {thresholds['warning']}",
            )

        # Check minimum thresholds (for metrics like cache hit rate)
        elif "min" in thresholds and metric.value < thresholds["min"]:
            alert = PerformanceAlert(
                metric_name=metric.name,
                threshold_type="min",
                threshold_value=thresholds["min"],
                current_value=metric.value,
                timestamp=metric.timestamp,
                severity="critical",
                message=f"{metric.name} below minimum threshold: {metric.value} < {thresholds['min']}",
            )

        # Generate alert if needed
        if alert and self._should_alert(metric.name):
            self._alerts.append(alert)
            self._last_alert_time[metric.name] = alert.timestamp

            # Log the alert
            log_level = "warning" if alert.severity == "warning" else "error"
            getattr(self.logger, log_level)(alert.message)

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)

        for metric_name, metric_queue in self._metrics.items():
            # Remove old metrics from the front of the deque
            while metric_queue and metric_queue[0].timestamp < cutoff_time:
                metric_queue.popleft()

    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a performance metric.

        Args:
            name: Metric name
            value: Metric value
            metadata: Optional metadata for the metric
        """
        metric = PerformanceMetric(
            name=name, value=value, timestamp=datetime.utcnow(), metadata=metadata or {}
        )

        # Store metric
        if name not in self._metrics:
            self._metrics[name] = deque()

        self._metrics[name].append(metric)

        # Check thresholds
        self._check_thresholds(metric)

        # Periodic cleanup
        if len(self._metrics[name]) % 100 == 0:
            self._cleanup_old_metrics()

    async def _collect_database_metrics(self) -> Dict[str, Any]:
        """Collect database-related performance metrics."""
        metrics = {}

        try:
            # Connection pool metrics
            pool = await get_connection_pool()
            pool_stats = await pool.get_pool_stats()

            # Extract key metrics
            if pool_stats:
                current_stats = pool_stats.get("current_stats", {})
                pool_config = pool_stats.get("pool_config", {})

                # Connection pool utilization
                if current_stats.get("total_connections") and pool_config.get("max_connections"):
                    utilization = (
                        current_stats["total_connections"] / pool_config["max_connections"]
                    )
                    self.record_metric("connection_pool_utilization", utilization)
                    metrics["connection_pool_utilization"] = utilization

                # Query performance
                if current_stats.get("average_query_time_ms"):
                    self.record_metric(
                        "database_query_time_ms", current_stats["average_query_time_ms"]
                    )
                    metrics["database_query_time_ms"] = current_stats["average_query_time_ms"]

                # Connection counts
                metrics["active_connections"] = current_stats.get("active_connections", 0)
                metrics["idle_connections"] = current_stats.get("idle_connections", 0)
                metrics["total_queries"] = current_stats.get("queries_executed", 0)

        except Exception as e:
            self.logger.warning(f"Failed to collect database metrics: {str(e)}")
            self.record_metric("database_error_rate", 1.0)

        return metrics

    async def _collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect cache-related performance metrics."""
        metrics = {}

        try:
            # Redis metrics
            redis_client = await get_redis_client()
            if redis_client:
                redis_info = await redis_client.get_info()

                if redis_info:
                    # Memory usage
                    memory_usage_mb = redis_info.get("used_memory", 0) / (1024 * 1024)
                    self.record_metric("redis_memory_usage_mb", memory_usage_mb)
                    metrics["redis_memory_usage_mb"] = memory_usage_mb

                    # Connection count
                    metrics["redis_connected_clients"] = redis_info.get("connected_clients", 0)

                # Cache performance from application metrics
                # (Would be populated by the query cache during operations)
                metrics["cache_operations"] = redis_info.get("total_commands_processed", 0)

        except Exception as e:
            self.logger.warning(f"Failed to collect cache metrics: {str(e)}")

        return metrics

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level performance metrics."""
        metrics = {}

        try:
            import psutil

            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_usage_percent", memory.percent)
            metrics["system_memory_usage_percent"] = memory.percent

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system_cpu_usage_percent", cpu_percent)
            metrics["system_cpu_usage_percent"] = cpu_percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("system_disk_usage_percent", disk_percent)
            metrics["system_disk_usage_percent"] = disk_percent

        except ImportError:
            self.logger.debug("psutil not available for system metrics")
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {str(e)}")

        return metrics

    async def _monitoring_loop(self):
        """Main monitoring loop that collects metrics periodically."""
        while not self._stop_event.is_set():
            try:
                start_time = time.time()

                # Collect all metrics
                database_metrics = await self._collect_database_metrics()
                cache_metrics = await self._collect_cache_metrics()
                system_metrics = await self._collect_system_metrics()

                collection_time = time.time() - start_time
                self.record_metric("metric_collection_time_ms", collection_time * 1000)

                # Log summary periodically
                total_metrics = len(self._metrics)
                if total_metrics > 0 and total_metrics % 50 == 0:
                    self.logger.info(
                        f"Performance monitoring: {total_metrics} metric types, "
                        f"{len(self._alerts)} alerts, collection_time={collection_time:.3f}s"
                    )

                # Wait for next collection
                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.collection_interval)

    async def start_monitoring(self):
        """Start the performance monitoring service."""
        if self._monitoring_task and not self._monitoring_task.done():
            self.logger.warning("Monitoring already started")
            return

        self._stop_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info(
            f"Performance monitoring started: "
            f"interval={self.collection_interval}s, retention={self.retention_hours}h"
        )

    async def stop_monitoring(self):
        """Stop the performance monitoring service."""
        if self._monitoring_task:
            self._stop_event.set()
            self._monitoring_task.cancel()

            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

            self.logger.info("Performance monitoring stopped")

    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[PerformanceMetric]:
        """
        Get historical data for a specific metric.

        Args:
            metric_name: Name of the metric
            hours: Number of hours of history to return

        Returns:
            List of metrics within the time range
        """
        if metric_name not in self._metrics:
            return []

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self._metrics[metric_name] if m.timestamp >= cutoff_time]

    def get_metric_summary(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """
        Get statistical summary for a metric.

        Args:
            metric_name: Name of the metric
            hours: Number of hours to analyze

        Returns:
            Statistical summary of the metric
        """
        history = self.get_metric_history(metric_name, hours)

        if not history:
            return {"error": f"No data available for metric: {metric_name}"}

        values = [m.value for m in history]

        return {
            "metric_name": metric_name,
            "time_range_hours": hours,
            "data_points": len(values),
            "min_value": min(values),
            "max_value": max(values),
            "average": sum(values) / len(values),
            "latest_value": values[-1],
            "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "decreasing",
        }

    def get_active_alerts(self, hours: int = 1) -> List[PerformanceAlert]:
        """Get alerts from the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self._alerts if alert.timestamp >= cutoff_time]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_config": {
                "retention_hours": self.retention_hours,
                "collection_interval_seconds": self.collection_interval,
                "alert_cooldown_minutes": self.alert_cooldown_minutes,
            },
            "metric_summaries": {},
            "recent_alerts": self.get_active_alerts(hours=1),
            "system_health": "healthy",
        }

        # Generate summaries for key metrics
        key_metrics = [
            "database_query_time_ms",
            "connection_pool_utilization",
            "cache_hit_rate",
            "system_memory_usage_percent",
            "system_cpu_usage_percent",
        ]

        for metric_name in key_metrics:
            if metric_name in self._metrics:
                report["metric_summaries"][metric_name] = self.get_metric_summary(metric_name)

        # Determine overall system health
        critical_alerts = [a for a in report["recent_alerts"] if a.severity == "critical"]
        if critical_alerts:
            report["system_health"] = "critical"
        elif len(report["recent_alerts"]) > 0:
            report["system_health"] = "warning"

        return report

    async def get_optimization_suggestions(self) -> List[str]:
        """Generate performance optimization suggestions based on metrics."""
        suggestions = []

        # Analyze recent metrics for optimization opportunities
        try:
            # Database query performance
            query_time_summary = self.get_metric_summary("database_query_time_ms", hours=1)
            if query_time_summary.get("average", 0) > 500:
                suggestions.append(
                    "Database queries are slow (avg > 500ms). Consider optimizing queries, "
                    "adding more connections, or implementing better caching."
                )

            # Cache hit rate
            cache_summary = self.get_metric_summary("cache_hit_rate", hours=1)
            if cache_summary.get("average", 1.0) < 0.8:
                suggestions.append(
                    "Cache hit rate is low (< 80%). Consider increasing cache TTL, "
                    "optimizing cache keys, or pre-warming frequently accessed data."
                )

            # Connection pool utilization
            pool_summary = self.get_metric_summary("connection_pool_utilization", hours=1)
            if pool_summary.get("average", 0) > 0.8:
                suggestions.append(
                    "Connection pool utilization is high (> 80%). Consider increasing "
                    "max_connections or optimizing query patterns."
                )

            # System resources
            cpu_summary = self.get_metric_summary("system_cpu_usage_percent", hours=1)
            if cpu_summary.get("average", 0) > 80:
                suggestions.append(
                    "CPU usage is high (> 80%). Consider scaling horizontally "
                    "or optimizing CPU-intensive operations."
                )

            memory_summary = self.get_metric_summary("system_memory_usage_percent", hours=1)
            if memory_summary.get("average", 0) > 85:
                suggestions.append(
                    "Memory usage is high (> 85%). Consider increasing available memory "
                    "or optimizing memory-intensive operations."
                )

        except Exception as e:
            self.logger.warning(f"Failed to generate optimization suggestions: {str(e)}")
            suggestions.append("Unable to generate suggestions due to insufficient metrics data.")

        return suggestions


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


async def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor

    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        await _performance_monitor.start_monitoring()

    return _performance_monitor


async def stop_performance_monitoring():
    """Stop the global performance monitoring."""
    global _performance_monitor

    if _performance_monitor:
        await _performance_monitor.stop_monitoring()
        _performance_monitor = None
