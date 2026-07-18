"""Performance monitoring dashboards for observability."""

import json
import time
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from assistant.config import get_settings
from assistant.infrastructure.cache.redis_client import get_redis_client
from assistant.infrastructure.database.connection_pool import get_connection_pool
from assistant.infrastructure.monitoring.performance_monitor import get_performance_monitor
from assistant.infrastructure.observability import (
    get_enhanced_logger,
    get_prometheus_metrics,
    get_tracer,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
logger = get_enhanced_logger(__name__)
settings = get_settings()


@router.get("/", response_class=HTMLResponse)
async def main_dashboard() -> str:
    """
    Main performance monitoring dashboard.

    Returns:
        HTML dashboard with real-time performance metrics
    """
    try:
        # Get dashboard data
        dashboard_data = await get_dashboard_data()

        # Generate HTML dashboard
        html_content = _generate_dashboard_html(dashboard_data)

        return html_content

    except Exception as e:
        logger.error(f"Failed to generate dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")


@router.get("/data")
async def get_dashboard_data() -> Dict[str, Any]:
    """
    Get dashboard data in JSON format for API consumption.

    Returns:
        Complete dashboard data including metrics, health, and performance
    """
    try:
        dashboard_data = {
            "timestamp": time.time(),
            "service": "sales-assistant",
            "overview": {},
            "metrics": {},
            "performance": {},
            "health": {},
            "traces": {},
            "alerts": [],
        }

        # Get overview data
        dashboard_data["overview"] = await _get_overview_data()

        # Get metrics data
        dashboard_data["metrics"] = await _get_metrics_data()

        # Get performance data
        dashboard_data["performance"] = await _get_performance_data()

        # Get health data
        dashboard_data["health"] = await _get_health_data()

        # Get traces data
        dashboard_data["traces"] = await _get_traces_data()

        # Get alerts
        dashboard_data["alerts"] = await _get_alerts_data()

        return dashboard_data

    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard data retrieval failed: {str(e)}")


@router.get("/metrics-overview")
async def get_metrics_overview() -> Dict[str, Any]:
    """
    Get a high-level overview of all metrics.

    Returns:
        Metrics overview with key performance indicators
    """
    try:
        # Get Prometheus metrics summary
        prometheus_metrics = get_prometheus_metrics()
        metrics_summary = prometheus_metrics.get_metrics_summary()

        # Get performance monitoring data
        performance_monitor = await get_performance_monitor()
        performance_report = performance_monitor.get_performance_report()

        # Get tracer summary
        tracer = get_tracer()
        trace_summary = tracer.get_trace_summary()

        return {
            "timestamp": time.time(),
            "prometheus": {
                "collectors": metrics_summary.get("registry_collector_count", 0),
                "collection_errors": sum(metrics_summary.get("collection_errors", {}).values()),
                "service_health": metrics_summary.get("service_health", {}),
            },
            "performance_monitoring": {
                "system_health": performance_report.get("system_health", "unknown"),
                "recent_alerts": len(performance_report.get("recent_alerts", [])),
                "metric_summaries": len(performance_report.get("metric_summaries", {})),
            },
            "distributed_tracing": {
                "enabled": trace_summary.get("enabled", False),
                "sample_rate": trace_summary.get("sample_rate", 0),
                "completed_traces": trace_summary.get("completed_traces_count", 0),
                "current_trace_active": trace_summary.get("current_trace_id") is not None,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get metrics overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics overview retrieval failed: {str(e)}")


@router.get("/performance-chart")
async def get_performance_chart_data(
    metric_name: str = Query(..., description="Name of the metric to chart"),
    hours: int = Query(default=1, le=24, description="Hours of data to include"),
) -> Dict[str, Any]:
    """
    Get performance chart data for a specific metric.

    Args:
        metric_name: Name of the metric to chart
        hours: Number of hours of data to include

    Returns:
        Chart data with timestamps and values
    """
    try:
        performance_monitor = await get_performance_monitor()

        # Get metric history
        history = performance_monitor.get_metric_history(metric_name, hours)
        summary = performance_monitor.get_metric_summary(metric_name, hours)

        # Format for charting
        chart_data = {
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
            "chart_config": {
                "type": "line",
                "x_axis": "timestamp",
                "y_axis": "value",
                "title": f"{metric_name} - Last {hours} hour(s)",
            },
        }

        return chart_data

    except Exception as e:
        logger.error(f"Failed to get chart data for {metric_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chart data retrieval failed: {str(e)}")


@router.get("/alerts")
async def get_alerts_dashboard() -> Dict[str, Any]:
    """
    Get alerts dashboard data.

    Returns:
        Current alerts and alert history
    """
    try:
        performance_monitor = await get_performance_monitor()

        # Get active alerts
        active_alerts = performance_monitor.get_active_alerts(hours=1)
        all_alerts = performance_monitor.get_active_alerts(hours=24)

        # Categorize alerts by severity
        critical_alerts = [a for a in active_alerts if a.severity == "critical"]
        warning_alerts = [a for a in active_alerts if a.severity == "warning"]

        return {
            "timestamp": time.time(),
            "summary": {
                "total_active": len(active_alerts),
                "critical_count": len(critical_alerts),
                "warning_count": len(warning_alerts),
                "last_24h_count": len(all_alerts),
            },
            "active_alerts": [
                {
                    "metric_name": alert.metric_name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "threshold_value": alert.threshold_value,
                    "current_value": alert.current_value,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in active_alerts
            ],
            "alert_history": [
                {
                    "metric_name": alert.metric_name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in all_alerts[-20:]  # Last 20 alerts
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get alerts dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Alerts dashboard retrieval failed: {str(e)}")


# Helper functions for dashboard data


async def _get_overview_data() -> Dict[str, Any]:
    """Get high-level overview data."""
    try:
        # Service uptime (placeholder)
        uptime_seconds = 3600  # Would track actual uptime

        # Get basic stats
        prometheus_metrics = get_prometheus_metrics()
        metrics_summary = prometheus_metrics.get_metrics_summary()

        return {
            "service_name": "sales-assistant",
            "version": "1.0.0",
            "uptime_seconds": uptime_seconds,
            "status": "healthy",
            "observability": {
                "metrics_enabled": settings.enable_prometheus_metrics,
                "logging_enabled": settings.enable_structured_logging,
                "tracing_enabled": settings.enable_distributed_tracing,
            },
            "collection_errors": sum(metrics_summary.get("collection_errors", {}).values()),
        }
    except Exception as e:
        return {"error": str(e)}


async def _get_metrics_data() -> Dict[str, Any]:
    """Get metrics data for dashboard."""
    try:
        prometheus_metrics = get_prometheus_metrics()

        # Generate sample metrics text and parse key metrics
        metrics_text = prometheus_metrics.get_metrics_text()

        # Count different types of metrics
        lines = metrics_text.split("\n")
        help_lines = [line for line in lines if line.startswith("# HELP")]

        metric_types = {}
        for line in help_lines:
            if "http_" in line:
                metric_types["HTTP Metrics"] = metric_types.get("HTTP Metrics", 0) + 1
            elif "database_" in line or "db_" in line:
                metric_types["Database Metrics"] = metric_types.get("Database Metrics", 0) + 1
            elif "cache_" in line:
                metric_types["Cache Metrics"] = metric_types.get("Cache Metrics", 0) + 1
            elif "system_" in line:
                metric_types["System Metrics"] = metric_types.get("System Metrics", 0) + 1
            else:
                metric_types["Other Metrics"] = metric_types.get("Other Metrics", 0) + 1

        return {
            "total_metrics": len(help_lines),
            "metric_categories": metric_types,
            "metrics_size_bytes": len(metrics_text),
            "collection_status": "active",
        }
    except Exception as e:
        return {"error": str(e)}


async def _get_performance_data() -> Dict[str, Any]:
    """Get performance monitoring data."""
    try:
        performance_monitor = await get_performance_monitor()
        performance_report = performance_monitor.get_performance_report()

        # Get key metrics summaries
        key_metrics = [
            "database_query_time_ms",
            "connection_pool_utilization",
            "system_memory_usage_percent",
            "system_cpu_usage_percent",
        ]

        metrics_data = {}
        for metric_name in key_metrics:
            summary = performance_monitor.get_metric_summary(metric_name, hours=1)
            if "error" not in summary:
                metrics_data[metric_name] = {
                    "current": summary.get("latest_value", 0),
                    "average": summary.get("average", 0),
                    "max": summary.get("max_value", 0),
                    "data_points": summary.get("data_points", 0),
                }

        return {
            "system_health": performance_report.get("system_health", "unknown"),
            "monitoring_active": True,
            "recent_alerts": len(performance_report.get("recent_alerts", [])),
            "key_metrics": metrics_data,
        }
    except Exception as e:
        return {"error": str(e)}


async def _get_health_data() -> Dict[str, Any]:
    """Get health status data."""
    try:
        health_checks = {
            "database": {"status": "unknown"},
            "redis": {"status": "unknown"},
            "metrics": {"status": "unknown"},
            "tracing": {"status": "unknown"},
        }

        # Check Redis
        try:
            redis_client = await get_redis_client()
            await redis_client.ping()
            health_checks["redis"]["status"] = "healthy"
        except Exception:
            health_checks["redis"]["status"] = "unhealthy"

        # Check Database
        try:
            connection_pool = await get_connection_pool()
            pool_health = await connection_pool.health_check()
            health_checks["database"]["status"] = pool_health.get("status", "unknown")
        except Exception:
            health_checks["database"]["status"] = "unhealthy"

        # Check Metrics
        try:
            prometheus_metrics = get_prometheus_metrics()
            metrics_summary = prometheus_metrics.get_metrics_summary()
            total_errors = sum(metrics_summary.get("collection_errors", {}).values())
            health_checks["metrics"]["status"] = "healthy" if total_errors < 10 else "degraded"
        except Exception:
            health_checks["metrics"]["status"] = "unhealthy"

        # Check Tracing
        try:
            tracer = get_tracer()
            health_checks["tracing"]["status"] = "healthy"
        except Exception:
            health_checks["tracing"]["status"] = "unhealthy"

        return health_checks
    except Exception as e:
        return {"error": str(e)}


async def _get_traces_data() -> Dict[str, Any]:
    """Get distributed tracing data."""
    try:
        tracer = get_tracer()
        trace_summary = tracer.get_trace_summary()
        recent_traces = tracer.get_completed_traces(10)

        return {
            "enabled": trace_summary.get("enabled", False),
            "sample_rate": trace_summary.get("sample_rate", 0),
            "completed_traces": trace_summary.get("completed_traces_count", 0),
            "current_trace_active": trace_summary.get("current_trace_id") is not None,
            "recent_traces": [
                {
                    "trace_id": trace.trace_id[:8] + "...",
                    "duration_ms": trace.duration_ms,
                    "span_count": trace.span_count,
                    "error_count": trace.error_count,
                    "start_time": trace.start_time.isoformat(),
                }
                for trace in recent_traces
            ],
        }
    except Exception as e:
        return {"error": str(e)}


async def _get_alerts_data() -> List[Dict[str, Any]]:
    """Get alerts data."""
    try:
        performance_monitor = await get_performance_monitor()
        active_alerts = performance_monitor.get_active_alerts(hours=1)

        return [
            {
                "metric_name": alert.metric_name,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "threshold_value": alert.threshold_value,
                "current_value": alert.current_value,
            }
            for alert in active_alerts[-5:]  # Latest 5 alerts
        ]
    except Exception as e:
        return [{"error": str(e)}]


def _generate_dashboard_html(dashboard_data: Dict[str, Any]) -> str:
    """Generate HTML dashboard content."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales Assistant - Performance Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .status-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .status-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .status-healthy {{
            border-left: 4px solid #4CAF50;
        }}
        .status-warning {{
            border-left: 4px solid #FF9800;
        }}
        .status-error {{
            border-left: 4px solid #F44336;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #666;
        }}
        .metric-label {{
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
        }}
        .alerts-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .alert-item {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid #F44336;
        }}
        .refresh-info {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }}
        .json-data {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin-top: 20px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 400px;
            overflow-y: auto;
        }}
    </style>
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {{
            window.location.reload();
        }}, 30000);
    </script>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Sales Assistant Performance Dashboard</h1>
            <p>Real-time monitoring and observability</p>
            <p><strong>Last Updated:</strong> {datetime.fromtimestamp(dashboard_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="status-grid">
            <div class="status-card status-healthy">
                <h3>Service Overview</h3>
                <div class="metric-value">{dashboard_data['overview'].get('status', 'Unknown').title()}</div>
                <div class="metric-label">Service Status</div>
                <p>Uptime: {dashboard_data['overview'].get('uptime_seconds', 0)//3600}h {(dashboard_data['overview'].get('uptime_seconds', 0)%3600)//60}m</p>
            </div>

            <div class="status-card status-healthy">
                <h3>Metrics Collection</h3>
                <div class="metric-value">{dashboard_data['metrics'].get('total_metrics', 0)}</div>
                <div class="metric-label">Total Metrics</div>
                <p>Collection Errors: {dashboard_data['overview'].get('collection_errors', 0)}</p>
            </div>

            <div class="status-card status-{'healthy' if dashboard_data['performance'].get('system_health') == 'healthy' else 'warning'}">
                <h3>System Health</h3>
                <div class="metric-value">{dashboard_data['performance'].get('system_health', 'Unknown').title()}</div>
                <div class="metric-label">Performance Status</div>
                <p>Active Alerts: {dashboard_data['performance'].get('recent_alerts', 0)}</p>
            </div>

            <div class="status-card status-{'healthy' if dashboard_data['traces'].get('enabled') else 'warning'}">
                <h3>Distributed Tracing</h3>
                <div class="metric-value">{'Enabled' if dashboard_data['traces'].get('enabled') else 'Disabled'}</div>
                <div class="metric-label">Tracing Status</div>
                <p>Completed Traces: {dashboard_data['traces'].get('completed_traces', 0)}</p>
            </div>
        </div>

        <div class="status-grid">
            <div class="status-card status-{'healthy' if dashboard_data['health'].get('database', {}).get('status') == 'healthy' else 'error'}">
                <h3>Database</h3>
                <div class="metric-value">{dashboard_data['health'].get('database', {}).get('status', 'Unknown').title()}</div>
                <div class="metric-label">Connection Status</div>
            </div>

            <div class="status-card status-{'healthy' if dashboard_data['health'].get('redis', {}).get('status') == 'healthy' else 'error'}">
                <h3>Redis Cache</h3>
                <div class="metric-value">{dashboard_data['health'].get('redis', {}).get('status', 'Unknown').title()}</div>
                <div class="metric-label">Cache Status</div>
            </div>

            <div class="status-card status-{'healthy' if dashboard_data['health'].get('metrics', {}).get('status') == 'healthy' else 'warning'}">
                <h3>Metrics System</h3>
                <div class="metric-value">{dashboard_data['health'].get('metrics', {}).get('status', 'Unknown').title()}</div>
                <div class="metric-label">Collection Status</div>
            </div>

            <div class="status-card status-{'healthy' if dashboard_data['health'].get('tracing', {}).get('status') == 'healthy' else 'warning'}">
                <h3>Tracing System</h3>
                <div class="metric-value">{dashboard_data['health'].get('tracing', {}).get('status', 'Unknown').title()}</div>
                <div class="metric-label">Tracing Status</div>
            </div>
        </div>

        {'<div class="alerts-section"><h3>Active Alerts</h3>' + ''.join(['<div class="alert-item"><strong>' + alert.get('metric_name', 'Unknown') + '</strong>: ' + alert.get('message', 'No message') + '</div>' for alert in dashboard_data.get('alerts', [])]) + '</div>' if dashboard_data.get('alerts') else ''}

        <div class="refresh-info">
            <p>Dashboard auto-refreshes every 30 seconds | <a href="/dashboard/data">View Raw Data</a> | <a href="/metrics">Prometheus Metrics</a> | <a href="/observability/health">Health Checks</a></p>
        </div>

        <div class="json-data">
            <h4>Dashboard Data (JSON)</h4>
            <pre>{json.dumps(dashboard_data, indent=2, default=str)}</pre>
        </div>
    </div>
</body>
</html>
"""
