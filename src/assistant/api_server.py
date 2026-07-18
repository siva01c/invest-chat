"""FastAPI application with enterprise-grade architecture.

This module provides the main FastAPI application with comprehensive middleware,
routing, and service integration for the Sales Assistant project.

Features:
    - Enterprise middleware stack (security, rate limiting, caching, observability)
    - Structured logging with correlation IDs
    - Background task processing with graceful shutdown
    - Comprehensive health monitoring and metrics
    - Production-ready configuration management

Example:
    Run the application:
        $ python -m assistant.api_server

    Or with uvicorn:
        $ uvicorn assistant.api_server:app --host 0.0.0.0 --port 5000
"""

import pathlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from assistant.api.middleware.correlation_middleware import CorrelationMiddleware
from assistant.api.middleware.enhanced_security import EnhancedSecurityMiddleware
from assistant.api.middleware.error_handler import ErrorHandlingMiddleware
from assistant.api.middleware.prometheus_middleware import PrometheusMiddleware
from assistant.api.middleware.rate_limit import RateLimitMiddleware
from assistant.api.middleware.response_cache import ResponseCacheMiddleware
from assistant.api.middleware.tracing_middleware import TracingMiddleware
from assistant.api.routes import (
    auth,
    chat,
    contact,
    dashboard,
    health,
    knowledge,
    mcp,
    metrics,
    observability,
    tasks,
    teams,
    tracing,
)
from assistant.config import get_settings
from assistant.core.config.services import cleanup_services, initialize_services
from assistant.infrastructure.observability import setup_correlation_aware_logging
from assistant.infrastructure.tasks.worker_manager import get_worker_manager

# Get application settings
settings = get_settings()

# Initialize dependency injection (sync parts only)
# Async initialization will happen in startup event

# Set up enhanced logging with correlation IDs
if settings.enable_structured_logging:
    setup_correlation_aware_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events.

    This context manager handles startup and shutdown events for the FastAPI application.
    It initializes services and background workers on startup, and cleans them up on shutdown.
    """
    # Startup logic
    try:
        await initialize_services()
    except Exception as e:
        print(f"Warning: Failed to initialize services: {e}")

    try:
        worker_manager = get_worker_manager()
        await worker_manager.start_default_workers()
    except Exception as e:
        print(f"Warning: Failed to start background workers: {e}")

    yield

    # Shutdown logic
    try:
        worker_manager = get_worker_manager()
        await worker_manager.stop_all_workers(timeout=30.0)
    except Exception as e:
        print(f"Warning: Error stopping background workers: {e}")

    cleanup_services()


# Create FastAPI app with configuration
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    docs_url=None if settings.disable_docs else "/docs",
    redoc_url=None if settings.disable_docs else "/redoc",
    openapi_url=None if settings.disable_docs else "/openapi.json",
    lifespan=lifespan,
)

# Add error handling middleware (first in chain)
app.add_middleware(ErrorHandlingMiddleware)

# Add correlation ID middleware (very early for request tracing)
if settings.enable_structured_logging:
    app.add_middleware(
        CorrelationMiddleware,
        correlation_header_name=settings.correlation_id_header,
        response_header_name=settings.correlation_id_header,
        enable_request_logging=True,
    )

# Add distributed tracing middleware (after correlation for trace context)
if settings.enable_distributed_tracing:
    app.add_middleware(
        TracingMiddleware,
        enable_tracing=True,
        exclude_paths={"/health", "/metrics", "/favicon.ico", "/tracing"},
    )

# Add Prometheus metrics middleware (after correlation for accurate tracing)
if settings.enable_prometheus_metrics:
    app.add_middleware(
        PrometheusMiddleware,
        enable_request_size_metrics=settings.prometheus_enable_request_size,
        enable_response_size_metrics=settings.prometheus_enable_response_size,
        track_in_progress_requests=settings.prometheus_track_in_progress,
        exclude_paths=set(settings.prometheus_exclude_paths),
    )

# Add response caching middleware (after metrics for accurate measurement)
if settings.enable_response_caching:
    app.add_middleware(
        ResponseCacheMiddleware,
        default_ttl_seconds=settings.response_cache_ttl,
        max_cache_size_mb=settings.response_cache_size_mb,
        enable_compression=settings.enable_cache_compression,
    )

# Add rate limiting middleware (before security middleware)
app.add_middleware(RateLimitMiddleware, enable_rate_limiting=settings.enable_redis_rate_limiting)

# Configure CORS from settings (must be first for preflight handling)
app.add_middleware(CORSMiddleware, **settings.cors_config)

# Add enhanced security middleware with comprehensive protection
app.add_middleware(
    EnhancedSecurityMiddleware,
    enable_csrf=settings.enable_csrf_protection,
    enable_xss_protection=True,
    enable_content_validation=True,
    max_request_size=settings.max_request_size,
    trusted_hosts=settings.trusted_hosts,
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Make sure static directory exists
static_dir = pathlib.Path("static")
if not static_dir.exists():
    static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(contact.router)
app.include_router(health.router)
app.include_router(knowledge.router)
app.include_router(metrics.router)
app.include_router(tracing.router)
app.include_router(observability.router)
app.include_router(dashboard.router)
app.include_router(tasks.router)
app.include_router(mcp.router)
app.include_router(teams.router)


def main() -> None:
    """Main entry point for the Sales Assistant application.

    This function starts the FastAPI application using uvicorn server
    with configuration loaded from environment variables and settings.

    The server configuration includes:
        - Host and port from application settings
        - Log level configuration
        - Automatic module reloading in development mode

    Environment Variables:
        HOST: Server host (default: 0.0.0.0)
        PORT: Server port (default: 5000)
        LOG_LEVEL: Logging level (default: INFO)

    Example:
        Run the application:
            $ python -m assistant.api_server

        Or directly:
            $ python src/assistant/api_server.py
    """
    uvicorn.run(
        "assistant.api_server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
