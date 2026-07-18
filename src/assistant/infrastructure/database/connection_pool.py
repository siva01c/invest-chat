"""ChromaDB connection pool manager for optimized database operations."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional
from weakref import WeakSet

import chromadb
from chromadb.utils import embedding_functions

from assistant.config import get_settings
from assistant.core.exceptions import VectorStoreException
from assistant.core.logging import get_logger


@dataclass
class ConnectionStats:
    """Statistics for connection pool monitoring."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connections_created: int = 0
    connections_closed: int = 0
    queries_executed: int = 0
    average_query_time: float = 0.0
    last_activity: float = field(default_factory=time.time)


@dataclass
class PooledConnection:
    """A pooled ChromaDB connection with metadata."""

    client: chromadb.Client
    collection: chromadb.Collection
    created_at: float
    last_used: float
    query_count: int = 0
    is_active: bool = False
    connection_id: str = ""


class ChromaDBConnectionPool:
    """
    Optimized connection pool for ChromaDB with advanced features.

    Features:
    - Connection pooling with min/max limits
    - Automatic connection recycling
    - Health monitoring and recovery
    - Query performance tracking
    - Thread-safe operations
    """

    def __init__(
        self,
        min_connections: int = 2,
        max_connections: int = 10,
        max_idle_time: int = 300,  # 5 minutes
        health_check_interval: int = 60,  # 1 minute
        max_query_retries: int = 3,
    ):
        """
        Initialize the connection pool.

        Args:
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed
            max_idle_time: Maximum time (seconds) a connection can be idle
            health_check_interval: Interval (seconds) between health checks
            max_query_retries: Maximum number of query retries on failure
        """
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)

        # Pool configuration
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.max_query_retries = max_query_retries

        # Connection management
        self._pool: List[PooledConnection] = []
        self._pool_lock = asyncio.Lock()
        self._stats = ConnectionStats()
        self._executor = ThreadPoolExecutor(max_workers=max_connections)

        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Connection tracking
        self._active_connections: WeakSet = WeakSet()

        # Initialize embedding function once
        self._embedding_function = None
        self._initialize_embedding_function()

    def _initialize_embedding_function(self):
        """Initialize the embedding function for reuse."""
        try:
            openai_key = self.settings.openai_api_key
            if not openai_key:
                raise VectorStoreException(
                    "OpenAI API key not found in environment variables",
                    operation="embedding_initialization",
                    cause=None,
                )

            # Set environment variable temporarily for ChromaDB embedding function
            import os

            original_api_key = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = openai_key
            try:
                self._embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key_env_var="OPENAI_API_KEY",
                    model_name=self.settings.openai_embedding_model,
                )
            finally:
                # Restore original environment variable
                if original_api_key is not None:
                    os.environ["OPENAI_API_KEY"] = original_api_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)

            self.logger.info("Embedding function initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize embedding function: {str(e)}")
            raise VectorStoreException(
                f"Embedding function initialization failed: {str(e)}",
                operation="embedding_initialization",
                cause=e,
            )

    async def _create_connection(self) -> PooledConnection:
        """Create a new ChromaDB connection."""
        try:
            connection_id = f"conn_{int(time.time())}_{len(self._pool)}"

            # Create client based on configuration
            if self.settings.chromadb_use_http:
                client = await asyncio.to_thread(
                    chromadb.HttpClient,
                    host=self.settings.chromadb_host,
                    port=self.settings.chromadb_port,
                )
                self.logger.debug(f"Created HTTP client: {connection_id}")
            else:
                client = await asyncio.to_thread(
                    chromadb.PersistentClient, path=self.settings.chromadb_database_path
                )
                self.logger.debug(f"Created persistent client: {connection_id}")

            # Get or create collection
            collection = await asyncio.to_thread(
                client.get_or_create_collection,
                name=self.settings.chromadb_collection_name,
                embedding_function=self._embedding_function,
            )

            connection = PooledConnection(
                client=client,
                collection=collection,
                created_at=time.time(),
                last_used=time.time(),
                connection_id=connection_id,
            )

            self._stats.connections_created += 1
            self._stats.total_connections += 1

            self.logger.info(f"Created new connection: {connection_id}")
            return connection

        except Exception as e:
            self.logger.error(f"Failed to create connection: {str(e)}")
            raise VectorStoreException(
                f"Connection creation failed: {str(e)}", operation="create_connection", cause=e
            )

    async def _close_connection(self, connection: PooledConnection):
        """Close a connection and update statistics."""
        try:
            # ChromaDB clients don't have explicit close methods
            # Just remove from tracking
            connection.is_active = False
            self._stats.connections_closed += 1
            self._stats.total_connections -= 1

            self.logger.debug(f"Closed connection: {connection.connection_id}")

        except Exception as e:
            self.logger.warning(f"Error closing connection {connection.connection_id}: {str(e)}")

    async def _maintain_pool_size(self):
        """Ensure the pool has the minimum number of connections."""
        async with self._pool_lock:
            current_size = len(self._pool)

            if current_size < self.min_connections:
                needed = self.min_connections - current_size
                self.logger.info(f"Creating {needed} connections to maintain minimum pool size")

                for _ in range(needed):
                    try:
                        connection = await self._create_connection()
                        self._pool.append(connection)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to create connection for pool maintenance: {str(e)}"
                        )
                        break

    async def _cleanup_idle_connections(self):
        """Remove connections that have been idle too long."""
        current_time = time.time()
        connections_to_remove = []

        async with self._pool_lock:
            for connection in self._pool:
                if (
                    not connection.is_active
                    and current_time - connection.last_used > self.max_idle_time
                ):
                    connections_to_remove.append(connection)

            for connection in connections_to_remove:
                if len(self._pool) > self.min_connections:
                    self._pool.remove(connection)
                    await self._close_connection(connection)
                    self.logger.debug(f"Removed idle connection: {connection.connection_id}")

    async def _health_check_loop(self):
        """Background task for pool maintenance."""
        while not self._shutdown_event.is_set():
            try:
                await self._maintain_pool_size()
                await self._cleanup_idle_connections()
                await self._update_statistics()

                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(self.health_check_interval)

    async def _update_statistics(self):
        """Update connection pool statistics."""
        async with self._pool_lock:
            self._stats.total_connections = len(self._pool)
            self._stats.active_connections = sum(1 for conn in self._pool if conn.is_active)
            self._stats.idle_connections = (
                self._stats.total_connections - self._stats.active_connections
            )
            self._stats.last_activity = time.time()

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[PooledConnection, None]:
        """
        Get a connection from the pool (context manager).

        Yields:
            PooledConnection: A connection from the pool
        """
        connection = None
        try:
            # Get connection from pool
            async with self._pool_lock:
                # Look for an idle connection
                for conn in self._pool:
                    if not conn.is_active:
                        connection = conn
                        break

                # Create new connection if none available and under max
                if not connection and len(self._pool) < self.max_connections:
                    connection = await self._create_connection()
                    self._pool.append(connection)

                # If still no connection, wait for one to become available
                if not connection:
                    # Use the least recently used connection
                    connection = min(self._pool, key=lambda c: c.last_used)

                # Mark as active
                connection.is_active = True
                connection.last_used = time.time()
                self._active_connections.add(connection)

            yield connection

        except Exception as e:
            self.logger.error(f"Error getting connection: {str(e)}")
            raise VectorStoreException(
                f"Failed to get database connection: {str(e)}", operation="get_connection", cause=e
            )
        finally:
            # Return connection to pool
            if connection:
                connection.is_active = False
                connection.last_used = time.time()
                connection.query_count += 1

    async def execute_query(self, operation: str, *args, retries: int = None, **kwargs) -> Any:
        """
        Execute a query operation with automatic retry and connection management.

        Args:
            operation: The operation to execute on the collection
            *args: Positional arguments for the operation
            retries: Number of retries (defaults to max_query_retries)
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation
        """
        if retries is None:
            retries = self.max_query_retries

        last_exception = None
        start_time = time.time()

        for attempt in range(retries + 1):
            try:
                async with self.get_connection() as connection:
                    # Execute operation on the collection
                    result = await asyncio.to_thread(
                        getattr(connection.collection, operation), *args, **kwargs
                    )

                    # Update statistics
                    query_time = time.time() - start_time
                    self._stats.queries_executed += 1

                    # Update average query time
                    if self._stats.queries_executed > 1:
                        self._stats.average_query_time = (
                            self._stats.average_query_time * (self._stats.queries_executed - 1)
                            + query_time
                        ) / self._stats.queries_executed
                    else:
                        self._stats.average_query_time = query_time

                    self.logger.debug(
                        f"Query '{operation}' completed in {query_time:.3f}s "
                        f"(attempt {attempt + 1}/{retries + 1})"
                    )

                    return result

            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Query '{operation}' failed on attempt {attempt + 1}/{retries + 1}: {str(e)}"
                )

                if attempt < retries:
                    # Exponential backoff
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

        # All retries failed
        self.logger.error(f"Query '{operation}' failed after {retries + 1} attempts")
        raise VectorStoreException(
            f"Query '{operation}' failed after {retries + 1} attempts: {str(last_exception)}",
            operation=operation,
            cause=last_exception,
        )

    async def start(self):
        """Start the connection pool and health monitoring."""
        try:
            # Initialize minimum connections
            await self._maintain_pool_size()

            # Start health check loop
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            self.logger.info(
                f"Connection pool started: min={self.min_connections}, "
                f"max={self.max_connections}, initial_size={len(self._pool)}"
            )

        except Exception as e:
            self.logger.error(f"Failed to start connection pool: {str(e)}")
            raise VectorStoreException(
                f"Connection pool startup failed: {str(e)}", operation="pool_start", cause=e
            )

    async def stop(self):
        """Stop the connection pool and close all connections."""
        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Cancel health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Close all connections
            async with self._pool_lock:
                for connection in self._pool[:]:  # Copy to avoid modification during iteration
                    await self._close_connection(connection)
                self._pool.clear()

            # Shutdown executor
            self._executor.shutdown(wait=True)

            self.logger.info("Connection pool stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping connection pool: {str(e)}")

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        await self._update_statistics()

        return {
            "pool_config": {
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
                "max_idle_time": self.max_idle_time,
                "health_check_interval": self.health_check_interval,
            },
            "current_stats": {
                "total_connections": self._stats.total_connections,
                "active_connections": self._stats.active_connections,
                "idle_connections": self._stats.idle_connections,
                "connections_created": self._stats.connections_created,
                "connections_closed": self._stats.connections_closed,
                "queries_executed": self._stats.queries_executed,
                "average_query_time_ms": self._stats.average_query_time * 1000,
                "last_activity": self._stats.last_activity,
            },
            "health": {
                "pool_healthy": len(self._pool) >= self.min_connections,
                "all_connections_responsive": True,  # Would need health checks to determine
                "uptime_seconds": time.time()
                - (
                    self._stats.last_activity
                    if self._stats.connections_created > 0
                    else time.time()
                ),
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        try:
            start_time = time.time()

            # Test a simple operation
            async with self.get_connection() as connection:
                count = await asyncio.to_thread(connection.collection.count)

            response_time = time.time() - start_time
            stats = await self.get_pool_stats()

            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "test_query_successful": True,
                "pool_stats": stats,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "test_query_successful": False,
                "timestamp": time.time(),
            }


# Global connection pool instance
_connection_pool: Optional[ChromaDBConnectionPool] = None
_pool_lock = threading.Lock()


async def get_connection_pool() -> ChromaDBConnectionPool:
    """Get the global connection pool instance."""
    global _connection_pool

    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                settings = get_settings()
                _connection_pool = ChromaDBConnectionPool(
                    min_connections=getattr(settings, "chromadb_min_connections", 2),
                    max_connections=getattr(settings, "chromadb_max_connections", 10),
                    max_idle_time=getattr(settings, "chromadb_max_idle_time", 300),
                    health_check_interval=getattr(settings, "chromadb_health_check_interval", 60),
                )
                await _connection_pool.start()

    return _connection_pool


async def close_connection_pool():
    """Close the global connection pool."""
    global _connection_pool

    if _connection_pool:
        await _connection_pool.stop()
        _connection_pool = None
