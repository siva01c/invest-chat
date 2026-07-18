"""Background task worker for processing jobs from the task queue."""

import asyncio
import signal
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from assistant.config import get_settings
from assistant.infrastructure.observability import (
    CorrelationContext,
    get_enhanced_logger,
    get_prometheus_metrics,
    traceable,
)

from .task_queue import Task, get_task_queue
from .task_registry import get_task_registry


class TaskWorker:
    """
    Background task worker for processing jobs from Redis queue.

    Features:
    - Concurrent task processing
    - Graceful shutdown handling
    - Task timeout management
    - Metrics collection and monitoring
    - Error handling and retry logic
    - Health monitoring
    """

    def __init__(
        self,
        queue_name: str = "default",
        worker_id: Optional[str] = None,
        concurrency: int = 4,
        poll_interval: float = 1.0,
    ):
        """
        Initialize task worker.

        Args:
            queue_name: Name of the task queue to process
            worker_id: Unique worker identifier
            concurrency: Number of concurrent tasks to process
            poll_interval: Polling interval in seconds
        """
        self.settings = get_settings()
        self.queue_name = queue_name
        self.worker_id = worker_id or f"worker-{int(time.time())}"
        self.concurrency = concurrency
        self.poll_interval = poll_interval

        self.logger = get_enhanced_logger(self.__class__.__name__)
        self.metrics = get_prometheus_metrics()
        self.task_registry = get_task_registry()

        # Worker state
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._worker_tasks: List[asyncio.Task] = []
        self._active_tasks: Dict[str, asyncio.Task] = {}

        # Statistics
        self._stats = {
            "tasks_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "started_at": None,
            "last_task_at": None,
        }

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        try:
            # Handle SIGTERM and SIGINT for graceful shutdown
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, self._signal_handler)
        except ValueError:
            # Signal handling might not work in all environments (e.g., Jupyter)
            self.logger.debug("Could not setup signal handlers")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop())

    async def start(self):
        """Start the task worker."""
        if self._running:
            self.logger.warning("Worker is already running")
            return

        self._running = True
        self._stats["started_at"] = datetime.utcnow()
        self._shutdown_event.clear()

        self.logger.info(
            f"Starting task worker: {self.worker_id} "
            f"(queue: {self.queue_name}, concurrency: {self.concurrency})"
        )

        # Start worker tasks
        for i in range(self.concurrency):
            worker_task = asyncio.create_task(
                self._worker_loop(f"{self.worker_id}-{i}"), name=f"worker-{i}"
            )
            self._worker_tasks.append(worker_task)

        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_loop(), name="monitor")
        self._worker_tasks.append(monitor_task)

        # Track worker start
        self.metrics.track_business_event(
            "worker_started",
            "worker",
            self.worker_id,
            "start",
            True,
            {"queue_name": self.queue_name, "concurrency": self.concurrency},
        )

        try:
            # Wait for shutdown
            await self._shutdown_event.wait()
        finally:
            await self._cleanup()

    async def stop(self, timeout: float = 30.0):
        """
        Stop the task worker gracefully.

        Args:
            timeout: Maximum time to wait for tasks to complete
        """
        if not self._running:
            return

        self.logger.info(f"Stopping task worker: {self.worker_id}")

        # Signal shutdown
        self._running = False
        self._shutdown_event.set()

        # Wait for active tasks to complete
        if self._active_tasks:
            self.logger.info(f"Waiting for {len(self._active_tasks)} active tasks to complete...")

            try:
                # Wait for active tasks with timeout
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks.values(), return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout waiting for tasks, forcing shutdown")
                # Cancel remaining tasks
                for task in self._active_tasks.values():
                    if not task.done():
                        task.cancel()

        # Cancel worker tasks
        for task in self._worker_tasks:
            if not task.done():
                task.cancel()

        # Wait for worker tasks to finish
        try:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Error during worker shutdown: {str(e)}")

        # Track worker stop
        self.metrics.track_business_event(
            "worker_stopped",
            "worker",
            self.worker_id,
            "stop",
            True,
            {"tasks_processed": self._stats["tasks_processed"]},
        )

        self.logger.info(f"Task worker stopped: {self.worker_id}")

    async def _worker_loop(self, worker_name: str):
        """
        Main worker loop for processing tasks.

        Args:
            worker_name: Name of this worker instance
        """
        task_queue = await get_task_queue(self.queue_name)

        self.logger.debug(f"Worker loop started: {worker_name}")

        while self._running:
            try:
                # Dequeue a task
                task = await task_queue.dequeue(timeout_seconds=self.poll_interval)

                if task is None:
                    # No task available, continue polling
                    continue

                # Process the task
                await self._process_task(task, worker_name)

            except asyncio.CancelledError:
                self.logger.debug(f"Worker loop cancelled: {worker_name}")
                break
            except Exception as e:
                self.logger.error(f"Error in worker loop {worker_name}: {str(e)}", exc_info=True)
                # Brief pause to prevent rapid error loops
                await asyncio.sleep(1.0)

        self.logger.debug(f"Worker loop ended: {worker_name}")

    @traceable("process_background_task")
    async def _process_task(self, task: Task, worker_name: str):
        """
        Process a single task.

        Args:
            task: Task to process
            worker_name: Name of the worker processing the task
        """
        task_queue = await get_task_queue(self.queue_name)
        start_time = time.time()

        # Set up correlation context for the task
        with CorrelationContext.correlation_scope(
            correlation_id=task.task_id,
            context={
                "task_name": task.name,
                "worker_id": self.worker_id,
                "worker_name": worker_name,
                "queue_name": self.queue_name,
            },
        ):
            self.logger.info(
                f"Processing task: {task.name} (ID: {task.task_id}, worker: {worker_name})"
            )

            # Create async task for processing
            process_task = asyncio.create_task(
                self._execute_task(task), name=f"task-{task.task_id}"
            )

            # Track active task
            self._active_tasks[task.task_id] = process_task

            try:
                # Execute with optional timeout
                if task.timeout_seconds:
                    result = await asyncio.wait_for(process_task, timeout=task.timeout_seconds)
                else:
                    result = await process_task

                # Task completed successfully
                duration_ms = (time.time() - start_time) * 1000
                await task_queue.complete_task(task, result, duration_ms)

                # Update statistics
                self._stats["tasks_processed"] += 1
                self._stats["tasks_completed"] += 1
                self._stats["last_task_at"] = datetime.utcnow()

                # Track metrics
                self.metrics.track_performance_metric(
                    f"task_execution_{task.name}", duration_ms, True
                )

                self.logger.info(
                    f"Task completed: {task.name} (ID: {task.task_id}, duration: {duration_ms:.2f}ms)"
                )

            except asyncio.TimeoutError:
                # Task timed out
                duration_ms = (time.time() - start_time) * 1000
                error_msg = f"Task timed out after {task.timeout_seconds} seconds"

                await task_queue.fail_task(task, error_msg, retry=True, duration_ms=duration_ms)

                self._stats["tasks_processed"] += 1
                self._stats["tasks_failed"] += 1

                self.logger.error(f"Task timed out: {task.name} (ID: {task.task_id})")

            except Exception as e:
                # Task failed with exception
                duration_ms = (time.time() - start_time) * 1000
                error_msg = str(e)

                retry_result = await task_queue.fail_task(
                    task, error_msg, retry=True, duration_ms=duration_ms
                )

                self._stats["tasks_processed"] += 1
                if retry_result:
                    self._stats["tasks_retried"] += 1
                else:
                    self._stats["tasks_failed"] += 1

                # Track metrics
                self.metrics.track_performance_metric(
                    f"task_execution_{task.name}", duration_ms, False
                )

                self.logger.error(
                    f"Task failed: {task.name} (ID: {task.task_id}, error: {error_msg})",
                    exc_info=True,
                )

            finally:
                # Remove from active tasks
                self._active_tasks.pop(task.task_id, None)

    async def _execute_task(self, task: Task) -> Any:
        """
        Execute the actual task function.

        Args:
            task: Task to execute

        Returns:
            Task execution result
        """
        # Get the task function from registry
        task_func = self.task_registry.get_task(task.name)

        if task_func is None:
            raise ValueError(f"Task '{task.name}' is not registered")

        # Execute the task
        return await self.task_registry.execute_task(task.name, *task.args, **task.kwargs)

    async def _monitor_loop(self):
        """Monitor worker health and performance."""
        while self._running:
            try:
                # Log worker statistics periodically
                self.logger.debug(
                    f"Worker stats: {self.worker_id} - "
                    f"processed: {self._stats['tasks_processed']}, "
                    f"active: {len(self._active_tasks)}, "
                    f"completed: {self._stats['tasks_completed']}, "
                    f"failed: {self._stats['tasks_failed']}"
                )

                # Update worker metrics
                self.metrics.update_service_health(f"worker_{self.worker_id}", True)

                # Sleep before next monitor cycle
                await asyncio.sleep(30.0)  # Monitor every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {str(e)}")
                await asyncio.sleep(5.0)

    async def _cleanup(self):
        """Cleanup worker resources."""
        self.logger.debug(f"Cleaning up worker: {self.worker_id}")

        # Clear active tasks
        self._active_tasks.clear()

        # Clear worker tasks
        self._worker_tasks.clear()

        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get worker statistics.

        Returns:
            Worker statistics
        """
        uptime_seconds = 0
        if self._stats["started_at"]:
            uptime_seconds = (datetime.utcnow() - self._stats["started_at"]).total_seconds()

        return {
            "worker_id": self.worker_id,
            "queue_name": self.queue_name,
            "concurrency": self.concurrency,
            "running": self._running,
            "active_tasks": len(self._active_tasks),
            "uptime_seconds": uptime_seconds,
            "tasks_processed": self._stats["tasks_processed"],
            "tasks_completed": self._stats["tasks_completed"],
            "tasks_failed": self._stats["tasks_failed"],
            "tasks_retried": self._stats["tasks_retried"],
            "started_at": (
                self._stats["started_at"].isoformat() if self._stats["started_at"] else None
            ),
            "last_task_at": (
                self._stats["last_task_at"].isoformat() if self._stats["last_task_at"] else None
            ),
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform worker health check.

        Returns:
            Health check results
        """
        try:
            task_queue = await get_task_queue(self.queue_name)
            queue_stats = await task_queue.get_queue_stats()

            return {
                "worker_id": self.worker_id,
                "status": "healthy" if self._running else "stopped",
                "queue_stats": queue_stats,
                "worker_stats": self.get_stats(),
                "registered_tasks": len(self.task_registry.list_tasks()),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "worker_id": self.worker_id,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
