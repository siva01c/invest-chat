"""Worker manager for background task processing."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from assistant.config import get_settings
from assistant.infrastructure.observability import get_enhanced_logger

from .task_worker import TaskWorker


class WorkerManager:
    """
    Manager for background task workers.

    Features:
    - Multiple worker management
    - Worker health monitoring
    - Graceful shutdown coordination
    - Worker statistics aggregation
    """

    def __init__(self) -> None:
        """Initialize worker manager."""
        self.settings = get_settings()
        self.logger = get_enhanced_logger(self.__class__.__name__)

        self._workers: Dict[str, TaskWorker] = {}
        self._worker_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start_worker(
        self, worker_id: Optional[str] = None, queue_name: str = "default", concurrency: int = 4
    ) -> str:
        """
        Start a new worker.

        Args:
            worker_id: Unique worker identifier
            queue_name: Queue name to process
            concurrency: Number of concurrent tasks

        Returns:
            Worker ID
        """
        if worker_id is None:
            worker_id = f"worker-{len(self._workers) + 1}"

        if worker_id in self._workers:
            raise ValueError(f"Worker {worker_id} already exists")

        # Create worker
        worker = TaskWorker(queue_name=queue_name, worker_id=worker_id, concurrency=concurrency)

        # Start worker in background
        worker_task = asyncio.create_task(worker.start(), name=f"worker-{worker_id}")

        # Store worker and task
        self._workers[worker_id] = worker
        self._worker_tasks[worker_id] = worker_task

        self.logger.info(
            f"Started worker: {worker_id} (queue: {queue_name}, concurrency: {concurrency})"
        )

        return worker_id

    async def stop_worker(self, worker_id: str, timeout: float = 30.0) -> bool:
        """
        Stop a specific worker.

        Args:
            worker_id: Worker ID
            timeout: Shutdown timeout

        Returns:
            True if stopped successfully
        """
        if worker_id not in self._workers:
            return False

        worker = self._workers[worker_id]
        worker_task = self._worker_tasks[worker_id]

        try:
            # Stop worker gracefully
            await worker.stop(timeout=timeout)

            # Wait for worker task to complete
            await worker_task

            # Remove from tracking
            del self._workers[worker_id]
            del self._worker_tasks[worker_id]

            self.logger.info(f"Stopped worker: {worker_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop worker {worker_id}: {str(e)}")
            return False

    async def start_default_workers(self) -> List[str]:
        """
        Start default set of workers.

        Returns:
            List of worker IDs
        """
        worker_ids = []

        # Start main worker for default queue
        worker_id = await self.start_worker(
            worker_id="main-worker", queue_name="default", concurrency=4
        )
        worker_ids.append(worker_id)

        # Start email worker for email-specific tasks
        worker_id = await self.start_worker(
            worker_id="email-worker",
            queue_name="default",  # Can use same queue or separate email queue
            concurrency=2,
        )
        worker_ids.append(worker_id)

        self._running = True
        self.logger.info(f"Started {len(worker_ids)} default workers")

        return worker_ids

    async def stop_all_workers(self, timeout: float = 30.0) -> None:
        """
        Stop all workers gracefully.

        Args:
            timeout: Shutdown timeout per worker
        """
        if not self._workers:
            return

        self.logger.info(f"Stopping {len(self._workers)} workers...")

        # Stop all workers concurrently
        stop_tasks = []
        for worker_id in list(self._workers.keys()):
            stop_task = asyncio.create_task(
                self.stop_worker(worker_id, timeout), name=f"stop-{worker_id}"
            )
            stop_tasks.append(stop_task)

        # Wait for all workers to stop
        try:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"Error during worker shutdown: {str(e)}")

        self._running = False
        self._shutdown_event.set()

        self.logger.info("All workers stopped")

    def get_worker_stats(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific worker.

        Args:
            worker_id: Worker ID

        Returns:
            Worker statistics or None if not found
        """
        worker = self._workers.get(worker_id)
        if worker:
            return worker.get_stats()
        return None

    def get_all_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all workers.

        Returns:
            Dictionary of worker ID to statistics
        """
        return {worker_id: worker.get_stats() for worker_id, worker in self._workers.items()}

    async def get_worker_health(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get health status for a specific worker.

        Args:
            worker_id: Worker ID

        Returns:
            Worker health status or None if not found
        """
        worker = self._workers.get(worker_id)
        if worker:
            return await worker.health_check()
        return None

    async def get_all_worker_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status for all workers.

        Returns:
            Dictionary of worker ID to health status
        """
        health_checks = {}

        for worker_id, worker in self._workers.items():
            try:
                health_checks[worker_id] = await worker.health_check()
            except Exception as e:
                health_checks[worker_id] = {
                    "worker_id": worker_id,
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }

        return health_checks

    def list_workers(self) -> List[Dict[str, Any]]:
        """
        List all workers with basic information.

        Returns:
            List of worker information
        """
        return [
            {
                "worker_id": worker_id,
                "queue_name": worker.queue_name,
                "concurrency": worker.concurrency,
                "running": worker._running,
            }
            for worker_id, worker in self._workers.items()
        ]

    def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get manager-level statistics.

        Returns:
            Manager statistics
        """
        total_tasks_processed = sum(
            worker.get_stats()["tasks_processed"] for worker in self._workers.values()
        )

        total_tasks_completed = sum(
            worker.get_stats()["tasks_completed"] for worker in self._workers.values()
        )

        total_tasks_failed = sum(
            worker.get_stats()["tasks_failed"] for worker in self._workers.values()
        )

        total_active_tasks = sum(
            worker.get_stats()["active_tasks"] for worker in self._workers.values()
        )

        return {
            "manager_running": self._running,
            "total_workers": len(self._workers),
            "total_tasks_processed": total_tasks_processed,
            "total_tasks_completed": total_tasks_completed,
            "total_tasks_failed": total_tasks_failed,
            "total_active_tasks": total_active_tasks,
            "workers": self.list_workers(),
        }

    async def restart_worker(self, worker_id: str) -> bool:
        """
        Restart a specific worker.

        Args:
            worker_id: Worker ID

        Returns:
            True if restarted successfully
        """
        if worker_id not in self._workers:
            return False

        # Get current worker configuration
        worker = self._workers[worker_id]
        queue_name = worker.queue_name
        concurrency = worker.concurrency

        # Stop the worker
        stopped = await self.stop_worker(worker_id)
        if not stopped:
            return False

        # Start new worker with same configuration
        try:
            await self.start_worker(worker_id, queue_name, concurrency)
            self.logger.info(f"Restarted worker: {worker_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restart worker {worker_id}: {str(e)}")
            return False


# Global worker manager instance
_worker_manager: Optional[WorkerManager] = None


def get_worker_manager() -> WorkerManager:
    """Get the global worker manager instance."""
    global _worker_manager

    if _worker_manager is None:
        _worker_manager = WorkerManager()

    return _worker_manager
