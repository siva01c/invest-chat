"""Redis-based task queue system for background job processing."""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

from assistant.config import get_settings
from assistant.infrastructure.cache.redis_client import get_redis_client
from assistant.infrastructure.observability import get_enhanced_logger, get_prometheus_metrics


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskResult:
    """Task execution result."""

    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            status=TaskStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            duration_ms=data.get("duration_ms"),
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Task:
    """Task definition for background processing."""

    task_id: str
    name: str
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    delay_seconds: float = 0
    max_retries: int = 3
    retry_delay_seconds: float = 60
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "args": self.args,
            "kwargs": self.kwargs,
            "priority": self.priority.value,
            "delay_seconds": self.delay_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            args=tuple(data.get("args", [])),
            kwargs=data.get("kwargs", {}),
            priority=TaskPriority(data.get("priority", TaskPriority.NORMAL.value)),
            delay_seconds=data.get("delay_seconds", 0),
            max_retries=data.get("max_retries", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 60),
            timeout_seconds=data.get("timeout_seconds"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            scheduled_at=(
                datetime.fromisoformat(data["scheduled_at"]) if data.get("scheduled_at") else None
            ),
        )


class TaskQueue:
    """
    Redis-based task queue for background job processing.

    Features:
    - Priority-based task scheduling
    - Delayed task execution
    - Retry mechanisms with exponential backoff
    - Task status tracking and monitoring
    - Graceful shutdown handling
    - Metrics collection and monitoring
    """

    def __init__(self, queue_name: str = "default"):
        """
        Initialize task queue.

        Args:
            queue_name: Name of the task queue
        """
        self.settings = get_settings()
        self.queue_name = queue_name
        self.logger = get_enhanced_logger(self.__class__.__name__)
        self.metrics = get_prometheus_metrics()

        # Redis keys
        self.pending_key = f"tasks:{queue_name}:pending"
        self.running_key = f"tasks:{queue_name}:running"
        self.completed_key = f"tasks:{queue_name}:completed"
        self.failed_key = f"tasks:{queue_name}:failed"
        self.delayed_key = f"tasks:{queue_name}:delayed"
        self.results_key = f"tasks:{queue_name}:results"

        # Queue statistics
        self._stats = {
            "tasks_enqueued": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
        }

    async def enqueue(
        self,
        task_name: str,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay_seconds: float = 0,
        max_retries: int = 3,
        retry_delay_seconds: float = 60,
        timeout_seconds: Optional[float] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Enqueue a task for background processing.

        Args:
            task_name: Name of the task to execute
            *args: Positional arguments for the task
            priority: Task priority level
            delay_seconds: Delay before task execution
            max_retries: Maximum retry attempts
            retry_delay_seconds: Delay between retries
            timeout_seconds: Task execution timeout
            task_id: Custom task ID (generated if not provided)
            metadata: Additional task metadata
            **kwargs: Keyword arguments for the task

        Returns:
            Task ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        # Create task
        task = Task(
            task_id=task_id,
            name=task_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            delay_seconds=delay_seconds,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {},
        )

        # Calculate scheduled time
        if delay_seconds > 0:
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=delay_seconds)

        redis_client = await get_redis_client()

        try:
            # Serialize task
            task_data = json.dumps(task.to_dict(), default=str)

            if delay_seconds > 0:
                # Add to delayed queue with score as timestamp
                score = time.time() + delay_seconds
                await redis_client.zadd(self.delayed_key, {task_data: score})
                self.logger.info(
                    f"Enqueued delayed task: {task_name} (ID: {task_id}, delay: {delay_seconds}s)"
                )
            else:
                # Add to pending queue with priority
                await redis_client.zadd(self.pending_key, {task_data: priority.value})
                self.logger.info(
                    f"Enqueued task: {task_name} (ID: {task_id}, priority: {priority.name})"
                )

            # Update statistics
            self._stats["tasks_enqueued"] += 1

            # Track metrics
            self.metrics.track_business_event(
                "task_enqueued",
                "task",
                task_id,
                "enqueue",
                True,
                {"task_name": task_name, "priority": priority.name, "delayed": delay_seconds > 0},
            )

            return task_id

        except Exception as e:
            self.logger.error(f"Failed to enqueue task {task_name}: {str(e)}", exc_info=True)
            raise

    async def dequeue(self, timeout_seconds: float = 10.0) -> Optional[Task]:
        """
        Dequeue a task for processing.

        Args:
            timeout_seconds: Timeout for blocking dequeue operation

        Returns:
            Task to process or None if timeout
        """
        redis_client = await get_redis_client()

        try:
            # First, move any ready delayed tasks to pending queue
            await self._process_delayed_tasks()

            # Get highest priority task from pending queue
            result = await redis_client.bzpopmax(self.pending_key, timeout=timeout_seconds)

            if not result:
                return None

            # Parse task data
            _, task_data, _ = result
            task_dict = json.loads(task_data)
            task = Task.from_dict(task_dict)

            # Move task to running queue
            await redis_client.zadd(self.running_key, {task_data: time.time()})

            self.logger.debug(f"Dequeued task: {task.name} (ID: {task.task_id})")
            return task

        except Exception as e:
            self.logger.error(f"Failed to dequeue task: {str(e)}", exc_info=True)
            return None

    async def complete_task(
        self, task: Task, result: Any = None, duration_ms: Optional[float] = None
    ):
        """
        Mark a task as completed.

        Args:
            task: The completed task
            result: Task execution result
            duration_ms: Task execution duration in milliseconds
        """
        redis_client = await get_redis_client()

        try:
            # Create task result
            task_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                completed_at=datetime.utcnow(),
                duration_ms=duration_ms,
            )

            # Store result
            result_data = json.dumps(task_result.to_dict(), default=str)
            await redis_client.hset(self.results_key, task.task_id, result_data)

            # Move from running to completed
            task_data = json.dumps(task.to_dict(), default=str)
            await redis_client.zrem(self.running_key, task_data)
            await redis_client.zadd(self.completed_key, {task_data: time.time()})

            # Update statistics
            self._stats["tasks_completed"] += 1

            # Track metrics
            self.metrics.track_business_event(
                "task_completed",
                "task",
                task.task_id,
                "complete",
                True,
                {"task_name": task.name, "duration_ms": duration_ms},
            )

            self.logger.info(
                f"Completed task: {task.name} (ID: {task.task_id}, duration: {duration_ms}ms)"
            )

        except Exception as e:
            self.logger.error(f"Failed to complete task {task.task_id}: {str(e)}", exc_info=True)

    async def fail_task(
        self, task: Task, error: str, retry: bool = True, duration_ms: Optional[float] = None
    ) -> bool:
        """
        Mark a task as failed and optionally retry.

        Args:
            task: The failed task
            error: Error message
            retry: Whether to retry the task
            duration_ms: Task execution duration in milliseconds

        Returns:
            True if task was retried, False if marked as failed
        """
        redis_client = await get_redis_client()

        try:
            # Check if we should retry
            current_retries = task.metadata.get("retry_count", 0)
            should_retry = retry and current_retries < task.max_retries

            if should_retry:
                # Increment retry count
                task.metadata["retry_count"] = current_retries + 1
                task.metadata["last_error"] = error
                task.metadata["last_retry_at"] = datetime.utcnow().isoformat()

                # Calculate retry delay (exponential backoff)
                retry_delay = task.retry_delay_seconds * (2**current_retries)
                task.scheduled_at = datetime.utcnow() + timedelta(seconds=retry_delay)

                # Add back to delayed queue
                task_data = json.dumps(task.to_dict(), default=str)
                score = time.time() + retry_delay
                await redis_client.zadd(self.delayed_key, {task_data: score})

                # Remove from running queue
                await redis_client.zrem(self.running_key, task_data)

                # Update statistics
                self._stats["tasks_retried"] += 1

                self.logger.warning(
                    f"Retrying task: {task.name} (ID: {task.task_id}, attempt: {current_retries + 1}/{task.max_retries}, delay: {retry_delay}s)"
                )

                return True

            else:
                # Mark as permanently failed
                task_result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=error,
                    completed_at=datetime.utcnow(),
                    duration_ms=duration_ms,
                    retry_count=current_retries,
                )

                # Store result
                result_data = json.dumps(task_result.to_dict(), default=str)
                await redis_client.hset(self.results_key, task.task_id, result_data)

                # Move from running to failed
                task_data = json.dumps(task.to_dict(), default=str)
                await redis_client.zrem(self.running_key, task_data)
                await redis_client.zadd(self.failed_key, {task_data: time.time()})

                # Update statistics
                self._stats["tasks_failed"] += 1

                # Track metrics
                self.metrics.track_business_event(
                    "task_failed",
                    "task",
                    task.task_id,
                    "fail",
                    False,
                    {"task_name": task.name, "error": error, "retry_count": current_retries},
                )

                self.logger.error(
                    f"Task failed permanently: {task.name} (ID: {task.task_id}, error: {error})"
                )

                return False

        except Exception as e:
            self.logger.error(
                f"Failed to handle task failure {task.task_id}: {str(e)}", exc_info=True
            )
            return False

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Get task execution result.

        Args:
            task_id: Task ID

        Returns:
            Task result or None if not found
        """
        redis_client = await get_redis_client()

        try:
            result_data = await redis_client.hget(self.results_key, task_id)
            if result_data:
                result_dict = json.loads(result_data)
                return TaskResult.from_dict(result_dict)
            return None

        except Exception as e:
            self.logger.error(f"Failed to get task result {task_id}: {str(e)}")
            return None

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Queue statistics
        """
        redis_client = await get_redis_client()

        try:
            # Count tasks in each queue
            pending_count = await redis_client.zcard(self.pending_key)
            running_count = await redis_client.zcard(self.running_key)
            completed_count = await redis_client.zcard(self.completed_key)
            failed_count = await redis_client.zcard(self.failed_key)
            delayed_count = await redis_client.zcard(self.delayed_key)

            return {
                "queue_name": self.queue_name,
                "pending_tasks": pending_count,
                "running_tasks": running_count,
                "completed_tasks": completed_count,
                "failed_tasks": failed_count,
                "delayed_tasks": delayed_count,
                "total_enqueued": self._stats["tasks_enqueued"],
                "total_completed": self._stats["tasks_completed"],
                "total_failed": self._stats["tasks_failed"],
                "total_retried": self._stats["tasks_retried"],
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get queue stats: {str(e)}")
            return {"error": str(e)}

    async def _process_delayed_tasks(self):
        """Process delayed tasks that are ready to run."""
        redis_client = await get_redis_client()

        try:
            current_time = time.time()

            # Get tasks that are ready to run
            ready_tasks = await redis_client.zrangebyscore(
                self.delayed_key, 0, current_time, withscores=False
            )

            for task_data in ready_tasks:
                try:
                    # Parse task
                    task_dict = json.loads(task_data)
                    task = Task.from_dict(task_dict)

                    # Move to pending queue
                    await redis_client.zadd(self.pending_key, {task_data: task.priority.value})
                    await redis_client.zrem(self.delayed_key, task_data)

                    self.logger.debug(
                        f"Moved delayed task to pending: {task.name} (ID: {task.task_id})"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to process delayed task: {str(e)}")
                    # Remove corrupted task from delayed queue
                    await redis_client.zrem(self.delayed_key, task_data)

        except Exception as e:
            self.logger.error(f"Failed to process delayed tasks: {str(e)}")

    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """
        Clean up old completed and failed tasks.

        Args:
            max_age_hours: Maximum age of tasks to keep
        """
        redis_client = await get_redis_client()

        try:
            cutoff_time = time.time() - (max_age_hours * 3600)

            # Clean up completed tasks
            completed_removed = await redis_client.zremrangebyscore(
                self.completed_key, 0, cutoff_time
            )

            # Clean up failed tasks
            failed_removed = await redis_client.zremrangebyscore(self.failed_key, 0, cutoff_time)

            if completed_removed > 0 or failed_removed > 0:
                self.logger.info(
                    f"Cleaned up old tasks: {completed_removed} completed, {failed_removed} failed"
                )

        except Exception as e:
            self.logger.error(f"Failed to cleanup tasks: {str(e)}")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


async def get_task_queue(queue_name: str = "default") -> TaskQueue:
    """Get the global task queue instance."""
    global _task_queue

    if _task_queue is None or _task_queue.queue_name != queue_name:
        _task_queue = TaskQueue(queue_name)

    return _task_queue
