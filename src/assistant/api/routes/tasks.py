"""Background task monitoring and management endpoints."""

import asyncio
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from assistant.config import get_settings
from assistant.infrastructure.observability import get_enhanced_logger
from assistant.infrastructure.tasks import EmailTasks
from assistant.infrastructure.tasks.task_queue import TaskPriority, get_task_queue
from assistant.infrastructure.tasks.task_registry import get_task_registry
from assistant.infrastructure.tasks.worker_manager import get_worker_manager

router = APIRouter(prefix="/tasks", tags=["tasks"])
logger = get_enhanced_logger(__name__)
settings = get_settings()


@router.get("/status")
async def get_task_system_status() -> Dict[str, Any]:
    """
    Get overall task system status.

    Returns:
        Task system status and statistics
    """
    try:
        task_queue = await get_task_queue()
        task_registry = get_task_registry()

        # Get queue statistics
        queue_stats = await task_queue.get_queue_stats()

        # Get registered tasks
        registered_tasks = task_registry.list_tasks()

        return {
            "service": "background_tasks",
            "timestamp": time.time(),
            "status": "operational",
            "queue_stats": queue_stats,
            "registered_tasks": {"count": len(registered_tasks), "tasks": registered_tasks},
            "system_info": {
                "default_queue": "default",
                "redis_backend": True,
                "async_processing": True,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get task system status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Task system status retrieval failed: {str(e)}"
        )


@router.get("/queue/{queue_name}/stats")
async def get_queue_statistics(queue_name: str = "default") -> Dict[str, Any]:
    """
    Get detailed statistics for a specific queue.

    Args:
        queue_name: Name of the task queue

    Returns:
        Detailed queue statistics
    """
    try:
        task_queue = await get_task_queue(queue_name)
        stats = await task_queue.get_queue_stats()

        return {"queue_name": queue_name, "timestamp": time.time(), **stats}

    except Exception as e:
        logger.error(f"Failed to get queue stats for {queue_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Queue statistics retrieval failed: {str(e)}")


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str) -> Dict[str, Any]:
    """
    Get the result of a specific task.

    Args:
        task_id: Task ID

    Returns:
        Task execution result
    """
    try:
        task_queue = await get_task_queue()
        result = await task_queue.get_task_result(task_id)

        if result is None:
            raise HTTPException(status_code=404, detail=f"Task result not found: {task_id}")

        return {"task_id": task_id, "result": result.to_dict(), "retrieved_at": time.time()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task result {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task result retrieval failed: {str(e)}")


@router.post("/email/send")
async def send_email_task(
    recipient: str,
    subject: str,
    body: str,
    body_html: Optional[str] = None,
    priority: str = "normal",
    delay_seconds: float = 0,
) -> Dict[str, Any]:
    """
    Enqueue an email sending task.

    Args:
        recipient: Email recipient
        subject: Email subject
        body: Email body (plain text)
        body_html: Email body (HTML, optional)
        priority: Task priority (low, normal, high, urgent)
        delay_seconds: Delay before sending

    Returns:
        Task ID and enqueue status
    """
    try:
        # Validate priority
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "urgent": TaskPriority.URGENT,
        }

        if priority not in priority_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority: {priority}. Must be one of: {list(priority_map.keys())}",
            )

        # Create email tasks instance
        email_tasks = EmailTasks()

        # Enqueue the email
        task_id = await email_tasks.enqueue_email(
            recipient=recipient,
            subject=subject,
            body=body,
            body_html=body_html,
            priority=priority_map[priority],
            delay_seconds=delay_seconds,
            metadata={"source": "api", "endpoint": "/tasks/email/send"},
        )

        return {
            "task_id": task_id,
            "status": "enqueued",
            "recipient": recipient,
            "subject": subject,
            "priority": priority,
            "delay_seconds": delay_seconds,
            "enqueued_at": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enqueue email task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Email task enqueue failed: {str(e)}")


@router.post("/email/notification")
async def send_chat_notification(
    user_message: str,
    classification: str,
    language: str = "en",
    user_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enqueue a chat notification email.

    Args:
        user_message: The user's message
        classification: Message classification
        language: Message language (en, cs)
        user_info: User information dictionary

    Returns:
        Task ID and enqueue status
    """
    try:
        # Create email tasks instance
        email_tasks = EmailTasks()

        # Enqueue the notification
        task_id = await email_tasks.enqueue_chat_notification(
            user_message=user_message,
            classification=classification,
            language=language,
            user_info=user_info or {},
        )

        return {
            "task_id": task_id,
            "status": "enqueued",
            "classification": classification,
            "language": language,
            "enqueued_at": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to enqueue chat notification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat notification enqueue failed: {str(e)}")


@router.post("/email/bulk")
async def send_bulk_emails(
    recipients: List[str],
    subject: str,
    body: str,
    body_html: Optional[str] = None,
    batch_size: int = Query(default=10, le=50),
    delay_between_batches: float = Query(default=60, le=300),
) -> Dict[str, Any]:
    """
    Enqueue bulk email sending tasks.

    Args:
        recipients: List of email recipients
        subject: Email subject
        body: Email body (plain text)
        body_html: Email body (HTML, optional)
        batch_size: Number of emails per batch
        delay_between_batches: Delay between batches in seconds

    Returns:
        Task IDs and enqueue status
    """
    try:
        # Validate batch size
        if len(recipients) > 1000:
            raise HTTPException(
                status_code=400, detail="Maximum 1000 recipients allowed per bulk operation"
            )

        # Create email tasks instance
        email_tasks = EmailTasks()

        # Enqueue bulk emails
        task_ids = await email_tasks.enqueue_bulk_emails(
            recipients=recipients,
            subject=subject,
            body=body,
            body_html=body_html,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches,
        )

        return {
            "task_ids": task_ids,
            "status": "enqueued",
            "total_recipients": len(recipients),
            "total_tasks": len(task_ids),
            "batch_size": batch_size,
            "delay_between_batches": delay_between_batches,
            "enqueued_at": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enqueue bulk emails: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk email enqueue failed: {str(e)}")


@router.post("/test")
async def create_test_task(
    task_name: str = "test_task", delay_seconds: float = 0, should_fail: bool = False
) -> Dict[str, Any]:
    """
    Create a test task for debugging and monitoring.

    Args:
        task_name: Name of the test task
        delay_seconds: Delay before execution
        should_fail: Whether the task should fail (for testing retry logic)

    Returns:
        Task ID and enqueue status
    """
    try:
        task_queue = await get_task_queue()

        # Create test task data
        test_data = {
            "message": f"Test task created at {time.time()}",
            "should_fail": should_fail,
            "delay_seconds": delay_seconds,
        }

        # Enqueue test task
        task_id = await task_queue.enqueue(
            "test_task",
            test_data,
            priority=TaskPriority.LOW,
            delay_seconds=delay_seconds,
            metadata={"source": "api", "endpoint": "/tasks/test"},
        )

        return {
            "task_id": task_id,
            "status": "enqueued",
            "task_name": task_name,
            "should_fail": should_fail,
            "delay_seconds": delay_seconds,
            "enqueued_at": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to create test task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test task creation failed: {str(e)}")


@router.get("/health")
async def task_system_health() -> Dict[str, Any]:
    """
    Health check for the task system.

    Returns:
        Health status of task system components
    """
    try:
        health_status: Dict[str, Any] = {
            "service": "task_system",
            "timestamp": time.time(),
            "status": "healthy",
            "components": {},
        }

        # Check task queue
        try:
            task_queue = await get_task_queue()
            queue_stats = await task_queue.get_queue_stats()

            health_status["components"]["task_queue"] = {
                "status": "healthy",
                "pending_tasks": queue_stats.get("pending_tasks", 0),
                "running_tasks": queue_stats.get("running_tasks", 0),
            }
        except Exception as e:
            health_status["components"]["task_queue"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "degraded"

        # Check task registry
        try:
            task_registry = get_task_registry()
            registered_tasks = task_registry.list_tasks()

            health_status["components"]["task_registry"] = {
                "status": "healthy",
                "registered_tasks": len(registered_tasks),
            }
        except Exception as e:
            health_status["components"]["task_registry"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        logger.error(f"Task system health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.delete("/queue/{queue_name}/cleanup")
async def cleanup_queue(
    queue_name: str = "default", max_age_hours: int = Query(default=24, le=168)  # Max 1 week
) -> Dict[str, Any]:
    """
    Clean up old completed and failed tasks from a queue.

    Args:
        queue_name: Name of the queue to clean up
        max_age_hours: Maximum age of tasks to keep (in hours)

    Returns:
        Cleanup results
    """
    try:
        task_queue = await get_task_queue(queue_name)

        # Perform cleanup
        await task_queue.cleanup_completed_tasks(max_age_hours)

        return {
            "status": "completed",
            "queue_name": queue_name,
            "max_age_hours": max_age_hours,
            "cleaned_at": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to cleanup queue {queue_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Queue cleanup failed: {str(e)}")


# Test task for demonstration
from assistant.infrastructure.tasks.task_registry import task


@task(  # type: ignore[misc]
    name="test_task",
    description="Test task for debugging and monitoring",
    timeout_seconds=10,
    max_retries=2,
)
async def test_task(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple test task for debugging.

    Args:
        test_data: Test data dictionary

    Returns:
        Test result
    """
    logger = get_enhanced_logger("test_task")

    # Simulate work
    await asyncio.sleep(1)

    # Check if task should fail
    if test_data.get("should_fail", False):
        raise Exception("Test task intentionally failed")

    result = {
        "status": "completed",
        "message": test_data.get("message", "Test completed"),
        "processed_at": time.time(),
    }

    logger.info("Test task completed successfully")
    return result


# Worker management endpoints


@router.get("/workers")
async def list_workers() -> Dict[str, Any]:
    """
    List all active workers.

    Returns:
        List of workers with their status
    """
    try:
        worker_manager = get_worker_manager()
        workers = worker_manager.list_workers()
        manager_stats = worker_manager.get_manager_stats()

        return {"workers": workers, "manager_stats": manager_stats, "timestamp": time.time()}

    except Exception as e:
        logger.error(f"Failed to list workers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Worker listing failed: {str(e)}")


@router.get("/workers/{worker_id}/stats")
async def get_worker_stats(worker_id: str) -> Dict[str, Any]:
    """
    Get statistics for a specific worker.

    Args:
        worker_id: Worker ID

    Returns:
        Worker statistics
    """
    try:
        worker_manager = get_worker_manager()
        stats = worker_manager.get_worker_stats(worker_id)

        if stats is None:
            raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

        return {"worker_id": worker_id, "stats": stats, "timestamp": time.time()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get worker stats for {worker_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Worker stats retrieval failed: {str(e)}")


@router.get("/workers/{worker_id}/health")
async def get_worker_health(worker_id: str) -> Dict[str, Any]:
    """
    Get health status for a specific worker.

    Args:
        worker_id: Worker ID

    Returns:
        Worker health status
    """
    try:
        worker_manager = get_worker_manager()
        health = await worker_manager.get_worker_health(worker_id)

        if health is None:
            raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

        return health

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get worker health for {worker_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Worker health check failed: {str(e)}")


@router.get("/workers/health")
async def get_all_workers_health() -> Dict[str, Any]:
    """
    Get health status for all workers.

    Returns:
        Health status of all workers
    """
    try:
        worker_manager = get_worker_manager()
        health_checks = await worker_manager.get_all_worker_health()

        # Determine overall health
        overall_status = "healthy"
        unhealthy_workers = []

        for worker_id, health in health_checks.items():
            if health.get("status") != "healthy":
                overall_status = "degraded"
                unhealthy_workers.append(worker_id)

        return {
            "overall_status": overall_status,
            "total_workers": len(health_checks),
            "unhealthy_workers": unhealthy_workers,
            "workers": health_checks,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to get all workers health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workers health check failed: {str(e)}")


@router.post("/workers")
async def start_worker(
    queue_name: str = "default",
    concurrency: int = Query(default=4, ge=1, le=10),
    worker_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Start a new worker.

    Args:
        queue_name: Queue name to process
        concurrency: Number of concurrent tasks
        worker_id: Optional worker ID

    Returns:
        Worker start status
    """
    try:
        worker_manager = get_worker_manager()
        worker_id = await worker_manager.start_worker(
            worker_id=worker_id, queue_name=queue_name, concurrency=concurrency
        )

        return {
            "status": "started",
            "worker_id": worker_id,
            "queue_name": queue_name,
            "concurrency": concurrency,
            "started_at": time.time(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start worker: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Worker start failed: {str(e)}")


@router.delete("/workers/{worker_id}")
async def stop_worker(
    worker_id: str, timeout: float = Query(default=30.0, ge=1, le=300)
) -> Dict[str, Any]:
    """
    Stop a specific worker.

    Args:
        worker_id: Worker ID
        timeout: Shutdown timeout in seconds

    Returns:
        Worker stop status
    """
    try:
        worker_manager = get_worker_manager()
        success = await worker_manager.stop_worker(worker_id, timeout)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Worker not found or failed to stop: {worker_id}"
            )

        return {"status": "stopped", "worker_id": worker_id, "stopped_at": time.time()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop worker {worker_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Worker stop failed: {str(e)}")


@router.post("/workers/{worker_id}/restart")
async def restart_worker(worker_id: str) -> Dict[str, Any]:
    """
    Restart a specific worker.

    Args:
        worker_id: Worker ID

    Returns:
        Worker restart status
    """
    try:
        worker_manager = get_worker_manager()
        success = await worker_manager.restart_worker(worker_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Worker not found or failed to restart: {worker_id}"
            )

        return {"status": "restarted", "worker_id": worker_id, "restarted_at": time.time()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart worker {worker_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Worker restart failed: {str(e)}")
