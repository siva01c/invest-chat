"""Task registry for managing registered background tasks."""

import asyncio
from functools import wraps
from typing import Any, Callable, Dict, Optional

from assistant.infrastructure.observability import get_enhanced_logger


class TaskRegistry:
    """
    Registry for background tasks.

    Features:
    - Task registration and discovery
    - Type checking and validation
    - Task metadata management
    - Automatic function wrapping
    """

    def __init__(self):
        """Initialize task registry."""
        self.logger = get_enhanced_logger(self.__class__.__name__)
        self._tasks: Dict[str, Callable] = {}
        self._task_metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 60,
    ):
        """
        Decorator to register a task function.

        Args:
            name: Task name (defaults to function name)
            description: Task description
            timeout_seconds: Task timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay_seconds: Delay between retries

        Returns:
            Decorated function
        """

        def decorator(func: Callable):
            task_name = name or func.__name__

            # Validate function
            if not callable(func):
                raise ValueError(f"Task {task_name} must be callable")

            # Check if it's async
            is_async = asyncio.iscoroutinefunction(func)

            # Store task metadata
            self._task_metadata[task_name] = {
                "function": func,
                "description": description or func.__doc__ or "No description",
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
                "retry_delay_seconds": retry_delay_seconds,
                "is_async": is_async,
                "module": func.__module__,
                "qualname": func.__qualname__,
            }

            # Wrap function for execution
            if is_async:

                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)

                self._tasks[task_name] = async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return func(*args, **kwargs)

                self._tasks[task_name] = sync_wrapper

            self.logger.info(f"Registered task: {task_name} ({'async' if is_async else 'sync'})")
            return func

        return decorator

    def get_task(self, name: str) -> Optional[Callable]:
        """
        Get a registered task by name.

        Args:
            name: Task name

        Returns:
            Task function or None if not found
        """
        return self._tasks.get(name)

    def get_task_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get task metadata.

        Args:
            name: Task name

        Returns:
            Task metadata or None if not found
        """
        return self._task_metadata.get(name)

    def is_registered(self, name: str) -> bool:
        """
        Check if a task is registered.

        Args:
            name: Task name

        Returns:
            True if task is registered
        """
        return name in self._tasks

    def list_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered tasks with metadata.

        Returns:
            Dictionary of task names and their metadata
        """
        return {
            name: {
                "description": metadata["description"],
                "timeout_seconds": metadata["timeout_seconds"],
                "max_retries": metadata["max_retries"],
                "retry_delay_seconds": metadata["retry_delay_seconds"],
                "is_async": metadata["is_async"],
                "module": metadata["module"],
            }
            for name, metadata in self._task_metadata.items()
        }

    async def execute_task(self, name: str, *args, **kwargs) -> Any:
        """
        Execute a registered task.

        Args:
            name: Task name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Task execution result

        Raises:
            ValueError: If task is not registered
        """
        if not self.is_registered(name):
            raise ValueError(f"Task '{name}' is not registered")

        task_func = self._tasks[name]
        metadata = self._task_metadata[name]

        try:
            if metadata["is_async"]:
                result = await task_func(*args, **kwargs)
            else:
                # Run sync function in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: task_func(*args, **kwargs))

            return result

        except Exception as e:
            self.logger.error(f"Task execution failed: {name} - {str(e)}", exc_info=True)
            raise

    def unregister(self, name: str) -> bool:
        """
        Unregister a task.

        Args:
            name: Task name

        Returns:
            True if task was unregistered, False if not found
        """
        if name in self._tasks:
            del self._tasks[name]
            del self._task_metadata[name]
            self.logger.info(f"Unregistered task: {name}")
            return True
        return False


# Global task registry instance
_task_registry = TaskRegistry()


def get_task_registry() -> TaskRegistry:
    """Get the global task registry instance."""
    return _task_registry


def task(
    name: Optional[str] = None,
    description: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 60,
):
    """
    Decorator to register a background task.

    Args:
        name: Task name (defaults to function name)
        description: Task description
        timeout_seconds: Task timeout in seconds
        max_retries: Maximum retry attempts
        retry_delay_seconds: Delay between retries

    Returns:
        Decorated function

    Example:
        @task(name="send_email", max_retries=5)
        async def send_email_task(recipient: str, subject: str, body: str):
            # Implementation here
            pass
    """
    return _task_registry.register(
        name=name,
        description=description,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )
