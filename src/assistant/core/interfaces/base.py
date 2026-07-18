"""Base service interfaces and service registry for dependency injection."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class BaseService(ABC):
    """Base interface for all services."""

    @abstractmethod
    def get_service_name(self) -> str:
        """Return the unique service name."""

    async def initialize(self) -> None:
        """Initialize the service. Override if needed."""

    def cleanup(self) -> None:
        """Cleanup resources. Override if needed."""


class ServiceRegistry:
    """Dependency injection container for service registration and resolution."""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._interfaces: Dict[Type, str] = {}

    def register_singleton(
        self, interface: Type[T], implementation: T, name: Optional[str] = None
    ) -> None:
        """Register a singleton service instance."""
        service_name = name or interface.__name__
        self._singletons[service_name] = implementation
        self._interfaces[interface] = service_name

    def register_factory(
        self, interface: Type[T], factory: Callable[[], T], name: Optional[str] = None
    ) -> None:
        """Register a factory function for creating service instances."""
        service_name = name or interface.__name__
        self._factories[service_name] = factory
        self._interfaces[interface] = service_name

    def register_transient(
        self, interface: Type[T], implementation_class: Type[T], name: Optional[str] = None
    ) -> None:
        """Register a transient service (new instance each time)."""
        service_name = name or interface.__name__

        def factory():
            instance = implementation_class()
            return instance

        self._factories[service_name] = factory
        self._interfaces[interface] = service_name

    def get(self, interface: Type[T], name: Optional[str] = None) -> T:
        """Get a service instance by interface or name."""
        service_name = name or self._interfaces.get(interface)

        if not service_name:
            raise ValueError(f"Service not registered: {interface}")

        # Check singletons first
        if service_name in self._singletons:
            return self._singletons[service_name]

        # Check factories
        if service_name in self._factories:
            return self._factories[service_name]()

        raise ValueError(f"Service not found: {service_name}")

    def get_by_name(self, name: str) -> Any:
        """Get a service instance by name."""
        if name in self._singletons:
            return self._singletons[name]

        if name in self._factories:
            return self._factories[name]()

        raise ValueError(f"Service not found: {name}")

    def is_registered(self, interface: Type, name: Optional[str] = None) -> bool:
        """Check if a service is registered."""
        service_name = name or self._interfaces.get(interface)
        return service_name is not None and (
            service_name in self._singletons or service_name in self._factories
        )

    def cleanup_all(self) -> None:
        """Cleanup all registered singleton services."""
        for service in self._singletons.values():
            if isinstance(service, BaseService):
                try:
                    service.cleanup()
                except Exception as e:
                    print(f"Error cleaning up service {service}: {e}")

    def list_services(self) -> Dict[str, str]:
        """List all registered services."""
        services = {}
        for interface, name in self._interfaces.items():
            services[name] = interface.__name__
        return services


# Global service registry instance
_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry instance."""
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry


def register_service(
    interface: Type[T], implementation: T, singleton: bool = True, name: Optional[str] = None
) -> None:
    """Convenience function to register a service."""
    registry = get_service_registry()
    if singleton:
        registry.register_singleton(interface, implementation, name)
    else:
        registry.register_transient(interface, type(implementation), name)


def get_service(interface: Type[T], name: Optional[str] = None) -> T:
    """Convenience function to get a service."""
    registry = get_service_registry()
    return registry.get(interface, name)
