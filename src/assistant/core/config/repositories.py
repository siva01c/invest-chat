"""Repository configuration and factory for dependency injection."""

from typing import Optional

from assistant.config import get_settings
from assistant.core.interfaces.base import ServiceRegistry, get_service_registry
from assistant.core.interfaces.repository import (
    IConfigurationRepository,
    IConversationRepository,
    IKnowledgeRepository,
    IVectorRepository,
)
from assistant.infrastructure.database.vector_store import VectorStore
from assistant.infrastructure.repositories.conversation_repository import ConversationRepository
from assistant.infrastructure.repositories.vector_repository import VectorStoreRepository


def configure_repositories(registry: Optional[ServiceRegistry] = None) -> ServiceRegistry:
    """
    Configure and register all repositories with the dependency injection container.

    Args:
        registry: Optional existing registry to use. If None, creates a new one.

    Returns:
        Configured service registry
    """
    if registry is None:
        registry = get_service_registry()

    # Get settings for configuration
    settings = get_settings()

    # Register Vector Repository as singleton
    vector_store = VectorStore(
        collection_name=settings.chromadb_collection_name,
        database_path=settings.chromadb_database_path,
    )
    vector_repository = VectorStoreRepository(vector_store=vector_store)
    registry.register_singleton(IVectorRepository, vector_repository, "VectorRepository")

    # Register Conversation Repository as singleton
    conversation_repository = ConversationRepository(max_history_per_session=settings.max_history)
    registry.register_singleton(
        IConversationRepository, conversation_repository, "ConversationRepository"
    )

    return registry


def get_vector_repository() -> IVectorRepository:
    """
    Get the configured vector repository instance.

    Returns:
        Configured vector repository
    """
    registry = get_service_registry()

    # Check if repositories are configured
    if not registry.is_registered(IVectorRepository):
        configure_repositories(registry)

    return registry.get(IVectorRepository)


def get_conversation_repository() -> IConversationRepository:
    """
    Get the conversation repository instance.

    Returns:
        Conversation repository
    """
    registry = get_service_registry()

    # Check if repositories are configured
    if not registry.is_registered(IConversationRepository):
        configure_repositories(registry)

    return registry.get(IConversationRepository)


def get_configuration_repository() -> IConfigurationRepository:
    """
    Get the configuration repository instance.

    Returns:
        Configuration repository
    """
    registry = get_service_registry()

    # Check if repositories are configured
    if not registry.is_registered(IConfigurationRepository):
        configure_repositories(registry)

    return registry.get(IConfigurationRepository)


def get_knowledge_repository() -> IKnowledgeRepository:
    """
    Get the knowledge repository instance.

    Returns:
        Knowledge repository
    """
    registry = get_service_registry()

    # Check if repositories are configured
    if not registry.is_registered(IKnowledgeRepository):
        configure_repositories(registry)

    return registry.get(IKnowledgeRepository)


def initialize_repositories() -> None:
    """Initialize all repositories in the correct order."""
    # This function is called once at application startup
    # to ensure all repositories are properly configured
    configure_repositories()
    print("Repositories initialized successfully")


# Convenience function for backwards compatibility
def setup_repository_injection() -> ServiceRegistry:
    """
    Set up repository dependency injection for the application.

    Returns:
        Configured service registry
    """
    return configure_repositories()
