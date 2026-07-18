"""Service registration and dependency injection configuration."""

from typing import Optional

from assistant.config import get_settings
from assistant.config.loader import ConfigManager
from assistant.core.config.repositories import configure_repositories
from assistant.core.interfaces.base import ServiceRegistry, get_service_registry
from assistant.core.interfaces.chat import (
    IChatService,
    IClassificationService,
    IConversationService,
    IEmailService,
)
from assistant.core.interfaces.infrastructure import IConfigurationManager, ILLMClient, IVectorStore
from assistant.core.services.classification_service import ClassificationService
from assistant.core.services.conversation_service import ConversationService
from assistant.core.services.email_service import EmailService
from assistant.core.services.mcp_service import MCPService
from assistant.core.services.refactored_chat_service import RefactoredAIService
from assistant.infrastructure.database.vector_store import VectorStore
from assistant.infrastructure.llm.openai_client import OpenAIClient


def configure_services(registry: Optional[ServiceRegistry] = None) -> ServiceRegistry:
    """
    Configure and register all services with the dependency injection container.

    Args:
        registry: Optional existing registry to use. If None, creates a new one.

    Returns:
        Configured service registry
    """
    if registry is None:
        registry = get_service_registry()

    # Get settings for configuration
    settings = get_settings()

    # Register infrastructure services as singletons
    config_manager = ConfigManager()
    registry.register_singleton(IConfigurationManager, config_manager, "ConfigurationManager")

    vector_store = VectorStore(
        collection_name=settings.chromadb_collection_name,
        database_path=settings.chromadb_database_path,
    )
    registry.register_singleton(IVectorStore, vector_store, "VectorStore")

    # Register LLM client as singleton
    llm_client = OpenAIClient()
    registry.register_singleton(ILLMClient, llm_client, "OpenAIClient")

    # Register core services as singletons
    classification_service = ClassificationService()
    registry.register_singleton(
        IClassificationService, classification_service, "ClassificationService"
    )

    conversation_service = ConversationService(
        max_history=settings.max_history, context_window=settings.context_window
    )
    registry.register_singleton(IConversationService, conversation_service, "ConversationService")

    email_service = EmailService()
    registry.register_singleton(IEmailService, email_service, "EmailService")

    # Register MCP service if enabled
    if settings.enable_mcp and settings.apify_api_key:
        mcp_service = MCPService(
            apify_api_key=settings.apify_api_key, enabled_tools=settings.mcp_enabled_tools
        )
        registry.register_singleton(MCPService, mcp_service, "MCPService")

    # Register main chat service with dependency injection
    chat_service = RefactoredAIService(
        classification_service=classification_service,
        conversation_service=conversation_service,
        email_service=email_service,
    )
    registry.register_singleton(IChatService, chat_service, "ChatService")

    # Configure repositories
    configure_repositories(registry)

    return registry


def get_chat_service() -> IChatService:
    """
    Get the configured chat service instance.

    Returns:
        Configured chat service
    """
    registry = get_service_registry()

    # Check if services are configured
    if not registry.is_registered(IChatService):
        configure_services(registry)

    return registry.get(IChatService)


def get_classification_service() -> IClassificationService:
    """
    Get the classification service instance.

    Returns:
        Classification service
    """
    registry = get_service_registry()

    # Check if services are configured
    if not registry.is_registered(IClassificationService):
        configure_services(registry)

    return registry.get(IClassificationService)


def get_conversation_service() -> IConversationService:
    """
    Get the conversation service instance.

    Returns:
        Conversation service
    """
    registry = get_service_registry()

    # Check if services are configured
    if not registry.is_registered(IConversationService):
        configure_services(registry)

    return registry.get(IConversationService)


def get_mcp_service() -> Optional[MCPService]:
    """
    Get the MCP service instance if available.

    Returns:
        MCP service instance or None if not configured
    """
    registry = get_service_registry()

    # Check if services are configured
    if not registry.is_registered(IChatService):  # Use IChatService as a proxy for configuration
        configure_services(registry)

    # Return MCP service if registered
    if registry.is_registered(MCPService):
        return registry.get(MCPService)
    return None


def get_email_service() -> IEmailService:
    """
    Get the email service instance.

    Returns:
        Email service
    """
    registry = get_service_registry()

    # Check if services are configured
    if not registry.is_registered(IEmailService):
        configure_services(registry)

    return registry.get(IEmailService)


def get_vector_store() -> IVectorStore:
    """
    Get the vector store instance.

    Returns:
        Vector store
    """
    registry = get_service_registry()

    # Check if services are configured
    if not registry.is_registered(IVectorStore):
        configure_services(registry)

    return registry.get(IVectorStore)


def get_llm_client() -> ILLMClient:
    """
    Get the LLM client instance.

    Returns:
        LLM client
    """
    registry = get_service_registry()
    return registry.get(ILLMClient)


def get_configuration_manager() -> IConfigurationManager:
    """
    Get the configuration manager instance.

    Returns:
        Configuration manager
    """
    registry = get_service_registry()
    return registry.get(IConfigurationManager)


def cleanup_services() -> None:
    """Clean up all registered services."""
    registry = get_service_registry()
    registry.cleanup_all()


async def initialize_services() -> None:
    """Initialize all services in the correct order."""
    # This function is called once at application startup
    # to ensure all services are properly configured
    configure_services()

    # Initialize MCP service if available
    mcp_service = get_mcp_service()
    if mcp_service:
        try:
            await mcp_service.initialize()
        except Exception as e:
            # Log error but don't fail startup
            import logging

            logging.getLogger(__name__).warning(f"Failed to initialize MCP service: {e}")

    # Optionally perform any additional initialization
    print("Services initialized successfully")


# Convenience function for backwards compatibility
def setup_dependency_injection() -> ServiceRegistry:
    """
    Set up dependency injection for the application.

    Returns:
        Configured service registry
    """
    return configure_services()
