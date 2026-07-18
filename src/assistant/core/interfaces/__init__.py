"""Service interfaces and protocols for dependency injection."""

from .base import BaseService, ServiceRegistry
from .chat import IChatService, IClassificationService, IConversationService, IEmailService
from .infrastructure import IConfigurationManager, ILLMClient, IVectorStore

__all__ = [
    "BaseService",
    "ServiceRegistry",
    "IChatService",
    "IClassificationService",
    "IConversationService",
    "IEmailService",
    "IVectorStore",
    "ILLMClient",
    "IConfigurationManager",
]
