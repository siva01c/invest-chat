"""Chat service interfaces for dependency injection."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .base import BaseService


class IClassificationService(BaseService):
    """Interface for message classification service."""

    @abstractmethod
    async def classify_message_extended(
        self, user_text: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Classify user message into appropriate category with extended information."""

    @abstractmethod
    def get_action_for_category(self, category: str) -> str:
        """Get the action type for a given category."""

    @abstractmethod
    def is_technical_category(self, category: str) -> bool:
        """Check if category should maintain conversation history."""


class IConversationService(BaseService):
    """Interface for conversation management service."""

    @abstractmethod
    def add_interaction(
        self, user_message: str, assistant_response: str, category: Optional[str] = None
    ) -> None:
        """Add interaction to chat history."""

    @abstractmethod
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get recent interactions for context."""

    @abstractmethod
    def get_conversation_category(self) -> Optional[str]:
        """Get the current conversation category."""

    @abstractmethod
    def set_conversation_category(self, category: str) -> None:
        """Set the conversation category."""

    @abstractmethod
    def clear_history(self) -> None:
        """Clear conversation history."""

    @abstractmethod
    def handle_inappropriate_request(self, user_text: str) -> Dict[str, Any]:
        """Handle inappropriate requests with polite redirection."""

    @abstractmethod
    async def generate_conversation_summary(self, user_message: str) -> str:
        """Generate a comprehensive summary of the full conversation."""

    @abstractmethod
    def check_greeting(self, user_text: str) -> Optional[Dict[str, Any]]:
        """Check if message is a greeting and return appropriate response."""

    @abstractmethod
    def check_curiosity_response(self, user_text: str) -> Optional[Dict[str, Any]]:
        """Check if user responded with curiosity to a technical question."""


class IEmailService(BaseService):
    """Interface for email service."""

    @abstractmethod
    def check_email_confirmation_context(
        self, last_response: str, user_text: str
    ) -> Optional[Dict[str, Any]]:
        """Check if we're in an email confirmation context."""

    @abstractmethod
    async def process_email_confirmation(
        self, user_text: str, session_id: str, conversation_service: IConversationService
    ) -> Dict[str, Any]:
        """Process email confirmation and send message."""

    @abstractmethod
    async def handle_forward_message(
        self, user_text: str, session_id: str, conversation_service: IConversationService
    ) -> Dict[str, Any]:
        """Handle simple message forwarding."""

    @abstractmethod
    def handle_cybersecurity_urgent(
        self, user_text: str, session_id: str, conversation_service: IConversationService
    ) -> Dict[str, Any]:
        """Handle urgent cybersecurity requests."""

    @abstractmethod
    def handle_service_details(
        self, user_text: str, conversation_service: IConversationService
    ) -> Dict[str, Any]:
        """Handle service detail requests and contact information."""


class IChatService(BaseService):
    """Interface for main chat service."""

    @abstractmethod
    async def chat(self, user_text: str, session_id: Optional[str] = None) -> str:
        """Process user input and generate response."""

    @abstractmethod
    async def handle_user_request(
        self, user_text: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle user requests with modular service architecture."""
