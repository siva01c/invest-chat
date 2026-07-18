import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChatEntry:
    """Represents a single chat interaction with user message and assistant response"""

    user_message: str
    assistant_response: str
    timestamp: Optional[float] = None
    category: Optional[str] = None

    def __post_init__(self):
        """Set timestamp to current time if not provided"""
        if self.timestamp is None:
            self.timestamp = time.time()


class ChatHistory:
    def __init__(self, max_history: int = 10):
        """
        Initialize chat history with maximum number of entries to store

        Args:
            max_history (int): Maximum number of chat interactions to keep in history
        """
        self.max_history = max_history
        self.history: deque[ChatEntry] = deque(maxlen=max_history)
        self.conversation_category: Optional[str] = None

    def add_interaction(
        self, user_message: str, assistant_response: str, category: Optional[str] = None
    ) -> None:
        """
        Add a new chat interaction to the history

        Args:
            user_message (str): The message from the user
            assistant_response (str): The response from the assistant
            category (str, optional): The conversation category for this interaction
        """
        entry = ChatEntry(
            user_message=user_message, assistant_response=assistant_response, category=category
        )
        self.history.append(entry)

    # This method could be async if we were storing history in a database
    async def add_interaction_async(
        self, user_message: str, assistant_response: str, category: Optional[str] = None
    ) -> None:
        """
        Add a new chat interaction to the history asynchronously

        Args:
            user_message (str): The message from the user
            assistant_response (str): The response from the assistant
            category (str, optional): The conversation category for this interaction
        """
        # Create entry
        entry = ChatEntry(
            user_message=user_message, assistant_response=assistant_response, category=category
        )
        # For demonstration purposes, using to_thread even though this is an in-memory operation
        # In a real application, this would be a database call
        await asyncio.to_thread(self.history.append, entry)

    def get_user_messages(self) -> List[str]:
        """Get all user messages in chronological order"""
        return [entry.user_message for entry in self.history]

    async def get_user_messages_async(self) -> List[str]:
        """Get all user messages in chronological order asynchronously"""
        return await asyncio.to_thread(lambda: [entry.user_message for entry in self.history])

    def get_assistant_responses(self) -> List[str]:
        """Get all assistant responses in chronological order"""
        return [entry.assistant_response for entry in self.history]

    async def get_assistant_responses_async(self) -> List[str]:
        """Get all assistant responses in chronological order asynchronously"""
        return await asyncio.to_thread(lambda: [entry.assistant_response for entry in self.history])

    def get_last_n_interactions(self, n: int) -> List[ChatEntry]:
        """
        Get the last n chat interactions

        Args:
            n (int): Number of interactions to retrieve

        Returns:
            List of the last n ChatEntry objects
        """
        return list(self.history)[-n:]

    async def get_last_n_interactions_async(self, n: int) -> List[ChatEntry]:
        """
        Get the last n chat interactions asynchronously

        Args:
            n (int): Number of interactions to retrieve

        Returns:
            List of the last n ChatEntry objects
        """
        return await asyncio.to_thread(lambda: list(self.history)[-n:])

    def clear_history(self) -> None:
        """Clear all chat history and conversation category"""
        self.history.clear()
        self.conversation_category = None

    async def clear_history_async(self) -> None:
        """Clear all chat history and conversation category asynchronously"""
        await asyncio.to_thread(self.history.clear)
        self.conversation_category = None

    def get_full_history(self) -> List[ChatEntry]:
        """Get all chat interactions as a list"""
        return list(self.history)

    async def get_full_history_async(self) -> List[ChatEntry]:
        """Get all chat interactions as a list asynchronously"""
        return await asyncio.to_thread(list, self.history)

    def set_conversation_category(self, category: str) -> None:
        """
        Set the conversation category for this chat session

        Args:
            category (str): The category to set (e.g., 'drupal', 'ai', 'cybersecurity')
        """
        self.conversation_category = category

    def get_conversation_category(self) -> Optional[str]:
        """
        Get the current conversation category

        Returns:
            Optional[str]: The current conversation category or None if not set
        """
        return self.conversation_category

    def has_conversation_category(self) -> bool:
        """
        Check if a conversation category is set

        Returns:
            bool: True if category is set, False otherwise
        """
        return self.conversation_category is not None

    def clear_conversation_category(self) -> None:
        """Clear the conversation category"""
        self.conversation_category = None

    def __len__(self) -> int:
        """Return the number of interactions in history"""
        return len(self.history)

    def __str__(self) -> str:
        """String representation of the chat history"""
        if not self.history:
            return "Chat history is empty"

        output = []
        for i, entry in enumerate(self.history, 1):
            output.append(f"Interaction {i}:")
            output.append(f"User: {entry.user_message}")
            output.append(f"Assistant: {entry.assistant_response}")
            output.append("")

        return "\n".join(output)
