import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChatEntry:
    """Represents a single chat interaction with user message and assistant response."""

    user_message: str
    assistant_response: str
    timestamp: float = field(default_factory=time.time)


class ChatHistory:
    """In-memory sliding-window chat history.

    Keeps the last *max_history* interactions for conversation context.
    One instance per user session — do not share across sessions.
    """

    def __init__(self, max_history: int = 10):
        """Initialize chat history.

        Args:
            max_history: Maximum number of chat interactions to keep.
        """
        self.max_history = max_history
        self.history: deque[ChatEntry] = deque(maxlen=max_history)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_interaction(self, user_message: str, assistant_response: str) -> None:
        """Append a new interaction to history."""
        self.history.append(ChatEntry(user_message=user_message, assistant_response=assistant_response))

    def clear_history(self) -> None:
        """Remove all stored interactions."""
        self.history.clear()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_last_n_interactions(self, n: int) -> List[ChatEntry]:
        """Return the last *n* interactions in chronological order."""
        return list(self.history)[-n:]

    def get_full_history(self) -> List[ChatEntry]:
        """Return all stored interactions."""
        return list(self.history)

    def get_user_messages(self) -> List[str]:
        """Return all user messages in chronological order."""
        return [e.user_message for e in self.history]

    def get_assistant_responses(self) -> List[str]:
        """Return all assistant responses in chronological order."""
        return [e.assistant_response for e in self.history]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.history)

    def __str__(self) -> str:
        if not self.history:
            return "Chat history is empty"
        lines = []
        for i, entry in enumerate(self.history, 1):
            lines += [
                f"Interaction {i}:",
                f"  User: {entry.user_message}",
                f"  Assistant: {entry.assistant_response}",
                "",
            ]
        return "\n".join(lines)
