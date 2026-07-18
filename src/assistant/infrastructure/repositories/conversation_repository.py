"""Conversation repository implementation for chat history management."""

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from assistant.core.exceptions import DatabaseException, ErrorCode
from assistant.core.interfaces.repository import ChatInteraction, IConversationRepository
from assistant.core.logging import get_logger, log_service_method
from assistant.services.chat_history import ChatEntry, ChatHistory


@dataclass
class ConversationInteraction:
    """Implementation of ChatInteraction protocol."""

    id: Optional[str]
    user_message: str
    assistant_response: str
    timestamp: datetime
    category: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


class ConversationRepository(IConversationRepository):
    """Repository for managing conversation history and chat interactions."""

    def __init__(self, max_history_per_session: int = 100):
        """
        Initialize the conversation repository.

        Args:
            max_history_per_session: Maximum interactions to keep per session
        """
        self.logger = get_logger(self.__class__.__name__)
        self.max_history = max_history_per_session

        # In-memory storage for conversations
        # In production, this would be backed by a database
        self._conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        self._conversation_categories: Dict[str, str] = {}
        self._interaction_index: Dict[str, ConversationInteraction] = {}

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "ConversationRepository"

    @log_service_method()
    async def save_interaction(
        self,
        user_message: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a chat interaction."""
        if not user_message or not user_message.strip():
            raise DatabaseException(
                "User message cannot be empty",
                operation="save_interaction",
                error_code=ErrorCode.INVALID_INPUT,
            )

        if not assistant_response or not assistant_response.strip():
            raise DatabaseException(
                "Assistant response cannot be empty",
                operation="save_interaction",
                error_code=ErrorCode.INVALID_INPUT,
            )

        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create interaction
        interaction = ConversationInteraction(
            id=str(uuid.uuid4()),
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.utcnow(),
            category=category,
            session_id=session_id,
            metadata=metadata or {},
        )

        # Store interaction
        self._conversations[session_id].append(interaction)
        self._interaction_index[interaction.id] = interaction

        # Update conversation category if provided
        if category:
            self._conversation_categories[session_id] = category

        self.logger.info(f"Saved interaction {interaction.id} for session {session_id}")
        return interaction.id

    @log_service_method()
    async def get_conversation_history(
        self, session_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[ChatInteraction]:
        """Retrieve conversation history for a session."""
        if not session_id or not session_id.strip():
            raise DatabaseException(
                "Session ID cannot be empty",
                operation="get_conversation_history",
                error_code=ErrorCode.INVALID_INPUT,
            )

        if session_id not in self._conversations:
            return []

        interactions = list(self._conversations[session_id])

        # Apply offset and limit
        start = offset
        end = offset + limit if limit is not None else None
        result = interactions[start:end]

        self.logger.debug(f"Retrieved {len(result)} interactions for session {session_id}")
        return result

    @log_service_method()
    async def get_recent_interactions(
        self, session_id: str, count: int = 5
    ) -> List[ChatInteraction]:
        """Get the most recent interactions for a session."""
        if count <= 0:
            raise DatabaseException(
                "Count must be positive",
                operation="get_recent_interactions",
                error_code=ErrorCode.INVALID_INPUT,
            )

        if session_id not in self._conversations:
            return []

        interactions = list(self._conversations[session_id])
        # Get the last 'count' interactions
        recent = interactions[-count:] if len(interactions) > count else interactions

        self.logger.debug(f"Retrieved {len(recent)} recent interactions for session {session_id}")
        return recent

    @log_service_method()
    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if not session_id or not session_id.strip():
            raise DatabaseException(
                "Session ID cannot be empty",
                operation="clear_conversation",
                error_code=ErrorCode.INVALID_INPUT,
            )

        try:
            # Remove interactions from index
            if session_id in self._conversations:
                for interaction in self._conversations[session_id]:
                    if interaction.id in self._interaction_index:
                        del self._interaction_index[interaction.id]

            # Clear conversation and category
            if session_id in self._conversations:
                del self._conversations[session_id]
            if session_id in self._conversation_categories:
                del self._conversation_categories[session_id]

            self.logger.info(f"Cleared conversation for session {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear conversation {session_id}: {str(e)}")
            raise DatabaseException(
                f"Failed to clear conversation: {str(e)}",
                operation="clear_conversation",
                error_code=ErrorCode.DATABASE_QUERY_ERROR,
                details={"session_id": session_id},
                cause=e,
            )

    async def get_conversation_summary(
        self, session_id: str, interaction_limit: Optional[int] = None
    ) -> Optional[str]:
        """Get a summary of the conversation."""
        interactions = await self.get_conversation_history(session_id, limit=interaction_limit)

        if not interactions:
            return None

        # Create a simple summary
        total_interactions = len(interactions)
        categories = set(i.category for i in interactions if i.category)
        first_interaction = interactions[0] if interactions else None
        last_interaction = interactions[-1] if interactions else None

        summary_parts = [f"Conversation with {total_interactions} interactions"]

        if categories:
            summary_parts.append(f"Categories: {', '.join(categories)}")

        if first_interaction:
            summary_parts.append(f"Started: {first_interaction.timestamp.isoformat()}")

        if last_interaction:
            summary_parts.append(f"Last activity: {last_interaction.timestamp.isoformat()}")

        return "; ".join(summary_parts)

    async def set_conversation_category(self, session_id: str, category: str) -> bool:
        """Set the category for a conversation."""
        if not session_id or not session_id.strip():
            raise DatabaseException(
                "Session ID cannot be empty",
                operation="set_conversation_category",
                error_code=ErrorCode.INVALID_INPUT,
            )

        if not category or not category.strip():
            raise DatabaseException(
                "Category cannot be empty",
                operation="set_conversation_category",
                error_code=ErrorCode.INVALID_INPUT,
            )

        self._conversation_categories[session_id] = category
        self.logger.debug(f"Set category '{category}' for session {session_id}")
        return True

    async def get_conversation_category(self, session_id: str) -> Optional[str]:
        """Get the category for a conversation."""
        if not session_id or not session_id.strip():
            return None

        return self._conversation_categories.get(session_id)

    @log_service_method()
    async def search_conversations(
        self,
        query: str,
        limit: int = 10,
        category_filter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[ChatInteraction]:
        """Search conversations by content."""
        if not query or not query.strip():
            raise DatabaseException(
                "Search query cannot be empty",
                operation="search_conversations",
                error_code=ErrorCode.INVALID_INPUT,
            )

        query_lower = query.lower()
        results = []

        # Search through all interactions
        for interaction in self._interaction_index.values():
            # Apply filters
            if category_filter and interaction.category != category_filter:
                continue

            if date_from and interaction.timestamp < date_from:
                continue

            if date_to and interaction.timestamp > date_to:
                continue

            # Search in message content
            if (
                query_lower in interaction.user_message.lower()
                or query_lower in interaction.assistant_response.lower()
            ):
                results.append(interaction)

            if len(results) >= limit:
                break

        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x.timestamp, reverse=True)

        self.logger.info(f"Search for '{query}' returned {len(results)} results")
        return results[:limit]

    async def get_conversation_stats(
        self,
        session_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get conversation statistics."""
        interactions_to_analyze = []

        if session_id:
            # Get interactions for specific session
            if session_id in self._conversations:
                interactions_to_analyze = list(self._conversations[session_id])
        else:
            # Get all interactions
            interactions_to_analyze = list(self._interaction_index.values())

        # Apply date filters
        if date_from or date_to:
            filtered_interactions = []
            for interaction in interactions_to_analyze:
                if date_from and interaction.timestamp < date_from:
                    continue
                if date_to and interaction.timestamp > date_to:
                    continue
                filtered_interactions.append(interaction)
            interactions_to_analyze = filtered_interactions

        # Calculate statistics
        total_interactions = len(interactions_to_analyze)
        unique_sessions = len(set(i.session_id for i in interactions_to_analyze if i.session_id))
        categories = {}

        for interaction in interactions_to_analyze:
            if interaction.category:
                categories[interaction.category] = categories.get(interaction.category, 0) + 1

        # Calculate average message lengths
        user_msg_lengths = [len(i.user_message) for i in interactions_to_analyze]
        assistant_msg_lengths = [len(i.assistant_response) for i in interactions_to_analyze]

        avg_user_msg_length = (
            sum(user_msg_lengths) / len(user_msg_lengths) if user_msg_lengths else 0
        )
        avg_assistant_msg_length = (
            sum(assistant_msg_lengths) / len(assistant_msg_lengths) if assistant_msg_lengths else 0
        )

        stats = {
            "total_interactions": total_interactions,
            "unique_sessions": unique_sessions,
            "categories": categories,
            "avg_user_message_length": round(avg_user_msg_length, 2),
            "avg_assistant_message_length": round(avg_assistant_msg_length, 2),
        }

        if interactions_to_analyze:
            stats["date_range"] = {
                "earliest": min(i.timestamp for i in interactions_to_analyze).isoformat(),
                "latest": max(i.timestamp for i in interactions_to_analyze).isoformat(),
            }

        return stats

    # Migration methods for existing ChatHistory compatibility
    def from_chat_history(self, chat_history: ChatHistory, session_id: str) -> None:
        """
        Migrate data from existing ChatHistory to repository.

        Args:
            chat_history: Existing ChatHistory instance
            session_id: Session ID to assign to the interactions
        """
        try:
            # Get full history from ChatHistory
            history = chat_history.get_full_history()

            for entry in history:
                interaction = ConversationInteraction(
                    id=str(uuid.uuid4()),
                    user_message=entry.user_message,
                    assistant_response=entry.assistant_response,
                    timestamp=(
                        datetime.fromtimestamp(entry.timestamp)
                        if entry.timestamp
                        else datetime.utcnow()
                    ),
                    category=entry.category,
                    session_id=session_id,
                )

                self._conversations[session_id].append(interaction)
                self._interaction_index[interaction.id] = interaction

            # Migrate conversation category
            if (
                hasattr(chat_history, "conversation_category")
                and chat_history.conversation_category
            ):
                self._conversation_categories[session_id] = chat_history.conversation_category

            self.logger.info(
                f"Migrated {len(history)} interactions from ChatHistory to session {session_id}"
            )

        except Exception as e:
            self.logger.error(f"Failed to migrate ChatHistory: {str(e)}")
            raise DatabaseException(
                f"Failed to migrate chat history: {str(e)}",
                operation="from_chat_history",
                error_code=ErrorCode.DATABASE_QUERY_ERROR,
                cause=e,
            )

    def to_chat_history(self, session_id: str) -> ChatHistory:
        """
        Convert repository data to ChatHistory format for backward compatibility.

        Args:
            session_id: Session ID to convert

        Returns:
            ChatHistory instance with the session's interactions
        """
        chat_history = ChatHistory(max_history=self.max_history)

        if session_id in self._conversations:
            for interaction in self._conversations[session_id]:
                entry = ChatEntry(
                    user_message=interaction.user_message,
                    assistant_response=interaction.assistant_response,
                    timestamp=interaction.timestamp.timestamp(),
                    category=interaction.category,
                )
                chat_history.history.append(entry)

        # Set conversation category
        if session_id in self._conversation_categories:
            chat_history.conversation_category = self._conversation_categories[session_id]

        return chat_history
