"""Repository interfaces for data access abstraction."""

from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from .base import BaseService


class Document(Protocol):
    """Protocol for document objects stored in repositories."""

    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: Optional[datetime]


class SearchResult(Protocol):
    """Protocol for search result objects."""

    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    distance: float


class ChatInteraction(Protocol):
    """Protocol for chat interaction objects."""

    id: Optional[str]
    user_message: str
    assistant_response: str
    timestamp: datetime
    category: Optional[str]
    session_id: Optional[str]


class IVectorRepository(BaseService):
    """Interface for vector-based document storage and retrieval."""

    @abstractmethod
    async def store_document(
        self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a single document in the vector store.

        Args:
            document_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata associated with the document

        Returns:
            True if storage was successful, False otherwise
        """

    @abstractmethod
    async def store_documents(self, documents: List[Document]) -> int:
        """
        Store multiple documents in the vector store.

        Args:
            documents: List of documents to store

        Returns:
            Number of documents successfully stored
        """

    @abstractmethod
    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            metadata_filter: Optional metadata filters to apply

        Returns:
            List of search results ordered by similarity
        """

    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.

        Args:
            document_id: Unique identifier of the document

        Returns:
            Document if found, None otherwise
        """

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            document_id: Unique identifier of the document to delete

        Returns:
            True if deletion was successful, False otherwise
        """

    @abstractmethod
    async def get_all_documents(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> List[Document]:
        """
        Retrieve all documents from the vector store.

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of documents
        """

    @abstractmethod
    async def count_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the vector store.

        Args:
            metadata_filter: Optional metadata filters to apply

        Returns:
            Number of documents matching the filter
        """

    @abstractmethod
    async def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific document.

        Args:
            document_id: Unique identifier of the document
            metadata: New metadata to set

        Returns:
            True if update was successful, False otherwise
        """


class IConversationRepository(BaseService):
    """Interface for conversation and chat history management."""

    @abstractmethod
    async def save_interaction(
        self,
        user_message: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a chat interaction.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            session_id: Optional session identifier
            category: Optional conversation category
            metadata: Optional additional metadata

        Returns:
            Unique identifier for the saved interaction
        """

    @abstractmethod
    async def get_conversation_history(
        self, session_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[ChatInteraction]:
        """
        Retrieve conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to return
            offset: Number of interactions to skip

        Returns:
            List of chat interactions ordered by timestamp
        """

    @abstractmethod
    async def get_recent_interactions(
        self, session_id: str, count: int = 5
    ) -> List[ChatInteraction]:
        """
        Get the most recent interactions for a session.

        Args:
            session_id: Session identifier
            count: Number of recent interactions to retrieve

        Returns:
            List of recent chat interactions
        """

    @abstractmethod
    async def clear_conversation(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if clearing was successful, False otherwise
        """

    @abstractmethod
    async def get_conversation_summary(
        self, session_id: str, interaction_limit: Optional[int] = None
    ) -> Optional[str]:
        """
        Get a summary of the conversation.

        Args:
            session_id: Session identifier
            interaction_limit: Limit number of interactions to include

        Returns:
            Conversation summary or None if no conversation exists
        """

    @abstractmethod
    async def set_conversation_category(self, session_id: str, category: str) -> bool:
        """
        Set the category for a conversation.

        Args:
            session_id: Session identifier
            category: Conversation category to set

        Returns:
            True if setting was successful, False otherwise
        """

    @abstractmethod
    async def get_conversation_category(self, session_id: str) -> Optional[str]:
        """
        Get the category for a conversation.

        Args:
            session_id: Session identifier

        Returns:
            Conversation category or None if not set
        """

    @abstractmethod
    async def search_conversations(
        self,
        query: str,
        limit: int = 10,
        category_filter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[ChatInteraction]:
        """
        Search conversations by content.

        Args:
            query: Search query
            limit: Maximum number of results
            category_filter: Optional category filter
            date_from: Optional start date filter
            date_to: Optional end date filter

        Returns:
            List of matching chat interactions
        """

    @abstractmethod
    async def get_conversation_stats(
        self,
        session_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get conversation statistics.

        Args:
            session_id: Optional session filter
            date_from: Optional start date filter
            date_to: Optional end date filter

        Returns:
            Dictionary with conversation statistics
        """


class IConfigurationRepository(BaseService):
    """Interface for configuration data management."""

    @abstractmethod
    async def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration by name.

        Args:
            config_name: Name of the configuration

        Returns:
            Configuration data or None if not found
        """

    @abstractmethod
    async def set_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """
        Set configuration data.

        Args:
            config_name: Name of the configuration
            config_data: Configuration data to set

        Returns:
            True if setting was successful, False otherwise
        """

    @abstractmethod
    async def get_translations(self, language_code: str) -> Optional[Dict[str, Any]]:
        """
        Get translations for a language.

        Args:
            language_code: Language code (e.g., 'en', 'cs')

        Returns:
            Translation data or None if not found
        """

    @abstractmethod
    async def get_prompt_template(self, template_name: str) -> Optional[str]:
        """
        Get a prompt template by name.

        Args:
            template_name: Name of the prompt template

        Returns:
            Prompt template content or None if not found
        """

    @abstractmethod
    async def cache_invalidate(self, cache_key: Optional[str] = None) -> bool:
        """
        Invalidate configuration cache.

        Args:
            cache_key: Optional specific cache key to invalidate

        Returns:
            True if invalidation was successful, False otherwise
        """


class IKnowledgeRepository(BaseService):
    """Interface for knowledge base data management."""

    @abstractmethod
    async def store_knowledge_item(
        self,
        item_id: str,
        content: Dict[str, Any],
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Store a knowledge base item.

        Args:
            item_id: Unique identifier for the item
            content: Knowledge item content
            category: Optional category
            tags: Optional tags

        Returns:
            True if storage was successful, False otherwise
        """

    @abstractmethod
    async def get_knowledge_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a knowledge base item by ID.

        Args:
            item_id: Unique identifier of the item

        Returns:
            Knowledge item or None if not found
        """

    @abstractmethod
    async def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base items.

        Args:
            query: Search query
            category: Optional category filter
            tags: Optional tag filters
            limit: Maximum number of results

        Returns:
            List of matching knowledge items
        """

    @abstractmethod
    async def get_all_knowledge(
        self, category: Optional[str] = None, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all knowledge base items.

        Args:
            category: Optional category filter
            limit: Maximum number of items to return
            offset: Number of items to skip

        Returns:
            List of knowledge items
        """

    @abstractmethod
    async def update_knowledge_item(
        self, item_id: str, content: Dict[str, Any], merge: bool = False
    ) -> bool:
        """
        Update a knowledge base item.

        Args:
            item_id: Unique identifier of the item
            content: New content
            merge: If True, merge with existing content

        Returns:
            True if update was successful, False otherwise
        """

    @abstractmethod
    async def delete_knowledge_item(self, item_id: str) -> bool:
        """
        Delete a knowledge base item.

        Args:
            item_id: Unique identifier of the item to delete

        Returns:
            True if deletion was successful, False otherwise
        """


# Generic repository interface for future extensions
class IRepository(BaseService):
    """Generic repository interface for CRUD operations."""

    @abstractmethod
    async def create(self, entity: Any) -> Any:
        """Create a new entity."""

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[Any]:
        """Get entity by ID."""

    @abstractmethod
    async def update(self, entity: Any) -> bool:
        """Update an existing entity."""

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity by ID."""

    @abstractmethod
    async def list_all(
        self, limit: Optional[int] = None, offset: int = 0, filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """List entities with optional filtering and pagination."""

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filtering."""
