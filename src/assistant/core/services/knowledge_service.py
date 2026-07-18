"""Knowledge service using repository pattern for data access."""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from assistant.core.exceptions import ErrorCode, ServiceException
from assistant.core.interfaces.base import BaseService
from assistant.core.interfaces.repository import IConversationRepository, IVectorRepository
from assistant.core.logging import get_logger, log_service_method


class KnowledgeService(BaseService):
    """Service for knowledge management using repository pattern."""

    def __init__(
        self, vector_repository: IVectorRepository, conversation_repository: IConversationRepository
    ):
        """
        Initialize the knowledge service with repositories.

        Args:
            vector_repository: Repository for vector-based document storage
            conversation_repository: Repository for conversation management
        """
        self.logger = get_logger(self.__class__.__name__)
        self.vector_repository = vector_repository
        self.conversation_repository = conversation_repository

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "KnowledgeService"

    @log_service_method()
    async def search_knowledge(
        self, query: str, limit: int = 5, score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant information.

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of relevant knowledge items
        """
        try:
            # Use repository to search for similar documents
            search_results = await self.vector_repository.search_similar(
                query=query, limit=limit, score_threshold=score_threshold
            )

            # Format results for service response
            knowledge_items = []
            for result in search_results:
                knowledge_items.append(
                    {
                        "id": result.id,
                        "content": result.content,
                        "metadata": result.metadata,
                        "relevance_score": result.score,
                        "source": "knowledge_base",
                    }
                )

            self.logger.info(f"Found {len(knowledge_items)} knowledge items for query: '{query}'")
            return knowledge_items

        except Exception as e:
            raise ServiceException(
                f"Failed to search knowledge base: {str(e)}",
                service_name=self.get_service_name(),
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"query": query, "limit": limit, "threshold": score_threshold},
                cause=e,
            )

    @log_service_method()
    async def store_knowledge_item(
        self,
        content: str,
        title: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store a new knowledge item.

        Args:
            content: Knowledge content
            title: Optional title
            category: Optional category
            tags: Optional tags

        Returns:
            Unique identifier of the stored item
        """
        try:
            # Generate document ID
            import uuid

            doc_id = str(uuid.uuid4())

            # Prepare metadata
            metadata = {
                "title": title or f"Knowledge item {doc_id[:8]}",
                "category": category or "general",
                "tags": tags or [],
                "created_at": datetime.now(UTC).isoformat(),
                "type": "knowledge_item",
            }

            # Store using repository
            success = await self.vector_repository.store_document(
                document_id=doc_id, content=content, metadata=metadata
            )

            if not success:
                raise ServiceException(
                    "Failed to store knowledge item",
                    service_name=self.get_service_name(),
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                )

            self.logger.info(f"Stored knowledge item: {doc_id}")
            return doc_id

        except Exception as e:
            raise ServiceException(
                f"Failed to store knowledge item: {str(e)}",
                service_name=self.get_service_name(),
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"content_length": len(content)},
                cause=e,
            )

    @log_service_method()
    async def get_conversation_context(
        self, session_id: str, context_limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get conversation context for knowledge retrieval.

        Args:
            session_id: Session identifier
            context_limit: Number of recent interactions to include

        Returns:
            List of conversation context items
        """
        try:
            # Get recent interactions using repository
            recent_interactions = await self.conversation_repository.get_recent_interactions(
                session_id=session_id, count=context_limit
            )

            # Format context for response
            context_items = []
            for interaction in recent_interactions:
                context_items.append(
                    {
                        "id": interaction.id,
                        "user_message": interaction.user_message,
                        "assistant_response": interaction.assistant_response,
                        "timestamp": interaction.timestamp.isoformat(),
                        "category": interaction.category,
                    }
                )

            self.logger.debug(
                f"Retrieved {len(context_items)} context items for session {session_id}"
            )
            return context_items

        except Exception as e:
            raise ServiceException(
                f"Failed to get conversation context: {str(e)}",
                service_name=self.get_service_name(),
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"session_id": session_id, "context_limit": context_limit},
                cause=e,
            )

    @log_service_method()
    async def enhanced_search(
        self,
        query: str,
        session_id: Optional[str] = None,
        include_conversation_context: bool = True,
        knowledge_limit: int = 5,
        context_limit: int = 3,
    ) -> Dict[str, Any]:
        """
        Perform enhanced search combining knowledge base and conversation context.

        Args:
            query: Search query
            session_id: Optional session for context
            include_conversation_context: Whether to include conversation context
            knowledge_limit: Maximum knowledge items to return
            context_limit: Maximum context items to return

        Returns:
            Combined search results with knowledge and context
        """
        try:
            results = {
                "query": query,
                "timestamp": datetime.now(UTC).isoformat(),
                "knowledge_items": [],
                "conversation_context": [],
            }

            # Search knowledge base
            knowledge_items = await self.search_knowledge(query=query, limit=knowledge_limit)
            results["knowledge_items"] = knowledge_items

            # Include conversation context if requested and session provided
            if include_conversation_context and session_id:
                context_items = await self.get_conversation_context(
                    session_id=session_id, context_limit=context_limit
                )
                results["conversation_context"] = context_items

            # Add summary statistics
            results["summary"] = {
                "total_knowledge_items": len(knowledge_items),
                "total_context_items": len(results["conversation_context"]),
                "has_high_relevance": any(
                    item["relevance_score"] > 0.9 for item in knowledge_items
                ),
            }

            self.logger.info(f"Enhanced search completed for query: '{query}'")
            return results

        except Exception as e:
            raise ServiceException(
                f"Enhanced search failed: {str(e)}",
                service_name=self.get_service_name(),
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={
                    "query": query,
                    "session_id": session_id,
                    "include_context": include_conversation_context,
                },
                cause=e,
            )

    @log_service_method()
    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with knowledge base statistics
        """
        try:
            # Get document count
            total_documents = await self.vector_repository.count_documents()

            # Get documents by category (if metadata filtering is supported)
            category_counts = {}
            try:
                all_docs = await self.vector_repository.get_all_documents(limit=1000)
                for doc in all_docs:
                    category = doc.metadata.get("category", "unknown")
                    category_counts[category] = category_counts.get(category, 0) + 1
            except Exception:
                # Fallback if get_all_documents is not fully implemented
                category_counts = {"total": total_documents}

            stats = {
                "total_documents": total_documents,
                "categories": category_counts,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            self.logger.info("Retrieved knowledge base statistics")
            return stats

        except Exception as e:
            raise ServiceException(
                f"Failed to get knowledge statistics: {str(e)}",
                service_name=self.get_service_name(),
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                cause=e,
            )
