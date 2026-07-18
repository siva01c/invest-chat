"""Enhanced knowledge service with async vector store and Redis caching."""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from assistant.core.exceptions import ErrorCode, ServiceException
from assistant.core.interfaces.base import BaseService
from assistant.core.interfaces.repository import IConversationRepository, IVectorRepository
from assistant.core.logging import get_logger, log_service_method
from assistant.infrastructure.cache.query_cache import CachedQueryMixin
from assistant.infrastructure.database.async_vector_store import AsyncVectorStore


class CachedKnowledgeService(BaseService, CachedQueryMixin):
    """Enhanced knowledge service with caching and async optimization."""

    def __init__(
        self,
        vector_repository: Optional[IVectorRepository] = None,
        conversation_repository: Optional[IConversationRepository] = None,
        use_async_vector_store: bool = True,
    ):
        """
        Initialize the cached knowledge service.

        Args:
            vector_repository: Repository for vector-based document storage
            conversation_repository: Repository for conversation management
            use_async_vector_store: Whether to use the new async vector store
        """
        BaseService.__init__(self)
        CachedQueryMixin.__init__(self)

        self.logger = get_logger(self.__class__.__name__)
        self.vector_repository = vector_repository
        self.conversation_repository = conversation_repository

        # Initialize async vector store if enabled
        self.async_vector_store = None
        if use_async_vector_store:
            self.async_vector_store = AsyncVectorStore()

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "CachedKnowledgeService"

    @log_service_method()
    async def search_knowledge(
        self, query: str, limit: int = 5, score_threshold: float = 0.7, use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base with caching for repeated queries.

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            use_cache: Whether to use caching

        Returns:
            List of relevant knowledge items
        """
        if not use_cache:
            return await self._search_knowledge_direct(query, limit, score_threshold)

        # Use cached operation
        return await self.cached_operation(
            operation_name="search_knowledge",
            operation_func=self._search_knowledge_direct,
            ttl=300,  # 5 minutes cache
            query=query,
            limit=limit,
            score_threshold=score_threshold,
        )

    async def _search_knowledge_direct(
        self, query: str, limit: int, score_threshold: float
    ) -> List[Dict[str, Any]]:
        """Direct knowledge search without caching."""
        try:
            knowledge_items = []

            if self.async_vector_store:
                # Use async vector store
                search_results = await self.async_vector_store.search_similar_text(
                    query=query, limit=limit, score_threshold=score_threshold
                )

                for result in search_results:
                    knowledge_items.append(
                        {
                            "id": result.id,
                            "content": result.content,
                            "metadata": result.metadata,
                            "relevance_score": result.score,
                            "source": "async_knowledge_base",
                        }
                    )

            elif self.vector_repository:
                # Fallback to repository pattern
                search_results = await self.vector_repository.search_similar(
                    query=query, limit=limit, score_threshold=score_threshold
                )

                for result in search_results:
                    knowledge_items.append(
                        {
                            "id": result.id,
                            "content": result.content,
                            "metadata": result.metadata,
                            "relevance_score": result.score,
                            "source": "repository_knowledge_base",
                        }
                    )

            else:
                raise ServiceException(
                    "No vector store or repository available",
                    service_name=self.get_service_name(),
                    error_code=ErrorCode.CONFIGURATION_ERROR,
                )

            self.logger.info(
                f"Found {len(knowledge_items)} knowledge items for query: '{query[:50]}...'"
            )
            return knowledge_items

        except Exception as e:
            raise ServiceException(
                f"Failed to search knowledge base: {str(e)}",
                service_name=self.get_service_name(),
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={"query": query[:100], "limit": limit, "threshold": score_threshold},
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
        Store a new knowledge item and invalidate relevant caches.

        Args:
            content: Knowledge content
            title: Optional title
            category: Optional category
            tags: Optional tags

        Returns:
            Unique identifier of the stored item
        """
        try:
            import uuid

            doc_id = str(uuid.uuid4())

            metadata = {
                "title": title or f"Knowledge item {doc_id[:8]}",
                "category": category or "general",
                "tags": tags or [],
                "created_at": datetime.now(UTC).isoformat(),
                "type": "knowledge_item",
            }

            if self.async_vector_store:
                # Use async vector store
                success = await self.async_vector_store.add_documents(
                    documents=[content], metadatas=[metadata], ids=[doc_id]
                )
            elif self.vector_repository:
                # Fallback to repository
                success = await self.vector_repository.store_document(
                    document_id=doc_id, content=content, metadata=metadata
                )
            else:
                raise ServiceException(
                    "No vector store or repository available",
                    service_name=self.get_service_name(),
                    error_code=ErrorCode.CONFIGURATION_ERROR,
                )

            if not success:
                raise ServiceException(
                    "Failed to store knowledge item",
                    service_name=self.get_service_name(),
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                )

            # Invalidate search caches since new content was added
            await self.invalidate_cache_pattern("search_knowledge*")

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
        self, session_id: str, context_limit: int = 3, use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get conversation context with caching.

        Args:
            session_id: Session identifier
            context_limit: Number of recent interactions to include
            use_cache: Whether to use caching

        Returns:
            List of conversation context items
        """
        if not use_cache:
            return await self._get_conversation_context_direct(session_id, context_limit)

        # Use cached operation with shorter TTL for conversation data
        return await self.cached_operation(
            operation_name="get_conversation_context",
            operation_func=self._get_conversation_context_direct,
            ttl=60,  # 1 minute cache for conversation data
            session_id=session_id,
            context_limit=context_limit,
        )

    async def _get_conversation_context_direct(
        self, session_id: str, context_limit: int
    ) -> List[Dict[str, Any]]:
        """Direct conversation context retrieval without caching."""
        try:
            if not self.conversation_repository:
                return []

            recent_interactions = await self.conversation_repository.get_recent_interactions(
                session_id=session_id, count=context_limit
            )

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
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Enhanced search with caching for the combined results.

        Args:
            query: Search query
            session_id: Optional session for context
            include_conversation_context: Whether to include conversation context
            knowledge_limit: Maximum knowledge items to return
            context_limit: Maximum context items to return
            use_cache: Whether to use caching

        Returns:
            Combined search results with knowledge and context
        """
        if not use_cache:
            return await self._enhanced_search_direct(
                query, session_id, include_conversation_context, knowledge_limit, context_limit
            )

        # Use cached operation with custom TTL
        return await self.cached_operation(
            operation_name="enhanced_search",
            operation_func=self._enhanced_search_direct,
            ttl=180,  # 3 minutes cache for combined search
            query=query,
            session_id=session_id,
            include_conversation_context=include_conversation_context,
            knowledge_limit=knowledge_limit,
            context_limit=context_limit,
        )

    async def _enhanced_search_direct(
        self,
        query: str,
        session_id: Optional[str],
        include_conversation_context: bool,
        knowledge_limit: int,
        context_limit: int,
    ) -> Dict[str, Any]:
        """Direct enhanced search without caching."""
        try:
            results = {
                "query": query,
                "timestamp": datetime.now(UTC).isoformat(),
                "knowledge_items": [],
                "conversation_context": [],
                "cache_used": False,
            }

            # Search knowledge base (using internal caching)
            knowledge_items = await self.search_knowledge(
                query=query,
                limit=knowledge_limit,
                use_cache=True,  # Use internal caching for knowledge search
            )
            results["knowledge_items"] = knowledge_items

            # Include conversation context if requested
            if include_conversation_context and session_id:
                context_items = await self.get_conversation_context(
                    session_id=session_id,
                    context_limit=context_limit,
                    use_cache=True,  # Use internal caching for context
                )
                results["conversation_context"] = context_items

            # Add summary statistics
            results["summary"] = {
                "total_knowledge_items": len(knowledge_items),
                "total_context_items": len(results["conversation_context"]),
                "has_high_relevance": any(
                    item["relevance_score"] > 0.9 for item in knowledge_items
                ),
                "search_source": "async_vector_store" if self.async_vector_store else "repository",
            }

            self.logger.info(f"Enhanced search completed for query: '{query[:50]}...'")
            return results

        except Exception as e:
            raise ServiceException(
                f"Enhanced search failed: {str(e)}",
                service_name=self.get_service_name(),
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                details={
                    "query": query[:100],
                    "session_id": session_id,
                    "include_context": include_conversation_context,
                },
                cause=e,
            )

    @log_service_method()
    async def get_knowledge_statistics(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base with caching.

        Args:
            use_cache: Whether to use caching

        Returns:
            Dictionary with knowledge base statistics
        """
        if not use_cache:
            return await self._get_knowledge_statistics_direct()

        # Cache statistics for longer since they change less frequently
        return await self.cached_operation(
            operation_name="get_knowledge_statistics",
            operation_func=self._get_knowledge_statistics_direct,
            ttl=600,  # 10 minutes cache for statistics
        )

    async def _get_knowledge_statistics_direct(self) -> Dict[str, Any]:
        """Direct knowledge statistics retrieval without caching."""
        try:
            if self.async_vector_store:
                # Use async vector store stats
                stats = await self.async_vector_store.get_collection_stats()
                return stats

            elif self.vector_repository:
                # Get document count from repository
                total_documents = await self.vector_repository.count_documents()

                # Get documents by category (basic implementation)
                category_counts = {}
                try:
                    all_docs = await self.vector_repository.get_all_documents(limit=1000)
                    for doc in all_docs:
                        category = doc.metadata.get("category", "unknown")
                        category_counts[category] = category_counts.get(category, 0) + 1
                except Exception:
                    category_counts = {"total": total_documents}

                stats = {
                    "total_documents": total_documents,
                    "categories": category_counts,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "source": "repository",
                }

            else:
                stats = {
                    "error": "No vector store or repository available",
                    "total_documents": 0,
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

    @log_service_method()
    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get comprehensive health status for the knowledge service.

        Returns:
            Health status with performance metrics
        """
        try:
            health = {
                "service_name": self.get_service_name(),
                "status": "healthy",
                "timestamp": datetime.now(UTC).isoformat(),
                "components": {},
            }

            # Check async vector store health
            if self.async_vector_store:
                vector_health = await self.async_vector_store.health_check()
                health["components"]["async_vector_store"] = vector_health
                if vector_health["status"] != "healthy":
                    health["status"] = "degraded"

            # Check cache health
            cache = await self._get_query_cache()
            if cache:
                try:
                    cache_stats = await cache.get_cache_stats()
                    health["components"]["query_cache"] = {
                        "status": "healthy",
                        "stats": cache_stats,
                    }
                except Exception as e:
                    health["components"]["query_cache"] = {"status": "unhealthy", "error": str(e)}
                    health["status"] = "degraded"

            # Check repository health (if available)
            if self.vector_repository:
                try:
                    # Simple health check - try to count documents
                    count = await self.vector_repository.count_documents()
                    health["components"]["vector_repository"] = {
                        "status": "healthy",
                        "document_count": count,
                    }
                except Exception as e:
                    health["components"]["vector_repository"] = {
                        "status": "unhealthy",
                        "error": str(e),
                    }
                    health["status"] = "degraded"

            return health

        except Exception as e:
            return {
                "service_name": self.get_service_name(),
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def clear_all_caches(self) -> Dict[str, Any]:
        """
        Clear all caches and return statistics.

        Returns:
            Cache clearing results
        """
        try:
            results = {"cleared_caches": [], "errors": []}

            # Clear query cache
            cache = await self._get_query_cache()
            if cache:
                try:
                    before_stats = await cache.get_cache_stats()
                    cleared_count = await self.invalidate_cache_pattern("*")
                    results["cleared_caches"].append(
                        {
                            "cache_type": "query_cache",
                            "entries_cleared": cleared_count,
                            "total_entries_before": before_stats.get("total_entries", 0),
                        }
                    )
                except Exception as e:
                    results["errors"].append(f"Query cache clear failed: {str(e)}")

            # Clear vector store cache
            if self.async_vector_store:
                try:
                    await self.async_vector_store.clear_cache()
                    results["cleared_caches"].append(
                        {"cache_type": "vector_store_cache", "status": "cleared"}
                    )
                except Exception as e:
                    results["errors"].append(f"Vector store cache clear failed: {str(e)}")

            self.logger.info(f"Cleared {len(results['cleared_caches'])} caches")
            return results

        except Exception as e:
            return {"error": str(e), "cleared_caches": [], "errors": [str(e)]}
