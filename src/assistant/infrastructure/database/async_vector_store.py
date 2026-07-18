"""Async-optimized ChromaDB vector store implementation with connection pooling."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from assistant.config import get_settings
from assistant.core.exceptions import ErrorCode, VectorStoreException
from assistant.core.logging import get_logger, log_service_method


@dataclass
class SearchResult:
    """Search result from vector store."""

    id: str
    content: str
    metadata: Dict[str, Any]
    distance: float
    score: float


class AsyncVectorStore:
    """Async-optimized vector store with connection pooling and caching."""

    def __init__(
        self, collection_name: str = None, database_path: str = None, max_workers: int = 4
    ):
        """
        Initialize the async vector store.

        Args:
            collection_name: ChromaDB collection name
            database_path: Database path for persistent client
            max_workers: Maximum number of worker threads
        """
        self.logger = get_logger(self.__class__.__name__)
        self.settings = get_settings()

        # Use settings if parameters not provided
        self.collection_name = collection_name or self.settings.chromadb_collection_name
        self.database_path = database_path or self.settings.chromadb_database_path

        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Connection management
        self._client = None
        self._collection = None
        self._lock = asyncio.Lock()

        # Simple cache for frequently accessed data
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes

        # Performance metrics
        self._metrics = {"queries": 0, "cache_hits": 0, "cache_misses": 0, "avg_query_time": 0.0}

    async def _get_client(self):
        """Get or create ChromaDB client with connection pooling."""
        if self._client is None:
            async with self._lock:
                if self._client is None:  # Double-check pattern
                    await self._initialize_client()
        return self._client

    async def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Get OpenAI API key
            openai_key = self.settings.openai_api_key
            if not openai_key:
                raise VectorStoreException(
                    "OpenAI API key not found in environment variables",
                    collection_name=self.collection_name,
                    operation="initialization",
                    error_code=ErrorCode.CONFIGURATION_ERROR,
                )

            # Create embedding function using environment variable to avoid deprecation warnings
            import os

            original_api_key = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = openai_key
            try:
                embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key_env_var="OPENAI_API_KEY",
                    model_name=self.settings.openai_embedding_model,
                )
            finally:
                # Restore original environment variable
                if original_api_key is not None:
                    os.environ["OPENAI_API_KEY"] = original_api_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)

            # Choose client type based on configuration
            if self.settings.chromadb_use_http:
                # HTTP client for containerized ChromaDB
                self._client = await asyncio.to_thread(
                    chromadb.HttpClient,
                    host=self.settings.chromadb_host,
                    port=self.settings.chromadb_port,
                )
                self.logger.info(
                    f"Using HTTP ChromaDB client: {self.settings.chromadb_host}:{self.settings.chromadb_port}"
                )
            else:
                # Persistent client for local ChromaDB
                self._client = await asyncio.to_thread(
                    chromadb.PersistentClient, path=self.database_path
                )
                self.logger.info(f"Using Persistent ChromaDB client: {self.database_path}")

            # Get or create collection
            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=self.collection_name,
                embedding_function=embedding_function,
            )

            self.logger.info(f"AsyncVectorStore initialized: collection={self.collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise VectorStoreException(
                f"ChromaDB initialization failed: {str(e)}",
                collection_name=self.collection_name,
                operation="initialization",
                error_code=ErrorCode.DATABASE_CONNECTION_ERROR,
                cause=e,
            )

    async def _get_collection(self):
        """Get collection with lazy initialization."""
        if self._collection is None:
            await self._get_client()
        return self._collection

    def _cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for operation."""
        key_parts = [operation]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return ":".join(key_parts)

    def _get_cache(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        if key in self._cache:
            cached_time, value = self._cache[key]
            if time.time() - cached_time < self._cache_timeout:
                self._metrics["cache_hits"] += 1
                return value
            else:
                # Remove expired item
                del self._cache[key]

        self._metrics["cache_misses"] += 1
        return None

    def _set_cache(self, key: str, value: Any):
        """Set item in cache with timestamp."""
        self._cache[key] = (time.time(), value)

    @log_service_method()
    async def search_similar_text(
        self, query: str, limit: int = 3, score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar text with caching and async optimization.

        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        start_time = time.time()

        # Check cache first
        cache_key = self._cache_key("search", query=query, limit=limit, threshold=score_threshold)
        cached_result = self._get_cache(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for query: '{query[:50]}...'")
            return cached_result

        try:
            collection = await self._get_collection()

            # Perform search in thread pool
            results = await asyncio.to_thread(
                collection.query, query_texts=[query], n_results=limit
            )

            # Process results
            search_results = []
            if results and results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results.get("metadatas", [[{}] * len(documents)])[0]
                distances = results.get("distances", [[1.0] * len(documents)])[0]
                ids = results.get("ids", [[""] * len(documents)])[0]

                for i, (doc, metadata, distance, doc_id) in enumerate(
                    zip(documents, metadatas, distances, ids)
                ):
                    # Convert distance to similarity score (1 - normalized_distance)
                    score = max(0.0, 1.0 - distance)

                    if score >= score_threshold:
                        search_results.append(
                            SearchResult(
                                id=doc_id or f"result_{i}",
                                content=doc,
                                metadata=metadata or {},
                                distance=distance,
                                score=score,
                            )
                        )

            # Cache results
            self._set_cache(cache_key, search_results)

            # Update metrics
            query_time = time.time() - start_time
            self._metrics["queries"] += 1
            self._metrics["avg_query_time"] = (
                self._metrics["avg_query_time"] * (self._metrics["queries"] - 1) + query_time
            ) / self._metrics["queries"]

            self.logger.info(
                f"Vector search completed: query='{query[:50]}...', results={len(search_results)}, "
                f"time={query_time:.3f}s"
            )

            return search_results

        except Exception as e:
            self.logger.error(f"Vector search failed: {str(e)}")
            raise VectorStoreException(
                f"Search operation failed: {str(e)}",
                collection_name=self.collection_name,
                operation="search_similar_text",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                details={"query": query[:100], "limit": limit, "threshold": score_threshold},
                cause=e,
            )

    @log_service_method()
    async def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Add documents to vector store with batch optimization.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            True if successful
        """
        try:
            collection = await self._get_collection()

            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}_{int(time.time())}" for i in range(len(documents))]

            # Ensure metadatas match documents length
            if metadatas is None:
                metadatas = [{}] * len(documents)
            elif len(metadatas) != len(documents):
                metadatas = metadatas[: len(documents)] + [{}] * (len(documents) - len(metadatas))

            # Add documents in thread pool
            await asyncio.to_thread(
                collection.add, documents=documents, metadatas=metadatas, ids=ids
            )

            # Clear cache since new documents were added
            self._cache.clear()

            self.logger.info(f"Added {len(documents)} documents to vector store")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise VectorStoreException(
                f"Add documents operation failed: {str(e)}",
                collection_name=self.collection_name,
                operation="add_documents",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                details={"document_count": len(documents)},
                cause=e,
            )

    @log_service_method()
    async def get_all_records(self) -> List[Dict[str, Any]]:
        """
        Get all records from the collection with caching.

        Returns:
            List of all records
        """
        # Check cache first
        cache_key = self._cache_key("get_all_records")
        cached_result = self._get_cache(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            collection = await self._get_collection()

            # Get all records in thread pool
            results = await asyncio.to_thread(collection.get)

            # Process results
            records = []
            if results and results.get("documents"):
                documents = results["documents"]
                metadatas = results.get("metadatas", [{}] * len(documents))
                ids = results.get("ids", [""] * len(documents))

                for doc, metadata, doc_id in zip(documents, metadatas, ids):
                    records.append({"id": doc_id, "content": doc, "metadata": metadata or {}})

            # Cache results
            self._set_cache(cache_key, records)

            self.logger.info(f"Retrieved {len(records)} records from vector store")
            return records

        except Exception as e:
            self.logger.error(f"Failed to get all records: {str(e)}")
            raise VectorStoreException(
                f"Get all records operation failed: {str(e)}",
                collection_name=self.collection_name,
                operation="get_all_records",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                cause=e,
            )

    @log_service_method()
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            document_id: ID of document to delete

        Returns:
            True if successful
        """
        try:
            collection = await self._get_collection()

            await asyncio.to_thread(collection.delete, ids=[document_id])

            # Clear cache since document was deleted
            self._cache.clear()

            self.logger.info(f"Deleted document: {document_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise VectorStoreException(
                f"Delete document operation failed: {str(e)}",
                collection_name=self.collection_name,
                operation="delete_document",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                details={"document_id": document_id},
                cause=e,
            )

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics and performance metrics.

        Returns:
            Dictionary with stats and metrics
        """
        try:
            collection = await self._get_collection()

            # Get collection count
            count_result = await asyncio.to_thread(collection.count)

            stats = {
                "collection_name": self.collection_name,
                "document_count": count_result,
                "cache_size": len(self._cache),
                "performance_metrics": self._metrics.copy(),
                "client_type": "HTTP" if self.settings.chromadb_use_http else "Persistent",
                "connection_info": {
                    "host": (
                        self.settings.chromadb_host if self.settings.chromadb_use_http else "local"
                    ),
                    "port": (
                        self.settings.chromadb_port if self.settings.chromadb_use_http else None
                    ),
                },
            }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "performance_metrics": self._metrics.copy(),
            }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on vector store.

        Returns:
            Health check results
        """
        try:
            start_time = time.time()

            # Test connection
            collection = await self._get_collection()

            # Test basic operation
            test_result = await asyncio.to_thread(collection.count)

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "collection_name": self.collection_name,
                "document_count": test_result,
                "response_time_ms": response_time * 1000,
                "client_type": "HTTP" if self.settings.chromadb_use_http else "Persistent",
                "cache_stats": {
                    "size": len(self._cache),
                    "hit_rate": (
                        self._metrics["cache_hits"]
                        / (self._metrics["cache_hits"] + self._metrics["cache_misses"])
                        if (self._metrics["cache_hits"] + self._metrics["cache_misses"]) > 0
                        else 0
                    ),
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "collection_name": self.collection_name,
                "error": str(e),
                "client_type": "HTTP" if self.settings.chromadb_use_http else "Persistent",
            }

    async def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
        self.logger.info("Vector store cache cleared")

    async def close(self):
        """Close the vector store and clean up resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
        self._cache.clear()
        self.logger.info("AsyncVectorStore closed")

    # Additional convenience methods for compatibility
    async def retrieve_context(self, query: str, limit: int = 3) -> str:
        """Retrieve context from vector store (compatibility method)."""
        results = await self.search_similar_text(query, limit=limit)
        if not results:
            return ""

        context_parts = []
        for result in results:
            context_parts.append(result.content)

        return "\n\n".join(context_parts)

    async def store_embeddings(
        self, texts: List[str], embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """Store embeddings (compatibility method - uses OpenAI embeddings internally)."""
        return await self.add_documents(texts)
