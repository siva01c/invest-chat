"""Optimized vector store with advanced connection pooling and query optimization."""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from assistant.config import get_settings
from assistant.core.exceptions import VectorStoreException
from assistant.core.logging import get_logger, log_service_method
from assistant.infrastructure.cache.query_cache import QueryCache
from assistant.infrastructure.cache.redis_client import get_redis_client
from assistant.infrastructure.database.connection_pool import get_connection_pool


@dataclass
class OptimizedSearchResult:
    """Enhanced search result with performance metadata."""

    id: str
    content: str
    metadata: Dict[str, Any]
    distance: float
    score: float
    source: str = "database"
    query_time_ms: float = 0.0
    cache_hit: bool = False


@dataclass
class QueryOptimization:
    """Query optimization configuration."""

    use_cache: bool = True
    cache_ttl: int = 300  # 5 minutes
    enable_query_rewriting: bool = True
    enable_result_filtering: bool = True
    max_results_cache: int = 100
    score_threshold_optimization: bool = True


class OptimizedVectorStore:
    """
    Production-optimized vector store with advanced features.

    Features:
    - Connection pooling for improved performance
    - Intelligent query caching with Redis
    - Query optimization and rewriting
    - Result filtering and ranking
    - Performance monitoring and metrics
    - Batch operations for bulk inserts
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        optimization_config: Optional[QueryOptimization] = None,
    ):
        """
        Initialize the optimized vector store.

        Args:
            collection_name: ChromaDB collection name
            optimization_config: Query optimization configuration
        """
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)

        self.collection_name = collection_name or self.settings.chromadb_collection_name
        self.optimization_config = optimization_config or QueryOptimization()

        # Performance tracking
        self._query_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_query_time": 0.0,
            "optimized_queries": 0,
        }

        # Query cache
        self._query_cache: Optional[QueryCache] = None

    async def _get_query_cache(self) -> Optional[QueryCache]:
        """Get the query cache instance."""
        if not self._query_cache and self.optimization_config.use_cache:
            try:
                redis_client = await get_redis_client()
                self._query_cache = QueryCache(redis_client)
            except Exception as e:
                self.logger.warning(f"Failed to initialize query cache: {str(e)}")
        return self._query_cache

    def _optimize_query(self, query: str, limit: int, score_threshold: float) -> Dict[str, Any]:
        """
        Optimize query parameters for better performance.

        Args:
            query: Search query
            limit: Number of results requested
            score_threshold: Minimum score threshold

        Returns:
            Optimized query parameters
        """
        optimized = {
            "query": query.strip(),
            "limit": limit,
            "score_threshold": score_threshold,
            "optimizations_applied": [],
        }

        if self.optimization_config.enable_query_rewriting:
            # Remove redundant whitespace
            original_query = optimized["query"]
            optimized["query"] = " ".join(original_query.split())

            if original_query != optimized["query"]:
                optimized["optimizations_applied"].append("whitespace_normalization")

            # Expand common abbreviations (could be made configurable)
            abbreviations = {
                "AI": "artificial intelligence",
                "ML": "machine learning",
                "API": "application programming interface",
            }

            for abbr, expansion in abbreviations.items():
                if abbr in optimized["query"]:
                    optimized["query"] = optimized["query"].replace(abbr, f"{abbr} {expansion}")
                    optimized["optimizations_applied"].append(f"expanded_{abbr}")

        if self.optimization_config.score_threshold_optimization:
            # Adjust score threshold based on query length
            query_length = len(optimized["query"].split())

            if query_length <= 2:
                # Short queries need higher threshold
                optimized["score_threshold"] = max(score_threshold, 0.8)
                optimized["optimizations_applied"].append("short_query_threshold_boost")
            elif query_length >= 10:
                # Long queries can use lower threshold
                optimized["score_threshold"] = min(score_threshold, 0.6)
                optimized["optimizations_applied"].append("long_query_threshold_reduction")

        # Optimize limit based on expected result quality
        if limit > 20:
            optimized["limit"] = min(limit, 50)  # Cap at reasonable maximum
            optimized["optimizations_applied"].append("limit_optimization")

        if optimized["optimizations_applied"]:
            self._query_stats["optimized_queries"] += 1

        return optimized

    def _filter_and_rank_results(
        self, results: List[Dict[str, Any]], query: str, score_threshold: float
    ) -> List[OptimizedSearchResult]:
        """
        Apply additional filtering and ranking to search results.

        Args:
            results: Raw search results from ChromaDB
            query: Original search query
            score_threshold: Score threshold for filtering

        Returns:
            Filtered and ranked results
        """
        if not results:
            return []

        # Convert to OptimizedSearchResult objects
        search_results = []
        query_terms = set(query.lower().split())

        for i, result in enumerate(results):
            # Calculate relevance score (1 - normalized distance)
            distance = result.get("distance", 1.0)
            score = max(0.0, 1.0 - distance)

            # Apply score threshold
            if score < score_threshold:
                continue

            # Create result object
            search_result = OptimizedSearchResult(
                id=result.get("id", f"result_{i}"),
                content=result.get("document", ""),
                metadata=result.get("metadata", {}),
                distance=distance,
                score=score,
                source="database",
            )

            # Additional ranking factors
            content_lower = search_result.content.lower()

            # Boost score for exact phrase matches
            if query.lower() in content_lower:
                search_result.score *= 1.2
                search_result.metadata["boost_reason"] = "exact_phrase_match"

            # Boost score for multiple term matches
            matching_terms = sum(1 for term in query_terms if term in content_lower)
            if matching_terms > 1:
                term_boost = 1 + (matching_terms - 1) * 0.1
                search_result.score *= term_boost
                search_result.metadata["matching_terms"] = matching_terms

            # Consider metadata relevance
            if search_result.metadata:
                metadata_text = " ".join(str(v).lower() for v in search_result.metadata.values())
                metadata_matches = sum(1 for term in query_terms if term in metadata_text)
                if metadata_matches > 0:
                    search_result.score *= 1 + metadata_matches * 0.05
                    search_result.metadata["metadata_matches"] = metadata_matches

            search_results.append(search_result)

        # Sort by score (descending)
        search_results.sort(key=lambda x: x.score, reverse=True)

        return search_results

    @log_service_method()
    async def search_similar_text(
        self, query: str, limit: int = 5, score_threshold: float = 0.7, use_cache: bool = None
    ) -> List[OptimizedSearchResult]:
        """
        Search for similar text with advanced optimization.

        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            use_cache: Override cache usage setting

        Returns:
            List of optimized search results
        """
        start_time = time.time()
        use_cache = use_cache if use_cache is not None else self.optimization_config.use_cache

        try:
            # Optimize query parameters
            optimized_params = self._optimize_query(query, limit, score_threshold)
            optimized_query = optimized_params["query"]
            optimized_limit = optimized_params["limit"]
            optimized_threshold = optimized_params["score_threshold"]

            # Generate cache key
            cache_key = f"search:{hash(optimized_query)}:{optimized_limit}:{optimized_threshold}"

            # Try cache first
            cached_result = None
            if use_cache:
                query_cache = await self._get_query_cache()
                if query_cache:
                    cached_result = await query_cache.get(
                        "vector_search",
                        query=optimized_query,
                        limit=optimized_limit,
                        threshold=optimized_threshold,
                    )

            if cached_result is not None:
                self._query_stats["cache_hits"] += 1
                self.logger.debug(f"Cache hit for query: '{optimized_query[:50]}...'")

                # Convert cached results back to OptimizedSearchResult objects
                results = []
                for item in cached_result:
                    result = OptimizedSearchResult(**item)
                    result.cache_hit = True
                    result.query_time_ms = (time.time() - start_time) * 1000
                    results.append(result)

                return results

            # Cache miss - query database
            self._query_stats["cache_misses"] += 1

            # Get connection pool and execute query
            pool = await get_connection_pool()

            # Execute the search query
            raw_results = await pool.execute_query(
                "query",
                query_texts=[optimized_query],
                n_results=optimized_limit * 2,  # Get more results for better filtering
            )

            # Process raw results
            processed_results = []
            if raw_results and raw_results.get("documents") and raw_results["documents"][0]:
                documents = raw_results["documents"][0]
                metadatas = raw_results.get("metadatas", [[{}] * len(documents)])[0]
                distances = raw_results.get("distances", [[1.0] * len(documents)])[0]
                ids = raw_results.get("ids", [[""] * len(documents)])[0]

                for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
                    processed_results.append(
                        {
                            "id": doc_id or f"result_{len(processed_results)}",
                            "document": doc,
                            "metadata": metadata or {},
                            "distance": distance,
                        }
                    )

            # Apply filtering and ranking
            search_results = self._filter_and_rank_results(
                processed_results, optimized_query, optimized_threshold
            )

            # Limit to requested number of results
            search_results = search_results[:limit]

            # Add performance metadata
            query_time = time.time() - start_time
            for result in search_results:
                result.query_time_ms = query_time * 1000
                result.metadata["optimizations_applied"] = optimized_params["optimizations_applied"]

            # Cache results
            if use_cache and query_cache:
                # Convert to serializable format for caching
                cache_data = []
                for result in search_results:
                    cache_data.append(
                        {
                            "id": result.id,
                            "content": result.content,
                            "metadata": result.metadata,
                            "distance": result.distance,
                            "score": result.score,
                            "source": result.source,
                            "query_time_ms": result.query_time_ms,
                            "cache_hit": False,
                        }
                    )

                await query_cache.set(
                    "vector_search",
                    cache_data,
                    ttl=self.optimization_config.cache_ttl,
                    query=optimized_query,
                    limit=optimized_limit,
                    threshold=optimized_threshold,
                )

            # Update statistics
            self._query_stats["total_queries"] += 1
            if self._query_stats["total_queries"] > 1:
                self._query_stats["average_query_time"] = (
                    self._query_stats["average_query_time"]
                    * (self._query_stats["total_queries"] - 1)
                    + query_time
                ) / self._query_stats["total_queries"]
            else:
                self._query_stats["average_query_time"] = query_time

            self.logger.info(
                f"Optimized search completed: query='{optimized_query[:50]}...', "
                f"results={len(search_results)}, time={query_time:.3f}s, "
                f"optimizations={len(optimized_params['optimizations_applied'])}"
            )

            return search_results

        except Exception as e:
            self.logger.error(f"Optimized search failed: {str(e)}")
            raise VectorStoreException(
                f"Optimized search operation failed: {str(e)}",
                operation="search_similar_text",
                cause=e,
            )

    @log_service_method()
    async def batch_add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Add documents in optimized batches for better performance.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
            batch_size: Number of documents per batch

        Returns:
            Results of batch operations
        """
        try:
            total_docs = len(documents)
            batches_processed = 0
            total_time = 0.0

            # Prepare metadatas and IDs
            if metadatas is None:
                metadatas = [{}] * total_docs
            elif len(metadatas) != total_docs:
                metadatas = metadatas[:total_docs] + [{}] * (total_docs - len(metadatas))

            if ids is None:
                ids = [f"doc_{i}_{int(time.time())}" for i in range(total_docs)]

            # Process in batches
            pool = await get_connection_pool()

            for i in range(0, total_docs, batch_size):
                start_time = time.time()
                batch_end = min(i + batch_size, total_docs)

                batch_docs = documents[i:batch_end]
                batch_metadata = metadatas[i:batch_end]
                batch_ids = ids[i:batch_end]

                # Execute batch add
                await pool.execute_query(
                    "add", documents=batch_docs, metadatas=batch_metadata, ids=batch_ids
                )

                batch_time = time.time() - start_time
                total_time += batch_time
                batches_processed += 1

                self.logger.debug(
                    f"Processed batch {batches_processed}: {len(batch_docs)} documents "
                    f"in {batch_time:.3f}s"
                )

            # Clear cache since new documents were added
            query_cache = await self._get_query_cache()
            if query_cache:
                await query_cache.invalidate_pattern("vector_search:*")

            results = {
                "total_documents": total_docs,
                "batches_processed": batches_processed,
                "total_time_seconds": total_time,
                "average_batch_time": (
                    total_time / batches_processed if batches_processed > 0 else 0
                ),
                "documents_per_second": total_docs / total_time if total_time > 0 else 0,
            }

            self.logger.info(
                f"Batch add completed: {total_docs} documents in {batches_processed} batches, "
                f"total_time={total_time:.3f}s, rate={results['documents_per_second']:.1f} docs/sec"
            )

            return results

        except Exception as e:
            self.logger.error(f"Batch add documents failed: {str(e)}")
            raise VectorStoreException(
                f"Batch add operation failed: {str(e)}", operation="batch_add_documents", cause=e
            )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            # Get connection pool stats
            pool = await get_connection_pool()
            pool_stats = await pool.get_pool_stats()

            # Get cache stats
            cache_stats = {}
            query_cache = await self._get_query_cache()
            if query_cache:
                cache_stats = await query_cache.get_cache_stats()

            # Combine all metrics
            metrics = {
                "query_performance": {
                    **self._query_stats,
                    "cache_hit_rate": (
                        self._query_stats["cache_hits"]
                        / (self._query_stats["cache_hits"] + self._query_stats["cache_misses"])
                        if (self._query_stats["cache_hits"] + self._query_stats["cache_misses"]) > 0
                        else 0
                    ),
                    "optimization_rate": (
                        self._query_stats["optimized_queries"] / self._query_stats["total_queries"]
                        if self._query_stats["total_queries"] > 0
                        else 0
                    ),
                },
                "connection_pool": pool_stats,
                "cache_performance": cache_stats,
                "collection_info": {"name": self.collection_name, "optimization_enabled": True},
                "timestamp": datetime.utcnow().isoformat(),
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {str(e)}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for the optimized vector store."""
        try:
            start_time = time.time()

            # Test basic search operation
            test_results = await self.search_similar_text(
                "test query", limit=1, score_threshold=0.0, use_cache=False
            )

            # Get connection pool health
            pool = await get_connection_pool()
            pool_health = await pool.health_check()

            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "search_test_successful": True,
                "results_returned": len(test_results),
                "connection_pool_health": pool_health,
                "cache_available": self._query_cache is not None,
                "optimization_config": {
                    "cache_enabled": self.optimization_config.use_cache,
                    "query_rewriting": self.optimization_config.enable_query_rewriting,
                    "result_filtering": self.optimization_config.enable_result_filtering,
                },
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "search_test_successful": False,
                "timestamp": time.time(),
            }
