"""Vector repository implementation using ChromaDB."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from assistant.core.exceptions import ErrorCode, VectorStoreException
from assistant.core.interfaces.repository import Document, IVectorRepository, SearchResult
from assistant.core.logging import get_logger, log_service_method
from assistant.infrastructure.database.vector_store import VectorStore


@dataclass
class VectorDocument:
    """Implementation of Document protocol for vector store."""

    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class VectorSearchResult:
    """Implementation of SearchResult protocol."""

    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    distance: float

    @classmethod
    def from_chroma_result(cls, chroma_result: tuple) -> "VectorSearchResult":
        """Create SearchResult from ChromaDB result tuple."""
        doc_id, distance, metadata, content, _ = chroma_result
        score = 1.0 - distance  # Convert distance to similarity score
        return cls(
            id=doc_id, content=content, metadata=metadata or {}, score=score, distance=distance
        )


class VectorStoreRepository(IVectorRepository):
    """Repository implementation for vector-based document storage using ChromaDB."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        collection_name: str = "knowledge_base",
        database_path: str = "chromadb",
    ):
        """
        Initialize the vector repository.

        Args:
            vector_store: Optional existing VectorStore instance
            collection_name: Name of the ChromaDB collection
            database_path: Path to ChromaDB database
        """
        self.logger = get_logger(self.__class__.__name__)
        self.vector_store = vector_store or VectorStore(
            collection_name=collection_name, database_path=database_path
        )

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "VectorStoreRepository"

    @log_service_method()
    async def store_document(
        self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a single document in the vector store."""
        if not document_id or not document_id.strip():
            raise VectorStoreException(
                "Document ID cannot be empty",
                operation="store_document",
                error_code=ErrorCode.INVALID_INPUT,
            )

        if not content or not content.strip():
            raise VectorStoreException(
                "Document content cannot be empty",
                operation="store_document",
                error_code=ErrorCode.INVALID_INPUT,
            )

        try:
            # Create a document-like object for the existing VectorStore
            doc = type(
                "Document",
                (),
                {"id": document_id, "knowledge": [content], "metadata": metadata or {}},
            )()

            await self.vector_store.store_embeddings([doc])
            self.logger.info(f"Successfully stored document: {document_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store document {document_id}: {str(e)}")
            raise VectorStoreException(
                f"Failed to store document: {str(e)}",
                operation="store_document",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                details={"document_id": document_id},
                cause=e,
            )

    @log_service_method()
    async def store_documents(self, documents: List[Document]) -> int:
        """Store multiple documents in the vector store."""
        if not documents:
            self.logger.warning("No documents provided for storage")
            return 0

        stored_count = 0
        failed_count = 0

        for doc in documents:
            try:
                await self.store_document(doc.id, doc.content, doc.metadata)
                stored_count += 1
            except Exception as e:
                failed_count += 1
                self.logger.error(f"Failed to store document {doc.id}: {str(e)}")

        self.logger.info(f"Stored {stored_count}/{len(documents)} documents")
        if failed_count > 0:
            self.logger.warning(f"{failed_count} documents failed to store")

        return stored_count

    @log_service_method()
    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for documents similar to the query."""
        if not query or not query.strip():
            raise VectorStoreException(
                "Search query cannot be empty",
                operation="search_similar",
                error_code=ErrorCode.INVALID_INPUT,
            )

        if limit <= 0:
            raise VectorStoreException(
                "Limit must be positive",
                operation="search_similar",
                error_code=ErrorCode.INVALID_INPUT,
            )

        try:
            # Use the existing VectorStore search method
            raw_results = await self.vector_store.search_similar_text(query, limit)

            # Convert results to SearchResult objects
            search_results = []
            for result in raw_results:
                search_result = VectorSearchResult.from_chroma_result(result)

                # Apply score threshold if specified
                if score_threshold is not None and search_result.score < score_threshold:
                    continue

                # Apply metadata filter if specified
                if metadata_filter is not None:
                    if not self._matches_metadata_filter(search_result.metadata, metadata_filter):
                        continue

                search_results.append(search_result)

            self.logger.info(
                f"Search returned {len(search_results)} results for query: '{query[:50]}...'"
            )
            return search_results

        except Exception as e:
            self.logger.error(f"Search failed for query '{query[:50]}...': {str(e)}")
            raise VectorStoreException(
                f"Search operation failed: {str(e)}",
                operation="search_similar",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                details={"query_length": len(query), "limit": limit},
                cause=e,
            )

    def _matches_metadata_filter(
        self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filter criteria."""
        for key, expected_value in filter_criteria.items():
            if key not in metadata:
                return False
            if metadata[key] != expected_value:
                return False
        return True

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a specific document by ID."""
        if not document_id or not document_id.strip():
            raise VectorStoreException(
                "Document ID cannot be empty",
                operation="get_document",
                error_code=ErrorCode.INVALID_INPUT,
            )

        try:
            # ChromaDB doesn't have a direct get-by-id method in our implementation
            # We'll search for the document ID in metadata or use a workaround
            all_records = await self.vector_store.get_all_records()

            if hasattr(all_records, "ids") and document_id in all_records["ids"]:
                index = all_records["ids"].index(document_id)
                return VectorDocument(
                    id=document_id,
                    content=all_records["documents"][index] if "documents" in all_records else "",
                    metadata=all_records["metadatas"][index] if "metadatas" in all_records else {},
                )

            return None

        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {str(e)}")
            raise VectorStoreException(
                f"Failed to retrieve document: {str(e)}",
                operation="get_document",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                details={"document_id": document_id},
                cause=e,
            )

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        # Note: ChromaDB delete functionality would need to be implemented
        # in the underlying VectorStore class
        self.logger.warning(f"Delete operation not implemented for document: {document_id}")
        return False

    @log_service_method()
    async def get_all_documents(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> List[Document]:
        """Retrieve all documents from the vector store."""
        try:
            all_records = await self.vector_store.get_all_records()

            documents = []
            if hasattr(all_records, "ids"):
                ids = all_records["ids"]
                documents_data = all_records.get("documents", [])
                metadatas = all_records.get("metadatas", [])

                # Apply offset and limit
                start = offset
                end = offset + limit if limit is not None else None

                for i, doc_id in enumerate(ids[start:end]):
                    actual_index = start + i
                    content = (
                        documents_data[actual_index] if actual_index < len(documents_data) else ""
                    )
                    metadata = metadatas[actual_index] if actual_index < len(metadatas) else {}

                    documents.append(VectorDocument(id=doc_id, content=content, metadata=metadata))

            self.logger.info(f"Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to get all documents: {str(e)}")
            raise VectorStoreException(
                f"Failed to retrieve documents: {str(e)}",
                operation="get_all_documents",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                cause=e,
            )

    async def count_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the vector store."""
        try:
            all_records = await self.vector_store.get_all_records()

            if not hasattr(all_records, "ids"):
                return 0

            total_count = len(all_records["ids"])

            if metadata_filter is None:
                return total_count

            # Apply metadata filter
            filtered_count = 0
            metadatas = all_records.get("metadatas", [])

            for metadata in metadatas:
                if self._matches_metadata_filter(metadata or {}, metadata_filter):
                    filtered_count += 1

            return filtered_count

        except Exception as e:
            self.logger.error(f"Failed to count documents: {str(e)}")
            raise VectorStoreException(
                f"Failed to count documents: {str(e)}",
                operation="count_documents",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                cause=e,
            )

    async def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific document."""
        # Note: ChromaDB metadata update would need to be implemented
        # in the underlying VectorStore class
        self.logger.warning(f"Metadata update not implemented for document: {document_id}")
        return False
