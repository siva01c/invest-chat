"""ChromaDB vector store implementation."""

import asyncio
import os
from typing import Any, List

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import AsyncOpenAI

from assistant.config import get_settings
from assistant.core.exceptions import ErrorCode, VectorStoreException
from assistant.core.interfaces.infrastructure import IVectorStore
from assistant.core.logging import get_logger, log_service_method

load_dotenv()

MODEL = "gpt-4o-mini"
client = AsyncOpenAI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Record:
    """Data record for vector storage."""

    def __init__(self):
        self.knowledge = []
        self.metadata = []
        self.id = ""

    def to_dict(self):
        return {"knowledge": self.knowledge, "metadata": self.metadata, "id": self.id}


class VectorStore(IVectorStore):
    """Manages storage and retrieval of vector embeddings."""

    def __init__(
        self,
        collection_name: str = None,
        database_path: str = None,
    ):
        """Initialize the vector store."""
        self.logger = get_logger(self.__class__.__name__)
        self.settings = get_settings()

        # Use settings if parameters not provided
        self.collection_name = collection_name or self.settings.chromadb_collection_name
        self.database_path = database_path or self.settings.chromadb_database_path

        try:
            if not OPENAI_API_KEY:
                raise VectorStoreException(
                    "OpenAI API key not found in environment variables",
                    collection_name=self.collection_name,
                    operation="initialization",
                    error_code=ErrorCode.CONFIGURATION_ERROR,
                )

            # Use environment variable for API key to avoid deprecation warnings
            import os

            original_api_key = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            try:
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key_env_var="OPENAI_API_KEY", model_name="text-embedding-ada-002"
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
                self.chroma_client = chromadb.HttpClient(
                    host=self.settings.chromadb_host, port=self.settings.chromadb_port
                )
                self.logger.info(
                    f"Using HTTP ChromaDB client: {self.settings.chromadb_host}:{self.settings.chromadb_port}"
                )
            else:
                # Persistent client for local ChromaDB
                self.chroma_client = chromadb.PersistentClient(path=self.database_path)
                self.logger.info(f"Using Persistent ChromaDB client: {self.database_path}")

            self.about_me_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name, embedding_function=self.embedding_function
            )

            self.logger.info(f"Vector store initialized: collection={self.collection_name}")

        except Exception as e:
            if isinstance(e, VectorStoreException):
                raise
            raise VectorStoreException(
                f"Failed to initialize vector store: {str(e)}",
                collection_name=collection_name,
                operation="initialization",
                cause=e,
            )

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "VectorStore"

    @log_service_method()
    async def store_embeddings(self, documents) -> None:
        """Store documents using OpenAI embeddings."""
        if not documents:
            self.logger.warning("No documents provided for storage")
            return

        stored_count = 0
        failed_count = 0

        for document in documents:
            try:
                # Validate document structure
                if not hasattr(document, "knowledge") or not hasattr(document, "id"):
                    raise VectorStoreException(
                        "Invalid document structure: missing required fields",
                        collection_name=self.collection_name,
                        operation="store_embeddings",
                        error_code=ErrorCode.INVALID_INPUT,
                    )

                # Let ChromaDB handle the embedding generation
                await asyncio.to_thread(
                    self.about_me_collection.add,
                    documents=[" ".join(document.knowledge)],
                    metadatas=document.metadata,
                    ids=[document.id],
                )
                stored_count += 1

            except Exception as e:
                failed_count += 1
                self.logger.error(
                    f"Failed to store document {getattr(document, 'id', 'unknown')}: {str(e)}"
                )

        if failed_count > 0:
            self.logger.warning(
                f"Storage completed with {failed_count} failures out of {len(documents)} documents"
            )
        else:
            self.logger.info(f"Successfully stored {stored_count} documents")

    async def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant context for a given query."""
        # Run the blocking chromadb query in a thread pool
        results = await asyncio.to_thread(
            self.about_me_collection.query,
            query_texts=[query],  # Use query_texts instead of query_embeddings
            n_results=top_k,
            include=["documents"],
        )

        contexts = []
        for doc in results["documents"][0]:
            contexts.append(doc)

        return contexts

    @log_service_method()
    async def search_similar_text(self, search_text: str, n_results: int = 3) -> List[Any]:
        """Search for similar text asynchronously."""
        if not search_text or not search_text.strip():
            raise VectorStoreException(
                "Search text cannot be empty",
                collection_name=self.collection_name,
                operation="search_similar_text",
                error_code=ErrorCode.INVALID_INPUT,
            )

        if n_results <= 0:
            raise VectorStoreException(
                "Number of results must be positive",
                collection_name=self.collection_name,
                operation="search_similar_text",
                error_code=ErrorCode.INVALID_INPUT,
            )

        self.logger.info(f"Searching for similar text: '{search_text[:100]}...' (n={n_results})")

        try:
            # Run the blocking chromadb query in a thread pool
            results = await asyncio.to_thread(
                self.about_me_collection.query,
                query_texts=[search_text],
                n_results=n_results,
                include=["metadatas", "documents", "distances", "embeddings"],
            )

            sorted_results = []

            # Check if there are any results
            if results and results["ids"] and results["ids"][0]:
                # Create a list of tuples with all the result data
                sorted_results = list(
                    zip(
                        results["ids"][0],
                        results["distances"][0],
                        results["metadatas"][0],
                        results["documents"][0],
                        results["embeddings"][0],
                    )
                )

                # Sort results by similarity score (1 - distance) in descending order
                sorted_results = sorted(
                    sorted_results,
                    key=lambda x: x[1],  # Sort by distance (lower is better)
                )

                self.logger.info(f"Found {len(sorted_results)} matching documents")
                for i, (id, distance, metadata, document, _) in enumerate(sorted_results):
                    similarity = 1 - distance
                    self.logger.debug(f"Match {i+1}: ID={id}, Similarity={similarity:.4f}")

            else:
                self.logger.info("No matching documents found")

            return sorted_results

        except Exception as e:
            raise VectorStoreException(
                f"Search operation failed: {str(e)}",
                collection_name=self.collection_name,
                operation="search_similar_text",
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                details={"search_text_length": len(search_text), "n_results": n_results},
                cause=e,
            )

    async def get_all_records(self) -> List[Any]:
        """Get all records from the collection."""
        # Run the blocking chromadb call in a thread pool
        records = await asyncio.to_thread(self.about_me_collection.get)
        return records


if __name__ == "__main__":

    async def main():
        store = VectorStore()
        # Get all documents
        all_docs = await store.get_all_records()
        print(all_docs)

        # Test search
        # results = await store.search_similar_text("Drupal")
        # print(f"Found {len(results)} results")

    asyncio.run(main())
