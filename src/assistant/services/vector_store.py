import asyncio
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Record:
    """Represents a document item to be indexed into ChromaDB."""
    def __init__(self, doc_id: str, knowledge: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        self.id = doc_id
        self.knowledge = knowledge
        self.metadata = metadata or []

    def to_dict(self):
        return {"id": self.id, "knowledge": self.knowledge, "metadata": self.metadata}


class VectorStore:
    """Manages storage and retrieval of vector embeddings for Investment RAG."""

    def __init__(
        self,
        collection_name: str = "investment_knowledge",
        database_path: str = "chromadb",
    ):
        """Initialize the vector store with ChromaDB and OpenAI Embeddings."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback to dummy for initialization or testing if key not set yet
            api_key = "dummy"

        original_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key
        try:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key_env_var="OPENAI_API_KEY", model_name="text-embedding-3-small"
            )
        finally:
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

        self.chroma_client = chromadb.PersistentClient(path=database_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    async def store_embeddings(self, documents: List[Any]) -> int:
        """Store documents using OpenAI embeddings."""
        stored_count = 0
        for doc in documents:
            try:
                doc_id = getattr(doc, "id", None) or f"doc_{stored_count}"
                doc_text = " ".join(doc.knowledge) if isinstance(doc.knowledge, list) else str(doc.knowledge)
                metadatas = doc.metadata if hasattr(doc, "metadata") and doc.metadata else [{}]
                if isinstance(metadatas, list) and len(metadatas) > 0:
                    meta = metadatas[0] if isinstance(metadatas[0], dict) else {}
                else:
                    meta = {}

                await asyncio.to_thread(
                    self.collection.add,
                    documents=[doc_text],
                    metadatas=[meta],
                    ids=[doc_id],
                )
                stored_count += 1
            except Exception as e:
                print(f"Error storing document {getattr(doc, 'id', 'unknown')}: {e}")

        return stored_count

    async def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant document context for a given query."""
        results = await asyncio.to_thread(
            self.collection.query,
            query_texts=[query],
            n_results=top_k,
            include=["documents"],
        )

        contexts = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            for doc in results["documents"][0]:
                contexts.append(doc)

        return contexts

    async def search_similar_text(self, search_text: str, n_results: int = 3) -> List[Any]:
        """Search for similar text with distances and metadata."""
        try:
            results = await asyncio.to_thread(
                self.collection.query,
                query_texts=[search_text],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
            )

            sorted_results = []
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                sorted_results = list(
                    zip(
                        results["ids"][0],
                        results["distances"][0],
                        results["metadatas"][0],
                        results["documents"][0],
                    )
                )
                sorted_results = sorted(sorted_results, key=lambda x: x[1])

            return sorted_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    async def get_all_records(self) -> Dict[str, Any]:
        """Get all records from the collection."""
        return await asyncio.to_thread(self.collection.get)

    async def clear_collection(self) -> None:
        """Clear all documents from collection."""
        try:
            records = await self.get_all_records()
            if records and records.get("ids"):
                await asyncio.to_thread(self.collection.delete, ids=records["ids"])
        except Exception as e:
            print(f"Error clearing collection: {e}")
