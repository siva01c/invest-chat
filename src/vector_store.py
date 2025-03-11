from typing import List, Optional, Dict, Any
import torch
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI 
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

MODEL="gpt-4o-mini"
client = AsyncOpenAI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Record:
    def __init__(self):
        self.knowledge = []
        self.metadata = []
        self.id = ''

    def to_dict(self):
        return {
            "knowledge": self.knowledge,
            "metadata": self.metadata,
            "id": self.id
        }   

class VectorStore:
    """Manages storage and retrieval of vector embeddings."""

    def __init__(
        self, 
        collection_name: str = "about_me",
        database_path: str = "chromadb",
    ):
        """Initialize the vector store."""
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002"
        )

        self.chroma_client = chromadb.PersistentClient(path=database_path)
        self.about_me_collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    async def store_embeddings(self, documents) -> None:
        """Store documents using OpenAI embeddings."""
        for document in documents:
            try:
                # Let ChromaDB handle the embedding generation
                await asyncio.to_thread(
                    self.about_me_collection.add,
                    documents=[" ".join(document.knowledge)],
                    metadatas=document.metadata,
                    ids=[document.id]
                    # Remove the embeddings parameter - let ChromaDB generate them
                )
            except Exception as e:
                print(f"Error storing document: {str(e)}")

    async def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant context for a given query."""
        # Run the blocking chromadb query in a thread pool
        results = await asyncio.to_thread(
            self.about_me_collection.query,
            query_texts=[query],  # Use query_texts instead of query_embeddings
            n_results=top_k,
            include=["documents"]
        )
        
        contexts = []
        for doc in results['documents'][0]:
            contexts.append(doc)
        
        return contexts
    
    async def search_similar_text(self, search_text: str, n_results: int = 3) -> List[Any]:
        """Search for similar text asynchronously."""
        print(f"\nSearching for text similar to: '{search_text}'")
        
        try:
            # Run the blocking chromadb query in a thread pool
            results = await asyncio.to_thread(
                self.about_me_collection.query,
                query_texts=[search_text],
                n_results=n_results,
                include=["metadatas", "documents", "distances", "embeddings"]
            )
            
            sorted_results = []
            
            # Check if there are any results
            if results and results["ids"] and results["ids"][0]:
                print("\nMatching documents found:")
                
                # Create a list of tuples with all the result data
                sorted_results = list(zip(
                    results["ids"][0], 
                    results["distances"][0], 
                    results["metadatas"][0], 
                    results["documents"][0],
                    results["embeddings"][0]
                ))
                
                # Sort results by similarity score (1 - distance) in descending order
                sorted_results = sorted(
                    sorted_results,
                    key=lambda x: x[1],  # Sort by distance (lower is better)
                )    

                for i, (id, distance, metadata, document, _) in enumerate(sorted_results):
                    similarity = 1 - distance
                    print(f"\n{i+1}. Match Details:")
                    print(f"ID: {id}")
                    print(f"Similarity Score: {similarity:.4f}")
                    
                    if metadata:
                        print(f"Metadata: {metadata}")
        
                    if document:
                       print(f"Text Preview: {document[:1200]}...")
            
            else:
                print("No matching documents found")
            
            return sorted_results
        
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
        
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
        #results = await store.search_similar_text("Drupal")
        # print(f"Found {len(results)} results")
    
    asyncio.run(main())