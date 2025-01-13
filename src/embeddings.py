from typing import List, Optional, Dict
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class VectorStore:
    """Manages storage and retrieval of vector embeddings."""

    def __init__(
        self, 
        collection_name: str = "embeddings",
        database_path: str = "database",
        device: Optional[str] = None,
    ):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection to use.
            database_path: Path to the persistent database.
            model: Embedding model to use for generating embeddings.
            device: Device to run embedding computations on (e.g., "cpu" or "cuda").
        """
        self.client = chromadb.PersistentClient(
            path=database_path,
            settings=Settings(),
        )

        self.collection = self._get_or_create_collection(collection_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _get_or_create_collection(self, name: str):
        """Get or create a collection by name.
        
        Args:
            name: Name of the collection.
        
        Returns:
            The collection object.
        """
        collections = self.client.list_collections()
        for collection in collections:
            if collection.name == name:
                return collection
        return self.client.create_collection(name=name)
    
    def create_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for the provided texts.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            Tensor containing the embeddings.
        """
        if not texts:
            raise ValueError("The 'texts' argument cannot be empty.")

        # Example: Text preprocessing logic (to be implemented)
        processed_texts = self._preprocess_texts(texts)

        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model.encode(
                processed_texts,
                convert_to_tensor=True,
                device=self.device
            )

        return embeddings

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for embedding generation (e.g., tokenization, splitting).
        
        Args:
            texts: List of raw text strings.
        
        Returns:
            List of preprocessed text strings.
        """
        # Placeholder for text preprocessing logic
        return texts

    def store_embeddings(
        self, 
        embeddings: torch.Tensor, 
        metadata: List[Dict[str, str]], 
        documents: List[str]
    ) -> None:
        """Store embeddings in the vector database with metadata.
        
        Args:
            embeddings: Tensor of embeddings to store.
            metadata: List of metadata dictionaries corresponding to each embedding.
            documents: List of raw text or table data corresponding to each embedding.
        """
        if embeddings.size(0) != len(metadata) or embeddings.size(0) != len(documents):
            raise ValueError("Embeddings, metadata, and documents must have the same length.")

        embeddings_list = embeddings.cpu().numpy().tolist()

        for embedding, meta, document in zip(embeddings_list, metadata, documents):
            if "filename" not in meta or "page" not in meta:
                raise ValueError("Metadata must include 'filename' and 'page' keys.")

            self.collection.add(
                embeddings=[embedding],
                ids=[f"{meta['filename']}_page_{meta['page']}"],
                metadatas=[meta],
                documents=[document]
            )
            print(f"Stored embedding for {meta['filename']} page {meta['page']}")

    def retrieve_embeddings(self, query: List[float], top_k: int = 5) -> List[Dict]:
        """Retrieve the most similar embeddings from the database.
        
        Args:
            query: Query embedding as a list of floats (embedded vector).
            top_k: Number of top results to retrieve.
        
        Returns:
            List of dictionaries containing metadata and documents of top results.
        """
        results = self.collection.query(query_embeddings=[query], n_results=top_k)
        return results

if __name__ == "__main__":
    store = VectorStore()
