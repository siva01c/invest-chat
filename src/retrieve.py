from embeddings import VectorStore
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings


def print_collection_info(collection):
    """Print detailed information about a collection."""
    try:
        data = collection.get()
        print(f"\nCollection: {collection.name}")
        print(f"Total items: {len(data['ids'])}")
        print("\nSample of stored documents:")
        
        # Print first few documents with their IDs
        for i in range(min(3, len(data['ids']))):
            print(f"\nDocument {i+1}:")
            print(f"ID: {data['ids'][i]}")
            if data.get('metadatas') and data['metadatas'][i]:
                print(f"Metadata: {data['metadatas'][i]}")
            if data.get('documents') and data['documents'][i]:
                print(f"Text: {data['documents'][i][:140]}...")  # Show first 140 chars
    except Exception as e:
        print(f"Error getting collection info: {str(e)}")

def search_similar_text(search_text: str, n_results: int = 3):
    """Search for similar text with detailed results."""
    store = VectorStore()

    client = chromadb.PersistentClient(
        path="database",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
        
    store.collection = client.get_collection("embeddings")
    
    print(f"\nSearching for text similar to: '{search_text}'")
    
    try:
        # Create embedding for search text
        search_embedding = store.create_embeddings(search_text)
        search_embedding_np = search_embedding.cpu().numpy()
        
        # Ensure embedding is properly formatted
        if len(search_embedding_np.shape) > 1:
            search_embedding_np = search_embedding_np.flatten()
            
        # Query with metadata and documents included
        results = store.collection.query(
            query_embeddings=[search_embedding_np.tolist()],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        sorted_results = []
        # Print detailed results
        if results and results["ids"]:
            print("\nMatching documents found:")
            for i, (id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
                similarity = 1 - distance

            # Sort results by similarity score in descending order
            sorted_results = sorted(
                zip(results["ids"][0], results["distances"][0], results["metadatas"][0], results["documents"][0]),
                key=lambda x: 1 - x[1],
                reverse=False
            )

            for i, (id, distance, metadata, document) in enumerate(sorted_results):
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
        return None

if __name__ == "__main__":
    # Initialize components
    store = VectorStore()