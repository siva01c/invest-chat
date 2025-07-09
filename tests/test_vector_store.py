"""
Unit tests for the VectorStore class and embedding functionality.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import chromadb

from src.services.vector_store import VectorStore, Record


class TestRecord:
    """Test the Record class."""
    
    def test_record_init(self):
        """Test Record initialization."""
        record = Record()
        
        assert record.knowledge == []
        assert record.metadata == []
        assert record.id == ''
    
    def test_record_to_dict(self):
        """Test Record to_dict method."""
        record = Record()
        record.knowledge = ["test knowledge"]
        record.metadata = [{"key": "value"}]
        record.id = "test_id"
        
        result = record.to_dict()
        
        assert result == {
            "knowledge": ["test knowledge"],
            "metadata": [{"key": "value"}],
            "id": "test_id"
        }


class TestVectorStore:
    """Test the VectorStore class."""
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_init_default_params(self, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test VectorStore initialization with default parameters."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        assert store.about_me_collection == mock_collection
        mock_embedding_func.assert_called_once()
        mock_chroma_client.assert_called_once_with(path="chromadb")
        mock_client_instance.get_or_create_collection.assert_called_once_with(
            name="about_me",
            embedding_function=mock_embedding_func.return_value
        )
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_init_custom_params(self, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test VectorStore initialization with custom parameters."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore(
            collection_name="custom_collection",
            database_path="custom/path"
        )
        
        mock_chroma_client.assert_called_once_with(path="custom/path")
        mock_client_instance.get_or_create_collection.assert_called_once_with(
            name="custom_collection",
            embedding_function=mock_embedding_func.return_value
        )
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    @patch('asyncio.to_thread')
    async def test_store_embeddings_success(self, mock_to_thread, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test successful embedding storage."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock asyncio.to_thread to return None (success)
        mock_to_thread.return_value = None
        
        # Create test documents
        doc1 = Record()
        doc1.knowledge = ["test knowledge 1"]
        doc1.metadata = [{"key": "value1"}]
        doc1.id = "doc1"
        
        doc2 = Record()
        doc2.knowledge = ["test knowledge 2"]
        doc2.metadata = [{"key": "value2"}]
        doc2.id = "doc2"
        
        await store.store_embeddings([doc1, doc2])
        
        # Verify asyncio.to_thread was called for each document
        assert mock_to_thread.call_count == 2
        
        # Verify the calls to the collection.add method
        calls = mock_to_thread.call_args_list
        assert len(calls) == 2
        
        # Check first call
        first_call = calls[0]
        assert first_call[0][0] == mock_collection.add
        
        # Check second call
        second_call = calls[1]
        assert second_call[0][0] == mock_collection.add
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    @patch('asyncio.to_thread')
    async def test_store_embeddings_failure(self, mock_to_thread, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test embedding storage with failure."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock asyncio.to_thread to raise exception
        mock_to_thread.side_effect = Exception("Storage error")
        
        # Create test document
        doc = Record()
        doc.knowledge = ["test knowledge"]
        doc.metadata = [{"key": "value"}]
        doc.id = "doc1"
        
        # Should not raise exception, but print error
        with patch('builtins.print') as mock_print:
            await store.store_embeddings([doc])
            
            mock_print.assert_called_once()
            assert "Error storing document" in mock_print.call_args[0][0]
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    @patch('asyncio.to_thread')
    async def test_retrieve_context_success(self, mock_to_thread, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test successful context retrieval."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock query results
        mock_results = {
            'documents': [["Document 1", "Document 2", "Document 3"]]
        }
        mock_to_thread.return_value = mock_results
        
        result = await store.retrieve_context("test query", top_k=3)
        
        assert result == ["Document 1", "Document 2", "Document 3"]
        mock_to_thread.assert_called_once()
        
        # Verify the query call
        call_args = mock_to_thread.call_args
        assert call_args[0][0] == mock_collection.query
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    @patch('asyncio.to_thread')
    async def test_retrieve_context_empty(self, mock_to_thread, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test context retrieval with empty results."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock empty query results
        mock_results = {
            'documents': [[]]
        }
        mock_to_thread.return_value = mock_results
        
        result = await store.retrieve_context("test query")
        
        assert result == []
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    @patch('asyncio.to_thread')
    async def test_search_similar_text_success(self, mock_to_thread, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test successful similar text search."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock query results
        mock_results = {
            'ids': [["doc1", "doc2"]],
            'distances': [[0.1, 0.3]],
            'metadatas': [[{"key": "value1"}, {"key": "value2"}]],
            'documents': [["Document 1 content", "Document 2 content"]],
            'embeddings': [[[0.1, 0.2], [0.3, 0.4]]]
        }
        mock_to_thread.return_value = mock_results
        
        with patch('builtins.print') as mock_print:
            result = await store.search_similar_text("test query", n_results=2)
            
            # Check result structure
            assert len(result) == 2
            assert result[0][0] == "doc1"  # id
            assert result[0][1] == 0.1     # distance
            assert result[0][2] == {"key": "value1"}  # metadata
            assert result[0][3] == "Document 1 content"  # document
            
            # Verify results are sorted by distance
            assert result[0][1] <= result[1][1]
            
            # Verify print statements were called
            mock_print.assert_called()
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    @patch('asyncio.to_thread')
    async def test_search_similar_text_empty(self, mock_to_thread, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test similar text search with empty results."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock empty query results
        mock_results = {
            'ids': [[]],
            'distances': [[]],
            'metadatas': [[]],
            'documents': [[]],
            'embeddings': [[]]
        }
        mock_to_thread.return_value = mock_results
        
        with patch('builtins.print') as mock_print:
            result = await store.search_similar_text("test query")
            
            assert result == []
            mock_print.assert_called()
            
            # Check that "No matching documents found" was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any("No matching documents found" in call for call in print_calls)
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    @patch('asyncio.to_thread')
    async def test_search_similar_text_error(self, mock_to_thread, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test similar text search with error."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock asyncio.to_thread to raise exception
        mock_to_thread.side_effect = Exception("Search error")
        
        with patch('builtins.print') as mock_print:
            result = await store.search_similar_text("test query")
            
            assert result == []
            mock_print.assert_called()
            
            # Check that error was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any("Error during search" in call for call in print_calls)
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    @patch('asyncio.to_thread')
    async def test_get_all_records_success(self, mock_to_thread, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test successful retrieval of all records."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock get results
        mock_records = {
            'ids': ["doc1", "doc2"],
            'documents': ["Document 1", "Document 2"],
            'metadatas': [{"key": "value1"}, {"key": "value2"}]
        }
        mock_to_thread.return_value = mock_records
        
        result = await store.get_all_records()
        
        assert result == mock_records
        mock_to_thread.assert_called_once()
        
        # Verify the get call
        call_args = mock_to_thread.call_args
        assert call_args[0][0] == mock_collection.get
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    @patch('asyncio.to_thread')
    async def test_get_all_records_error(self, mock_to_thread, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test get all records with error."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock asyncio.to_thread to raise exception
        mock_to_thread.side_effect = Exception("Get records error")
        
        with pytest.raises(Exception, match="Get records error"):
            await store.get_all_records()


class TestVectorStoreIntegration:
    """Integration tests for VectorStore functionality."""
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    async def test_store_and_search_workflow(self, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test complete workflow of storing and searching embeddings."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock storage success
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None
            
            # Create test documents
            doc1 = Record()
            doc1.knowledge = ["Drupal is a content management system"]
            doc1.metadata = [{"topic": "drupal"}]
            doc1.id = "drupal_doc"
            
            doc2 = Record()
            doc2.knowledge = ["PHP is a programming language"]
            doc2.metadata = [{"topic": "php"}]
            doc2.id = "php_doc"
            
            await store.store_embeddings([doc1, doc2])
            
            # Verify storage was called
            assert mock_to_thread.call_count == 2
        
        # Mock search results
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_results = {
                'ids': [["drupal_doc"]],
                'distances': [[0.1]],
                'metadatas': [[{"topic": "drupal"}]],
                'documents': [["Drupal is a content management system"]],
                'embeddings': [[[0.1, 0.2, 0.3]]]
            }
            mock_to_thread.return_value = mock_results
            
            with patch('builtins.print'):
                results = await store.search_similar_text("What is Drupal?")
                
                assert len(results) == 1
                assert results[0][0] == "drupal_doc"
                assert results[0][3] == "Drupal is a content management system"
    
    @patch('services.vector_store.chromadb.PersistentClient')
    @patch('services.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    async def test_retrieve_context_integration(self, mock_embedding_func, mock_chroma_client, mock_env_vars):
        """Test context retrieval integration."""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        
        # Mock retrieve context results
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_results = {
                'documents': [[
                    "Drupal is a free, open-source content management system",
                    "It's written in PHP and uses a MySQL database",
                    "Drupal is highly customizable and extensible"
                ]]
            }
            mock_to_thread.return_value = mock_results
            
            contexts = await store.retrieve_context("Drupal CMS", top_k=3)
            
            assert len(contexts) == 3
            assert "Drupal is a free, open-source content management system" in contexts
            assert "written in PHP" in contexts[1]
            assert "customizable and extensible" in contexts[2]