"""
Shared test fixtures and configuration for the sales assistant tests.
"""
import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'EMAIL': 'test@example.com',
        'EMAIL_PWD': 'test_password',
        'EMAIL_RECEIVER': 'receiver@example.com',
        'SMTP_SERVER': 'smtp.test.com',
        'SMTP_PORT': '587',
        'OPENAI_API_KEY': 'test_openai_key'
    }):
        yield

@pytest.fixture
def mock_smtp_server():
    """Mock SMTP server for testing email functionality."""
    with patch('smtplib.SMTP') as mock_smtp, \
         patch('smtplib.SMTP_SSL') as mock_smtp_ssl:
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        mock_smtp_ssl.return_value = mock_server
        yield mock_server

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.AsyncOpenAI') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    with patch('assistant.services.vector_store.VectorStore') as mock_store:
        mock_instance = Mock()
        mock_instance.search_similar_text = AsyncMock(return_value=[
            ("id1", 0.1, {"source": "test"}, "Test document 1"),
            ("id2", 0.2, {"source": "test"}, "Test document 2")
        ])
        mock_store.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_chat_history():
    """Mock chat history for testing."""
    with patch('assistant.services.chat_history.ChatHistory') as mock_history:
        mock_instance = Mock()
        
        # Mock interaction objects
        mock_interaction = Mock()
        mock_interaction.user_message = "Test user message"
        mock_interaction.assistant_response = "Test assistant response"
        
        mock_instance.get_last_n_interactions.return_value = [mock_interaction]
        mock_instance.get_full_history.return_value = [mock_interaction]
        mock_instance.add_interaction.return_value = None
        mock_instance.clear_history.return_value = None
        
        mock_history.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {
            "post1": {
                "user": "test_user",
                "text": "Test post content",
                "metadata": "test metadata"
            }
        }
        import json
        json.dump(test_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

@pytest.fixture
def sample_chat_history():
    """Sample chat history for testing."""
    class MockChatEntry:
        def __init__(self, user_msg, assistant_msg):
            self.user_message = user_msg
            self.assistant_response = assistant_msg
    
    return [
        MockChatEntry("What services do you offer?", "I offer Drupal development services."),
        MockChatEntry("What are your rates?", "My rates vary by project scope.")
    ]

@pytest.fixture
def sample_user_context():
    """Sample user context for testing."""
    return {
        "session_id": "test_session_123",
        "ip_address": "192.168.1.1",
        "user_agent": "Mozilla/5.0 Test Browser"
    }