"""
Unit tests for the AIService class and chat functionality.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from assistant.core.services.chat_service import AIService
from assistant.utils.config_loader import load_json_data


class TestLoadJsonData:
    """Test the load_json_data function."""

    def test_load_json_data_success(self, temp_json_file):
        """Test successful JSON data loading."""
        data = load_json_data(temp_json_file)

        assert data is not None
        assert "post1" in data
        assert data["post1"]["user"] == "test_user"
        assert data["post1"]["text"] == "Test post content"

    def test_load_json_data_file_not_found(self):
        """Test JSON data loading with non-existent file."""
        data = load_json_data("non_existent_file.json")

        assert data == {}

    def test_load_json_data_invalid_json(self):
        """Test JSON data loading with invalid JSON."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            data = load_json_data(temp_path)
            assert data == {}
        finally:
            os.unlink(temp_path)


class TestAIService:
    """Test the AIService class."""

    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    def test_init_default_params(self, mock_load_json, mock_vector_store, mock_env_vars):
        """Test AIService initialization with default parameters."""
        mock_load_json.return_value = {"test": "data"}

        service = AIService()

        assert service.model_name == "gpt-4o-mini"
        assert service.temperature == 0
        assert service.context_window == 3
        assert service.data == {"test": "data"}
        mock_load_json.assert_called_once()

    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    def test_init_custom_params(self, mock_load_json, mock_vector_store, mock_env_vars):
        """Test AIService initialization with custom parameters."""
        mock_load_json.return_value = {"test": "data"}

        service = AIService(
            model_name="gpt-4",
            max_history=10,
            temperature=0.7,
            context_window=5,
            json_path="custom/path.json",
        )

        assert service.model_name == "gpt-4"
        assert service.temperature == 0.7
        assert service.context_window == 5
        mock_load_json.assert_called_with("custom/path.json")

    @patch("assistant.core.services.chat_service.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    @patch("os.getenv")
    def test_init_missing_openai_key(self, mock_getenv, mock_load_json, mock_vector_store):
        """Test AIService initialization with missing OpenAI key."""
        mock_load_json.return_value = {"test": "data"}
        mock_getenv.return_value = None  # No OPENAI_API_KEY

        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            AIService()

    @patch("assistant.core.services.chat_service.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    def test_prepare_knowledge_base_empty(self, mock_load_json, mock_vector_store, mock_env_vars):
        """Test knowledge base preparation with empty results."""
        mock_load_json.return_value = {"test": "data"}
        service = AIService()

        result = service._prepare_knowledge_base([])

        assert result == "No relevant information found."

    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    def test_prepare_knowledge_base_with_data(
        self, mock_load_json, mock_vector_store, mock_env_vars
    ):
        """Test knowledge base preparation with data."""
        mock_load_json.return_value = {"test": "data"}
        service = AIService()

        similar_texts = [
            ("id1", 0.1, {"meta": "data"}, "Document 1 content"),
            ("id2", 0.2, {"meta": "data"}, "Document 2 content"),
        ]

        result = service._prepare_knowledge_base(similar_texts)

        assert "Document 1 content" in result
        assert "Document 2 content" in result
        assert "\n\n" in result

    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    def test_create_system_prompt(self, mock_load_json, mock_vector_store, mock_env_vars):
        """Test system prompt creation."""
        mock_load_json.return_value = {"test": "data"}
        service = AIService()

        knowledge_base = "Test knowledge base content"
        prompt = service._create_system_prompt(knowledge_base)

        assert "Luděk Kvapil – Assistant System Prompt" in prompt
        assert "Test knowledge base content" in prompt
        assert "info@ludekkvapil.cz" in prompt
        assert "Drupal" in prompt
        assert "cybersecurity" in prompt

    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    def test_prepare_messages_basic(self, mock_load_json, mock_vector_store, mock_env_vars):
        """Test message preparation with basic input."""
        mock_load_json.return_value = {"test": "data"}
        service = AIService()
        service.chat_history.get_last_n_interactions = Mock(return_value=[])

        system_prompt = "Test system prompt"
        user_question = "What is Drupal?"
        knowledge_base = "Test knowledge base"

        messages = service._prepare_messages(system_prompt, user_question, knowledge_base)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[1]["role"] == "user"
        assert "What is Drupal?" in messages[1]["content"]

    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    def test_prepare_messages_with_history(self, mock_load_json, mock_vector_store, mock_env_vars):
        """Test message preparation with chat history."""
        mock_load_json.return_value = {"test": "data"}
        service = AIService()

        # Mock chat history
        class MockInteraction:
            def __init__(self, user_msg, assistant_msg):
                self.user_message = user_msg
                self.assistant_response = assistant_msg

        mock_interactions = [MockInteraction("Previous question", "Previous answer")]
        service.chat_history.get_last_n_interactions = Mock(return_value=mock_interactions)

        system_prompt = "Test system prompt"
        user_question = "What is Drupal?"
        knowledge_base = "Test knowledge base"

        messages = service._prepare_messages(system_prompt, user_question, knowledge_base)

        assert len(messages) == 4  # system + previous user + previous assistant + current user
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Previous question"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Previous answer"

    @pytest.mark.asyncio
    @patch("assistant.core.services.chat_service.load_json_data")
    @patch("assistant.core.services.chat_service.ChatCompletion")
    async def test_generate_response_success(
        self, mock_chat_completion, mock_load_json, mock_vector_store
    ):
        """Test successful response generation."""
        mock_load_json.return_value = {"test": "data"}
        service = AIService()

        # Mock ChatCompletion
        mock_completion_instance = Mock()
        mock_completion_instance.get_response = AsyncMock(return_value="Test response")
        mock_chat_completion.return_value = mock_completion_instance

        messages = [{"role": "user", "content": "Test question"}]

        result = await service._generate_response(messages)

        assert result == "Test response"
        mock_chat_completion.assert_called_once()
        mock_completion_instance.get_response.assert_called_once()

    @pytest.mark.asyncio
    @patch("assistant.core.services.chat_service.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    @patch("assistant.core.services.chat_service.ChatCompletion")
    async def test_generate_response_failure(
        self, mock_chat_completion, mock_load_json, mock_vector_store, mock_env_vars
    ):
        """Test response generation with failure."""
        mock_load_json.return_value = {"test": "data"}
        service = AIService()

        # Mock ChatCompletion to raise exception
        mock_completion_instance = Mock()
        mock_completion_instance.get_response = AsyncMock(side_effect=Exception("API Error"))
        mock_chat_completion.return_value = mock_completion_instance

        messages = [{"role": "user", "content": "Test question"}]

        result = await service._generate_response(messages)

        assert "AI generation failed" in result
        assert "API Error" in result

    @pytest.mark.skip(reason="Test needs better ChatCompletion mocking to avoid API calls")
    @pytest.mark.asyncio
    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    @patch("assistant.core.config.services.get_classification_service")
    @patch("assistant.infrastructure.llm.openai_client.ChatCompletion")
    async def test_chat_with_contexts(
        self,
        mock_chat_completion,
        mock_get_classification,
        mock_load_json,
        mock_vector_store,
        mock_env_vars,
    ):
        """Test chat function with provided contexts."""
        mock_load_json.return_value = {"test": "data"}

        # Mock classification service to return contexts
        mock_classification_service = Mock()
        mock_classification_service.classify_message_extended = AsyncMock(
            return_value={
                "contexts": ["Context 1", "Context 2"],
                "conversation_category": "drupal",
                "agent": False,
                "message": "",
            }
        )
        mock_get_classification.return_value = mock_classification_service

        service = AIService()

        # Mock ChatCompletion for response generation
        mock_completion_instance = Mock()
        mock_completion_instance.get_response = AsyncMock(return_value="Test response")
        mock_chat_completion.return_value = mock_completion_instance

        service.chat_history.add_interaction = Mock()

        result = await service.chat("What is Drupal?")

        assert result == "Test response"
        service.chat_history.add_interaction.assert_called_once()

    @pytest.mark.asyncio
    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    @patch("assistant.core.config.services.get_classification_service")
    async def test_chat_with_agent_response(
        self, mock_get_classification, mock_load_json, mock_vector_store, mock_env_vars
    ):
        """Test chat function with agent response."""
        mock_load_json.return_value = {"test": "data"}

        # Mock classification service to return agent response
        mock_classification_service = Mock()
        mock_classification_service.classify_message_extended = AsyncMock(
            return_value={
                "contexts": [],
                "conversation_category": "clear_chat",
                "agent": True,
                "message": "Agent response",
            }
        )
        mock_get_classification.return_value = mock_classification_service

        service = AIService()

        result = await service.chat("clear chat")

        assert result == "Agent response"

    @pytest.mark.skip(reason="Test needs better ChatCompletion mocking to avoid API calls")
    @pytest.mark.asyncio
    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    @patch("assistant.core.config.services.get_classification_service")
    @patch("assistant.infrastructure.llm.openai_client.ChatCompletion")
    async def test_chat_with_vector_search(
        self,
        mock_chat_completion,
        mock_get_classification,
        mock_load_json,
        mock_vector_store,
        mock_env_vars,
    ):
        """Test chat function with vector search."""
        mock_load_json.return_value = {"test": "data"}

        # Mock classification service to return no contexts
        mock_classification_service = Mock()
        mock_classification_service.classify_message_extended = AsyncMock(
            return_value={
                "contexts": [],
                "conversation_category": "drupal",
                "agent": False,
                "message": "",
            }
        )
        mock_get_classification.return_value = mock_classification_service

        service = AIService()

        # Mock vector store search
        service.store.search_similar_text = AsyncMock(
            return_value=[("id1", 0.1, {"meta": "data"}, "Drupal is a CMS")]
        )

        # Mock ChatCompletion for response generation
        mock_completion_instance = Mock()
        mock_completion_instance.get_response = AsyncMock(return_value="Drupal response")
        mock_chat_completion.return_value = mock_completion_instance

        service.chat_history.add_interaction = Mock()

        result = await service.chat("What is Drupal?")

        assert result == "Drupal response"
        service.store.search_similar_text.assert_called_once()

    @pytest.mark.skip(reason="Test needs proper mock_open setup for load_config context manager")
    @pytest.mark.asyncio
    @patch("assistant.infrastructure.database.vector_store.VectorStore")
    @patch("assistant.core.services.chat_service.load_json_data")
    @patch("assistant.core.config.services.get_classification_service")
    @patch("assistant.infrastructure.llm.openai_client.ChatCompletion")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=Mock)
    async def test_chat_logging(
        self,
        mock_open,
        mock_mkdir,
        mock_chat_completion,
        mock_get_classification,
        mock_load_json,
        mock_vector_store,
        mock_env_vars,
    ):
        """Test chat logging functionality."""
        mock_load_json.return_value = {"test": "data"}

        # Mock classification service
        mock_classification_service = Mock()
        mock_classification_service.classify_message_extended = AsyncMock(
            return_value={
                "contexts": [],
                "conversation_category": "drupal",
                "agent": False,
                "message": "",
            }
        )
        mock_get_classification.return_value = mock_classification_service

        service = AIService()

        # Mock vector search
        service.store.search_similar_text = AsyncMock(return_value=[])

        # Mock ChatCompletion for response generation
        mock_completion_instance = Mock()
        mock_completion_instance.get_response = AsyncMock(return_value="Test response")
        mock_chat_completion.return_value = mock_completion_instance

        service.chat_history.add_interaction = Mock()

        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        result = await service.chat("Test question")

        assert result == "Test response"
        mock_mkdir.assert_called_once()
        mock_file.write.assert_called_once()

        # Verify logged data structure
        logged_data = mock_file.write.call_args[0][0]
        assert '"user"' in logged_data
        assert '"assistant"' in logged_data
        assert '"system_prompt"' in logged_data
