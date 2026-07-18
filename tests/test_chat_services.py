"""
Unit tests for the AIService chat handlers (services, drupal, hugo, aws).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from assistant.core.services.chat_service import AIService


class TestAIServiceHandlers:
    """Test the AIService request handlers for different categories."""

    @pytest.fixture
    def ai_service(self):
        """Create AIService instance for testing."""
        with patch("assistant.infrastructure.database.vector_store.VectorStore"):
            with patch("assistant.utils.config_loader.load_json_data") as mock_load:
                mock_load.return_value = {}
                service = AIService()
                return service

    @pytest.mark.asyncio
    async def test_services_handler_czech(self, ai_service):
        """Test services category handler in Czech."""
        user_text = "chtěl bych chatbota"

        with patch(
            "assistant.core.config.services.get_classification_service"
        ) as mock_get_classification:
            mock_classification_service = AsyncMock()
            mock_classification_service.classify_message_extended.return_value = {
                "contexts": [],
                "conversation_category": "services",
                "agent": True,
                "message": "I would be happy to help you with your project. Can you describe more what your business problem is that you want to solve?",
            }
            mock_get_classification.return_value = mock_classification_service

            result = await ai_service.chat(user_text)

        assert (
            result
            == "I would be happy to help you with your project. Can you describe more what your business problem is that you want to solve?"
        )

    @pytest.mark.asyncio
    async def test_services_handler_english(self, ai_service):
        """Test services category handler in English."""
        user_text = "I would like a chatbot"

        with patch(
            "assistant.core.config.services.get_classification_service"
        ) as mock_get_classification:
            mock_classification_service = AsyncMock()
            mock_classification_service.classify_message_extended.return_value = {
                "contexts": [],
                "conversation_category": "services",
                "agent": True,
                "message": "I would be happy to help you with your project. Can you describe more what your business problem is that you want to solve?",
            }
            mock_get_classification.return_value = mock_classification_service

            result = await ai_service.chat(user_text)

        assert (
            result
            == "I would be happy to help you with your project. Can you describe more what your business problem is that you want to solve?"
        )

    @pytest.mark.asyncio
    async def test_drupal_handler_czech(self, ai_service):
        """Test drupal category handler in Czech."""
        user_text = "potřebuji drupal web"

        with patch(
            "assistant.core.config.services.get_classification_service"
        ) as mock_get_classification:
            mock_classification_service = AsyncMock()
            mock_classification_service.classify_message_extended.return_value = {
                "contexts": [],
                "conversation_category": "drupal",
                "agent": True,
                "message": "Drupal is my favorite content management system. What business problem does Drupal solve for you?",
            }
            mock_get_classification.return_value = mock_classification_service

            result = await ai_service.chat(user_text)

        assert (
            result
            == "Drupal is my favorite content management system. What business problem does Drupal solve for you?"
        )

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_drupal_handler_english(self, ai_service):
        """Test Drupal category handler in English."""
        user_text = "looking for Drupal developer"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "drupal"
            mock_completion.return_value = mock_chat

            with patch("assistant.services.chat.detect_language", return_value="en"):
                result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "Luděk Kvapil has extensive experience with **Drupal**" in result["message"]
        assert "Drupal 7, 8, 9, 10, and 11" in result["message"]
        assert "Migrations and upgrades" in result["message"]
        assert result["prompt_category"] == "drupal"

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_hugo_handler_czech(self, ai_service):
        """Test Hugo static sites handler in Czech."""
        user_text = "potřebuji jednoduchý web"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "technology_description"
            mock_completion.return_value = mock_chat

            with patch("assistant.services.chat.detect_language", return_value="cs"):
                result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "Hugo framework skvělá volba" in result["message"]
        assert "bleskově rychlé načítání" in result["message"]
        assert "portfolia, firemní prezentace, blogy" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_hugo_handler_english(self, ai_service):
        """Test Hugo static sites handler in English."""
        user_text = "need simple static website"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "technology_description"
            mock_completion.return_value = mock_chat

            with patch("assistant.services.chat.detect_language", return_value="en"):
                result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "Hugo framework is an excellent choice" in result["message"]
        assert "lightning-fast loading" in result["message"]
        assert "portfolios, business presentations, blogs" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_aws_handler_czech(self, ai_service):
        """Test AWS serverless handler in Czech."""
        user_text = "potřebuji serverless aplikaci"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "aws"
            mock_completion.return_value = mock_chat

            with patch("assistant.services.chat.detect_language", return_value="cs"):
                result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "AWS serverless technologiemi" in result["message"]
        assert "AWS Lambda funkce" in result["message"]
        assert "Platíte jen za využití" in result["message"]
        assert "Automatické škálování" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_aws_handler_english(self, ai_service):
        """Test AWS serverless handler in English."""
        user_text = "need serverless solution"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "aws"
            mock_completion.return_value = mock_chat

            with patch("assistant.services.chat.detect_language", return_value="en"):
                result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "AWS serverless technologies" in result["message"]
        assert "AWS Lambda functions" in result["message"]
        assert "Pay only for usage" in result["message"]
        assert "Automatic scaling" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_service_details_handler_with_context(self, ai_service):
        """Test service_details handler when there's service context in history."""
        user_text = "chci e-commerce web s AI doporučeními"

        # Mock chat history with service context
        mock_interaction = Mock()
        mock_interaction.assistant_response = "služeb poskytuje Luděk"
        ai_service.chat_history.get_last_n_interactions = Mock(return_value=[mock_interaction])

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "service_details"
            mock_completion.return_value = mock_chat

            with patch("assistant.services.chat.detect_language", return_value="cs"):
                result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "Děkuji za detaily" in result["message"]
        assert "Vývojem chatbotů" in result["message"]
        assert "konzultaci zdarma" in result["message"]
        assert "Chcete, abych poslal shrnutí" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_service_details_handler_without_context(self, ai_service):
        """Test service_details handler when there's no service context."""
        user_text = "více informací"

        # Mock empty chat history
        ai_service.chat_history.get_last_n_interactions = Mock(return_value=[])

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "service_details"
            mock_completion.return_value = mock_chat

            with patch("assistant.services.chat.detect_language", return_value="cs"):
                result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "konkrétní požadavky na projekt" in result["message"]
        assert "Můžete mi říci více" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_help_request_handler_czech(self, ai_service):
        """Test help_request handler in Czech."""
        user_text = "tomu já nerozumím, můžeš mi pomoct?"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "help_request"
            mock_completion.return_value = mock_chat

            with patch("assistant.services.chat.detect_language", return_value="cs"):
                result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "Chápu, že to může být složité" in result["message"]
        assert "Luděk Kvapil je expert" in result["message"]
        assert "Odbornou konzultaci" in result["message"]
        assert "info@ludekkvapil.cz" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_help_request_handler_english(self, ai_service):
        """Test help_request handler in English."""
        user_text = "I don't understand, can you help me?"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "help_request"
            mock_completion.return_value = mock_chat

            with patch("assistant.services.chat.detect_language", return_value="en"):
                result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "I understand this can be complex" in result["message"]
        assert "Luděk Kvapil is an expert" in result["message"]
        assert "Expert consultation" in result["message"]
        assert "info@ludekkvapil.cz" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_clear_chat_handler(self, ai_service):
        """Test clear_chat handler."""
        user_text = "clear chat"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "clear_chat"
            mock_completion.return_value = mock_chat

            result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "Conversation history has been cleared" in result["message"]
        assert result["prompt_category"] == "clear_chat"

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_greeting_handler(self, ai_service):
        """Test greeting handler."""
        user_text = "hello"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "greeting"
            mock_completion.return_value = mock_chat

            result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "Welcome! How can I assist you today?" in result["message"]
        assert "Luděk, his skills, or his services" in result["message"]
        assert result["prompt_category"] == "greeting"

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_common_knowledge_handler(self, ai_service):
        """Test common_knowledge handler."""
        user_text = "what is the capital of France?"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "common_knowledge"
            mock_completion.return_value = mock_chat

            result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "specifically designed to answer questions about Luděk Kvapil" in result["message"]
        assert "outside that scope" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_code_handler(self, ai_service):
        """Test code handler."""
        user_text = "write me a Python function"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "code"
            mock_completion.return_value = mock_chat

            result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is True
        assert "not designed to write or review code" in result["message"]
        assert "programming languages and technologies Luděk works" in result["message"]

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_default_response(self, ai_service):
        """Test default response for unmatched categories."""
        user_text = "some random text"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_chat = AsyncMock()
            mock_chat.get_response.return_value = "unknown_category"
            mock_completion.return_value = mock_chat

            result = await ai_service.handle_user_request(user_text)

        assert result["agent"] is False
        assert result["message"] == user_text
        assert result["prompt_category"] == "unknown_category"

    @pytest.mark.skip(
        reason="Test needs refactoring for new architecture - handle_user_request removed from AIService"
    )
    @pytest.mark.asyncio
    async def test_exception_handling(self, ai_service):
        """Test exception handling in handle_user_request."""
        user_text = "test exception"

        with patch("assistant.services.chat.ChatCompletion") as mock_completion:
            mock_completion.side_effect = Exception("Test exception")

            result = await ai_service.handle_user_request(user_text)

        assert result == user_text  # Returns original text on exception
