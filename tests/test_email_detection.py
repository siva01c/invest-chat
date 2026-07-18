"""Tests for improved email detection functionality."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from assistant.core.services.chat_service import AIService as ChatService
from assistant.services.chat_history import ChatHistory

# Test constants
LUDEK_EMAILS = ["info@ludekkvapil.cz", "ludek@ludekkvapil.cz"]
VALID_USER_EMAILS = [
    "siva01@seznam.cz",
    "test@example.com",
    "user.name+tag@domain.co.uk",
    "USER@EXAMPLE.COM",
    "User@Example.Com",
]
INVALID_EMAILS = ["invalid@email", "Hello world", "My phone is 123-456-7890"]


class TestEmailDetection:
    """Test cases for email detection improvements."""

    @pytest.fixture
    def chat_service(self):
        """Fixture providing a configured chat service instance."""
        chat_history = Mock(spec=ChatHistory)
        return ChatService(chat_history)

    @pytest.mark.parametrize("email_text", VALID_USER_EMAILS)
    def test_contains_email_address_valid_email(self, chat_service, email_text):
        """Test that _contains_email_address detects valid user emails."""
        assert chat_service._contains_email_address(email_text)

    @pytest.mark.parametrize(
        "email_text", LUDEK_EMAILS + [f"Contact him at {email}" for email in LUDEK_EMAILS]
    )
    def test_contains_email_address_filters_ludek_emails(self, chat_service, email_text):
        """Test that _contains_email_address filters out Luděk's emails."""
        assert not chat_service._contains_email_address(email_text)

    @pytest.mark.parametrize("text", INVALID_EMAILS)
    def test_contains_email_address_no_email(self, chat_service, text):
        """Test that _contains_email_address returns False for text without valid emails."""
        assert not chat_service._contains_email_address(text)

    def test_contains_email_address_mixed_content(self, chat_service):
        """Test email detection with mixed content containing both user and Luděk emails."""
        mixed_text = "My email is user@test.com, contact expert at info@ludekkvapil.cz"
        assert chat_service._contains_email_address(mixed_text)

    @pytest.fixture
    def mock_language_detection(self):
        """Fixture providing mocked language detection functions."""
        with (
            patch("assistant.agent.language_detection_agent.is_confirmation_yes") as mock_yes,
            patch("assistant.agent.language_detection_agent.is_confirmation_no") as mock_no,
            patch("assistant.agent.language_detection_agent.detect_language") as mock_detect,
        ):

            mock_yes.return_value = False
            mock_no.return_value = False
            mock_detect.return_value = "en"
            yield mock_yes, mock_no, mock_detect

    @pytest.fixture
    def mock_chat_history(self):
        """Fixture providing a mocked chat history."""
        chat_history = Mock(spec=ChatHistory)
        chat_history.get_full_history.return_value = [Mock(user_message="My website got hacked")]
        return chat_history

    @pytest.mark.asyncio
    async def test_email_triggers_forwarding_flow(
        self, chat_service, mock_language_detection, mock_chat_history
    ):
        """Test that providing an email triggers the forwarding flow."""
        mock_yes, mock_no, mock_detect = mock_language_detection

        # Update chat history for this test
        mock_chat_history.get_full_history.return_value = [
            Mock(user_message="My website got hacked"),
            Mock(user_message="siva01@seznam.cz"),
        ]
        chat_service.chat_history = mock_chat_history

        with patch.object(
            chat_service, "_generate_chat_summary", new_callable=AsyncMock
        ) as mock_summary:
            mock_summary.return_value = "Test summary"

            with patch(
                "assistant.agent.email_agent.SimpleMessageForwarder"
            ) as mock_forwarder_class:
                mock_forwarder = Mock()
                mock_forwarder.check_contact_info.return_value = None
                mock_forwarder.process_data.return_value = "Email sent successfully"
                mock_forwarder._extract_contact_info.return_value = "Email: siva01@seznam.cz"
                mock_forwarder_class.return_value = mock_forwarder

                # Test that email input triggers forwarding
                result = await chat_service.process_message("siva01@seznam.cz")

                # Verify forwarding was triggered
                assert result["agent"] is True
                assert "Email sent successfully" in result["message"]
                mock_forwarder_class.assert_called_once()
                mock_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_email_in_message_extracts_contact(
        self, chat_service, mock_language_detection, mock_chat_history
    ):
        """Test that email in message is properly extracted as contact."""
        chat_service.chat_history = mock_chat_history

        with patch.object(
            chat_service, "_generate_chat_summary", new_callable=AsyncMock
        ) as mock_summary:
            mock_summary.return_value = "Test summary"

            with patch(
                "assistant.agent.email_agent.SimpleMessageForwarder"
            ) as mock_forwarder_class:
                mock_forwarder = Mock()
                mock_forwarder.check_contact_info.return_value = None
                mock_forwarder.process_data.return_value = "Email sent successfully"
                mock_forwarder._extract_contact_info.return_value = "Email: siva01@seznam.cz"
                mock_forwarder_class.return_value = mock_forwarder

                # Test email extraction from current message
                result = await chat_service.process_message("Please contact me at siva01@seznam.cz")

                # Verify the result
                assert result["agent"] is True
                assert "Email sent successfully" in result["message"]
                # Verify that the forwarder was created with the correct message
                mock_forwarder_class.assert_called_once()
                call_args = mock_forwarder_class.call_args
                assert call_args[1]["user_message"] == "Please contact me at siva01@seznam.cz"

    @pytest.mark.parametrize("ludek_email", ["INFO@LUDEKKVAPIL.CZ", "Info@LudekKvapil.cz"])
    def test_email_detection_case_insensitive_filters_ludek(self, chat_service, ludek_email):
        """Test that email detection is case insensitive for filtering Luděk's emails."""
        assert not chat_service._contains_email_address(ludek_email)

    @pytest.mark.parametrize("user_email", ["USER@EXAMPLE.COM", "User@Example.Com"])
    def test_email_detection_case_insensitive_allows_users(self, chat_service, user_email):
        """Test that email detection allows user emails regardless of case."""
        assert chat_service._contains_email_address(user_email)
