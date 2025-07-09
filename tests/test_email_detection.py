"""Tests for improved email detection functionality."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.services.chat import ChatService
from src.services.chat_history import ChatHistory


class TestEmailDetection:
    """Test cases for email detection improvements."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chat_history = Mock(spec=ChatHistory)
        self.chat_service = ChatService(self.chat_history)

    def test_contains_email_address_valid_email(self):
        """Test that _contains_email_address detects valid emails."""
        # Valid user emails
        assert self.chat_service._contains_email_address("siva01@seznam.cz")
        assert self.chat_service._contains_email_address("My email is test@example.com")
        assert self.chat_service._contains_email_address("user.name+tag@domain.co.uk")
        
    def test_contains_email_address_filters_ludek_emails(self):
        """Test that _contains_email_address filters out Luděk's emails."""
        # Luděk's emails should be filtered out
        assert not self.chat_service._contains_email_address("info@ludekkvapil.cz")
        assert not self.chat_service._contains_email_address("ludek@ludekkvapil.cz")
        assert not self.chat_service._contains_email_address("Contact him at info@ludekkvapil.cz")
        
    def test_contains_email_address_no_email(self):
        """Test that _contains_email_address returns False for text without emails."""
        assert not self.chat_service._contains_email_address("Hello world")
        assert not self.chat_service._contains_email_address("My phone is 123-456-7890")
        assert not self.chat_service._contains_email_address("invalid@email")
        
    def test_contains_email_address_mixed_content(self):
        """Test email detection with mixed content."""
        # User email mixed with Luděk's email
        assert self.chat_service._contains_email_address("My email is user@test.com, contact expert at info@ludekkvapil.cz")
        
    @patch('services.chat.is_confirmation_yes')
    @patch('services.chat.is_confirmation_no')
    @patch('services.chat.detect_language')
    async def test_email_triggers_forwarding_flow(self, mock_detect_lang, mock_is_no, mock_is_yes):
        """Test that providing an email triggers the forwarding flow."""
        # Setup mocks
        mock_detect_lang.return_value = 'en'
        mock_is_yes.return_value = False
        mock_is_no.return_value = False
        
        # Mock chat history
        self.chat_history.get_full_history.return_value = [
            Mock(user_message="My website got hacked"),
            Mock(user_message="siva01@seznam.cz")
        ]
        
        # Mock _generate_chat_summary
        with patch.object(self.chat_service, '_generate_chat_summary', new_callable=AsyncMock) as mock_summary:
            mock_summary.return_value = "Test summary"
            
            # Mock SimpleMessageForwarder
            with patch('services.chat.SimpleMessageForwarder') as mock_forwarder_class:
                mock_forwarder = Mock()
                mock_forwarder.check_contact_info.return_value = None
                mock_forwarder.process_data.return_value = "Email sent successfully"
                mock_forwarder._extract_contact_info.return_value = "Email: siva01@seznam.cz"
                mock_forwarder_class.return_value = mock_forwarder
                
                # Test that email input triggers forwarding
                result = await self.chat_service.process_message("siva01@seznam.cz")
                
                # Verify forwarding was triggered
                assert result["agent"] == True
                assert "Email sent successfully" in result["message"]
                mock_forwarder_class.assert_called_once()
                mock_summary.assert_called_once()

    @patch('services.chat.is_confirmation_yes')
    @patch('services.chat.is_confirmation_no') 
    @patch('services.chat.detect_language')
    async def test_email_in_message_extracts_contact(self, mock_detect_lang, mock_is_no, mock_is_yes):
        """Test that email in message is properly extracted as contact."""
        # Setup mocks
        mock_detect_lang.return_value = 'en'
        mock_is_yes.return_value = False
        mock_is_no.return_value = False
        
        # Mock chat history
        self.chat_history.get_full_history.return_value = [
            Mock(user_message="My website got hacked")
        ]
        
        with patch.object(self.chat_service, '_generate_chat_summary', new_callable=AsyncMock) as mock_summary:
            mock_summary.return_value = "Test summary"
            
            with patch('services.chat.SimpleMessageForwarder') as mock_forwarder_class:
                mock_forwarder = Mock()
                mock_forwarder.check_contact_info.return_value = None
                mock_forwarder.process_data.return_value = "Email sent successfully"
                mock_forwarder._extract_contact_info.return_value = "Email: siva01@seznam.cz"
                mock_forwarder_class.return_value = mock_forwarder
                
                # Test email extraction from current message
                await self.chat_service.process_message("Please contact me at siva01@seznam.cz")
                
                # Verify contact was set from current message
                assert mock_forwarder.user_contact == "Email: siva01@seznam.cz"
                
    def test_email_detection_case_insensitive(self):
        """Test that email detection is case insensitive for filtering."""
        # Test case variations of Luděk's email
        assert not self.chat_service._contains_email_address("INFO@LUDEKKVAPIL.CZ")
        assert not self.chat_service._contains_email_address("Info@LudekKvapil.cz")
        
        # But user emails should still work
        assert self.chat_service._contains_email_address("USER@EXAMPLE.COM")
        assert self.chat_service._contains_email_address("User@Example.Com")