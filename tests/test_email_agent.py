"""
Unit tests for the SimpleMessageForwarder class and email functionality.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, Mock, patch

import pytest

from assistant.agent.email_agent import SimpleMessageForwarder, send_email, send_simple_message


class TestSendEmail:
    """Test the send_email function."""

    def test_send_email_success_ssl(self, mock_env_vars, mock_smtp_server):
        """Test successful email sending with SSL."""
        mock_smtp_server.login.return_value = None
        mock_smtp_server.send_message.return_value = None

        with patch.dict("os.environ", {"SMTP_PORT": "465"}):
            result = send_email("Test Subject", "Test Message")

        assert result == "Email sent successfully!"
        mock_smtp_server.login.assert_called_once()
        mock_smtp_server.send_message.assert_called_once()
        mock_smtp_server.quit.assert_called_once()

    def test_send_email_success_starttls(self, mock_env_vars, mock_smtp_server):
        """Test successful email sending with STARTTLS."""
        mock_smtp_server.login.return_value = None
        mock_smtp_server.send_message.return_value = None
        mock_smtp_server.starttls.return_value = None

        result = send_email("Test Subject", "Test Message")

        assert result == "Email sent successfully!"
        mock_smtp_server.starttls.assert_called_once()
        mock_smtp_server.login.assert_called_once()
        mock_smtp_server.send_message.assert_called_once()

    def test_send_email_missing_env_vars(self):
        """Test email sending with missing environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            result = send_email("Test Subject", "Test Message")

        assert "Missing required email configuration" in result

    def test_send_email_invalid_port(self, mock_env_vars):
        """Test email sending with invalid port."""
        with patch.dict("os.environ", {"SMTP_PORT": "invalid"}):
            result = send_email("Test Subject", "Test Message")

        assert "Invalid SMTP_PORT value" in result

    def test_send_email_auth_error(self, mock_env_vars):
        """Test email sending with authentication error."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value = mock_server
            mock_server.login.side_effect = smtplib.SMTPAuthenticationError(
                535, "Authentication failed"
            )

            result = send_email("Test Subject", "Test Message")

        assert "SMTP Authentication failed" in result

    def test_send_email_connection_error(self, mock_env_vars):
        """Test email sending with connection error."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.side_effect = smtplib.SMTPConnectError(421, "Service not available")

            result = send_email("Test Subject", "Test Message")

        assert "SMTP error occurred" in result


class TestSimpleMessageForwarder:
    """Test the SimpleMessageForwarder class."""

    def test_init_basic(self):
        """Test basic initialization."""
        forwarder = SimpleMessageForwarder("Hello world")

        assert forwarder.user_message == "Hello world"
        assert forwarder.chat_history == []
        assert forwarder.user_context == {}
        assert forwarder.detected_language == "en"

    def test_init_with_context(self, sample_chat_history, sample_user_context):
        """Test initialization with chat history and context."""
        forwarder = SimpleMessageForwarder(
            "Test message", chat_history=sample_chat_history, user_context=sample_user_context
        )

        assert forwarder.user_message == "Test message"
        assert forwarder.chat_history == sample_chat_history
        assert forwarder.user_context == sample_user_context

    def test_detect_language_english(self):
        """Test language detection for English."""
        forwarder = SimpleMessageForwarder("Hello, how are you?")
        assert forwarder.detected_language == "en"

    def test_detect_language_czech(self):
        """Test language detection for Czech."""
        forwarder = SimpleMessageForwarder("Ahoj, jak se máš?")
        assert forwarder.detected_language == "cs"

    def test_detect_language_czech_characters(self):
        """Test language detection with Czech characters."""
        forwarder = SimpleMessageForwarder("Potřebuji pomoc s projektem")
        assert forwarder.detected_language == "cs"

    def test_extract_contact_info_email(self):
        """Test contact info extraction for email."""
        forwarder = SimpleMessageForwarder("Contact me at john@example.com")
        contact = forwarder._extract_contact_info(forwarder.user_message)

        assert contact is not None
        assert "Email: john@example.com" in contact

    def test_extract_contact_info_phone(self):
        """Test contact info extraction for phone."""
        forwarder = SimpleMessageForwarder("Call me at 555-123-4567")
        contact = forwarder._extract_contact_info(forwarder.user_message)

        assert contact is not None
        assert "Phone: 555-123-4567" in contact

    def test_extract_contact_info_linkedin(self):
        """Test contact info extraction for LinkedIn."""
        forwarder = SimpleMessageForwarder("My LinkedIn is https://linkedin.com/in/johndoe")
        contact = forwarder._extract_contact_info(forwarder.user_message)

        assert contact is not None
        assert "LinkedIn: https://linkedin.com/in/johndoe" in contact

    def test_extract_contact_info_none(self):
        """Test contact info extraction with no contact info."""
        forwarder = SimpleMessageForwarder("This is just a regular message")
        contact = forwarder._extract_contact_info(forwarder.user_message)

        assert contact is None

    def test_needs_contact_info_strong_indicators_en(self):
        """Test contact info need detection with strong English indicators."""
        from assistant.agent.language_detection_agent import needs_contact_info

        message = "Please contact me about this project"
        assert needs_contact_info(message) == True

    def test_needs_contact_info_strong_indicators_cs(self):
        """Test contact info need detection with strong Czech indicators."""
        from assistant.agent.language_detection_agent import needs_contact_info

        message = "Kontaktujte mě ohledně projektu"
        assert needs_contact_info(message) == True

    def test_needs_contact_info_business_indicators(self):
        """Test contact info need detection with business indicators."""
        from assistant.agent.language_detection_agent import needs_contact_info

        message = "I'm interested in your services"
        assert needs_contact_info(message) == True

    def test_needs_contact_info_questions(self):
        """Test contact info need detection with questions."""
        from assistant.agent.language_detection_agent import needs_contact_info

        message = "What services do you offer?"
        assert needs_contact_info(message) == False

    def test_is_incomplete_message_request_en(self):
        """Test incomplete message request detection in English."""
        from assistant.agent.language_detection_agent import is_email_request

        message = "send him a message"
        assert is_email_request(message) == True

    def test_is_incomplete_message_request_cs(self):
        """Test incomplete message request detection in Czech."""
        from assistant.agent.language_detection_agent import is_email_request

        message = "pošli mu zprávu"
        assert is_email_request(message) == True

    def test_is_incomplete_message_request_complete(self):
        """Test incomplete message request detection with complete message."""
        from assistant.agent.language_detection_agent import is_email_request

        message = "Hi Ludek, I need help with my project. Contact me at john@example.com"
        assert is_email_request(message) == False

    def test_check_contact_info_incomplete_request(self):
        """Test check_contact_info with incomplete message request."""
        forwarder = SimpleMessageForwarder("send him message")
        result = forwarder.check_contact_info()

        assert result is not None
        assert "provide" in result.lower()

    def test_check_contact_info_needs_contact_missing(self):
        """Test check_contact_info when contact info is needed but missing."""
        forwarder = SimpleMessageForwarder("Please contact me about this project")
        result = forwarder.check_contact_info()

        assert result is not None
        assert "email" in result.lower() or "phone" in result.lower()

    def test_check_contact_info_complete(self):
        """Test check_contact_info with complete message."""
        forwarder = SimpleMessageForwarder(
            "Hi, I need help with my project. Contact me at john@example.com"
        )
        result = forwarder.check_contact_info()

        # Should return confirmation message when contact info is found
        assert result is not None
        assert "I have your email" in result
        assert "john@example.com" in result

    def test_generate_subject_project(self):
        """Test subject generation for project inquiries."""
        forwarder = SimpleMessageForwarder("I have a project for you")
        subject = forwarder._generate_subject()

        assert "Project Inquiry" in subject
        assert "💼" in subject

    def test_generate_subject_drupal(self):
        """Test subject generation for Drupal inquiries."""
        forwarder = SimpleMessageForwarder("I need help with Drupal development")
        subject = forwarder._generate_subject()

        assert "Drupal Development" in subject
        assert "🔧" in subject

    def test_generate_subject_security(self):
        """Test subject generation for security inquiries."""
        forwarder = SimpleMessageForwarder("I need cybersecurity consulting")
        subject = forwarder._generate_subject()

        assert "Cybersecurity" in subject
        assert "🔒" in subject

    def test_generate_subject_ai(self):
        """Test subject generation for AI inquiries."""
        forwarder = SimpleMessageForwarder("I need help with AI and LLM")
        subject = forwarder._generate_subject()

        assert "AI/LLM" in subject
        assert "🤖" in subject

    def test_generate_subject_default(self):
        """Test subject generation for generic messages."""
        forwarder = SimpleMessageForwarder("Hello there")
        subject = forwarder._generate_subject()

        assert "Chat Summary from User" in subject
        assert "💬" in subject

    def test_generate_email_body_basic(self):
        """Test email body generation with basic message."""
        forwarder = SimpleMessageForwarder("Test message")
        body = forwarder._generate_email_body()

        assert "CONVERSATION SUMMARY:" in body
        assert "CONTEXT:" in body
        assert "Sales Assistant Chat" in body
        assert "First message." in body

    def test_generate_email_body_with_contact(self):
        """Test email body generation with contact info."""
        forwarder = SimpleMessageForwarder("Contact me at john@example.com")
        forwarder.user_contact = "Email: john@example.com"
        body = forwarder._generate_email_body()

        assert "User Contact: Email: john@example.com" in body

    def test_generate_email_body_with_context(self, sample_user_context):
        """Test email body generation with user context."""
        forwarder = SimpleMessageForwarder("Test message", user_context=sample_user_context)
        body = forwarder._generate_email_body()

        assert "Session Info:" in body
        assert "test_session_123" in body

    def test_generate_email_body_with_history(self, sample_chat_history):
        """Test email body generation with chat history."""
        forwarder = SimpleMessageForwarder("Test message", chat_history=sample_chat_history)
        body = forwarder._generate_email_body()

        assert "Summary of 2 interactions." in body
        assert "CONVERSATION SUMMARY:" in body

    @patch("assistant.agent.email_agent.send_email")
    def test_process_data_success(self, mock_send_email, mock_env_vars):
        """Test successful message processing."""
        mock_send_email.return_value = "Email sent successfully!"

        # Test the confirmation flow - first provide contact, then confirm
        forwarder = SimpleMessageForwarder("yes", chat_history=[], user_context={})
        # Simulate that contact was provided in previous interaction
        forwarder.user_contact = "Email: john@example.com"
        result = forwarder.process_data()

        assert "sent to luděk" in result.lower() or "odesláno" in result.lower()
        mock_send_email.assert_called_once()

    @patch("assistant.agent.email_agent.send_email")
    def test_process_data_email_failure(self, mock_send_email, mock_env_vars):
        """Test message processing with email failure."""
        mock_send_email.return_value = "SMTP Authentication failed"

        # Test email failure after confirmation
        forwarder = SimpleMessageForwarder("yes", chat_history=[], user_context={})
        # Simulate that contact was provided in previous interaction
        forwarder.user_contact = "Email: john@example.com"
        result = forwarder.process_data()

        assert "issue" in result.lower() or "chyba" in result.lower()
        mock_send_email.assert_called_once()

    @patch("assistant.agent.email_agent.send_email")
    def test_send_simple_message_with_context(
        self, mock_send_email, mock_env_vars, sample_chat_history, sample_user_context
    ):
        """Test simple message sending with context."""
        mock_send_email.return_value = "Email sent successfully!"

        # Test confirmation flow - message with contact info should ask for confirmation
        result = send_simple_message(
            "Hi, I need help with my project. Contact me at john@example.com",
            chat_history=sample_chat_history,
            user_context=sample_user_context,
        )

        # Should return confirmation message, not send email yet
        assert "I have your email" in result
        assert "john@example.com" in result
        mock_send_email.assert_not_called()

    def test_send_simple_message_incomplete(self):
        """Test simple message sending with incomplete request."""
        result = send_simple_message("send him a message")

        assert "provide" in result.lower()


class TestLocalizationMessages:
    """Test localized message responses."""

    def test_localized_message_english(self):
        """Test English localized messages."""
        from assistant.agent.language_detection_agent import get_localized_message

        contact_msg = get_localized_message("contact_needed", "en")
        assert "email address or phone number" in contact_msg

        forwarded_msg = get_localized_message("summary_sent", "en")
        assert "sent to Luděk" in forwarded_msg

    def test_localized_message_czech(self):
        """Test Czech localized messages."""
        from assistant.agent.language_detection_agent import get_localized_message

        contact_msg = get_localized_message("contact_needed", "cs")
        assert "email nebo telefon" in contact_msg

        forwarded_msg = get_localized_message("summary_sent", "cs")
        assert "odesláno Luďkovi" in forwarded_msg
