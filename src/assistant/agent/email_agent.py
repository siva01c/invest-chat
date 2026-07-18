import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv

from assistant.agent.agents import Agent
from assistant.agent.language_detection_agent import (
    detect_language,
    get_localized_message,
    is_confirmation_no,
    is_confirmation_yes,
    needs_contact_info,
)
from assistant.utils.log_sanitizer import SanitizedFormatter, safe_error, safe_info

# Load environment variables
load_dotenv()

# Configure logging
# Configure secure logging with sanitization
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(SanitizedFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def send_email(subject: str, message: str) -> str:
    """
    Send email using Gigaserver SMTP configuration.

    Args:
        subject: Email subject line
        message: Email body content

    Returns:
        Success or error message
    """
    # Email configuration
    sender_email = os.getenv("EMAIL")
    receiver_email = os.getenv("EMAIL_RECEIVER")
    password = os.getenv("EMAIL_PWD")
    # SMTP server configuration
    smtp_server = os.getenv("SMTP_SERVER", "mail.gigaserver.cz")
    smtp_port_str = os.getenv("SMTP_PORT", "587")

    # Validate required environment variables
    if not all([sender_email, receiver_email, password]):
        error_msg = "Missing required email configuration. Check EMAIL, EMAIL_RECEIVER, and EMAIL_PWD environment variables."
        safe_error(logger, error_msg)
        return error_msg

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        error_msg = f"Invalid SMTP_PORT value: {smtp_port_str}. Must be a number."
        safe_error(logger, error_msg)
        return error_msg

    # Create message
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # Add message body
    msg.attach(MIMEText(message, "plain"))

    server = None
    try:
        safe_info(logger, "Attempting SMTP connection to [REDACTED]:[REDACTED]")

        # Create SMTP session with debug logging
        if smtp_port == 465:
            safe_info(logger, "Using SMTP_SSL for secure connection")
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            safe_info(logger, "Using SMTP with STARTTLS for secure connection")
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()

        # Debug mode disabled for production security
        # server.set_debuglevel(1)  # Only enable for local debugging

        safe_info(logger, "Attempting login with email: [REDACTED]")
        # Login to the server
        server.login(sender_email, password)

        safe_info(logger, "Login successful, sending email...")
        # Send email
        server.send_message(msg)
        safe_info(logger, "Email sent successfully to [REDACTED]")
        return "Email sent successfully!"

    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP Authentication failed: {str(e)}"
        safe_error(logger, error_msg)
        return error_msg
    except smtplib.SMTPException as e:
        error_msg = f"SMTP error occurred: {str(e)}"
        safe_error(logger, error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        safe_error(logger, error_msg)
        return error_msg

    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass


class SimpleMessageForwarder(Agent):
    """
    Simple agent for forwarding user messages directly to Luděk via email.
    No complex email composition - just forward the message with context.
    """

    def __init__(
        self,
        user_message: str,
        chat_history: Optional[List] = None,
        user_context: Optional[dict] = None,
        chat_summary: Optional[str] = None,
    ):
        """
        Initialize the simple message forwarder.

        Args:
            user_message: The user's message to forward
            chat_history: Optional chat history for context
            user_context: Optional additional context (IP, session info, etc.)
            chat_summary: Optional pre-generated chat summary
        """
        self.user_message = user_message
        self.chat_history = chat_history or []
        self.user_context = user_context or {}
        self.user_contact = None
        self.chat_summary = chat_summary
        self.config = self._load_config()

        # Improved language detection - check user message and chat history
        self.detected_language = self._detect_language_with_context()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            config_path = Path(__file__).parent.parent / "data" / "config.yml"
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def _detect_language_with_context(self) -> str:
        """
        Detect language using current message and chat history context.

        Returns:
            'cs' for Czech, 'en' for English
        """
        # First try to detect from current message
        current_lang = detect_language(self.user_message)

        # If current message detection is confident (contains clear language indicators), use it
        current_lower = self.user_message.lower()

        # Strong English indicators
        english_indicators = [
            "my email is",
            "email me at",
            "contact me at",
            "my phone",
            "call me",
            "reach me",
        ]
        if any(indicator in current_lower for indicator in english_indicators):
            return "en"

        # Strong Czech indicators
        czech_indicators = [
            "můj email je",
            "email mám",
            "kontaktujte mě",
            "volejte mi",
            "napište mi",
        ]
        if any(indicator in current_lower for indicator in czech_indicators):
            return "cs"

        # If current message is unclear, check chat history
        if self.chat_history and len(self.chat_history) > 0:
            # Get recent messages from chat history
            recent_messages = []
            for interaction in self.chat_history[-3:]:  # Last 3 interactions
                if hasattr(interaction, "user_message"):
                    recent_messages.append(interaction.user_message)

            # Detect language from recent chat context
            if recent_messages:
                combined_text = " ".join(recent_messages)
                history_lang = detect_language(combined_text)
                return history_lang

        # Fallback to current message detection
        return current_lang

    def _extract_contact_info(self, message: str) -> Optional[str]:
        """Extract email, phone, or other contact info from message."""
        import re

        # Email pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, message)

        # Filter out Luděk's email addresses (don't use these as user contact)
        ludek_emails = ["info@ludekkvapil.cz", "ludek@ludekkvapil.cz"]
        emails = [
            email for email in emails if email.lower() not in [e.lower() for e in ludek_emails]
        ]

        # Phone pattern (various formats)
        phone_pattern = r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b|(?:\+?[0-9]{1,3}[-.\s]?)?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b"
        phones = re.findall(phone_pattern, message)

        # LinkedIn profile pattern
        linkedin_pattern = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+"
        linkedin = re.findall(linkedin_pattern, message)

        contact_info = []
        if emails:
            contact_info.extend([f"Email: {email}" for email in emails])
        if phones:
            contact_info.extend([f"Phone: {phone.strip()}" for phone in phones])
        if linkedin:
            contact_info.extend([f"LinkedIn: {url}" for url in linkedin])

        return "; ".join(contact_info) if contact_info else None

    def check_contact_info(self) -> Optional[str]:
        """
        Check if contact info is available and handle confirmation flow.

        Returns:
            None if ready to send (contact + confirmation)
            String message for next step (ask email, ask confirmation, or cancel)
        """
        # Handle confirmation responses first
        if is_confirmation_yes(self.user_message):
            # User confirmed, but we need to check if we have contact from previous interaction
            # This should be handled by chat service checking chat history for contact
            return None  # Proceed to send

        if is_confirmation_no(self.user_message):
            return get_localized_message("send_cancelled", self.detected_language)

        # Extract any contact info from the message
        self.user_contact = self._extract_contact_info(self.user_message)

        # If we have contact info, ask for confirmation with summary
        if self.user_contact:
            # Use provided summary or create a simple one
            summary = self.chat_summary or "Conversation about Luděk Kvapil's services"
            return get_localized_message(
                "show_summary_for_confirmation", self.detected_language
            ).format(contact=self.user_contact, summary=summary)

        # No contact info, ask for email
        return get_localized_message("need_email_only", self.detected_language)

    def process_data(self) -> str:
        """
        Process and send the user message directly to Luděk.

        Returns:
            Success or error message
        """
        try:
            # Check if we need contact info first
            contact_request = self.check_contact_info()
            if contact_request:
                return contact_request

            # Generate email subject
            subject = self._generate_subject()

            # Generate email body with context
            email_body = self._generate_email_body()

            # Send the email
            result = send_email(subject, email_body)

            if "successfully" in result.lower():
                safe_info(logger, "Chat summary forwarded successfully")
                return get_localized_message("summary_sent", self.detected_language)
            else:
                safe_error(logger, f"Failed to forward message: {result}")
                return get_localized_message("forwarding_error", self.detected_language)

        except Exception as e:
            error_msg = f"Error in message forwarding: {str(e)}"
            safe_error(logger, error_msg)
            return get_localized_message("forwarding_error", self.detected_language)

    def _generate_subject(self) -> str:
        """Generate an appropriate email subject based on the chat content."""
        # Look through summary and chat to determine the main topic
        all_content = self.user_message.lower()
        if self.chat_summary:
            all_content += " " + self.chat_summary.lower()
        elif self.chat_history:
            for interaction in self.chat_history:
                if hasattr(interaction, "user_message"):
                    all_content += " " + interaction.user_message.lower()

        # Check for specific keywords to customize subject
        if any(word in all_content for word in ["project", "hire", "work", "collaboration"]):
            return "💼 Chat Summary - Project Inquiry"
        elif any(word in all_content for word in ["drupal", "development", "website"]):
            return "🔧 Chat Summary - Drupal Development"
        elif any(word in all_content for word in ["security", "cybersecurity", "penetration"]):
            return "🔒 Chat Summary - Cybersecurity Inquiry"
        elif any(word in all_content for word in ["ai", "llm", "chatbot", "rag"]):
            return "🤖 Chat Summary - AI/LLM Inquiry"
        elif any(word in all_content for word in ["question", "ask", "help"]):
            return "❓ Chat Summary - Customer Questions"
        else:
            return "💬 Chat Summary from User"

    def _generate_email_body(self) -> str:
        """Generate the email body with user message and context."""
        try:
            # Load email template from file
            template_path = Path(__file__).parent.parent / "data" / "prompts" / "email_template.md"
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        except FileNotFoundError:
            # Use fallback template from config
            email_config = self.config.get("email_config", {})
            template = email_config.get(
                "fallback_template", "Hello Luděk,\n\n{chat_summary}\n\nBest regards,\nBot"
            )

        # Get text snippets from config
        snippets = self.config.get("email_config", {}).get("text_snippets", {})

        # Prepare template variables
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        language_name = "Czech" if self.detected_language == "cs" else "English"

        # Contact info section
        contact_info = ""
        if self.user_contact:
            prefix = snippets.get("contact_info_prefix", "• User Contact: ")
            contact_info = f"{prefix}{self.user_contact}\n"

        # User context section
        user_context_info = ""
        if self.user_context:
            prefix = snippets.get("session_info_prefix", "• Session Info: ")
            user_context_info = f"{prefix}{self.user_context}\n"

        # Conversation note
        if self.chat_history and len(self.chat_history) > 0:
            template_text = snippets.get(
                "conversation_note_multiple", "Summary of {count} interactions."
            )
            conversation_note = template_text.format(count=len(self.chat_history))
        else:
            conversation_note = snippets.get("conversation_note_single", "First message.")

        # Contact warning
        contact_warning = ""
        if needs_contact_info(self.user_message) and not self.user_contact:
            contact_warning = snippets.get("contact_warning", "")

        # Chat summary
        no_summary_text = snippets.get("no_summary_available", "No summary available")

        # Format template
        return template.format(
            chat_summary=self.chat_summary or no_summary_text,
            timestamp=timestamp,
            contact_info=contact_info,
            user_context_info=user_context_info,
            language_name=language_name,
            detected_language=self.detected_language,
            conversation_note=conversation_note,
            contact_warning=contact_warning,
        )


def send_simple_message(
    user_message: str,
    chat_history: Optional[List] = None,
    user_context: Optional[dict] = None,
    chat_summary: Optional[str] = None,
) -> str:
    """
    Convenience function to quickly send a user message to Luděk.

    Args:
        user_message: The user's message to forward
        chat_history: Optional chat history for context
        user_context: Optional additional context
        chat_summary: Optional chat summary to send

    Returns:
        Success or error message
    """
    forwarder = SimpleMessageForwarder(user_message, chat_history, user_context, chat_summary)
    return forwarder.process_data()


# Example usage and testing
if __name__ == "__main__":
    # Test the simple message forwarder
    test_message = (
        "Hi, I'm interested in your Drupal development services. Can you help with a project?"
    )

    # Mock chat history for testing
    class MockChatEntry:
        def __init__(self, user_msg, assistant_msg):
            self.user_message = user_msg
            self.assistant_response = assistant_msg

    test_history = [
        MockChatEntry(
            "What services do you offer?",
            "I offer Drupal development, cybersecurity consulting, and AI solutions.",
        ),
        MockChatEntry(
            "What are your rates?",
            "My rates vary depending on the project scope. Let's discuss your specific needs.",
        ),
    ]

    result = send_simple_message(test_message, test_history, {"session_id": "test123"})
    print(f"Result: {result}")
