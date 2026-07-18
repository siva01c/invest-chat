"""Email service for handling message forwarding and confirmations."""

import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from assistant.core.interfaces.chat import IConversationService, IEmailService
from assistant.utils.log_sanitizer import sanitize_for_logging


def load_translations(language_code: str) -> dict:
    """Load translations from YAML file for specified language."""
    try:
        translations_path = (
            Path(__file__).parent.parent.parent / "data" / "translations" / f"{language_code}.yml"
        )
        with open(translations_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Translation file for {language_code} not found.")
        return {}
    except yaml.YAMLError:
        print(f"Error: Invalid YAML format in translation file for {language_code}.")
        return {}


class EmailService(IEmailService):
    """Service for handling email forwarding and confirmation flows."""

    def __init__(self):
        """Initialize the email service."""
        self.config = self._load_config()

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "EmailService"

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            config_path = Path(__file__).parent.parent.parent / "data" / "config.yml"
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Config file not found.")
            return {}
        except yaml.YAMLError:
            print(f"Error: Invalid YAML format in config file.")
            return {}

    def check_email_confirmation_context(
        self, last_response: str, user_text: str
    ) -> Optional[Dict[str, any]]:
        """
        Check if we're in an email confirmation context and handle accordingly.

        Args:
            last_response: The last assistant response
            user_text: Current user input

        Returns:
            Response dict if in confirmation context, None otherwise
        """
        # Check if we're in email confirmation state
        confirmation_indicators = [
            "napište 'ano'",
            "type 'yes'",
            "shrnutí luďkovi",
            "summary to luděk",
        ]
        in_confirmation_state = any(
            indicator in last_response.lower() for indicator in confirmation_indicators
        )

        if not in_confirmation_state:
            return None

        # Handle confirmation/cancellation in email context
        from assistant.agent.language_detection_agent import is_confirmation_no, is_confirmation_yes

        if not (is_confirmation_yes(user_text) or is_confirmation_no(user_text)):
            return None

        return {"requires_email_processing": True}

    async def process_email_confirmation(
        self, user_text: str, session_id: str, conversation_service: IConversationService
    ) -> Dict[str, any]:
        """
        Process email confirmation and send message.

        Args:
            user_text: User's confirmation text
            session_id: Session identifier
            conversation_service: Reference to conversation service

        Returns:
            Response dict with confirmation result
        """
        try:
            from assistant.agent.email_agent import SimpleMessageForwarder

            chat_history = conversation_service.chat_history.get_full_history()
            conversation_summary = await conversation_service.generate_conversation_summary(
                user_text
            )

            # Find contact info from previous messages in chat history
            user_contact = self._extract_contact_info(chat_history)

            # Create a mock forwarder with the contact info to send the email
            forwarder = SimpleMessageForwarder(
                user_message=user_text,
                chat_history=chat_history,
                user_context={"session_id": session_id or "unknown", "source": "chat"},
                chat_summary=conversation_summary,
            )

            # Set the contact info we found
            if user_contact:
                forwarder.user_contact = user_contact

            # Call process_data which will handle confirmation and send email
            result = forwarder.process_data()

            conversation_service.add_interaction(user_text, result)
            return {"agent": True, "message": result, "prompt_category": "email_confirmation"}

        except Exception as e:
            print(f"Email confirmation error: {sanitize_for_logging(str(e))}")
            error_msg = self.config.get("response_messages", {}).get("email_error", "Email error.")
            conversation_service.add_interaction(user_text, error_msg)
            return {"agent": True, "message": error_msg, "prompt_category": "email_error"}

    def _extract_contact_info(self, chat_history: List) -> Optional[str]:
        """Extract contact information from chat history."""
        for interaction in reversed(chat_history):
            if hasattr(interaction, "user_message"):
                email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                emails = re.findall(email_pattern, interaction.user_message)
                if emails:
                    # Filter out Luděk's emails
                    ludek_emails = [
                        "info@ludekkvapil.cz",
                        "ludek@ludekkvapil.cz",
                        "kvapilludek@gmail.com",
                    ]
                    user_emails = [
                        email
                        for email in emails
                        if email.lower() not in [e.lower() for e in ludek_emails]
                    ]
                    if user_emails:
                        return f"Email: {user_emails[0]}"
        return None

    async def handle_forward_message(
        self, user_text: str, session_id: str, conversation_service: IConversationService
    ) -> Dict[str, any]:
        """
        Handle simple message forwarding.

        Args:
            user_text: User message to forward
            session_id: Session identifier
            conversation_service: Reference to conversation service

        Returns:
            Response dict with forwarding result
        """
        try:
            from assistant.agent.email_agent import send_simple_message

            chat_history = conversation_service.chat_history.get_full_history()
            # Generate accurate conversation summary
            conversation_summary = await conversation_service.generate_conversation_summary(
                user_text
            )
            result = send_simple_message(
                user_message=user_text,
                chat_history=chat_history,
                user_context={"session_id": session_id or "unknown", "source": "chat"},
                chat_summary=conversation_summary,
            )
            conversation_service.add_interaction(user_text, result)
            return {"agent": True, "message": result, "contexts": []}

        except Exception as e:
            print(f"Forward message error: {sanitize_for_logging(str(e))}")
            error_msg = self.config.get("response_messages", {}).get("email_error", "Email error.")
            conversation_service.add_interaction(user_text, error_msg)
            return {"agent": True, "message": error_msg, "contexts": []}

    def handle_cybersecurity_urgent(
        self, user_text: str, session_id: str, conversation_service: IConversationService
    ) -> Dict[str, any]:
        """
        Handle urgent cybersecurity requests.

        Args:
            user_text: User message about cybersecurity
            session_id: Session identifier
            conversation_service: Reference to conversation service

        Returns:
            Response dict with handling result
        """
        # Check if user provided contact info in their response
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, user_text)

        if emails:
            # User provided contact info along with technical details, forward immediately
            try:
                from assistant.agent.email_agent import send_simple_message

                chat_history = conversation_service.chat_history.get_full_history()
                conversation_summary = conversation_service.generate_conversation_summary(user_text)
                result = send_simple_message(
                    user_message=user_text,
                    chat_history=chat_history,
                    user_context={"session_id": session_id or "unknown", "source": "chat"},
                    chat_summary=conversation_summary,
                )
                conversation_service.add_interaction(user_text, result)
                return {"agent": True, "message": result, "contexts": []}

            except Exception as e:
                print(f"Cybersecurity urgent forward error: {sanitize_for_logging(str(e))}")
                error_msg = self.config.get("response_messages", {}).get(
                    "email_error", "Email error."
                )
                conversation_service.add_interaction(user_text, error_msg)
                return {"agent": True, "message": error_msg, "contexts": []}
        else:
            # Technical details provided but no contact info, ask for it
            from assistant.agent.language_detection_agent import detect_language

            language = detect_language(user_text)
            if language == "cs":
                msg = "Děkuji za technické detaily! **Prosím uveďte ještě vaše kontaktní údaje (email nebo telefon)**, abych mohl předat všechny informace Luďkovi pro okamžitou pomoc."
            else:
                msg = "Thank you for the technical details! **Please also provide your contact information (email or phone)** so I can forward all information to Luděk for immediate assistance."

            conversation_service.add_interaction(user_text, msg)
            return {"agent": True, "message": msg, "contexts": []}

    def handle_service_details(
        self, user_text: str, conversation_service: IConversationService
    ) -> Dict[str, any]:
        """
        Handle service detail requests and contact information.

        Args:
            user_text: User message with service details
            conversation_service: Reference to conversation service

        Returns:
            Response dict with handling result
        """
        from assistant.agent.language_detection_agent import detect_language

        language = detect_language(user_text)

        # Check if user already provided contact info in their response
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        phone_pattern = r"[\+]?[1-9]?[0-9]{7,15}"
        emails = re.findall(email_pattern, user_text)
        phones = re.findall(phone_pattern, user_text)

        if emails or phones:
            # User provided contact info along with project details, forward immediately
            try:
                from assistant.agent.email_agent import send_simple_message

                chat_history = conversation_service.chat_history.get_full_history()
                conversation_summary = conversation_service.generate_conversation_summary(user_text)
                result = send_simple_message(
                    user_message=user_text,
                    chat_history=chat_history,
                    user_context={"session_id": "unknown", "source": "chat"},
                    chat_summary=conversation_summary,
                )
                conversation_service.add_interaction(user_text, result)
                return {"agent": True, "message": result, "contexts": []}

            except Exception as e:
                print(f"Service forward error: {sanitize_for_logging(str(e))}")
                error_msg = self.config.get("response_messages", {}).get(
                    "email_error", "Email error."
                )
                conversation_service.add_interaction(user_text, error_msg)
                return {"agent": True, "message": error_msg, "contexts": []}
        else:
            # Project details provided but no contact info, ask for it
            if language == "cs":
                msg = "Děkuji za informace o vašem projektu! **Prosím uveďte ještě vaše kontaktní údaje (email nebo telefon)**, abych mohl předat všechny informace Luďkovi, který vás bude kontaktovat ohledně realizace."
            else:
                msg = "Thank you for the project information! **Please also provide your contact information (email or phone)** so I can forward all details to Luděk, who will contact you about implementation."

            conversation_service.add_interaction(user_text, msg)
            return {"agent": True, "message": msg, "contexts": []}
