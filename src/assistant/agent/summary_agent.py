import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from assistant.agent.language_detection_agent import detect_language


class SummaryAgent:
    """
    Agent responsible for generating conversation summaries when OpenAI fails.
    Extracts key project details, contact information, and creates meaningful summaries.
    Uses configuration from config.yml for all settings.
    """

    def __init__(self):
        # Load merged configuration from both language files
        self._load_merged_config()

    def _load_merged_config(self):
        """Load and merge configuration from both language translation files"""
        try:
            # Load Czech and English configurations
            cs_translations = self._load_translations("cs")
            en_translations = self._load_translations("en")

            cs_config = cs_translations.get("summary_agent", {})
            en_config = en_translations.get("summary_agent", {})

            # Merge project types from both languages
            self.project_types = {}
            self.project_types.update(cs_config.get("project_types", {}))
            self.project_types.update(en_config.get("project_types", {}))

            # Store language-specific keywords
            self.project_keywords = {
                "czech": cs_config.get("project_keywords", []),
                "english": en_config.get("project_keywords", []),
            }

            # Store language-specific business indicators
            self.business_indicators = {
                "czech": cs_config.get("business_indicators", []),
                "english": en_config.get("business_indicators", []),
            }

            # Use excluded emails from either file (they should be the same)
            self.excluded_emails = cs_config.get(
                "excluded_emails", en_config.get("excluded_emails", [])
            )

        except Exception as e:
            print(f"Error loading summary agent config: {str(e)}")
            # Fallback configuration
            self.project_types = {"web": "web development", "ai": "AI solution"}
            self.project_keywords = {"czech": ["web"], "english": ["website"]}
            self.business_indicators = {"czech": ["chceme"], "english": ["we want"]}
            self.excluded_emails = ["info@ludekkvapil.cz"]

    def _load_translations(self, language_code: str) -> Dict[str, Any]:
        """Load translations from language-specific YAML file"""
        try:
            translations_path = (
                Path(__file__).parent.parent / "data" / "translations" / f"{language_code}.yml"
            )
            with open(translations_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading translations for {language_code}: {str(e)}")
            return {}

    def extract_contact_info(self, interactions: List) -> Optional[str]:
        """
        Extract user contact information from conversation history.

        Args:
            interactions: List of chat interactions

        Returns:
            Contact information string or None
        """
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        phone_pattern = r"[\+]?[1-9]?[0-9]{7,15}"

        for interaction in interactions:
            if hasattr(interaction, "user_message"):
                # Extract emails
                emails = re.findall(email_pattern, interaction.user_message)
                if emails:
                    user_emails = [
                        email
                        for email in emails
                        if email.lower() not in [e.lower() for e in self.excluded_emails]
                    ]
                    if user_emails:
                        return f"Email: {user_emails[0]}"

                # Extract phone numbers
                phones = re.findall(phone_pattern, interaction.user_message)
                if phones:
                    return f"Telefon: {phones[0]}"

        return None

    def detect_project_type(self, all_text: str) -> Optional[str]:
        """
        Detect project type from conversation text.

        Args:
            all_text: Combined conversation text

        Returns:
            Detected project type or None
        """
        text_lower = all_text.lower()

        for keyword, project_type in self.project_types.items():
            if keyword in text_lower:
                return project_type

        return None

    def extract_key_messages(self, all_messages: List[str], language: str) -> List[str]:
        """
        Extract messages containing key project-related keywords or business context.

        Args:
            all_messages: List of all user messages
            language: Detected conversation language

        Returns:
            List of relevant messages
        """
        keywords = self.project_keywords.get(language, self.project_keywords["english"])

        key_messages = []

        # Include all messages that contain keywords
        for msg in all_messages:
            if any(word in msg.lower() for word in keywords):
                key_messages.append(msg)

        # If we have less than 3 messages and conversation is longer, include additional context
        if len(key_messages) < 3 and len(all_messages) > 2:
            # Include messages that might be business requirements or context
            lang_indicators = self.business_indicators.get(
                language, self.business_indicators["english"]
            )

            for msg in all_messages:
                if msg not in key_messages and any(
                    indicator in msg.lower() for indicator in lang_indicators
                ):
                    key_messages.append(msg)

        # Ensure we include the first message (often contains main request)
        if all_messages and all_messages[0] not in key_messages:
            key_messages.insert(0, all_messages[0])

        return key_messages[:3]  # Limit to 3 most relevant messages

    def generate_summary(self, user_message: str, interactions: List) -> str:
        """
        Generate a meaningful conversation summary.

        Args:
            user_message: Current user message
            interactions: List of chat interactions

        Returns:
            Generated summary string
        """
        try:
            # Collect all user messages
            all_messages = [user_message]
            for interaction in interactions:
                if hasattr(interaction, "user_message"):
                    all_messages.append(interaction.user_message)

            # Extract contact information
            contact_info = self.extract_contact_info(interactions)

            # Detect project type
            all_text = " ".join(all_messages)
            detected_project = self.detect_project_type(all_text)

            # Detect language
            conversation_language = detect_language(all_text)
            lang_code = "czech" if conversation_language == "cs" else "english"

            # Load templates from translation files
            lang_code_file = "cs" if conversation_language == "cs" else "en"
            translations = self._load_translations(lang_code_file)
            lang_templates = translations.get("summary_templates", {})

            # Create enhanced summary using templates
            if len(all_messages) == 1:
                template = lang_templates.get("single_message", "Message: {message}")
                summary = template.format(message=f"{user_message[:100]}...")
            else:
                # Create main summary based on detected project
                if detected_project:
                    template = lang_templates.get(
                        "project_detected", "Customer interested in {project_type}."
                    )
                    summary = template.format(project_type=detected_project)
                else:
                    template = lang_templates.get(
                        "general_inquiry", "Customer inquiring about web services."
                    )
                    summary = template

                # Add key details from conversation
                if len(all_messages) > 2:
                    key_messages = self.extract_key_messages(all_messages, lang_code)
                    if key_messages:
                        template = lang_templates.get(
                            "key_requirements", " Key requirements: {requirements}"
                        )
                        summary += template.format(requirements="; ".join(key_messages))

                # Add contact info if found
                if contact_info:
                    template = lang_templates.get("contact_info", "\n\nContact: {contact}")
                    summary += template.format(contact=contact_info)

            return summary

        except Exception as e:
            print(f"Error in summary generation: {str(e)}")
            return f"Conversation summary: {user_message[:100]}..."
