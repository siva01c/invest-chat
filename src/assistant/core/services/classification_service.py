"""Message classification service for determining conversation categories."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from assistant.core.exceptions import (
    ClassificationException,
    ConfigurationException,
    ErrorCode,
    create_config_error,
    create_llm_error,
)
from assistant.core.interfaces.chat import IClassificationService
from assistant.core.logging import get_logger, log_service_method
from assistant.infrastructure.llm.openai_client import ChatCompletion


class ClassificationService(IClassificationService):
    """Service for classifying user messages into conversation categories."""

    def __init__(self):
        """Initialize the classification service."""
        self.logger = get_logger(self.__class__.__name__)
        self.config = self._load_config()

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "ClassificationService"

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / "data" / "config.yml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config is None:
                    raise ConfigurationException(
                        "Config file is empty",
                        config_file=str(config_path),
                        error_code=ErrorCode.CONFIG_PARSE_ERROR,
                    )
                return config
        except FileNotFoundError as e:
            raise create_config_error(str(config_path), e)
        except yaml.YAMLError as e:
            raise ConfigurationException(
                "Invalid YAML format in config file",
                config_file=str(config_path),
                error_code=ErrorCode.CONFIG_PARSE_ERROR,
                cause=e,
            )

    def _load_classification_prompt(self) -> str:
        """Load classification prompt from file."""
        classification_path = (
            Path(__file__).parent.parent.parent / "data" / "prompts" / "classification.md"
        )
        try:
            with open(classification_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            self.logger.warning(f"Classification prompt file not found: {classification_path}")
            return "You are a classification assistant. Classify user input appropriately."

    @log_service_method()
    async def classify_message(self, user_text: str) -> str:
        """
        Classify user message into appropriate category.

        Args:
            user_text: The user's message to classify

        Returns:
            Classification category string
        """
        if not user_text or not user_text.strip():
            raise ClassificationException(
                "Cannot classify empty message",
                user_input=user_text,
                error_code=ErrorCode.INVALID_INPUT,
            )

        classify_prompt = self._load_classification_prompt()

        # Use an LLM to classify the request type
        messages = [
            {"role": "system", "content": classify_prompt},
            {"role": "user", "content": user_text},
        ]

        try:
            chat = ChatCompletion(messages=messages, temperature=0)
            # Use the async client to classify the request type
            prompt_category = await chat.get_response(lower=True)
            # Clean up markdown formatting and extra characters
            prompt_category = prompt_category.strip("*").strip()

            self.logger.info(f"Classified message as: {prompt_category}")
            return prompt_category

        except Exception as e:
            error = create_llm_error("gpt-4o-mini", "classification", e)
            self.logger.error(f"Classification failed: {error}")
            # Return default fallback instead of raising to maintain service availability
            return "technical_answer"

    async def classify_message_extended(
        self, user_text: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extended classification that returns full context for chat processing.

        Args:
            user_text: The user's message to classify
            session_id: Optional session identifier

        Returns:
            Dictionary with contexts, conversation_category, agent, and message fields
        """
        # Get basic classification
        conversation_category = await self.classify_message(user_text)

        # Get the action for this category
        action = self.get_action_for_category(conversation_category)

        # Handle agent responses
        agent = False
        message = ""
        contexts = []

        if action == "agent_response":
            agent = True
            message = self._get_agent_response(conversation_category, user_text)
        elif action == "clear_chat":
            agent = True
            message = self._get_clear_chat_response()
        elif action == "summary":
            # For summary, we need to load data and return contexts
            contexts = self._get_summary_contexts()
        elif action == "ask_details":
            agent = True
            message = self._get_ask_details_response(conversation_category)

        return {
            "contexts": contexts,
            "conversation_category": conversation_category,
            "agent": agent,
            "message": message,
        }

    def get_action_for_category(self, category: str) -> str:
        """
        Get the action type for a given category.

        Args:
            category: The classification category

        Returns:
            Action type string
        """
        category_actions = self.config.get("category_actions", {})
        return category_actions.get(category, "technical_answer")

    def is_technical_category(self, category: str) -> bool:
        """
        Check if category should maintain conversation history.

        Args:
            category: The classification category

        Returns:
            True if category should maintain history
        """
        technical_categories = self.config.get("technical_categories_with_history", [])
        return category in technical_categories

    def _get_agent_response(self, category: str, user_text: str) -> str:
        """
        Get agent response for categories that require immediate responses.

        Args:
            category: The conversation category
            user_text: The user's original message

        Returns:
            The agent response message
        """
        # Load translations
        try:
            from assistant.utils.config_loader import load_translations

            translations = load_translations("en")
            lang = "en"  # Default to English, could be detected from user_text

            response_messages = translations.get(lang, {}).get("response_messages", {})

            if category == "common_knowledge":
                return response_messages.get(
                    "common_knowledge", "I can help with technical questions."
                )
            elif category == "code":
                return response_messages.get("code", "I cannot help with coding.")
            elif category == "inappropriate_request":
                return translations.get(lang, {}).get(
                    "inappropriate_response", "I can only help with technical topics."
                )

        except Exception:
            # Fallback responses
            if category == "common_knowledge":
                return "I focus on technical consulting and services by Luděk Kvapil."
            elif category == "code":
                return "Sorry, I can't help you with coding"
            elif category == "inappropriate_request":
                return "I can only help with technical topics."

        return ""

    def _get_clear_chat_response(self) -> str:
        """Get response for clear chat action."""
        try:
            from assistant.utils.config_loader import load_translations

            translations = load_translations("en")
            return translations.get("response_messages", {}).get("clear_chat", "Chat cleared.")
        except Exception:
            return "Chat cleared."

    def _get_summary_contexts(self) -> List[str]:
        """Get contexts for summary requests."""
        try:
            from pathlib import Path

            from assistant.utils.config_loader import load_json_data

            # Load posts data
            posts_path = Path(__file__).parent.parent.parent / "data" / "datasources" / "posts.json"
            posts_data = load_json_data(str(posts_path))

            if posts_data:
                contexts = []
                for post_id, post_data in posts_data.items():
                    if isinstance(post_data, dict) and "text" in post_data:
                        contexts.append(post_data["text"])
                return contexts

        except Exception:
            pass

        return []

    def _get_ask_details_response(self, category: str) -> str:
        """Get response for categories that need more details."""
        try:
            from assistant.utils.config_loader import load_translations

            translations = load_translations("en")

            service_detail_questions = translations.get("service_detail_questions", {})

            return service_detail_questions.get(
                category, "Can you provide more details about your request?"
            )

        except Exception:
            return "Can you provide more details about your request?"
