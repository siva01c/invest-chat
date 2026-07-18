"""Microsoft Teams service integration.

This module provides Teams bot functionality using the existing RAG chatbot architecture.
It acts as an adapter between Teams message events and the core chat service.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from microsoft.teams.api import MessageActivity, TypingActivityInput
from microsoft.teams.apps import ActivityContext, App
from microsoft.teams.devtools import DevToolsPlugin

from assistant.core.config.services import get_chat_service
from assistant.utils.log_sanitizer import sanitize_for_logging
from assistant.utils.markdown_sanitizer import sanitize_markdown_content

from .teams_cards import TeamsCardBuilder, TeamsMessageFormatter

logger = logging.getLogger(__name__)


class TeamsService:
    """Microsoft Teams service integration with RAG chatbot capabilities."""

    def __init__(self) -> None:
        """Initialize the Teams service."""
        self.app = App(plugins=[DevToolsPlugin()])
        self._setup_handlers()
        self._session_cache: Dict[str, str] = {}
        self.card_builder = TeamsCardBuilder()
        self.message_formatter = TeamsMessageFormatter()

    def _setup_handlers(self) -> None:
        """Set up Teams message handlers."""

        @self.app.on_message
        async def handle_message(ctx: ActivityContext[MessageActivity]) -> None:
            """Handle all incoming Teams messages."""
            try:
                await self._handle_teams_message(ctx)
            except Exception as e:
                logger.error(f"Error handling Teams message: {e}")
                error_card = self.card_builder.create_error_card(
                    "I encountered an error processing your message. Please try again."
                )
                await ctx.send_card(error_card)

        @self.app.on_conversation_update
        async def handle_conversation_update(ctx: ActivityContext) -> None:
            """Handle conversation updates including members added."""
            try:
                # Check if members were added
                if ctx.activity.members_added:
                    welcome_card = self.card_builder.create_welcome_card()
                    await ctx.send_card(welcome_card)
            except Exception as e:
                logger.error(f"Error handling conversation update: {e}")
                await ctx.send(
                    "Welcome! I'm your AI sales assistant. Ask me about Luděk's technical services!"
                )

        @self.app.on_card_action
        async def handle_card_actions(ctx: ActivityContext) -> None:
            """Handle adaptive card actions."""
            try:
                await self._handle_card_action(ctx)
            except Exception as e:
                logger.error(f"Error handling card action: {e}")
                await ctx.send("Sorry, I couldn't process that action. Please try again.")

    async def _handle_teams_message(self, ctx: ActivityContext[MessageActivity]) -> None:
        """Process Teams message using the existing RAG chatbot."""
        # Send typing indicator
        await ctx.reply(TypingActivityInput())

        # Extract message details
        user_message = ctx.activity.text or ""

        if not user_message.strip():
            await ctx.send(
                "I didn't receive any text in your message. Please send me a message and I'll be happy to help!"
            )
            return

        # Process the message using enhanced Teams features
        await self._process_user_message(ctx, user_message)

    def _get_session_id(self, user_id: str, conversation_id: str) -> str:
        """Generate or retrieve session ID for Teams conversation."""
        cache_key = f"{user_id}_{conversation_id}"

        if cache_key not in self._session_cache:
            # Create new session ID
            self._session_cache[cache_key] = f"teams_{uuid.uuid4()}"

        return self._session_cache[cache_key]

    def _format_teams_response(self, response: str) -> str:
        """Format response for Teams display."""
        # Sanitize markdown content
        formatted_response = sanitize_markdown_content(response)

        # Ensure response isn't too long for Teams (Teams has message limits)
        max_length = 4000  # Conservative limit for Teams messages
        if len(formatted_response) > max_length:
            formatted_response = formatted_response[: max_length - 3] + "..."

        return formatted_response

    async def start(self, host: str = "127.0.0.1", port: int = 3978) -> None:
        """Start the Teams bot service."""
        logger.info(f"Starting Teams bot service on {host}:{port}")
        try:
            await self.app.start(host=host, port=port)
        except Exception as e:
            logger.error(f"Failed to start Teams service: {e}")
            raise

    async def stop(self) -> None:
        """Stop the Teams bot service."""
        logger.info("Stopping Teams bot service")
        # Clean up session cache
        self._session_cache.clear()

    async def _handle_card_action(self, ctx: ActivityContext) -> None:
        """Handle adaptive card actions from users."""
        action_data = ctx.activity.value

        if not action_data:
            return

        action_type = action_data.get("action")

        if action_type == "view_services":
            services_card = self.card_builder.create_services_card()
            await ctx.send_card(services_card)

        elif action_type == "contact_info":
            contact_card = self.card_builder.create_contact_card()
            await ctx.send_card(contact_card)

        elif action_type == "request_quote":
            await ctx.send(
                "I'd be happy to provide a quote! Please describe your project requirements, and I'll connect you with Luděk for a detailed discussion."
            )

        elif action_type == "schedule_call":
            await ctx.send(
                "To schedule a call, please email info@ludekkvapil.cz with your preferred time slots. Luděk will respond within 24 hours to confirm the appointment."
            )

        elif action_type == "retry":
            await ctx.send(
                "Please feel free to ask your question again, and I'll do my best to help!"
            )

        elif action_data.get("quick_reply"):
            # Handle quick reply as a regular message
            quick_reply_text = action_data.get("quick_reply")
            # Create a mock context for the quick reply
            await self._process_user_message(ctx, quick_reply_text)

    async def _process_user_message(
        self, ctx: ActivityContext[MessageActivity], message_text: str
    ) -> None:
        """Process user message with enhanced Teams features."""
        # Extract message details
        user_id = ctx.activity.from_.id
        conversation_id = ctx.activity.conversation.id

        # Generate consistent session ID for conversation continuity
        session_id = self._get_session_id(user_id, conversation_id)

        # Log the interaction (sanitized)
        logger.info(
            f"Teams message from user {sanitize_for_logging(user_id)}: {sanitize_for_logging(message_text[:100])}"
        )

        try:
            # Get the chat service from dependency injection
            chat_service = get_chat_service()

            # Process the message using existing RAG capabilities
            response = await chat_service.chat(message_text, session_id=session_id)

            # Check if this is a website analysis response
            if "website analysis" in response.lower() or "title tag" in response.lower():
                await self._send_website_analysis_response(ctx, response, message_text)
            else:
                # Send regular formatted response
                await self._send_formatted_response(ctx, response)

            # Log successful interaction
            logger.info(f"Teams response sent to user {sanitize_for_logging(user_id)}")

        except Exception as e:
            logger.error(f"Error processing Teams message: {e}")
            error_card = self.card_builder.create_error_card(
                "I'm having trouble processing your request right now. Please try again in a moment."
            )
            await ctx.send_card(error_card)

    async def _send_website_analysis_response(
        self, ctx: ActivityContext[MessageActivity], response: str, original_message: str
    ) -> None:
        """Send website analysis response with rich card."""
        import re

        # Extract URL from original message
        url_pattern = r"(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
        urls = re.findall(url_pattern, original_message.lower())

        if urls:
            url = urls[0]
            # Parse response for structured data (simplified)
            analysis_data = {}

            if "title tag:" in response.lower():
                title_match = re.search(r"title tag:\s*(.+)", response, re.IGNORECASE)
                if title_match:
                    analysis_data["title"] = title_match.group(1).strip()

            if "meta description:" in response.lower():
                desc_match = re.search(r"meta description:\s*(.+)", response, re.IGNORECASE)
                if desc_match:
                    analysis_data["description"] = desc_match.group(1).strip()

            # Send analysis card
            analysis_card = self.card_builder.create_website_analysis_card(url, analysis_data)
            await ctx.send_card(analysis_card)

        # Also send the full response as text
        await self._send_formatted_response(ctx, response)

    async def _send_formatted_response(
        self, ctx: ActivityContext[MessageActivity], response: str
    ) -> None:
        """Send formatted response, potentially split into multiple messages."""
        # Format response for Teams
        formatted_response = self.message_formatter.add_teams_formatting(response)

        # Split long responses
        messages = self.message_formatter.format_long_response(formatted_response)

        for message in messages:
            await ctx.send(message)

        # Add quick reply suggestions for certain types of responses
        if any(keyword in response.lower() for keyword in ["services", "help", "what can"]):
            suggestions = ["View Services", "Contact Information", "Request Quote", "Portfolio"]
            quick_replies = self.message_formatter.create_quick_reply_suggestions(suggestions)
            await ctx.send_card(quick_replies)


# Global Teams service instance
_teams_service: Optional[TeamsService] = None


def get_teams_service() -> TeamsService:
    """Get or create Teams service instance."""
    global _teams_service
    if _teams_service is None:
        _teams_service = TeamsService()
    return _teams_service


async def start_teams_service(host: str = "127.0.0.1", port: int = 3978) -> None:
    """Start the Teams service."""
    service = get_teams_service()
    await service.start(host=host, port=port)


async def stop_teams_service() -> None:
    """Stop the Teams service."""
    global _teams_service
    if _teams_service:
        await _teams_service.stop()
        _teams_service = None
