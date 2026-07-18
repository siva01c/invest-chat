"""Teams-specific card builders and rich content functionality."""

from typing import Any, Dict, List


class TeamsCardBuilder:
    """Enhanced Teams card builder with advanced formatting capabilities."""

    @staticmethod
    def create_welcome_card() -> Dict[str, Any]:
        """Create a welcome card for new users."""
        return {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": "Welcome to Luděk's AI Sales Assistant! 🚀",
                        "weight": "Bolder",
                        "size": "Large",
                        "color": "Accent",
                    },
                    {
                        "type": "TextBlock",
                        "text": "I'm here to help you with:",
                        "weight": "Bolder",
                        "spacing": "Medium",
                    },
                    {
                        "type": "FactSet",
                        "facts": [
                            {
                                "title": "🔧 Drupal Development",
                                "value": "Custom solutions and architecture",
                            },
                            {
                                "title": "🤖 AI & Machine Learning",
                                "value": "LLM integration and RAG systems",
                            },
                            {"title": "☁️ DevOps & Cloud", "value": "AWS, Docker, CI/CD pipelines"},
                            {
                                "title": "🔒 Cybersecurity",
                                "value": "Security audits and consulting",
                            },
                        ],
                    },
                    {
                        "type": "TextBlock",
                        "text": "Ask me anything about these services or request a consultation!",
                        "wrap": True,
                        "spacing": "Medium",
                    },
                ],
                "actions": [
                    {
                        "type": "Action.Submit",
                        "title": "View Services",
                        "data": {"action": "view_services"},
                    },
                    {
                        "type": "Action.Submit",
                        "title": "Contact Info",
                        "data": {"action": "contact_info"},
                    },
                ],
            },
        }

    @staticmethod
    def create_services_card() -> Dict[str, Any]:
        """Create a detailed services overview card."""
        return {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": "Technical Services & Expertise",
                        "weight": "Bolder",
                        "size": "Large",
                    },
                    {
                        "type": "Container",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "🔧 **Drupal Development**",
                                "weight": "Bolder",
                                "color": "Accent",
                            },
                            {
                                "type": "TextBlock",
                                "text": "• Custom module development\n• Performance optimization\n• Migration services\n• Security hardening",
                                "wrap": True,
                                "spacing": "Small",
                            },
                        ],
                        "spacing": "Medium",
                    },
                    {
                        "type": "Container",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "🤖 **AI & Machine Learning**",
                                "weight": "Bolder",
                                "color": "Accent",
                            },
                            {
                                "type": "TextBlock",
                                "text": "• RAG system implementation\n• LLM integration\n• Chatbot development\n• AI consulting",
                                "wrap": True,
                                "spacing": "Small",
                            },
                        ],
                        "spacing": "Medium",
                    },
                    {
                        "type": "Container",
                        "items": [
                            {
                                "type": "TextBlock",
                                "text": "☁️ **DevOps & Cloud**",
                                "weight": "Bolder",
                                "color": "Accent",
                            },
                            {
                                "type": "TextBlock",
                                "text": "• AWS infrastructure\n• Docker containerization\n• CI/CD pipelines\n• Monitoring & observability",
                                "wrap": True,
                                "spacing": "Small",
                            },
                        ],
                        "spacing": "Medium",
                    },
                ],
                "actions": [
                    {
                        "type": "Action.Submit",
                        "title": "Request Quote",
                        "data": {"action": "request_quote"},
                    },
                    {
                        "type": "Action.OpenUrl",
                        "title": "Visit Website",
                        "url": "https://ludekkvapil.cz",
                    },
                ],
            },
        }

    @staticmethod
    def create_contact_card() -> Dict[str, Any]:
        """Create a contact information card."""
        return {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": "Contact Information",
                        "weight": "Bolder",
                        "size": "Large",
                    },
                    {
                        "type": "FactSet",
                        "facts": [
                            {"title": "📧 Email", "value": "info@ludekkvapil.cz"},
                            {"title": "🌐 Website", "value": "ludekkvapil.cz"},
                            {"title": "💼 LinkedIn", "value": "linkedin.com/in/ludekkvapil"},
                            {"title": "📍 Location", "value": "Prague, Czech Republic"},
                        ],
                    },
                    {
                        "type": "TextBlock",
                        "text": "Available for consultations and project discussions. Response time: typically within 24 hours.",
                        "wrap": True,
                        "spacing": "Medium",
                        "isSubtle": True,
                    },
                ],
                "actions": [
                    {
                        "type": "Action.OpenUrl",
                        "title": "Send Email",
                        "url": "mailto:info@ludekkvapil.cz?subject=Teams Bot Inquiry",
                    },
                    {
                        "type": "Action.Submit",
                        "title": "Schedule Call",
                        "data": {"action": "schedule_call"},
                    },
                ],
            },
        }

    @staticmethod
    def create_error_card(error_message: str) -> Dict[str, Any]:
        """Create an error card for displaying error messages."""
        return {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": "⚠️ Something went wrong",
                        "weight": "Bolder",
                        "color": "Attention",
                    },
                    {"type": "TextBlock", "text": error_message, "wrap": True, "spacing": "Medium"},
                    {
                        "type": "TextBlock",
                        "text": "Please try again or contact support if the issue persists.",
                        "wrap": True,
                        "spacing": "Small",
                        "isSubtle": True,
                    },
                ],
                "actions": [
                    {"type": "Action.Submit", "title": "Try Again", "data": {"action": "retry"}}
                ],
            },
        }

    @staticmethod
    def create_website_analysis_card(url: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a card displaying website analysis results."""
        facts = []

        if analysis_data.get("title"):
            facts.append({"title": "Title", "value": analysis_data["title"]})

        if analysis_data.get("description"):
            facts.append(
                {"title": "Description", "value": analysis_data["description"][:100] + "..."}
            )

        if analysis_data.get("meta"):
            meta_info = analysis_data["meta"]
            if isinstance(meta_info, dict):
                for key, value in list(meta_info.items())[:3]:  # Limit to first 3 meta items
                    facts.append({"title": key.title(), "value": str(value)[:50]})

        return {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": f"🔍 Website Analysis: {url}",
                        "weight": "Bolder",
                        "size": "Medium",
                    },
                    {
                        "type": "FactSet",
                        "facts": (
                            facts if facts else [{"title": "Status", "value": "Analysis completed"}]
                        ),
                    },
                ],
                "actions": [
                    {
                        "type": "Action.OpenUrl",
                        "title": "Visit Website",
                        "url": url if url.startswith("http") else f"https://{url}",
                    }
                ],
            },
        }

    @staticmethod
    def create_typing_card() -> Dict[str, Any]:
        """Create a typing indicator card."""
        return {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": "🤔 Analyzing your request...",
                        "weight": "Bolder",
                        "color": "Accent",
                    },
                    {
                        "type": "TextBlock",
                        "text": "Please wait while I process your message and gather relevant information.",
                        "wrap": True,
                        "isSubtle": True,
                    },
                ],
            },
        }


class TeamsMessageFormatter:
    """Utility class for formatting messages for Teams."""

    @staticmethod
    def format_long_response(response: str, max_length: int = 4000) -> List[str]:
        """Split long responses into multiple messages for Teams."""
        if len(response) <= max_length:
            return [response]

        # Split by paragraphs first
        paragraphs = response.split("\n\n")
        messages = []
        current_message = ""

        for paragraph in paragraphs:
            if len(current_message + paragraph) <= max_length:
                current_message += paragraph + "\n\n"
            else:
                if current_message:
                    messages.append(current_message.strip())
                current_message = paragraph + "\n\n"

        if current_message:
            messages.append(current_message.strip())

        return messages

    @staticmethod
    def add_teams_formatting(text: str) -> str:
        """Add Teams-specific formatting to text."""
        # Convert markdown-style formatting to Teams format
        # Teams supports a subset of markdown

        # Handle bold text first (replace ** with **)
        # Teams uses ** for bold, so no change needed

        # Handle italic text (replace single * with _)
        # But avoid replacing * inside ** pairs
        import re

        # Replace single asterisks that are not part of bold formatting
        text = re.sub(r"(?<!\*)\*(?!\*)", "_", text)

        # Code blocks - Teams supports ```
        # No changes needed for code blocks

        return text

    @staticmethod
    def create_quick_reply_suggestions(suggestions: List[str]) -> Dict[str, Any]:
        """Create quick reply suggestions for Teams."""
        actions = []
        for suggestion in suggestions[:4]:  # Limit to 4 suggestions
            actions.append(
                {"type": "Action.Submit", "title": suggestion, "data": {"quick_reply": suggestion}}
            )

        return {
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "text": "💡 Suggestions:",
                        "weight": "Bolder",
                        "size": "Small",
                    }
                ],
                "actions": actions,
            },
        }
