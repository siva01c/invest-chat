"""Main chat service implementation."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from assistant.infrastructure.database.vector_store import VectorStore
from assistant.infrastructure.llm.openai_client import ChatCompletion
from assistant.services.chat_history import ChatHistory
from assistant.utils.config_loader import (
    load_config,
    load_json_data,
    load_prompt_template,
)
from assistant.utils.markdown_sanitizer import sanitize_markdown_content


class AIService:
    """Main AI service for handling chat interactions."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_history: int = 5,
        temperature: float = 0,
        context_window: int = 3,
        json_path: Optional[str] = None,
    ) -> None:
        """Initialize the AI service."""
        self.model_name = model_name
        self.temperature = temperature
        self.context_window = context_window
        self._setup_environment()
        self.chat_history = ChatHistory(max_history=max_history)

        # Load default data if no path provided
        if json_path is None:
            json_path = str(
                Path(__file__).parent.parent.parent / "data" / "datasources" / "posts.json"
            )
        self.data = load_json_data(str(json_path))
        self.config = load_config()

        # Initialize vector store instance
        self.store = VectorStore()

    def _setup_environment(self) -> None:
        """Load environment variables from .env file."""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        dotenv_path = project_root / ".env"
        load_dotenv(dotenv_path)

        import os

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. Please set it in your .env file."
            )

    def _prepare_knowledge_base(
        self, similar_texts: List[Tuple[str, float, Dict[str, Any], str]]
    ) -> str:
        """Format retrieved texts into knowledge base string."""
        if not similar_texts:
            return "No relevant information found."

        # Format the texts - assuming similar_texts is a list of tuples from vector_store
        formatted_texts = []
        for item in similar_texts:
            if len(item) >= 4:  # id, distance, metadata, document
                document = item[3]
                formatted_texts.append(str(document))

        return "\n\n".join(formatted_texts)

    def _create_system_prompt(
        self, knowledge_base: str, conversation_category: Optional[str] = None
    ) -> str:
        """Create system prompt with knowledge base and category context."""
        try:
            system_template = load_prompt_template("system_prompt.md")

            # Add conversation category context if available
            category_context = ""
            services_context = ""

            # Define job-related categories that should include services.md
            job_related_categories = [
                "job_opportunity",
                "team_leadership",
                "technical_architecture",
                "services",
                "drupal",
                "ai",
                "llm",
                "rag",
                "aws",
                "devops",
                "cybersecurity",
                "cybersecurity_urgent",
                "automation",
                "person",
            ]

            # Load services.md for job-related conversations
            if conversation_category in job_related_categories:
                services_content = load_prompt_template("services.md")
                if services_content:
                    services_context = f"\n\n## Luděk's Services & Capabilities\n{services_content}"
                else:
                    print("Warning: services.md not found")

            if conversation_category:
                category_mapping = {
                    "drupal": "Drupal development and architecture",
                    "ai": "AI and machine learning solutions",
                    "llm": "Large Language Models and applications",
                    "rag": "Retrieval-Augmented Generation systems",
                    "aws": "AWS cloud architecture and DevOps",
                    "cybersecurity": "cybersecurity and security audits",
                    "cybersecurity_urgent": "urgent cybersecurity incidents",
                    "devops": "DevOps and automation",
                    "automation": "process automation and workflows",
                    "technical_architecture": "technical architecture and system design",
                    "job_opportunity": "job opportunities and employment",
                    "team_leadership": "team leadership and collaboration",
                    "services": "technical services and consulting",
                }

                topic = category_mapping.get(conversation_category, conversation_category)
                category_context = f"\n\nIMPORTANT: This conversation is focused on {topic}. Prioritize information and examples related to this topic in your responses."

            formatted_prompt = system_template.format(knowledge_base=knowledge_base)
            return formatted_prompt + services_context + category_context

        except Exception:
            category_info = f" focused on {conversation_category}" if conversation_category else ""
            return f"You are an assistant{category_info}. Knowledge base: {knowledge_base}"

    def _prepare_messages(
        self, system_prompt: str, user_question: str, knowledge_base: str
    ) -> List[Dict[str, str]]:
        """Prepare messages for the chat completion API."""
        messages = [{"role": "system", "content": system_prompt}]

        user_prompt = f"""
        Question: {user_question}

        Based on the provided context, please answer the question thoroughly and precisely.
        """

        # Add context from chat history
        last_interactions = self.chat_history.get_last_n_interactions(self.context_window)
        for interaction in last_interactions:
            messages.extend(
                [
                    {"role": "user", "content": interaction.user_message},
                    {"role": "assistant", "content": interaction.assistant_response},
                ]
            )

        messages.append({"role": "user", "content": user_prompt})

        return messages

    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI API."""
        try:
            chat = ChatCompletion(
                model=self.model_name, messages=messages, temperature=self.temperature
            )
            # Use the async client to classify the request type
            response = await chat.get_response()

            return response

        except Exception as e:
            return f"AI generation failed: {str(e)}"

    async def chat(self, user_text: str, session_id: Optional[str] = None) -> str:
        """Process user input and generate response."""
        # Clean user input
        question = user_text.strip()

        # Use dependency injection to get the classification service instead of legacy import
        try:
            from assistant.core.config.services import get_classification_service

            classification_service = get_classification_service()

            # Get classification and contexts from the proper service
            user_request = await classification_service.classify_message_extended(
                user_text, session_id=session_id
            )

            contexts = user_request.get("contexts", [])
            conversation_category = user_request.get("conversation_category", None)

            # If this is an agent action (like email sending), return the agent's response
            if user_request.get("agent"):
                return str(user_request.get("message", ""))

        except Exception as e:
            # Fallback to basic processing if classification service fails
            # Use a logger instead of print for proper error handling
            import logging

            logging.getLogger(__name__).warning(f"Classification service failed: {e}")
            contexts = []
            conversation_category = None

        # Get similar texts - ensure we await the async call
        if len(contexts) > 0:
            knowledge_base = " ".join(contexts)
        else:
            similar_texts = await self.store.search_similar_text(question)
            knowledge_base = self._prepare_knowledge_base(similar_texts)

            # If knowledge base is insufficient, try MCP enhancement
            if len(knowledge_base) < 100 or conversation_category == "website_analysis":
                if conversation_category == "website_analysis":
                    enhanced_knowledge = await self._enhance_with_website_analysis(question)
                else:
                    enhanced_knowledge = await self._enhance_with_mcp_search(
                        question, conversation_category
                    )

                if enhanced_knowledge:
                    knowledge_base = (
                        f"{knowledge_base}\n\nWeb Analysis Results:\n{enhanced_knowledge}"
                    )

        system_prompt = self._create_system_prompt(knowledge_base, conversation_category)
        messages = self._prepare_messages(system_prompt, question, knowledge_base)

        # Generate response - ensure we await the async call
        answer = await self._generate_response(messages)

        # Update chat history
        if answer and not answer.startswith("AI generation failed"):
            self.chat_history.add_interaction(question, answer, conversation_category)
            try:
                log_dir = Path("logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / "chat_history.log", "a") as log_file:
                    log_entry = {
                        "user": question,
                        "system_prompt": system_prompt,
                        "assistant": answer,
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                import logging

                logging.getLogger(__name__).error(f"Error writing to log file: {str(e)}")

        return sanitize_markdown_content(answer) if answer else answer

    async def _enhance_with_mcp_search(
        self, question: str, category: Optional[str] = None
    ) -> Optional[str]:
        """
        Enhance knowledge base with MCP web search when local knowledge is insufficient.

        Args:
            question: User's question
            category: Conversation category for context

        Returns:
            Enhanced knowledge content or None if MCP not available
        """
        try:
            # Import here to avoid circular dependencies
            from assistant.core.config.services import get_mcp_service

            mcp_service = get_mcp_service()
            if not mcp_service:
                return None

            # Determine if this is a question that would benefit from web search
            web_search_categories = [
                "technical_architecture",
                "services",
                "drupal",
                "ai",
                "llm",
                "rag",
                "aws",
                "devops",
                "cybersecurity",
                "automation",
                "website_analysis",
            ]

            if category not in web_search_categories:
                return None

            # Create a focused search query
            search_query = self._create_search_query(question, category)

            # Perform web search with content extraction
            search_results = await mcp_service.search_web_content(
                query=search_query, max_results=3, extract_content=True
            )

            if not search_results or not search_results.get("extracted_content"):
                return None

            # Format the extracted content for the knowledge base
            enhanced_content = []
            for content in search_results["extracted_content"]:
                if content.get("success") and content.get("content"):
                    site_name = content.get("site_name", "Web Source")
                    url = content.get("url", "")
                    # Extract meaningful content from the web result
                    web_content = self._extract_relevant_content(content["content"])
                    if web_content:
                        enhanced_content.append(f"Source: {site_name} ({url})\n{web_content}")

            if enhanced_content:
                return "\n\n".join(enhanced_content[:2])  # Limit to top 2 sources

            return None

        except Exception as e:
            # Log error but don't fail the chat request
            import logging

            logging.getLogger(__name__).warning(f"MCP enhancement failed: {e}")
            return None

    def _create_search_query(self, question: str, category: Optional[str] = None) -> str:
        """
        Create an optimized search query based on the user's question and category.

        Args:
            question: User's question
            category: Conversation category

        Returns:
            Optimized search query
        """
        # Extract key terms from the question
        key_terms = question.lower()

        # Add category-specific terms for better results
        category_terms = {
            "drupal": "Drupal development architecture",
            "ai": "artificial intelligence machine learning",
            "llm": "large language models LLM",
            "rag": "retrieval augmented generation RAG",
            "aws": "AWS cloud architecture",
            "devops": "DevOps automation",
            "cybersecurity": "cybersecurity security",
            "technical_architecture": "software architecture design",
            "website_analysis": "website analysis HTML meta tags",
        }

        if category and category in category_terms:
            return f"{key_terms} {category_terms[category]}"

        return key_terms

    def _extract_relevant_content(self, web_content: Dict[str, Any]) -> Optional[str]:
        """
        Extract relevant content from web search results.

        Args:
            web_content: Raw web content from MCP

        Returns:
            Extracted and formatted content
        """
        try:
            # Handle different content structures from web crawling
            if isinstance(web_content, dict):
                # Look for common content fields
                content_fields = ["content", "text", "body", "description", "summary"]
                for field in content_fields:
                    if field in web_content and web_content[field]:
                        content = str(web_content[field])
                        # Clean and truncate content
                        return self._clean_web_content(content)

                # If no direct content field, try to extract from results
                if "results" in web_content and isinstance(web_content["results"], list):
                    texts = []
                    for result in web_content["results"][:3]:
                        if isinstance(result, dict) and "text" in result:
                            texts.append(str(result["text"]))
                    if texts:
                        return self._clean_web_content(" ".join(texts))

            elif isinstance(web_content, str):
                return self._clean_web_content(web_content)

            return None

        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Content extraction failed: {e}")
            return None

    def _clean_web_content(self, content: str) -> str:
        """
        Clean and format web content for use in knowledge base.

        Args:
            content: Raw web content

        Returns:
            Cleaned content
        """
        # Remove excessive whitespace and clean up
        content = " ".join(content.split())

        # Limit length to avoid overwhelming the context
        max_length = 500
        if len(content) > max_length:
            content = content[:max_length] + "..."

        return content

    async def _generate_chat_summary(self) -> str:
        """Generate a summary of the chat history for email forwarding."""
        # Simple implementation for testing
        return "Test summary"

    async def process_message(self, user_text: str) -> dict:
        """
        Process user message and return structured response.

        This is a simplified version for testing that checks for emails
        and triggers forwarding if found.

        Args:
            user_text: The user's message

        Returns:
            Dict with agent flag and message
        """
        # Check if message contains a user email (not Luděk's)
        if self._contains_email_address(user_text):
            # Mock the forwarding flow for testing
            try:
                from assistant.agent.email_agent import SimpleMessageForwarder

                chat_summary = await self._generate_chat_summary()
                forwarder = SimpleMessageForwarder(
                    user_message=user_text,
                    chat_history=self.chat_history.get_full_history(),
                    chat_summary=chat_summary,
                )
                message = forwarder.process_data()
                return {"agent": True, "message": message}
            except Exception as e:
                return {"agent": False, "message": f"Error: {str(e)}"}

        # For non-email messages, return normal chat response
        response = await self.chat(user_text)
        return {"agent": False, "message": response}

    def _contains_email_address(self, text: str) -> bool:
        """
        Check if text contains valid email addresses, filtering out Luděk's emails.

        Args:
            text: Text to check for email addresses

        Returns:
            True if text contains valid user email addresses, False otherwise
        """
        # Email regex pattern
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

        # Find all email addresses in the text
        emails = re.findall(email_pattern, text)

        if not emails:
            return False

        # Luděk's email domains to filter out (case insensitive)
        excluded_domains = ["ludekkvapil.cz"]

        # Check if any email is NOT from excluded domains
        for email in emails:
            domain = email.split("@")[-1].lower()
            if domain not in excluded_domains:
                return True

        # All emails were from excluded domains
        return False

    async def _enhance_with_website_analysis(self, question: str) -> Optional[str]:
        """
        Perform direct website analysis using MCP tools.

        Args:
            question: User's question containing website URL

        Returns:
            Website analysis results or None if failed
        """
        try:
            # Import here to avoid circular dependencies
            import re

            from assistant.core.config.services import get_mcp_service

            mcp_service = get_mcp_service()
            if not mcp_service:
                return None

            # Extract URL from the question
            url_pattern = r"(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
            urls = re.findall(url_pattern, question.lower())

            if not urls:
                return None

            # Use the first URL found
            url = urls[0]
            if not url.startswith("http"):
                url = f"https://{url}"

            # Get website information using MCP
            website_info = await mcp_service.get_website_info(url)

            if website_info and website_info.get("success"):
                info_data = website_info.get("info", {})

                # Format the response based on what information is available
                analysis_parts = []

                if isinstance(info_data, dict):
                    # Look for title tag specifically
                    if "title" in info_data:
                        analysis_parts.append(f"Title tag: {info_data['title']}")

                    # Look for meta description
                    if "description" in info_data:
                        analysis_parts.append(f"Meta description: {info_data['description']}")

                    # Look for other meta information
                    if "meta" in info_data:
                        meta_info = info_data["meta"]
                        if isinstance(meta_info, dict):
                            for key, value in meta_info.items():
                                if key not in ["title", "description"]:
                                    analysis_parts.append(f"{key}: {value}")

                # If we have specific data, format it nicely
                if analysis_parts:
                    return f"Website analysis for {url}:\n" + "\n".join(analysis_parts)
                else:
                    # Fallback to raw info if available
                    return f"Website information for {url}: {str(info_data)}"

            return None

        except Exception as e:
            # Log error but don't fail the chat request
            import logging

            logging.getLogger(__name__).warning(f"Website analysis failed: {e}")
            return None
