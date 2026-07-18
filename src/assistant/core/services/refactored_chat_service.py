"""Refactored chat service using modular service architecture."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from assistant.config import get_settings
from assistant.config.loader import config_manager
from assistant.core.interfaces.chat import (
    IChatService,
    IClassificationService,
    IConversationService,
    IEmailService,
)
from assistant.core.services.classification_service import ClassificationService
from assistant.core.services.conversation_service import ConversationService
from assistant.core.services.email_service import EmailService
from assistant.infrastructure.database.vector_store import VectorStore
from assistant.infrastructure.llm.openai_client import ChatCompletion
from assistant.utils.markdown_sanitizer import sanitize_markdown_content


class RefactoredAIService(IChatService):
    """Refactored AI service with modular architecture and configuration management."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_history: Optional[int] = None,
        temperature: Optional[float] = None,
        context_window: Optional[int] = None,
        json_path: Optional[str] = None,
        classification_service: Optional[IClassificationService] = None,
        conversation_service: Optional[IConversationService] = None,
        email_service: Optional[IEmailService] = None,
    ):
        """Initialize the refactored AI service with configuration and dependency injection."""
        # Get settings instance
        self.settings = get_settings()

        # Use provided values or fallback to configuration
        self.model_name = model_name or self.settings.openai_model
        self.temperature = (
            temperature if temperature is not None else self.settings.openai_temperature
        )

        # Initialize services with dependency injection
        self.classification_service = classification_service or ClassificationService()
        self.conversation_service = conversation_service or ConversationService(
            max_history or self.settings.max_history, context_window or self.settings.context_window
        )
        self.email_service = email_service or EmailService()

        # Load data using configuration manager
        if json_path is None:
            self.data = config_manager.get_data("posts.json")
        else:
            # Custom path provided
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self.data = {}

        # Initialize vector store with configuration
        self.store = VectorStore(
            collection_name=self.settings.chromadb_collection_name,
            database_path=self.settings.chromadb_database_path,
        )

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "RefactoredAIService"

    def _prepare_knowledge_base(self, similar_texts: List) -> str:
        """Format retrieved texts into knowledge base string."""
        if not similar_texts:
            return "No relevant information found."

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
            # Load system prompt using configuration manager
            system_template = config_manager.get_prompt("system_prompt.md")

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
                try:
                    services_content = config_manager.get_prompt("services.md")
                    services_context = f"\n\n## Luděk's Services & Capabilities\n{services_content}"
                except Exception:
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

        except Exception as e:
            print(f"Error loading system prompt: {e}")
            category_info = f" focused on {conversation_category}" if conversation_category else ""
            return f"You are an assistant{category_info}. Knowledge base: {knowledge_base}"

    def _prepare_messages(self, system_prompt: str, user_question: str) -> List[Dict[str, str]]:
        """Prepare messages for the chat completion API."""
        messages = [{"role": "system", "content": system_prompt}]

        user_prompt = f"""
        Question: {user_question}

        Based on the provided context, please answer the question thoroughly and precisely.
        """

        # Add context from conversation service
        context_messages = self.conversation_service.get_context_messages()
        messages.extend(context_messages)
        messages.append({"role": "user", "content": user_prompt})

        return messages

    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI API."""
        try:
            chat = ChatCompletion(
                model=self.model_name, messages=messages, temperature=self.temperature
            )
            response = await chat.get_response()
            return response

        except Exception as e:
            return f"AI generation failed: {str(e)}"

    async def handle_user_request(
        self, user_text: str, session_id: Optional[str] = None
    ) -> Dict[str, any]:
        """Handle user request by classifying and routing to appropriate action."""
        # Initialize contexts list
        contexts = []

        # Classify the user message
        prompt_category = await self.classification_service.classify_message(user_text)
        action = self.classification_service.get_action_for_category(prompt_category)
        print(f"DEBUG: Category: {prompt_category}, Action: {action}")

        # Handle based on action type
        if action == "summary":
            # Convert JSON data to list of strings for the summary case
            for post_id, post_data in self.data.items():
                document_parts = [
                    f"User: {post_data.get('user', '')}",
                    f"Text: {post_data.get('text', '')}",
                    f"Metadata: {post_data.get('metadata', '')}",
                ]
                contexts.append("\n\n".join(document_parts))
            # Fall through to default return

        elif action == "forward_message":
            return await self.email_service.handle_forward_message(
                user_text, session_id, self.conversation_service
            )

        elif action == "agent_response":
            from assistant.agent.language_detection_agent import detect_language

            language = detect_language(user_text)
            lang_code = "cs" if language == "cs" else "en"
            translations = config_manager.get_translations(lang_code)

            if prompt_category == "common_knowledge":
                msg = translations.get("response_messages", {}).get(
                    "common_knowledge", "Outside scope."
                )
            elif prompt_category == "code":
                msg = translations.get("response_messages", {}).get(
                    "code", "Cannot help with code."
                )
            else:
                return self.conversation_service.handle_inappropriate_request(user_text)

            self.conversation_service.add_interaction(user_text, msg)
            return {"agent": True, "message": msg}

        elif action == "clear_chat":
            self.conversation_service.clear_history()
            from assistant.agent.language_detection_agent import detect_language

            language = detect_language(user_text)
            lang_code = "cs" if language == "cs" else "en"
            translations = config_manager.get_translations(lang_code)
            msg = translations.get("response_messages", {}).get("clear_chat", "Chat cleared.")
            return {"agent": True, "message": msg}

        elif action == "show_history":
            return await self._handle_show_history(user_text, session_id)

        elif action == "ask_details":
            return await self._handle_ask_details(user_text, prompt_category, session_id)

        # Default: technical_answer - Let the main chat system handle
        # Set conversation category for technical categories to enable categorization history
        if self.classification_service.is_technical_category(prompt_category):
            self.conversation_service.set_conversation_category(prompt_category)
            print(f"Set conversation category to: {prompt_category}")

        return {
            "agent": False,
            "message": user_text,
            "contexts": contexts,
            "conversation_category": prompt_category,
        }

    async def _handle_ask_details(
        self, user_text: str, prompt_category: str, session_id: Optional[str]
    ) -> Dict[str, any]:
        """Handle ask_details action with proper context checking."""
        # Check if we already asked for details recently by looking at the last response
        last_interactions = self.conversation_service.chat_history.get_last_n_interactions(1)
        if last_interactions:
            last_response = last_interactions[0].assistant_response

            from assistant.agent.language_detection_agent import detect_language

            language = detect_language(user_text)
            lang_code = "cs" if language == "cs" else "en"
            translations = config_manager.get_translations(lang_code)

            czech_ask_phrases = translations.get("conversation_phrases", {}).get(
                "ask_details_phrases", []
            )
            english_ask_phrases = (
                config_manager.get_translations("en")
                .get("conversation_phrases", {})
                .get("ask_details_phrases", [])
            )
            czech_cybersecurity_phrases = translations.get("conversation_phrases", {}).get(
                "cybersecurity_urgent_phrases", []
            )
            english_cybersecurity_phrases = (
                config_manager.get_translations("en")
                .get("conversation_phrases", {})
                .get("cybersecurity_urgent_phrases", [])
            )

            all_ask_phrases = czech_ask_phrases + english_ask_phrases
            all_cybersecurity_phrases = czech_cybersecurity_phrases + english_cybersecurity_phrases

            is_asking_details = any(phrase in last_response.lower() for phrase in all_ask_phrases)
            is_asking_cyber_details = any(
                phrase in last_response.lower() for phrase in all_cybersecurity_phrases
            )

            if is_asking_details or is_asking_cyber_details:
                if is_asking_cyber_details:
                    return self.email_service.handle_cybersecurity_urgent(
                        user_text, session_id, self.conversation_service
                    )
                else:
                    return self.email_service.handle_service_details(
                        user_text, self.conversation_service
                    )

        # Ask for more details about the service inquiry
        from assistant.agent.language_detection_agent import detect_language

        language = detect_language(user_text)

        lang_code = "cs" if language == "cs" else "en"
        translations = config_manager.get_translations(lang_code)
        service_questions = translations.get("service_detail_questions", {})

        if language == "cs":
            msg = service_questions.get(
                prompt_category, "Můžete mi říct více detailů o vašem projektu?"
            )
        else:
            msg = service_questions.get(
                prompt_category, "Could you tell me more details about your project?"
            )

        self.conversation_service.add_interaction(user_text, msg)
        return {"agent": True, "message": msg, "prompt_category": f"ask_details_{prompt_category}"}

    async def _handle_show_history(
        self, user_text: str, session_id: Optional[str]
    ) -> Dict[str, any]:
        """Handle show_history action by displaying conversation history."""
        try:
            # Get conversation history
            history = self.conversation_service.chat_history.get_full_history()

            if not history or len(history) == 0:
                msg = "No conversation history available. This appears to be the start of our conversation."
            else:
                # Filter out show_history interactions to avoid recursive history display
                filtered_history = [
                    interaction
                    for interaction in history
                    if not (
                        hasattr(interaction, "user_message")
                        and interaction.user_message
                        and any(
                            phrase in interaction.user_message.lower()
                            for phrase in [
                                "show me conversation history",
                                "show chat history",
                                "display previous messages",
                                "what did we talk about",
                            ]
                        )
                    )
                ]

                if not filtered_history:
                    msg = "No conversation history available. This appears to be the start of our conversation."
                else:
                    # Format history for display
                    history_lines = []
                    for i, interaction in enumerate(
                        filtered_history[-10:], 1
                    ):  # Show last 10 filtered interactions
                        if hasattr(interaction, "user_message") and hasattr(
                            interaction, "assistant_response"
                        ):
                            history_lines.append(f"{i}. **You:** {interaction.user_message}")
                            history_lines.append(
                                f"   **Assistant:** {interaction.assistant_response}"
                            )
                            history_lines.append("")  # Empty line for spacing

                    history_text = "\n".join(history_lines)

                    total_filtered = len(filtered_history)
                    if total_filtered > 10:
                        msg = f"Here are your last 10 messages from our conversation (total: {total_filtered} messages):\n\n{history_text}"
                    else:
                        msg = f"Here is our conversation history ({total_filtered} messages):\n\n{history_text}"

            self.conversation_service.add_interaction(user_text, msg)
            return {"agent": True, "message": msg, "prompt_category": "show_history"}

        except Exception as e:
            error_msg = f"Sorry, I couldn't retrieve the conversation history. Error: {str(e)}"
            self.conversation_service.add_interaction(user_text, error_msg)
            return {"agent": True, "message": error_msg, "prompt_category": "show_history"}

    async def chat(self, user_text: str, session_id: Optional[str] = None) -> str:
        """Process user input and generate response."""
        question = user_text.strip()

        user_request = await self.handle_user_request(user_text, session_id=session_id)

        contexts = user_request.get("contexts", [])
        conversation_category = user_request.get("conversation_category", None)
        if user_request.get("agent", False):
            return user_request.get("message", "")

        # Get similar texts - ensure we await the async call
        if len(contexts) > 0:
            knowledge_base = " ".join(contexts)
        else:
            similar_texts = await self.store.search_similar_text(
                question, self.settings.chromadb_search_results
            )
            knowledge_base = self._prepare_knowledge_base(similar_texts)

            # If knowledge base is insufficient or this is website analysis, try MCP enhancement
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
        messages = self._prepare_messages(system_prompt, question)

        # Generate response - ensure we await the async call
        answer = await self._generate_response(messages)

        # Update conversation history
        if answer and not answer.startswith("AI generation failed"):
            self.conversation_service.add_interaction(question, answer, conversation_category)

            # Log conversation if enabled
            if self.settings.enable_chat_logging:
                try:
                    log_dir = Path(self.settings.log_directory)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with open(log_dir / "chat_history.log", "a") as log_file:
                        log_entry = {
                            "user": question,
                            "system_prompt": system_prompt,
                            "assistant": answer,
                        }
                        log_file.write(json.dumps(log_entry) + "\n")
                except Exception as e:
                    print(f"Error writing to log file: {str(e)}")

        return sanitize_markdown_content(answer) if answer else answer

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
            print(f"Website analysis failed: {e}")
            return None

    async def _enhance_with_mcp_search(self, question: str, category: str = None) -> Optional[str]:
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
            print(f"MCP enhancement failed: {e}")
            return None

    def _create_search_query(self, question: str, category: str = None) -> str:
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

    def _extract_relevant_content(self, web_content: Dict) -> Optional[str]:
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
            print(f"Content extraction failed: {e}")
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
