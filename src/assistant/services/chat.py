from openai import AsyncOpenAI
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from assistant.services.vector_store import VectorStore
from assistant.services.chat_history import ChatHistory
import json
import yaml
from assistant.agent.email_agent import send_simple_message 
from assistant.services.completions import ChatCompletion

# Vector store will be initialized in the AIService constructor

def load_json_data(json_path: str) -> dict:
        """
        Load JSON data from file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Loaded JSON data or empty dict if error
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            print(f"Error: File {json_path} not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_path}.")
            return {}

def load_config() -> dict:
    """Load configuration from YAML file."""
    try:
        config_path = Path(__file__).parent.parent / 'data' / 'config.yml'
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found.")
        return {}
    except yaml.YAMLError:
        print(f"Error: Invalid YAML format in config file.")
        return {}

class AIService:
    def __init__(self, model_name: str = "gpt-4o-mini", 
                 max_history: int = 5, 
                 temperature: float = 0,
                 context_window: int = 3,
                 json_path: str = None):
        
        """
        Initialize the AI service.

        Args:
            model_name: OpenAI model to use
            max_history: Maximum number of chat interactions to store
            temperature: Temperature parameter for response generation
            context_window: Number of previous interactions to include in context
        """
        self.model_name = model_name
        self.temperature = temperature
        self.context_window = context_window
        self._setup_environment()
        self.chat_history = ChatHistory(max_history=max_history)
        # Load default data if no path provided
        if json_path is None:
            json_path = Path(__file__).parent.parent / 'data' / 'datasources' / 'posts.json'
        self.data = load_json_data(json_path=str(json_path))
        self.config = load_config()
        
        # Initialize vector store instance
        self.store = VectorStore()

    def _handle_inappropriate_request(self, user_text: str) -> dict:
        """
        Handle inappropriate requests with polite redirection.
        
        Args:
            user_text: The user's inappropriate request
            
        Returns:
            Response dictionary with polite redirection
        """
        from assistant.agent.language_detection_agent import detect_language
        language = detect_language(user_text)
        
        responses = self.config.get('inappropriate_responses', {})
        if language == 'cs':
            response = responses.get('czech', 'Omlouvám se, nemohu vám s tím pomoci.')
        else:
            response = responses.get('english', 'Sorry, I cannot help you with that.')
        
        self.chat_history.add_interaction(user_text, response)
        return {"agent": True, "message": response, "prompt_category": "inappropriate_request"}

    def _generate_conversation_summary(self, user_message: str) -> str:
        """
        Generate an accurate summary of the conversation based on actual content.
        
        Args:
            user_message: The current user message
            
        Returns:
            Accurate conversation summary
        """
        try:
            interactions = self.chat_history.get_full_history()
            
            # Collect all user messages for analysis
            all_user_messages = [user_message]
            for interaction in interactions:
                if hasattr(interaction, 'user_message'):
                    all_user_messages.append(interaction.user_message)
            
            conversation_text = ' '.join(all_user_messages).lower()
            
            # Detect conversation language
            from assistant.agent.language_detection_agent import detect_language
            conversation_language = detect_language(' '.join(all_user_messages))
            
            categories = self.config.get('conversation_categories', {})
            import re
            
            # First, collect all mentioned technologies
            mentioned_technologies = []
            technology_categories = ['drupal', 'hugo', 'wordpress', 'ai', 'aws', 'automation']
            
            for category_name in technology_categories:
                if category_name in categories:
                    category_data = categories[category_name]
                    keywords = category_data.get('keywords', [])
                    for keyword in keywords:
                        # Use case-insensitive search and word boundaries for better matching
                        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        if re.search(pattern, conversation_text):
                            # Map to proper display names
                            tech_display_name = {
                                'drupal': 'Drupal',
                                'hugo': 'Hugo', 
                                'wordpress': 'WordPress',
                                'ai': 'AI/LLM',
                                'aws': 'AWS',
                                'automation': 'Process Automation'
                            }.get(category_name, category_name.upper())
                            mentioned_technologies.append(tech_display_name)
                            break  # Only add each technology once
            
            # Then find the primary category
            primary_category = None
            primary_summary = None
            
            for category_name, category_data in categories.items():
                keywords = category_data.get('keywords', [])
                for keyword in keywords:
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    if re.search(pattern, conversation_text):
                        primary_category = category_name
                        primary_summary = category_data.get('summary', f'{category_name} inquiry')
                        break
                if primary_category:
                    break
            
            # If no category found, create intelligent summary based on content
            if not primary_summary:
                # Look for key business terms
                business_terms = {
                    'cs': {
                        'automatizaci': 'automatizace procesů',
                        'procesy': 'automatizace procesů', 
                        'integrovat': 'integrace',
                        'chatbot': 'chatbot',
                        'ai': 'AI řešení',
                        'firma': 'firemní řešení'
                    },
                    'en': {
                        'automation': 'process automation',
                        'integrate': 'integration',
                        'chatbot': 'chatbot development',
                        'ai': 'AI solutions',
                        'company': 'business solutions'
                    }
                }
                
                found_terms = []
                terms_dict = business_terms.get(conversation_language, business_terms['en'])
                
                for term, description in terms_dict.items():
                    if term in conversation_text:
                        found_terms.append(description)
                
                if found_terms:
                    if conversation_language == 'cs':
                        primary_summary = f"Dotaz na {', '.join(found_terms)}"
                    else:
                        primary_summary = f"Inquiry about {', '.join(found_terms)}"
            
            # Enhance summary with mentioned technologies
            if primary_summary and mentioned_technologies:
                tech_list = ', '.join(mentioned_technologies)
                # Add technology info to summary if not already present
                if not any(tech.lower() in primary_summary.lower() for tech in mentioned_technologies):
                    if conversation_language == 'cs':
                        primary_summary = f"{primary_summary} - Technologie: {tech_list}"
                    else:
                        primary_summary = f"{primary_summary} - Technologies: {tech_list}"
            
            # Final fallback with better context
            if not primary_summary:
                key_phrases = []
                # Extract meaningful phrases from user messages
                for msg in all_user_messages:
                    if len(msg.strip()) > 3 and msg.strip() not in ['čau', 'ahoj', 'hi', 'hello']:
                        key_phrases.append(msg.strip()[:50])
                
                if key_phrases:
                    if conversation_language == 'cs':
                        primary_summary = f"Obecný dotaz: {' | '.join(key_phrases[:2])}"
                    else:
                        primary_summary = f"General inquiry: {' | '.join(key_phrases[:2])}"
                else:
                    if conversation_language == 'cs':
                        primary_summary = "Obecný dotaz na služby"
                    else:
                        primary_summary = "General service inquiry"
            
            return primary_summary
                
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return f"Conversation summary: {user_message[:100]}..."

    def _setup_environment(self) -> None:
        """Load environment variables from .env file."""
        project_root = Path(__file__).parent.parent.parent.parent
        dotenv_path = project_root / '.env'
        load_dotenv(dotenv_path)
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

    def _prepare_knowledge_base(self, similar_texts: List) -> str:
        """
        Format retrieved texts into knowledge base string.

        Args:
            similar_texts: List of relevant text chunks

        Returns:
            Formatted knowledge base string
        """
        # Handle the actual structure of the similar_texts from vector store
        if not similar_texts:
            return "No relevant information found."
            
        # Format the texts - assuming similar_texts is a list of tuples from vector_store
        formatted_texts = []
        for item in similar_texts:
            if len(item) >= 4:  # id, distance, metadata, document
                document = item[3]
                formatted_texts.append(str(document))
        
        return "\n\n".join(formatted_texts)

    def _create_system_prompt(self, knowledge_base: str) -> str:
        try:
            system_prompt_path = Path(__file__).parent.parent / 'data' / 'prompts' / 'system_prompt.md'
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                system_template = f.read()
            return system_template.format(knowledge_base=knowledge_base)
        except FileNotFoundError:
            return f"You are an assistant. Knowledge base: {knowledge_base}"
    
    def _prepare_messages(self, system_prompt: str, user_question: str, knowledge_base: str) -> List[Dict[str, str]]:
        """
        Prepare messages for the chat completion API.

        Args:
            system_prompt: Formatted system prompt
            user_question: Current user question

        Returns:
            List of message dictionaries
        """
        messages = [{"role": "system", "content": system_prompt}]

        user_prompt = f"""      
        Question: {user_question}
        
        Based on the provided context, please answer the question thoroughly and precisely.
        """
        
        # Add context from chat history
        last_interactions = self.chat_history.get_last_n_interactions(self.context_window)
        for interaction in last_interactions:
            messages.extend([
                {"role": "user", "content": interaction.user_message},
                {"role": "assistant", "content": interaction.assistant_response}
            ])
        
        messages.append({"role": "user", "content": user_prompt})

        return messages

    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using OpenAI API.

        Args:
            messages: Prepared message list

        Returns:
            AI-generated response or error message
        """
        try:
            chat = ChatCompletion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            # Use the async client to classify the request type
            response = await chat.get_response()
            
            return response
            
        except Exception as e:
            return f"AI generation failed: {str(e)}"

    async def chat(self, user_text: str) -> str:
        """
        Process user input and generate response.

        Args:
            user_text: User's input text

        Returns:
            AI-generated response
        """
        # Clean user input
        question = user_text.strip()

        user_request = await self.handle_user_request(user_text)
                
        contexts = user_request.get("contexts", [])
        if user_request.get("agent", []):
            return user_request.get("message", [])
        
        # Get similar texts - ensure we await the async call
        if len(contexts) > 0:
            knowledge_base = ' ' . join(contexts)
        else:
            similar_texts = await self.store.search_similar_text(question)
            # Prepare prompts and messages
            knowledge_base = self._prepare_knowledge_base(similar_texts)
        
        system_prompt = self._create_system_prompt(knowledge_base)
        messages = self._prepare_messages(system_prompt, question, knowledge_base)
        
        # Generate response - ensure we await the async call
        answer = await self._generate_response(messages)
        
        # Update chat history
        if answer and not answer.startswith("AI generation failed"):
            self.chat_history.add_interaction(question, answer)
            try:
                log_dir = Path("logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / "chat_history.log", "a") as log_file:
                    log_entry = {
                        "user": question,
                        "system_prompt": system_prompt,
                        "assistant": answer
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"Error writing to log file: {str(e)}") 
            
        return answer
    
    async def handle_user_request(self, user_text: str) -> str:
        """
        Handle special user requests like clearing chat history.

        Args:
            user_text: User's input text.

        Returns:
            Response string based on the request.
        """

        # First, check for simple patterns that should never be misclassified
        user_text_lower = user_text.lower().strip()
        
        # Check for email confirmation context first
        last_interactions = self.chat_history.get_last_n_interactions(1)
        if last_interactions:
            last_response = last_interactions[0].assistant_response
            # Check if we're in email confirmation state
            confirmation_indicators = ['napište \'ano\'', 'type \'yes\'', 'shrnutí luďkovi', 'summary to luděk']
            in_confirmation_state = any(indicator in last_response.lower() for indicator in confirmation_indicators)
            
            if in_confirmation_state:
                # Handle confirmation/cancellation in email context
                from assistant.agent.language_detection_agent import is_confirmation_yes, is_confirmation_no
                if is_confirmation_yes(user_text) or is_confirmation_no(user_text):
                    # Forward to email agent for handling
                    try:
                        from assistant.agent.email_agent import send_simple_message
                        chat_history = self.chat_history.get_full_history()
                        conversation_summary = self._generate_conversation_summary(user_text)
                        
                        # Find contact info from previous messages in chat history
                        import re
                        user_contact = None
                        for interaction in reversed(chat_history):
                            if hasattr(interaction, 'user_message'):
                                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                                emails = re.findall(email_pattern, interaction.user_message)
                                if emails:
                                    # Filter out Luděk's emails
                                    ludek_emails = ['info@ludekkvapil.cz', 'ludek@ludekkvapil.cz', 'kvapilludek@gmail.com']
                                    user_emails = [email for email in emails if email.lower() not in [e.lower() for e in ludek_emails]]
                                    if user_emails:
                                        user_contact = f"Email: {user_emails[0]}"
                                        break
                        
                        # Create a mock forwarder with the contact info to send the email
                        from assistant.agent.email_agent import SimpleMessageForwarder
                        forwarder = SimpleMessageForwarder(
                            user_message=user_text,
                            chat_history=chat_history,
                            user_context={"timestamp": "now", "source": "chat"},
                            chat_summary=conversation_summary
                        )
                        
                        # Set the contact info we found
                        if user_contact:
                            forwarder.user_contact = user_contact
                        
                        # Call process_data which will handle confirmation and send email
                        result = forwarder.process_data()
                        
                        self.chat_history.add_interaction(user_text, result)
                        return {"agent": True, "message": result, "prompt_category": "email_confirmation"}
                    except Exception as e:
                        print(f"Email confirmation error: {str(e)}")
                        error_msg = self.config.get('response_messages', {}).get('email_error', 'Email error.')
                        self.chat_history.add_interaction(user_text, error_msg)
                        return {"agent": True, "message": error_msg, "prompt_category": "email_error"}
        
        # Load greetings from config
        greetings_config = self.config.get('greetings', {})
        english_greetings = greetings_config.get('english', [])
        czech_greetings = greetings_config.get('czech', [])
        
        if user_text_lower in english_greetings:
            greeting_msg = self.config.get('response_messages', {}).get('greeting_english', 'Welcome!')
            self.chat_history.add_interaction(user_text, greeting_msg)
            return {"agent": True, "message": greeting_msg, "prompt_category": "greeting"}
        elif user_text_lower in czech_greetings:
            greeting_msg = self.config.get('response_messages', {}).get('greeting_czech', 'Ahoj!')
            self.chat_history.add_interaction(user_text, greeting_msg)
            return {"agent": True, "message": greeting_msg, "prompt_category": "greeting"}
        
        try:
            classification_path = Path(__file__).parent.parent / 'data' / 'prompts' / 'classification.md'
            with open(classification_path, 'r', encoding='utf-8') as f:
                classify_prompt = f.read()
        except FileNotFoundError:
            classify_prompt = "You are a classification assistant. Classify user input appropriately."

        # Use an LLM to classify the request type
        messages = [
            {"role": "system", "content": classify_prompt},
            {"role": "user", "content": user_text}
        ]
        
        try:
            chat = ChatCompletion(
                messages=messages,
                temperature=0
            )
            # Use the async client to classify the request type
            prompt_category = await chat.get_response(lower=True)
            # Clean up markdown formatting and extra characters
            prompt_category = prompt_category.strip('*').strip()
            print(f"Prompt category: {prompt_category}")
            contexts = []
            
            # Get action mapping from config
            category_actions = self.config.get('category_actions', {})
            action = category_actions.get(prompt_category, 'technical_answer')
            
            # Handle based on action type
            if action == "summary":
                # Convert JSON data to list of strings for the summary case               
                for post_id, post_data in self.data.items():
                    document_parts = [
                        f"User: {post_data.get('user', '')}",
                        f"Text: {post_data.get('text', '')}",
                        f"Metadata: {post_data.get('metadata', '')}"
                        ]
                    contexts.append('\n\n'.join(document_parts))
            
            elif action == "forward_message":
                # Simple message forwarding - send user's message directly to Luděk
                try:
                    from assistant.agent.email_agent import send_simple_message
                    chat_history = self.chat_history.get_full_history()
                    # Generate accurate conversation summary
                    conversation_summary = self._generate_conversation_summary(user_text)
                    result = send_simple_message(
                        user_message=user_text,
                        chat_history=chat_history,
                        user_context={"timestamp": "now", "source": "chat"},
                        chat_summary=conversation_summary
                    )
                    self.chat_history.add_interaction(user_text, result)
                    return {"agent": True, "message": result, "contexts": contexts}
                except Exception as e:
                    print(f"Forward message error: {str(e)}")
                    error_msg = self.config.get('response_messages', {}).get('email_error', 'Email error.')
                    self.chat_history.add_interaction(user_text, error_msg)
                    return {"agent": True, "message": error_msg, "contexts": contexts}
            
            elif action == "agent_response":
                if prompt_category == "common_knowledge":
                    msg = self.config.get('response_messages', {}).get('common_knowledge', 'Outside scope.')
                elif prompt_category == "code":
                    msg = self.config.get('response_messages', {}).get('code', 'Cannot help with code.')
                else:
                    return self._handle_inappropriate_request(user_text)
                
                self.chat_history.add_interaction(user_text, msg)
                return {"agent": True, "message": msg}
            
            elif action == "clear_chat":
                self.chat_history.clear_history()
                msg = self.config.get('response_messages', {}).get('clear_chat', 'Chat cleared.')
                return {"agent": True, "message": msg}
            
            elif action == "ask_details":
                # Check if we already asked for details recently by looking at the last response
                last_interactions = self.chat_history.get_last_n_interactions(1)
                if last_interactions:
                    last_response = last_interactions[0].assistant_response
                    # Check if the last response was asking for more details
                    czech_ask_phrases = ['více o tom', 'více detailů', 'říct více', 'sdílet více']
                    english_ask_phrases = ['more details', 'tell me more', 'share more', 'more about']
                    
                    # Check for cybersecurity urgent technical details request
                    cybersecurity_urgent_phrases = ['doménu/url', 'domain/url', 'technologie/cms', 'technology/cms', 'kontaktní údaje', 'contact details', 'security incident', 'bezpečnostní incident', 'can help you fix this immediately', 'může okamžitě pomoci']
                    
                    is_asking_details = any(phrase in last_response.lower() for phrase in czech_ask_phrases + english_ask_phrases)
                    is_asking_cyber_details = any(phrase in last_response.lower() for phrase in cybersecurity_urgent_phrases)
                    
                    if is_asking_details or is_asking_cyber_details:
                        # User is providing details in response to our question
                        if is_asking_cyber_details:
                            # This is cybersecurity urgent details - treat as contact/forward message
                            from assistant.agent.language_detection_agent import detect_language
                            from assistant.agent.email_agent import send_simple_message
                            
                            # Check if user provided contact info in their response
                            import re
                            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                            emails = re.findall(email_pattern, user_text)
                            
                            if emails:
                                # User provided contact info along with technical details, forward immediately
                                try:
                                    chat_history = self.chat_history.get_full_history()
                                    conversation_summary = self._generate_conversation_summary(user_text)
                                    result = send_simple_message(
                                        user_message=user_text,
                                        chat_history=chat_history,
                                        user_context={"timestamp": "now", "source": "chat"},
                                        chat_summary=conversation_summary
                                    )
                                    self.chat_history.add_interaction(user_text, result)
                                    return {"agent": True, "message": result, "contexts": contexts}
                                except Exception as e:
                                    print(f"Cybersecurity urgent forward error: {str(e)}")
                                    error_msg = self.config.get('response_messages', {}).get('email_error', 'Email error.')
                                    self.chat_history.add_interaction(user_text, error_msg)
                                    return {"agent": True, "message": error_msg, "contexts": contexts}
                            else:
                                # Technical details provided but no contact info, ask for it
                                language = detect_language(user_text)
                                if language == 'cs':
                                    msg = "Děkuji za technické detaily! **Prosím uveďte ještě vaše kontaktní údaje (email nebo telefon)**, abych mohl předat všechny informace Luďkovi pro okamžitou pomoc."
                                else:
                                    msg = "Thank you for the technical details! **Please also provide your contact information (email or phone)** so I can forward all information to Luděk for immediate assistance."
                                
                                self.chat_history.add_interaction(user_text, msg)
                                return {"agent": True, "message": msg, "contexts": contexts}
                        else:
                            # Regular detail request - let AI handle
                            return {"agent": False, "message": user_text, "contexts": contexts}
                
                # Ask for more details about the service inquiry
                from assistant.agent.language_detection_agent import detect_language
                language = detect_language(user_text)
                
                service_questions = self.config.get('service_detail_questions', {})
                category_questions = service_questions.get(prompt_category, {})
                
                if language == 'cs':
                    msg = category_questions.get('czech', 'Můžete mi říct více detailů o vašem projektu?')
                else:
                    msg = category_questions.get('english', 'Could you tell me more details about your project?')
                
                self.chat_history.add_interaction(user_text, msg)
                return {"agent": True, "message": msg, "prompt_category": f"ask_details_{prompt_category}"}
            
            elif action == "handle_detailed_request":
                # User has provided detailed service requirements, let AI handle
                return {"agent": False, "message": user_text, "contexts": contexts}
            
            # Default: technical_answer - Let the main chat system handle
            return {"agent": False, "message": user_text, "contexts": contexts}

        except Exception as e:
            print(f"Error handling user request: {str(e)}")
            return user_text

