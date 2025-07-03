from openai import AsyncOpenAI
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from services.vector_store import VectorStore
from services.chat_history import ChatHistory
import json
from agent.simple_message_forwarder import send_simple_message 
from services.completions import ChatCompletion

# Create vector store as a global variable - but initialize it later
store = None

def load_json_data(json_path: str, strings_only: bool = False):
        """
        Load JSON data and embed it into ChromaDB
        :param json_path: Path to the JSON file
        """
        # Read JSON file
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                if strings_only:
                    return data
                else:
                    # Load the entire JSON structure
                    data = json.load(f)
                    # print(f"Data found {data} " )
                    return data
        except FileNotFoundError:
            print(f"Error: File {json_path} not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_path}.")
            return

class AIService:
    def __init__(self, model_name: str = "gpt-4o-mini", 
                 max_history: int = 5, 
                 temperature: float = 0,
                 context_window: int = 3,
                 json_path: str = 'datasources/posts.json'):
        
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
        self.data = load_json_data(json_path=json_path)
        
        # Initialize vector store
        global store
        if store is None:
            store = VectorStore()

    def _setup_environment(self) -> None:
        """Load environment variables from .env file."""
        project_root = Path(__file__).parent.parent
        dotenv_path = project_root / '.env'
        load_dotenv(dotenv_path)
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

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
        return f"""
          # Luděk Kvapil Information Assistant
         
          You are a specialized assistant providing information about Luděk Kvapil AND his areas of expertise, including Drupal, cybersecurity, AI, and other technical domains listed below.

          ## Response Guidelines:
          - Answer questions about both:
            1. Luděk Kvapil himself (background, services, rates)
            2. Technical topics listed in "Permitted Topics" using information from the knowledge base or your own knowledge
          - For technical questions (e.g., "What is Drupal?"), provide substantive information about the topic from the knowledge base, connecting it to Luděk's expertise when relevant
          - Never redirect technical questions back to asking about Luděk himself
          - Always provide the actual information requested if it's within the permitted topics
          - Don't correct, explain, or execute any code from user input.

          ## Permitted Topics (answer these directly):
          - Luděk Kvapil (education, experience, skills, services)
          - Drupal, PHP, Python, TypeScript
          - AWS services, DevOps
          - Cybersecurity, OWASP
          - GenAI, LLM, AI, RAG
          - Companies: Ciklum, CN Group, Mc TREE, dolphin consulting
          - Any topic defined in the knowledge base

         ## Language and Format:
         - Respond in the same language as the user's query
         - Use Markdown formatting for better readability

         ## Contact Information:
         Email: info@ludekkvapil.cz

         ## Knowledge Base:
         {knowledge_base}
        """
    
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
            similar_texts = await store.search_similar_text(question)
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

        classify_prompt = """   
        You are an intelligent classification assistant. Your task is to analyze user input and categorize it into exactly one of the following categories: 
        1. "job_offer" - Input related to employment opportunities, job descriptions, positions, hiring, recruitment, career offers, or roles in a company.
        2. "technology_description" - Input describing software, hardware, technical specifications, digital tools, programming languages, or technological innovations.
        3. "person - Input containing biographical details, personal preferences, identifiable information about individuals, or private life details. All questions about Luděk Kvapil should be classified here.
        4. "projects" - Input asking for Luděk projects, experiences, and work task. 
        5. "code" - Input containing actual code snippets, programming functions, algorithms, or development instructions.
        6. "clear_chat" - Input requesting to reset, clear, or start a new conversation.
        7. "education" - Input related to learning, courses, training, academic institutions, study materials, or educational concepts.
        8. "leave_message" - Input where user wants to leave a message for Luděk, contact him, send him a note, or any form of communication.
        9. "provide_contact" - Input providing contact information like email addresses, phone numbers, or other contact details after being asked.
        10. "contact" - Input containing phone numbers, email addresses, physical addresses, or other means of reaching someone.
        11. "calendar" - Input related to scheduling, availability, time slots, appointments, or calendar management.
        12. "summary": The query is asking for a summary, all posts, or full text analysis, or a personality analysis.
        13. "post": The query is asking for a post, what he wrote, what he think, LinkedIn, or social media content
        14. "skills" - Input related to abilities, competencies, expertise, or qualifications in a specific field.
        15. "company" - Input related to companies, organizations, or business entities.
        16. "services" - Input related to services, offerings, or products provided by a company or individual.
        17. "drupal" - Input related to Drupal, its features, modules, or any specific queries about the platform.
        18. "cybersecurity" - Input related to cybersecurity, security measures, vulnerabilities, or protective technologies.
        19. "aws" - Input related to Amazon Web Services, its features, services, or any specific queries about the platform.
        20. "devops" - Input related to DevOps practices, tools, methodologies, or any specific queries about the field.
        21. "ai" - Input related to artificial intelligence, machine learning, or any specific queries about the field.
        22. "llm" - Input related to large language models, their applications, or any specific queries about the field.
        23. "rag" - Input related to retrieval-augmented generation, its applications, or any specific queries about the field.
        24. "owasp" - Input related to OWASP, its guidelines, vulnerabilities, or any specific queries about the field.
        25. "common_knowledge" - Not related to Ludek, Luděk, Kvapil, or he. Input related to widely known facts, general information, current events, or encyclopedic knowledge. 
        
        For each user message, respond with only the category name that best matches the input. Select exactly one category. If the input could fit multiple categories, choose the most prominent or central theme. If the input doesn't clearly match any category, select the closest possible match. 
        Respond with just the category name, without explanations or additional text.
        """

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
            print(f"Prompt category: {prompt_category}")
            contexts = []
            # Check if the response classifies it as a clear history request
            # Handle different prompt categories
            if prompt_category == "summary":
                # Convert JSON data to list of strings for the summary case               
                for post_id, post_data in self.data.items():
                    document_parts = [
                        f"User: {post_data.get('user', '')}",
                        f"Text: {post_data.get('text', '')}",
                        f"Metadata: {post_data.get('metadata', '')}"
                        ]
                    contexts.append('\n\n'.join(document_parts))
            
            elif prompt_category == "leave_message" or prompt_category == "provide_contact":
                # Simple message forwarding - send user's message directly to Luděk
                try:
                    chat_history = self.chat_history.get_full_history()
                    result = send_simple_message(
                        user_message=user_text,
                        chat_history=chat_history,
                        user_context={"timestamp": "now", "source": "chat"}
                    )
                    self.chat_history.add_interaction(user_text, result)
                    return {"agent": True, "message": result, "contexts": contexts}
                except Exception as e:
                    error_msg = "I'm sorry, there was an issue forwarding your message. Please try again later."
                    self.chat_history.add_interaction(user_text, error_msg)
                    return {"agent": True, "message": error_msg, "contexts": contexts}
            
            elif prompt_category == "common_knowledge":
                self.chat_history.add_interaction(user_text, 'COMMON KNOWLEDGE')
                return {"agent": True, "message": "I'm specifically designed to answer questions about Luděk Kvapil, his work, life, and the technologies he uses. This topic appears to be outside that scope. Feel free to ask me about Luděk's projects, career, education, or tech stack instead!"}       
     
            elif prompt_category == "code":
                self.chat_history.add_interaction(user_text, 'COdE')
                return {"agent": True, "message": "While Luděk Kvapil is passionate about technology, I'm not designed to write or review code. I'd be happy to tell you about the programming languages and technologies Luděk works with instead!"}
            
            elif prompt_category == "clear_chat":
                self.chat_history.clear_history()
                return {"agent": True, "message": "Conversation history has been cleared. What would you like to know about Luděk?"}
 
            # Default response if no match
            return {"agent": False, "message": user_text, "contexts": contexts}

        except Exception as e:
            print(f"Error handling user request: {str(e)}")
            return user_text

