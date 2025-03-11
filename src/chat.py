from openai import AsyncOpenAI
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from src.vector_store import VectorStore
from src.chat_history import ChatHistory
import asyncio

# Create vector store as a global variable - but initialize it later
store = None

class AIService:
    def __init__(self, model_name: str = "gpt-4o-mini", 
                 max_history: int = 5, 
                 temperature: float = 0,
                 context_window: int = 3):
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
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_history = ChatHistory(max_history=max_history)
        
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
         return """
         You are a **sales assistant** specializing in providing information about **Luděk Kvapil** and his professional services. 
         
         ### **Response Guidelines:**
           - Only answer questions related to **Luděk Kvapil**, his **education**, **experience**, **skills**, and **services**.
           - Never add information that is not present in the **knowledge base**.
           - If user ask for hourly rate, respond with your knowledge about  man day rate.  
           - If the question is unrelated, respond: *"I'm sorry, but I can only provide information about Luděk Kvapil and his work."*
           - You may also answer general questions about **Drupal, AWS, OpenTofu**, and other **relevant technologies** explicitly defined by keywords in the knowledge base.
           - Detect the user's language and respond in the same language.
           - **Pronoun handling:**
             - "Ludek," "Luděk," "he," "she," or "they" always refer to **Luděk Kvapil**.

        #### **Knowledge Base:**
           {knowledge_base}

        Respond in **Markdown format** for better readability.
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
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            raise ValueError("No response generated")
            
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
        if user_request == "History cleared":
            return user_request
        
        # Get similar texts - ensure we await the async call
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
            
        return answer
    
    async def handle_user_request(self, user_text: str) -> str:
        """
        Handle special user requests like clearing chat history.

        Args:
            user_text: User's input text.

        Returns:
            Response string based on the request.
        """
        # Use an LLM to classify the request type
        messages = [
            {"role": "system", "content": "You are a helpful assistant. If the user asks to clear chat history, respond with 'clear history.' For all other requests, return 'uncategorized.'."},
            {"role": "user", "content": user_text}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )

            prompt_category = response.choices[0].message.content.lower()
            print(f"Prompt category: {prompt_category}")
            
            # Check if the response classifies it as a clear history request
            if "clear history" in prompt_category:
                self.chat_history.clear_history()
                return "History cleared"
            
            # Default response if no match
            return user_text
        except Exception as e:
            print(f"Error handling user request: {str(e)}")
            return user_text

# Usage example
if __name__ == "__main__":
    async def main():
        ai_service = AIService()
        response = await ai_service.chat("What are the best investment strategies for beginners?")
        print(response)
    
    asyncio.run(main())