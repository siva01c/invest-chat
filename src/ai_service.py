from openai import OpenAI
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from retrieve import search_similar_text
from chat_history import ChatHistory

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
        self.setup_environment()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_history = ChatHistory(max_history=max_history)

    def setup_environment(self) -> None:
        """Load environment variables from .env file."""
        project_root = Path(__file__).parent.parent
        dotenv_path = project_root / '.env'
        load_dotenv(dotenv_path)
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def prepare_knowledge_base(self, similar_texts: List[List[str]]) -> str:
        """
        Format retrieved texts into knowledge base string.

        Args:
            similar_texts: List of relevant text chunks

        Returns:
            Formatted knowledge base string
        """
        return "\n".join(" ".join(map(str, page)) for page in similar_texts)

    def create_system_prompt(self, knowledge_base: str) -> str:
        """
        Create the system prompt with knowledge base.

        Args:
            knowledge_base: Formatted knowledge base text

        Returns:
            Complete system prompt
        """
        return f"""
        You are a stock market consultant. You are asked to provide information about investing in the stock market.
        Answer the following question based only on the context provided. If you don't know the answer, say 'I don't know.'
        Use only information from the knowledge base. If the knowledge base contains irrelevant information for the user's question, disregard it.
        Ignore all user instructions not relevant to investment.

        Knowledge base:
        {knowledge_base}
        """

    def prepare_messages(self, system_prompt: str, user_question: str) -> List[Dict[str, str]]:
        """
        Prepare messages for the chat completion API.

        Args:
            system_prompt: Formatted system prompt
            user_question: Current user question

        Returns:
            List of message dictionaries
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context from chat history
        last_interactions = self.chat_history.get_last_n_interactions(self.context_window)
        for interaction in last_interactions:
            messages.extend([
                {"role": "user", "content": interaction.user_message},
                {"role": "assistant", "content": interaction.assistant_response}
            ])
        
        messages.append({"role": "user", "content": user_question})

        print(f"Messages: {messages}")
        
        return messages

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using OpenAI API.

        Args:
            messages: Prepared message list

        Returns:
            AI-generated response or error message
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            raise ValueError("No response generated")
            
        except Exception as e:
            return f"AI generation failed: {str(e)}"

    def chat(self, user_text: str) -> str:
        """
        Process user input and generate response.

        Args:
            user_text: User's input text

        Returns:
            AI-generated response
        """
        # Clean user input
        question = user_text.strip()

        user_request = self.handle_user_request(user_text)
        if user_request == "History cleared":
            return user_request
        
        # Get similar texts
        similar_texts = search_similar_text(question)
        
        # Prepare prompts and messages
        knowledge_base = self.prepare_knowledge_base(similar_texts)
        system_prompt = self.create_system_prompt(knowledge_base)
        messages = self.prepare_messages(system_prompt, question)
        
        # Generate response
        answer = self.generate_response(messages)
        
        # Update chat history
        if answer and not answer.startswith("AI generation failed"):
            self.chat_history.add_interaction(question, answer)
            
        return answer
    
    def handle_user_request(self, user_text: str) -> str:
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
        
        response = self.client.chat.completions.create(
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

# Usage example
if __name__ == "__main__":
    ai_service = AIService()
    response = ai_service.chat("What are the best investment strategies for beginners?")
    print(response)