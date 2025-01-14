from collections import deque
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ChatEntry:
    """Represents a single chat interaction with user message and assistant response"""
    user_message: str
    assistant_response: str
    timestamp: Optional[float] = None

class ChatHistory:
    def __init__(self, max_history: int = 10):
        """
        Initialize chat history with maximum number of entries to store
        
        Args:
            max_history (int): Maximum number of chat interactions to keep in history
        """
        self.max_history = max_history
        self.history: deque[ChatEntry] = deque(maxlen=max_history)
    
    def add_interaction(self, user_message: str, assistant_response: str) -> None:
        """
        Add a new chat interaction to the history
        
        Args:
            user_message (str): The message from the user
            assistant_response (str): The response from the assistant
        """
        entry = ChatEntry(
            user_message=user_message,
            assistant_response=assistant_response
        )
        self.history.append(entry)
    
    def get_user_messages(self) -> List[str]:
        """Get all user messages in chronological order"""
        return [entry.user_message for entry in self.history]
    
    def get_assistant_responses(self) -> List[str]:
        """Get all assistant responses in chronological order"""
        return [entry.assistant_response for entry in self.history]
    
    def get_last_n_interactions(self, n: int) -> List[ChatEntry]:
        """
        Get the last n chat interactions
        
        Args:
            n (int): Number of interactions to retrieve
            
        Returns:
            List of the last n ChatEntry objects
        """
        return list(self.history)[-n:]
    
    def clear_history(self) -> None:
        """Clear all chat history"""
        self.history.clear()
    
    def get_full_history(self) -> List[ChatEntry]:
        """Get all chat interactions as a list"""
        return list(self.history)
    
    def __len__(self) -> int:
        """Return the number of interactions in history"""
        return len(self.history)
    
    def __str__(self) -> str:
        """String representation of the chat history"""
        if not self.history:
            return "Chat history is empty"
        
        output = []
        for i, entry in enumerate(self.history, 1):
            output.append(f"Interaction {i}:")
            output.append(f"User: {entry.user_message}")
            output.append(f"Assistant: {entry.assistant_response}")
            output.append("")
        
        return "\n".join(output)