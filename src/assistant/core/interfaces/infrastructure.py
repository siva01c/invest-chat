"""Infrastructure service interfaces."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .base import BaseService


class IVectorStore(BaseService):
    """Interface for vector store operations."""

    @abstractmethod
    async def store_embeddings(self, documents: List[Any]) -> None:
        """Store documents using embeddings."""

    @abstractmethod
    async def search_similar_text(self, search_text: str, n_results: int = 3) -> List[Any]:
        """Search for similar text asynchronously."""

    @abstractmethod
    async def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant context for a given query."""

    @abstractmethod
    async def get_all_records(self) -> List[Any]:
        """Get all records from the collection."""


class ILLMClient(BaseService):
    """Interface for Large Language Model client."""

    @abstractmethod
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using LLM."""

    @abstractmethod
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for text."""


class IConfigurationManager(BaseService):
    """Interface for configuration management."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get main configuration."""

    @abstractmethod
    def get_translations(self, language_code: str) -> Dict[str, Any]:
        """Get translations for specified language."""

    @abstractmethod
    def get_prompt(self, filename: str) -> str:
        """Get prompt content."""

    @abstractmethod
    def get_data(self, filename: str) -> Dict[str, Any]:
        """Get JSON data."""

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear configuration cache."""
