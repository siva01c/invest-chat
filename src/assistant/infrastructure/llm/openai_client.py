"""OpenAI client implementation for LLM interactions."""

import os
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from assistant.core.interfaces.infrastructure import ILLMClient


class ChatCompletion:
    """OpenAI chat completion client."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        messages: List[Dict[str, str]] = None,
        temperature: float = 0,
        json_response: bool = False,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the chat completion client."""
        self.model = model
        self.messages = messages or []
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.json_response = json_response
        self.max_tokens = max_tokens

    async def get_response(self, lower: bool = False) -> str:
        """Get response from OpenAI API."""
        completion_args = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
        }

        if self.json_response:
            completion_args["response_format"] = {"type": "json_object"}

        if self.max_tokens:
            completion_args["max_tokens"] = self.max_tokens

        response = await self.client.chat.completions.create(**completion_args)

        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            if lower:
                return content.lower()
            else:
                return content
        raise ValueError("No response generated")


class OpenAIClient(ILLMClient):
    """OpenAI client implementing the ILLMClient interface."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client."""
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "OpenAIClient"

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion using OpenAI API."""
        completion_args = {
            "model": model or "gpt-4o-mini",
            "messages": messages,
            "temperature": temperature if temperature is not None else 0,
        }

        if max_tokens:
            completion_args["max_tokens"] = max_tokens

        response = await self.client.chat.completions.create(**completion_args)

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content or ""

        raise ValueError("No response generated")

    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for text using OpenAI API."""
        response = await self.client.embeddings.create(
            model=model or "text-embedding-ada-002", input=text
        )

        if response.data and len(response.data) > 0:
            return response.data[0].embedding

        raise ValueError("No embedding generated")
