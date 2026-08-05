"""Main investment chat service implementation."""

import os
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from assistant.services.chat_history import ChatHistory
from assistant.services.vector_store import VectorStore

load_dotenv()


class AIService:
    """Main AI service for handling investment chat interactions."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_history: int = 5,
        temperature: float = 0.2,
        context_window: int = 3,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.context_window = context_window
        self.chat_history = ChatHistory(max_history=max_history)
        self.vector_store = VectorStore(collection_name="investment_knowledge")
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _load_system_prompt(self, knowledge_base_text: str) -> str:
        """Load system prompt template and format with retrieved RAG context."""
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "prompts"
            / "system_prompt.md"
        )
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            return template.format(knowledge_base=knowledge_base_text)
        else:
            return (
                "Jsi Invest Chat AI Asistent pro téma osobních financí a investic.\n"
                f"Znalostní báze:\n{knowledge_base_text}"
            )

    def _prepare_messages(
        self, system_prompt: str, user_question: str
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]

        last_interactions = self.chat_history.get_last_n_interactions(self.context_window)
        for interaction in last_interactions:
            messages.extend(
                [
                    {"role": "user", "content": interaction.user_message},
                    {"role": "assistant", "content": interaction.assistant_response},
                ]
            )

        messages.append({"role": "user", "content": user_question})
        return messages

    async def chat(self, user_text: str) -> str:
        """Process user input, retrieve RAG context, and generate assistant response."""
        question = user_text.strip()
        if not question:
            return "Prosím zadejte dotaz k investování."

        # Retrieve RAG context
        contexts = await self.vector_store.retrieve_context(question, top_k=3)
        knowledge_base_text = (
            "\n\n".join(contexts) if contexts else "Žádné specifické podrobnosti nenalezeny."
        )

        system_prompt = self._load_system_prompt(knowledge_base_text)
        messages = self._prepare_messages(system_prompt, question)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            answer = response.choices[0].message.content or ""
            self.chat_history.add_interaction(question, answer)
            return answer

        except Exception as e:
            return f"Omlouvám se, při generování odpovědi došlo k chybě: {e}"

    async def stream_chat(self, user_text: str) -> AsyncGenerator[str, None]:
        """Stream assistant response tokens for real-time SSE UI."""
        question = user_text.strip()
        if not question:
            yield "Prosím zadejte dotaz k investování."
            return

        contexts = await self.vector_store.retrieve_context(question, top_k=3)
        knowledge_base_text = (
            "\n\n".join(contexts) if contexts else "Žádné specifické podrobnosti nenalezeny."
        )

        system_prompt = self._load_system_prompt(knowledge_base_text)
        messages = self._prepare_messages(system_prompt, question)

        full_response = []
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response.append(token)
                    yield token

            answer = "".join(full_response)
            self.chat_history.add_interaction(question, answer)

        except Exception as e:
            yield f"\n[Chyba při komunikaci s LLM: {e}]"
