import os

from openai import AsyncOpenAI


class ChatCompletion:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        messages: list = {},
        temperature: float = 0,
        json_response: bool = False,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.json_response = json_response

    async def get_response(self, lower=False):
        completion_args = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
        }

        if self.json_response:
            completion_args["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**completion_args)

        if response.choices and response.choices[0].message:
            if lower:
                return response.choices[0].message.content.lower()
            else:
                return response.choices[0].message.content
        raise ValueError("No response generated")
