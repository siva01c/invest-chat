"""Investment Chat API endpoints."""

import json
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from assistant.core.services.chat_service import AIService

router = APIRouter(prefix="/api/chat", tags=["chat"])
ai_service = AIService()


class ChatRequest(BaseModel):
    message: str = Field(..., description="Dotaz uživatele")
    messages: list = Field(default=[], description="Historie zpráv pro kompatibilitu UI")


class ChatResponse(BaseModel):
    response: str
    text: str


@router.post("", response_class=JSONResponse)
@router.post("/", response_class=JSONResponse)
async def chat_endpoint(payload: ChatRequest):
    """Synchronní chat endpoint pro generování odpovědí."""
    user_msg = payload.message
    if not user_msg and payload.messages:
        # Extract last user message if using messages list format
        for m in reversed(payload.messages):
            if isinstance(m, dict) and m.get("role") == "user":
                user_msg = m.get("content", "")
                break

    if not user_msg:
        raise HTTPException(status_code=400, detail="Zprávu nelze odeslat prázdnou.")

    response_text = await ai_service.chat(user_msg)
    return {"response": response_text, "text": response_text}


@router.post("/stream")
@router.get("/stream")
async def chat_stream_endpoint(request: Request, message: str = ""):
    """Streamovací endpoint pro Server-Sent Events (SSE)."""
    user_msg = message
    if not user_msg and request.method == "POST":
        try:
            body = await request.json()
            user_msg = body.get("message", "")
        except Exception:
            pass

    if not user_msg:
        raise HTTPException(status_code=400, detail="Dotaz nesmí být prázdný.")

    async def event_generator() -> AsyncGenerator[str, None]:
        async for token in ai_service.stream_chat(user_msg):
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
