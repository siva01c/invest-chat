"""Investment Chat API endpoints."""

import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from assistant.core.services.chat_service import AIService

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Per-session AIService registry — each browser session gets its own chat history.
# Keys are session IDs (UUID strings set via a cookie).
_session_registry: dict[str, AIService] = {}

SESSION_COOKIE_NAME = "invest_chat_session"


def _get_or_create_session(session_id: str | None) -> tuple[str, AIService]:
    """Return (session_id, AIService) for the given session ID.

    Creates a new session (and a new AIService with isolated history) when the
    session_id is unknown or absent.
    """
    if session_id and session_id in _session_registry:
        return session_id, _session_registry[session_id]

    new_id = str(uuid.uuid4())
    _session_registry[new_id] = AIService()
    return new_id, _session_registry[new_id]


class ChatRequest(BaseModel):
    message: str = Field(..., description="Dotaz uživatele")


class ChatResponse(BaseModel):
    response: str
    text: str


@router.post("", response_class=JSONResponse)
@router.post("/", response_class=JSONResponse)
async def chat_endpoint(
    payload: ChatRequest,
    response: Response,
    invest_chat_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Synchronní chat endpoint — každý uživatel má izolovanou historii."""
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Zprávu nelze odeslat prázdnou.")

    session_id, ai_service = _get_or_create_session(invest_chat_session)
    # Refresh / set the session cookie on every response.
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=3600 * 24,  # 24 hours
    )

    response_text = await ai_service.chat(user_msg)
    return {"response": response_text, "text": response_text}


@router.post("/stream")
@router.get("/stream")
async def chat_stream_endpoint(
    request: Request,
    message: str = "",
    invest_chat_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Streamovací endpoint pro Server-Sent Events (SSE) s per-session historií."""
    user_msg = message.strip()
    if not user_msg and request.method == "POST":
        try:
            body = await request.json()
            user_msg = body.get("message", "").strip()
        except Exception:
            pass

    if not user_msg:
        raise HTTPException(status_code=400, detail="Dotaz nesmí být prázdný.")

    session_id, ai_service = _get_or_create_session(invest_chat_session)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for token in ai_service.stream_chat(user_msg):
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    streaming_response = StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",   # Prevents nginx from buffering SSE chunks
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    # Set session cookie on SSE response as well.
    streaming_response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=3600 * 24,
    )
    return streaming_response
