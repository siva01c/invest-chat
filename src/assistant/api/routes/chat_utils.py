"""Shared utilities for chat API endpoints."""

import uuid
from json import JSONDecodeError
from typing import Any, Dict

from fastapi import HTTPException

from assistant.api.middleware.security import sanitize_input
from assistant.core.config.services import get_chat_service
from assistant.core.models import ChatRequest


async def process_chat_request(request_data) -> Dict[str, Any]:
    """
    Process a chat request with common validation and handling logic.

    Args:
        request_data: The parsed JSON request data

    Returns:
        Dictionary with the chat response in DeepChat-compatible format

    Raises:
        HTTPException: For validation or processing errors
    """
    # Normalize incoming shapes:
    # - If client sent a list of messages -> wrap into {'messages': [...]}
    # - If client sent a single message dict (e.g. {'type':'message','text':...}) -> wrap
    # - Otherwise expect the ChatRequest shape
    if isinstance(request_data, list):
        request_data = {"messages": request_data}
    elif isinstance(request_data, dict) and "messages" not in request_data:
        if (
            any(k in request_data for k in ("text", "content"))
            or request_data.get("type") == "message"
        ):
            request_data = {"messages": [request_data]}

    # Validate request data using Pydantic models
    try:
        validated_request = ChatRequest(**request_data)
    except Exception as e:
        # For Pydantic v2 the exception message can be structured; return as validation detail
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

    if not validated_request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Get the last message content with sanitization
    last_message_obj = validated_request.messages[-1]
    last_message = last_message_obj.content or last_message_obj.text or ""

    if not last_message.strip():
        raise HTTPException(status_code=400, detail="Empty message content")

    # Additional sanitization
    last_message = sanitize_input(last_message)

    # Extract or generate session ID
    session_id = validated_request.session_id or str(uuid.uuid4())

    # Get the chat service from dependency injection container
    ai_service = get_chat_service()

    # Process the message using the chat service
    response = await ai_service.chat(last_message, session_id=session_id)

    # Return in DeepChat-compatible format
    return {"text": response}


async def handle_chat_endpoint(request) -> Dict[str, Any]:
    # type: ignore[no-untyped-def]
    """
    Common handler for chat endpoints with standardized error handling.

    Args:
        request: The FastAPI request object

    Returns:
        JSON response dictionary

    Raises:
        HTTPException: For known errors
    """
    try:
        # Parse the incoming JSON request
        try:
            data = await request.json()
        except JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        # Process the request using shared logic
        return await process_chat_request(data)

    except HTTPException:
        # Re-raise known HTTP errors
        raise
    except Exception as e:
        # Log unexpected errors and return generic error
        import logging

        logging.getLogger(__name__).error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
