"""Chat endpoints for the API."""

from typing import Any, Dict, Union

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from assistant.api.dependencies import get_optional_user
from assistant.api.routes.chat_utils import handle_chat_endpoint
from assistant.core.models import UserInfo

router = APIRouter()


@router.options("/")
async def options_root() -> Response:
    """Handle OPTIONS request for CORS."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


@router.post("/", response_class=JSONResponse)
async def chat(
    request: Request, user: Union[UserInfo, None] = Depends(get_optional_user)
) -> Dict[str, Any]:
    """
    Chat endpoint compatible with DeepChat (optional authentication)

    Args:
        request: The incoming request with DeepChat format
        user: Current authenticated user (optional)

    Returns:
        JSON response in DeepChat-compatible format
    """
    return await handle_chat_endpoint(request)


@router.post("/public", response_class=JSONResponse)
async def chat_public(request: Request) -> Dict[str, Any]:
    """
    Public chat endpoint (no authentication required)

    This endpoint provides the same functionality as the main chat endpoint
    but without authentication requirements for backward compatibility.

    Args:
        request: The incoming request with DeepChat format

    Returns:
        JSON response in DeepChat-compatible format
    """
    return await handle_chat_endpoint(request)
