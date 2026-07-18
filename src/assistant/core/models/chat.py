"""Chat-related Pydantic models for data validation."""

import html
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from assistant.config import get_settings

# Get settings instance for configuration
settings = get_settings()

# Configuration constants from settings
MAX_MESSAGE_LENGTH = settings.max_message_length
MAX_MESSAGES_PER_REQUEST = settings.max_messages_per_request
MAX_SESSION_ID_LENGTH = settings.max_session_id_length


class MessageModel(BaseModel):
    """Individual message model with content validation."""

    content: Optional[str] = Field(None, max_length=MAX_MESSAGE_LENGTH)
    text: Optional[str] = Field(None, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("content", "text")
    @classmethod
    def sanitize_content(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize message content for security."""
        if v is None:
            return v
        # Basic HTML sanitization and cleanup
        sanitized = html.escape(v.strip())
        # Remove potential script injections
        sanitized = re.sub(
            r"<script[^>]*>.*?</script>", "", sanitized, flags=re.IGNORECASE | re.DOTALL
        )
        return sanitized


class ChatRequest(BaseModel):
    """Chat request model with message validation."""

    messages: List[MessageModel] = Field(..., min_items=1, max_items=MAX_MESSAGES_PER_REQUEST)
    session_id: Optional[str] = Field(None, max_length=MAX_SESSION_ID_LENGTH)

    @field_validator("session_id")
    @classmethod
    def sanitize_session_id(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize session ID to allow only safe characters."""
        if v is None:
            return v
        # Only allow alphanumeric and hyphens for session ID
        sanitized = re.sub(r"[^a-zA-Z0-9\-]", "", v)
        return sanitized[:MAX_SESSION_ID_LENGTH]


class ChatResponse(BaseModel):
    """Chat response model."""

    text: str
    session_id: Optional[str] = None
    metadata: Optional[dict] = None


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
    timestamp: int
    version: str


class KnowledgeBaseResponse(BaseModel):
    """Knowledge base response model."""

    message: List[dict]
