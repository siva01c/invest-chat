"""Authentication models for data validation and serialization."""

from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """Request model for user login."""

    username: str = Field(
        ..., min_length=1, max_length=50, description="Username for authentication"
    )
    password: str = Field(
        ..., min_length=1, max_length=100, description="Password for authentication"
    )


class TokenResponse(BaseModel):
    """Response model for token generation."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class TokenValidationResponse(BaseModel):
    """Response model for token validation."""

    valid: bool = Field(..., description="Whether the token is valid")
    user_id: Optional[str] = Field(None, description="User ID if token is valid")
    expires_at: Optional[str] = Field(None, description="Token expiration timestamp")


class UserInfo(BaseModel):
    """User information model."""

    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    role: str = Field(default="user", description="User role")
