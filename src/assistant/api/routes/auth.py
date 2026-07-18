"""Authentication endpoints for JWT-based authentication."""

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from assistant.core.models import LoginRequest, TokenResponse, TokenValidationResponse, UserInfo
from assistant.services.jwt_service import JWTService

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

# Initialize JWT service
jwt_service = JWTService()

# Simple user store - in production, use a proper database
USERS = {
    "ludek": {
        "username": "ludek",
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),  # Default password
        "user_id": "ludekkvapil",
        "role": "admin",
    }
}

# Allow override from environment
if os.getenv("AUTH_USERNAME") and os.getenv("AUTH_PASSWORD"):
    USERS[os.getenv("AUTH_USERNAME")] = {
        "username": os.getenv("AUTH_USERNAME"),
        "password_hash": hashlib.sha256(os.getenv("AUTH_PASSWORD").encode()).hexdigest(),
        "user_id": os.getenv("AUTH_USERNAME"),
        "role": "admin",
    }


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hashlib.sha256(plain_password.encode()).hexdigest() == password_hash


def authenticate_user(username: str, password: str) -> Dict[str, Any] | None:
    """Authenticate a user with username and password."""
    user = USERS.get(username)
    if not user:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return user


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """
    Authenticate user and return JWT token.

    Args:
        request: Login credentials

    Returns:
        JWT token response

    Raises:
        HTTPException: If credentials are invalid
    """
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate JWT token
    token_payload = {"user_id": user["user_id"], "username": user["username"], "role": user["role"]}

    expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    token = jwt_service.generate_token(token_payload, expiration_minutes=expiration_hours * 60)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expiration_hours * 3600,  # Convert to seconds
    )


@router.post("/validate", response_model=TokenValidationResponse)
async def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenValidationResponse:
    """
    Validate a JWT token.

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        Token validation response
    """
    try:
        payload = jwt_service.validate_token(credentials.credentials)

        return TokenValidationResponse(
            valid=True,
            user_id=payload.get("user_id"),
            expires_at=datetime.fromtimestamp(payload.get("exp", 0), timezone.utc).isoformat(),
        )
    except Exception as e:
        return TokenValidationResponse(valid=False, user_id=None, expires_at=None)


@router.get("/me", response_model=UserInfo)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInfo:
    """
    Get current user information from JWT token.

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        Current user information

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt_service.validate_token(credentials.credentials)

        return UserInfo(
            user_id=payload.get("user_id"),
            username=payload.get("username"),
            role=payload.get("role", "user"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_dependency(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInfo:
    """
    Dependency to get current authenticated user.

    This can be used as a dependency in protected endpoints.

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        Current user information

    Raises:
        HTTPException: If token is invalid or expired
    """
    return await get_current_user(credentials)
