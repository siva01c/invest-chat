"""Authentication dependencies for FastAPI endpoints."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from assistant.core.models import UserInfo
from assistant.services.jwt_service import JWTService

# Initialize security and JWT service
security = HTTPBearer()
jwt_service = JWTService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInfo:
    """
    Get current authenticated user from JWT token.

    This dependency can be used to protect endpoints requiring authentication.

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        Current user information

    Raises:
        HTTPException: If token is invalid, expired, or missing

    Example:
        ```python
        @router.get("/protected")
        async def protected_endpoint(user: UserInfo = Depends(get_current_user)):
            return {"message": f"Hello {user.username}"}
        ```
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


async def get_current_admin_user(user: UserInfo = Depends(get_current_user)) -> UserInfo:
    """
    Get current authenticated admin user.

    This dependency ensures the user has admin role.

    Args:
        user: Current authenticated user

    Returns:
        Current admin user information

    Raises:
        HTTPException: If user is not an admin

    Example:
        ```python
        @router.get("/admin-only")
        async def admin_endpoint(user: UserInfo = Depends(get_current_admin_user)):
            return {"message": f"Hello admin {user.username}"}
        ```
    """
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserInfo | None:
    """
    Get current user if authenticated, otherwise None.

    This dependency allows endpoints to work with or without authentication.

    Args:
        credentials: Bearer token from Authorization header (optional)

    Returns:
        Current user information if authenticated, None otherwise

    Example:
        ```python
        @router.get("/optional-auth")
        async def optional_auth_endpoint(user: UserInfo | None = Depends(get_optional_user)):
            if user:
                return {"message": f"Hello {user.username}"}
            else:
                return {"message": "Hello anonymous user"}
        ```
    """
    try:
        if not credentials:
            return None

        payload = jwt_service.validate_token(credentials.credentials)

        return UserInfo(
            user_id=payload.get("user_id"),
            username=payload.get("username"),
            role=payload.get("role", "user"),
        )
    except Exception:
        return None
