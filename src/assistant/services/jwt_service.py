import datetime
import os
from typing import Any, Dict, Optional

import jwt
from dotenv import load_dotenv

load_dotenv()


class JWTService:
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        """
        Initialize JWT service with secret key from environment or parameter.

        Args:
            secret_key: JWT secret key (optional, will use JWT_SECRET env var if not provided)
            algorithm: JWT algorithm (default: HS256)

        Raises:
            ValueError: If no secret key is provided and JWT_SECRET env var is not set
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET") or os.getenv("JWT_SECRET_KEY")
        if not self.secret_key:
            raise ValueError(
                "JWT secret key must be provided either as parameter or JWT_SECRET environment variable"
            )
        self.algorithm = algorithm

    def generate_token(self, payload: Dict[str, Any], expiration_minutes: int = 30) -> str:
        """
        Generate a JWT token with the given payload and expiration time.

        Args:
            payload: The payload to encode in the token
            expiration_minutes: Token expiration time in minutes (default: 30)

        Returns:
            JWT token string

        Raises:
            ValueError: If payload is empty or invalid
        """
        if not payload:
            raise ValueError("Payload cannot be empty")

        # Create a copy of payload to avoid modifying the original
        token_payload = payload.copy()
        token_payload["exp"] = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            minutes=expiration_minutes
        )
        token_payload["iat"] = datetime.datetime.now(datetime.timezone.utc)

        try:
            token = jwt.encode(token_payload, self.secret_key, algorithm=self.algorithm)
            return token
        except Exception as e:
            raise ValueError(f"Failed to generate token: {str(e)}")

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate and decode a JWT token.

        Args:
            token: JWT token string to validate

        Returns:
            Decoded token payload

        Raises:
            jwt.ExpiredSignatureError: If token has expired
            jwt.InvalidTokenError: If token is invalid
            ValueError: If token is empty or malformed
        """
        if not token:
            raise ValueError("Token cannot be empty")

        try:
            decoded_token = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return decoded_token
        except jwt.ExpiredSignatureError:
            raise jwt.ExpiredSignatureError("Token has expired")
        except jwt.InvalidTokenError:
            raise jwt.InvalidTokenError("Invalid token")
        except Exception as e:
            raise ValueError(f"Failed to validate token: {str(e)}")


# Example usage:
if __name__ == "__main__":
    try:
        # Initialize JWT service (will use JWT_SECRET_KEY environment variable)
        jwt_service = JWTService()

        # Generate token
        payload = {"user_id": "ludekkvapil", "role": "user"}
        token = jwt_service.generate_token(payload, expiration_minutes=60)
        print(f"Generated Token: {token}")

        # Validate the generated token
        try:
            decoded_payload = jwt_service.validate_token(token)
            print(f"Decoded Payload: {decoded_payload}")
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
            print(f"Token validation failed: {str(e)}")

    except ValueError as e:
        print(f"JWT Service Error: {str(e)}")
        print("Make sure to set JWT_SECRET environment variable")
