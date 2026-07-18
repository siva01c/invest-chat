"""Custom exception hierarchy for the assistant application."""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(Enum):
    """Standardized error codes for the application."""

    # Generic errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"

    # Service layer errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_INITIALIZATION_FAILED = "SERVICE_INITIALIZATION_FAILED"
    DEPENDENCY_NOT_FOUND = "DEPENDENCY_NOT_FOUND"

    # AI/LLM errors
    LLM_API_ERROR = "LLM_API_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_QUOTA_EXCEEDED = "LLM_QUOTA_EXCEEDED"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"

    # Data layer errors
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_QUERY_ERROR = "DATABASE_QUERY_ERROR"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"

    # Cache layer errors
    CACHE_CONNECTION_ERROR = "CACHE_CONNECTION_ERROR"
    CACHE_OPERATION_ERROR = "CACHE_OPERATION_ERROR"
    CACHE_TIMEOUT = "CACHE_TIMEOUT"

    # Configuration errors
    CONFIG_FILE_NOT_FOUND = "CONFIG_FILE_NOT_FOUND"
    CONFIG_PARSE_ERROR = "CONFIG_PARSE_ERROR"
    TRANSLATION_FILE_NOT_FOUND = "TRANSLATION_FILE_NOT_FOUND"

    # Email errors
    EMAIL_SEND_ERROR = "EMAIL_SEND_ERROR"
    EMAIL_AUTHENTICATION_ERROR = "EMAIL_AUTHENTICATION_ERROR"
    EMAIL_CONFIGURATION_ERROR = "EMAIL_CONFIGURATION_ERROR"

    # Classification errors
    CLASSIFICATION_ERROR = "CLASSIFICATION_ERROR"
    LANGUAGE_DETECTION_ERROR = "LANGUAGE_DETECTION_ERROR"

    # File processing errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_PARSE_ERROR = "FILE_PARSE_ERROR"
    FILE_READ_ERROR = "FILE_READ_ERROR"

    # Authentication errors
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    AUTH_ERROR = "AUTH_ERROR"

    # Rate limiting errors
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Security errors
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    CSRF_TOKEN_INVALID = "CSRF_TOKEN_INVALID"
    XSS_ATTEMPT_DETECTED = "XSS_ATTEMPT_DETECTED"
    MALICIOUS_CONTENT = "MALICIOUS_CONTENT"

    # External API errors
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    EXTERNAL_API_TIMEOUT = "EXTERNAL_API_TIMEOUT"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"


class AssistantException(Exception):
    """Base exception class for all assistant-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code.value,
            "details": self.details,
        }

        if self.cause:
            result["cause"] = {"type": type(self.cause).__name__, "message": str(self.cause)}

        return result

    def __str__(self) -> str:
        return f"{self.error_code.value}: {self.message}"


class ServiceException(AssistantException):
    """Exception for service layer errors."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.service_name = service_name
        if service_name:
            self.details["service_name"] = service_name


class LLMException(AssistantException):
    """Exception for LLM/AI-related errors."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.LLM_API_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.model_name = model_name
        if model_name:
            self.details["model_name"] = model_name


class DatabaseException(AssistantException):
    """Exception for database-related errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.DATABASE_CONNECTION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.operation = operation
        if operation:
            self.details["operation"] = operation


class VectorStoreException(DatabaseException):
    """Exception for vector store operations."""

    def __init__(
        self,
        message: str,
        collection_name: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, operation, ErrorCode.VECTOR_STORE_ERROR, details, cause)
        self.collection_name = collection_name
        if collection_name:
            self.details["collection_name"] = collection_name


class CacheException(AssistantException):
    """Exception for cache-related errors (Redis, etc.)."""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.CACHE_CONNECTION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.cache_key = cache_key
        self.operation = operation

        if cache_key:
            self.details["cache_key"] = cache_key
        if operation:
            self.details["operation"] = operation


class ConfigurationException(AssistantException):
    """Exception for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.CONFIGURATION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.config_file = config_file
        if config_file:
            self.details["config_file"] = config_file


class EmailException(AssistantException):
    """Exception for email-related errors."""

    def __init__(
        self,
        message: str,
        recipient: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.EMAIL_SEND_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.recipient = recipient
        if recipient:
            self.details["recipient"] = recipient


class ClassificationException(AssistantException):
    """Exception for message classification errors."""

    def __init__(
        self,
        message: str,
        user_input: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.CLASSIFICATION_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.user_input = user_input
        if user_input:
            # Don't store full user input for privacy, just length
            self.details["input_length"] = len(user_input)


class FileProcessingException(AssistantException):
    """Exception for file processing errors."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.FILE_NOT_FOUND,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.file_path = file_path
        self.operation = operation
        if file_path:
            self.details["file_path"] = file_path
        if operation:
            self.details["operation"] = operation


class AuthenticationException(AssistantException):
    """Exception for authentication-related errors."""

    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.AUTHENTICATION_FAILED,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.user_id = user_id
        if user_id:
            self.details["user_id"] = user_id


class RateLimitException(AssistantException):
    """Exception for rate limiting errors."""

    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, ErrorCode.RATE_LIMIT_EXCEEDED, details, cause)
        self.limit = limit
        self.window = window
        self.retry_after = retry_after

        if limit:
            self.details["limit"] = limit
        if window:
            self.details["window"] = window
        if retry_after:
            self.details["retry_after"] = retry_after


class SecurityException(AssistantException):
    """Exception for security-related errors."""

    def __init__(
        self,
        message: str,
        security_violation_type: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.SECURITY_VIOLATION,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.security_violation_type = security_violation_type
        if security_violation_type:
            self.details["violation_type"] = security_violation_type


class ExternalAPIException(AssistantException):
    """Exception for external API errors."""

    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        error_code: ErrorCode = ErrorCode.EXTERNAL_API_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)
        self.api_name = api_name
        self.status_code = status_code

        if api_name:
            self.details["api_name"] = api_name
        if status_code:
            self.details["status_code"] = status_code


# Convenience functions for creating common exceptions


def create_service_error(service_name: str, operation: str, cause: Exception) -> ServiceException:
    """Create a service exception with standardized formatting."""
    return ServiceException(
        message=f"Service '{service_name}' failed during '{operation}'",
        service_name=service_name,
        error_code=ErrorCode.SERVICE_UNAVAILABLE,
        details={"operation": operation},
        cause=cause,
    )


def create_llm_error(model_name: str, operation: str, cause: Exception) -> LLMException:
    """Create an LLM exception with standardized formatting."""
    return LLMException(
        message=f"LLM '{model_name}' failed during '{operation}'",
        model_name=model_name,
        error_code=ErrorCode.LLM_API_ERROR,
        details={"operation": operation},
        cause=cause,
    )


def create_config_error(config_file: str, cause: Exception) -> ConfigurationException:
    """Create a configuration exception with standardized formatting."""
    return ConfigurationException(
        message=f"Failed to load configuration from '{config_file}'",
        config_file=config_file,
        error_code=ErrorCode.CONFIG_FILE_NOT_FOUND,
        cause=cause,
    )


def create_file_error(file_path: str, operation: str, cause: Exception) -> FileProcessingException:
    """Create a file processing exception with standardized formatting."""
    return FileProcessingException(
        message=f"Failed to {operation} file '{file_path}'",
        file_path=file_path,
        operation=operation,
        error_code=ErrorCode.FILE_NOT_FOUND,
        cause=cause,
    )
