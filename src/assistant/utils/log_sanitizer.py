"""
Log sanitization utilities to protect sensitive information.
"""

import logging
import re
from typing import Any, Dict, List, Union

# Sensitive field patterns to redact
SENSITIVE_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    "session_id": r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b",
    "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "api_key": r"\b[A-Za-z0-9]{20,}\b",
}

# Sensitive field names to redact in dictionaries
SENSITIVE_FIELDS = {
    "password",
    "pwd",
    "token",
    "secret",
    "key",
    "email",
    "phone",
    "session_id",
    "user_id",
    "client_id",
    "api_key",
    "access_token",
}


def sanitize_string(text: str, redact_value: str = "[REDACTED]") -> str:
    """
    Sanitize a string by replacing sensitive patterns.

    Args:
        text: The text to sanitize
        redact_value: The replacement value for sensitive data

    Returns:
        Sanitized text with sensitive data redacted
    """
    if not isinstance(text, str):
        return str(text)

    sanitized = text

    # Apply regex patterns
    for pattern_name, pattern in SENSITIVE_PATTERNS.items():
        sanitized = re.sub(pattern, redact_value, sanitized, flags=re.IGNORECASE)

    return sanitized


def sanitize_dict(data: Dict[str, Any], redact_value: str = "[REDACTED]") -> Dict[str, Any]:
    """
    Sanitize a dictionary by redacting sensitive fields.

    Args:
        data: The dictionary to sanitize
        redact_value: The replacement value for sensitive data

    Returns:
        Sanitized dictionary with sensitive fields redacted
    """
    if not isinstance(data, dict):
        return data

    sanitized = {}

    for key, value in data.items():
        # Check if field name is sensitive
        if key.lower() in SENSITIVE_FIELDS:
            sanitized[key] = redact_value
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, redact_value)
        elif isinstance(value, list):
            sanitized[key] = sanitize_list(value, redact_value)
        elif isinstance(value, str):
            sanitized[key] = sanitize_string(value, redact_value)
        else:
            sanitized[key] = value

    return sanitized


def sanitize_list(data: List[Any], redact_value: str = "[REDACTED]") -> List[Any]:
    """
    Sanitize a list by redacting sensitive items.

    Args:
        data: The list to sanitize
        redact_value: The replacement value for sensitive data

    Returns:
        Sanitized list with sensitive items redacted
    """
    if not isinstance(data, list):
        return data

    sanitized = []

    for item in data:
        if isinstance(item, dict):
            sanitized.append(sanitize_dict(item, redact_value))
        elif isinstance(item, list):
            sanitized.append(sanitize_list(item, redact_value))
        elif isinstance(item, str):
            sanitized.append(sanitize_string(item, redact_value))
        else:
            sanitized.append(item)

    return sanitized


def sanitize_for_logging(
    data: Union[str, Dict, List, Any], redact_value: str = "[REDACTED]"
) -> Any:
    """
    Sanitize any data structure for safe logging.

    Args:
        data: The data to sanitize
        redact_value: The replacement value for sensitive data

    Returns:
        Sanitized data safe for logging
    """
    if isinstance(data, str):
        return sanitize_string(data, redact_value)
    elif isinstance(data, dict):
        return sanitize_dict(data, redact_value)
    elif isinstance(data, list):
        return sanitize_list(data, redact_value)
    else:
        return data


class SanitizedFormatter(logging.Formatter):
    """
    Custom logging formatter that automatically sanitizes log messages.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Sanitize the message
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = sanitize_string(record.msg)

        # Sanitize any arguments
        if hasattr(record, "args") and record.args:
            sanitized_args = []
            for arg in record.args:
                sanitized_args.append(sanitize_for_logging(arg))
            record.args = tuple(sanitized_args)

        return super().format(record)


def safe_log(logger: logging.Logger, level: int, message: str, *args, **kwargs) -> None:
    """
    Safe logging function that automatically sanitizes messages and arguments.

    Args:
        logger: The logger instance
        level: The logging level
        message: The message to log
        *args: Message arguments
        **kwargs: Additional logging kwargs
    """
    # Sanitize message and arguments
    sanitized_message = sanitize_string(message)
    sanitized_args = [sanitize_for_logging(arg) for arg in args]

    logger.log(level, sanitized_message, *sanitized_args, **kwargs)


# Convenience functions
def safe_info(logger: logging.Logger, message: str, *args, **kwargs) -> None:
    """Log info with sanitization."""
    safe_log(logger, logging.INFO, message, *args, **kwargs)


def safe_error(logger: logging.Logger, message: str, *args, **kwargs) -> None:
    """Log error with sanitization."""
    safe_log(logger, logging.ERROR, message, *args, **kwargs)


def safe_warning(logger: logging.Logger, message: str, *args, **kwargs):
    """Log warning with sanitization."""
    safe_log(logger, logging.WARNING, message, *args, **kwargs)


def safe_debug(logger: logging.Logger, message: str, *args, **kwargs):
    """Log debug with sanitization."""
    safe_log(logger, logging.DEBUG, message, *args, **kwargs)
