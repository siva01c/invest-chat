"""Enhanced security middleware with comprehensive protection measures."""

import json
import re
import secrets
import time
from typing import Any, Dict, List, Optional

import bleach
from bleach.css_sanitizer import CSSSanitizer
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from assistant.config import get_settings
from assistant.core.exceptions import ErrorCode, SecurityException
from assistant.core.logging import get_logger


class EnhancedSecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with comprehensive protection."""

    def __init__(
        self,
        app: ASGIApp,
        enable_csrf: bool = True,
        enable_xss_protection: bool = True,
        enable_content_validation: bool = True,
        enable_security_headers: bool = True,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB default
        trusted_hosts: Optional[List[str]] = None,
    ):
        """
        Initialize enhanced security middleware.

        Args:
            app: ASGI application
            enable_csrf: Enable CSRF protection
            enable_xss_protection: Enable XSS protection
            enable_content_validation: Enable content validation
            enable_security_headers: Enable security headers
            max_request_size: Maximum request size in bytes
            trusted_hosts: List of trusted host patterns
        """
        super().__init__(app)

        self.logger = get_logger(self.__class__.__name__)
        self.settings = get_settings()

        # Security feature flags
        self.enable_csrf = enable_csrf
        self.enable_xss_protection = enable_xss_protection
        self.enable_content_validation = enable_content_validation
        self.enable_security_headers = enable_security_headers

        # Request limits
        self.max_request_size = max_request_size
        self.max_json_depth = 10
        self.max_json_objects = 1000

        # Endpoint-specific size limits (in bytes)
        self.endpoint_size_limits = {
            "/chat": 100 * 1024,  # 100KB for chat messages
            "/knowledge": 500 * 1024,  # 500KB for knowledge queries
            "/health": 1024,  # 1KB for health checks
            "/metrics": 1024,  # 1KB for metrics
            "/tasks": 200 * 1024,  # 200KB for task data
            "default": max_request_size,
        }

        # Rate limiting for request sizes
        self.size_based_limits = {
            "small": (10 * 1024, 1000),  # <10KB: 1000 req/hour
            "medium": (100 * 1024, 100),  # <100KB: 100 req/hour
            "large": (1024 * 1024, 10),  # <1MB: 10 req/hour
            "xlarge": (max_request_size, 5),  # >1MB: 5 req/hour
        }

        # Content-type specific limits
        self.content_type_limits = {
            "application/json": 1024 * 1024,  # 1MB for JSON
            "text/plain": 100 * 1024,  # 100KB for text
            "application/x-www-form-urlencoded": 50 * 1024,  # 50KB for forms
            "multipart/form-data": 10 * 1024 * 1024,  # 10MB for file uploads
        }

        # Trusted hosts configuration
        self.trusted_hosts = set(
            trusted_hosts
            or [
                "localhost",
                "localhost:5000",
                "localhost:8000",
                "127.0.0.1",
                "127.0.0.1:5000",
                "127.0.0.1:8000",
                "ludekkvapil.cz",
                "*.ludekkvapil.cz",
                "ragchat.local",
            ]
        )

        # CSRF configuration
        self.csrf_token_length = 32
        self.csrf_header_name = "X-CSRF-Token"
        self.csrf_cookie_name = "csrf_token"
        self.csrf_exempt_paths = {"/health", "/health/detailed", "/docs", "/openapi.json"}

        # XSS protection configuration
        self.allowed_tags = {
            "b",
            "i",
            "u",
            "em",
            "strong",
            "p",
            "br",
            "span",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "ul",
            "ol",
            "li",
            "blockquote",
            "code",
            "pre",
        }
        self.allowed_attributes = {
            "*": ["class", "id"],
            "a": ["href", "title"],
            "img": ["src", "alt", "title", "width", "height"],
        }

        # Enhanced content validation patterns
        self.dangerous_patterns = [
            # JavaScript execution and XSS
            r"javascript:",
            r"on\w+\s*=",
            r"<script[^>]*>",
            r"</script>",
            r"eval\s*\(",
            r"Function\s*\(",
            r"setTimeout\s*\(",
            r"setInterval\s*\(",
            r"document\.cookie",
            r"document\.write",
            r"window\.location",
            r"location\.href",
            r"innerHTML\s*=",
            r"outerHTML\s*=",
            r"insertAdjacentHTML",
            r"createElement\s*\(",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<form[^>]*>",
            r"<input[^>]*>",
            r"<button[^>]*>",
            r"<link[^>]*>",
            r"<meta[^>]*>",
            r"<style[^>]*>",
            r"expression\s*\(",
            r"url\s*\(",
            r"@import",
            # Data URIs and dangerous protocols
            r"data:(?!image/(?:png|jpg|jpeg|gif|svg|webp))",
            r"vbscript:",
            r"file:",
            r"ftp:",
            r"mailto:.*[<>]",
            # SQL injection patterns (enhanced)
            r"\b(?:union|select|insert|update|delete|drop|create|alter|exec|execute|declare|cast|convert|char|varchar|nvarchar|concat|substring|ascii|hex|unhex|load_file|into\s+outfile)\b",
            r"--\s",
            r"/\*.*?\*/",
            r"\|\|",
            r"@@",
            r"\bor\s+\d+\s*=\s*\d+",
            r"\band\s+\d+\s*=\s*\d+",
            r'[\'"]\s*;\s*--',
            r'[\'"]\s*\|\|\s*[\'"]',
            r"0x[0-9a-fA-F]+",
            r"benchmark\s*\(",
            r"sleep\s*\(",
            r"waitfor\s+delay",
            # Command injection (enhanced)
            r"[;&|`$]",
            r"\$\(",
            r"`[^`]*`",
            r"\$\{[^}]*\}",
            r"%[0-9a-fA-F]{2}",
            r"\\x[0-9a-fA-F]{2}",
            r"\\[0-7]{3}",
            r"\bcat\s+",
            r"\bls\s+",
            r"\bpwd\b",
            r"\bwhoami\b",
            r"\bnetstat\b",
            r"\bps\s+",
            r"\btop\b",
            r"\bkill\s+",
            r"\brm\s+",
            r"\bmv\s+",
            r"\bcp\s+",
            r"\bchmod\s+",
            r"\bchown\s+",
            r"\bsu\s+",
            r"\bsudo\s+",
            # Path traversal (enhanced)
            r"\.\./.*\.\./",
            r"\.\.\\.*\.\.\\",
            r"\.\.[\\/]",
            r"[\\/]\.\.[\\/]",
            r"%2e%2e%2f",
            r"%2e%2e%5c",
            r"\.\.%2f",
            r"\.\.%5c",
            # LDAP injection
            r"[()&|!]",
            r"\*[^*]*\*",
            # XML/XXE injection
            r"<!ENTITY",
            r"<!DOCTYPE",
            r"<\?xml",
            r'SYSTEM\s+["\']',
            r'PUBLIC\s+["\']',
            # Server-side template injection
            r"\{\{.*\}\}",
            r"\{%.*%\}",
            r"\$\{.*\}",
            r"<%.*%>",
            # Log injection/CRLF
            r"%0d%0a",
            r"%0a",
            r"%0d",
            r"\r\n",
            r"\n\r",
            # NoSQL injection
            r"\$ne\b",
            r"\$gt\b",
            r"\$lt\b",
            r"\$regex\b",
            r"\$where\b",
            # Email injection
            r"bcc:",
            r"cc:",
            r"content-type:",
            r"mime-version:",
            r"x-mailer:",
            # Dangerous file extensions
            r"\.(exe|bat|cmd|com|pif|scr|vbs|js|jar|app|deb|pkg|dmg|ps1|sh|php|asp|aspx|jsp|jsp|cfm)$",
            # Protocol smuggling
            r"%2f%2f",
            r"/\*\*/",
            r"@[^@]*@",
            # Polyglot payloads
            r"jaVasCript:",
            r"vBsCrIpT:",
            r"J\s*a\s*v\s*a\s*S\s*c\s*r\s*i\s*p\s*t",
        ]

        # Compile patterns for performance
        self.dangerous_regex = re.compile(
            "|".join(self.dangerous_patterns), re.IGNORECASE | re.MULTILINE
        )

    @staticmethod
    def generate_csrf_token() -> str:
        """Generate a secure CSRF token."""
        return secrets.token_urlsafe(32)  # 32 bytes = 256 bits

    @staticmethod
    def validate_csrf_token(token: str) -> bool:
        """
        Validate CSRF token (simple validation for testing).

        Args:
            token: CSRF token string

        Returns:
            True if token is valid
        """
        if not token:
            return False

        # Basic validation - check if token is well-formed
        return len(token) >= 32 and token.replace("-", "").replace("_", "").isalnum()

    def validate_csrf_token_from_request(self, request: Request) -> bool:
        """
        Validate CSRF token from request.

        Args:
            request: HTTP request

        Returns:
            True if token is valid
        """
        if not self.enable_csrf:
            return True

        # Skip CSRF for exempt paths
        if request.url.path in self.csrf_exempt_paths:
            return True

        # Skip CSRF for GET requests
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return True

        # Get token from header
        token_header = request.headers.get(self.csrf_header_name)

        # Get token from cookie
        token_cookie = request.cookies.get(self.csrf_cookie_name)

        # Both tokens must be present and match
        if not token_header or not token_cookie:
            return False

        return secrets.compare_digest(token_header, token_cookie)

    def validate_host(self, request: Request) -> bool:
        """
        Validate request host against trusted hosts.

        Args:
            request: HTTP request

        Returns:
            True if host is trusted
        """
        host = request.headers.get("host", "").lower()
        if not host:
            return False

        # Check exact matches
        if host in self.trusted_hosts:
            return True

        # Check wildcard patterns
        for trusted in self.trusted_hosts:
            if trusted.startswith("*."):
                domain = trusted[2:]
                if host == domain or host.endswith("." + domain):
                    return True
            # Check IP range patterns like "172.22.*" or "172.22.*:*"
            elif "*" in trusted:
                import re

                # Handle patterns like "172.22.*:*" for IP with port wildcards
                if ":*" in trusted:
                    pattern = (
                        trusted.replace(".", r"\.")
                        .replace("*", r"[^:]*")
                        .replace(r"[^:]*:", r".*:")
                    )
                else:
                    # Handle patterns like "172.22.*" (without port)
                    pattern = trusted.replace(".", r"\.").replace("*", r"[^:]*")
                if re.match(f"^{pattern}$", host):
                    return True

        return False

    def sanitize_text_advanced(self, text: str) -> str:
        """
        Advanced text sanitization with XSS protection.

        Args:
            text: Input text

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Use bleach for comprehensive HTML sanitization
        css_sanitizer = CSSSanitizer(
            allowed_css_properties=[
                "color",
                "background-color",
                "font-size",
                "font-weight",
                "text-align",
                "margin",
                "padding",
            ]
        )

        sanitized = bleach.clean(
            text,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            css_sanitizer=css_sanitizer,
            strip=True,
            strip_comments=True,
        )

        # Additional pattern-based cleaning
        sanitized = self.dangerous_regex.sub("", sanitized)

        # Normalize whitespace
        sanitized = " ".join(sanitized.split())

        return sanitized

    def validate_json_structure(self, data: Any, depth: int = 0) -> bool:
        """
        Validate JSON structure for security.

        Args:
            data: JSON data to validate
            depth: Current recursion depth

        Returns:
            True if structure is safe
        """
        if depth > self.max_json_depth:
            return False

        if isinstance(data, dict):
            if len(data) > self.max_json_objects:
                return False
            for key, value in data.items():
                if not isinstance(key, str) or len(key) > 1000:
                    return False
                if not self.validate_json_structure(value, depth + 1):
                    return False

        elif isinstance(data, list):
            if len(data) > self.max_json_objects:
                return False
            for item in data:
                if not self.validate_json_structure(item, depth + 1):
                    return False

        elif isinstance(data, str):
            if len(data) > 100000:  # 100KB max for strings
                return False
            # Check for dangerous patterns in strings
            if self.dangerous_regex.search(data):
                return False

        return True

    def validate_content_type(self, request: Request) -> bool:
        """
        Validate request content type.

        Args:
            request: HTTP request

        Returns:
            True if content type is allowed
        """
        content_type = request.headers.get("content-type", "").lower()

        allowed_types = {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
        }

        # Extract main content type (remove charset, boundary, etc.)
        main_type = content_type.split(";")[0].strip()

        return main_type in allowed_types or not main_type

    async def validate_request_size(self, request: Request) -> bool:
        """
        Comprehensive request size validation with endpoint-specific limits.

        Args:
            request: HTTP request

        Returns:
            True if size is acceptable

        Raises:
            HTTPException: If request exceeds size limits
        """
        content_length = request.headers.get("content-length")

        if not content_length:
            return True

        try:
            length = int(content_length)
        except ValueError:
            self.logger.warning("Invalid content-length header")
            return False

        # Get endpoint-specific limit
        path = str(request.url.path)
        endpoint_limit = self.endpoint_size_limits.get(path, self.endpoint_size_limits["default"])

        # Check endpoint-specific limit first
        if length > endpoint_limit:
            self.logger.warning(
                f"Request size {length} exceeds endpoint limit {endpoint_limit} for {path}"
            )
            return False

        # Check content-type specific limits
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        content_limit = self.content_type_limits.get(content_type, self.max_request_size)

        if length > content_limit:
            self.logger.warning(
                f"Request size {length} exceeds content-type limit {content_limit} for {content_type}"
            )
            return False

        # Check global maximum
        if length > self.max_request_size:
            self.logger.warning(
                f"Request size {length} exceeds global limit {self.max_request_size}"
            )
            return False

        # Log large requests for monitoring
        if length > 1024 * 1024:  # 1MB
            self.logger.info(f"Large request detected: {length} bytes to {path}")

        return True

    def get_request_size_category(self, size: int) -> str:
        """
        Categorize request size for rate limiting.

        Args:
            size: Request size in bytes

        Returns:
            Size category string
        """
        for category, (limit, _) in self.size_based_limits.items():
            if size <= limit:
                return category
        return "xlarge"

    def add_security_headers(self, response: Response) -> Response:
        """
        Add security headers to response.

        Args:
            response: HTTP response

        Returns:
            Response with security headers
        """
        if not self.enable_security_headers:
            return response

        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]

        security_headers = {
            # Content Security Policy
            "Content-Security-Policy": "; ".join(csp_directives),
            # XSS Protection
            "X-XSS-Protection": "1; mode=block",
            # Content Type Options
            "X-Content-Type-Options": "nosniff",
            # Frame Options
            "X-Frame-Options": "DENY",
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions Policy
            "Permissions-Policy": "camera=(), microphone=(), geolocation=(), interest-cohort=()",
            # Strict Transport Security (if HTTPS)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            # Server identification
            "Server": "Assistant-API",
        }

        for header, value in security_headers.items():
            response.headers[header] = value

        # Add CSRF token to response if enabled
        if self.enable_csrf and not response.headers.get(self.csrf_cookie_name):
            csrf_token = self.generate_csrf_token()
            response.set_cookie(
                self.csrf_cookie_name,
                csrf_token,
                httponly=True,
                secure=True,
                samesite="strict",
                max_age=3600,  # 1 hour
            )

        return response

    async def validate_request_content(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Validate and sanitize request content.

        Args:
            request: HTTP request

        Returns:
            Validation results or None if valid
        """
        self.logger.info(f"validate_request_content called for {request.url.path}")
        if not self.enable_content_validation:
            return None

        # Skip validation for certain methods
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return None

        # Skip validation for certain endpoints that need body access
        if request.url.path in ("/public", "/chat", "/contact"):
            return None

        try:
            # Get request body (cache it to restore later)
            body = await request.body()

            # Restore the body for the endpoint handler
            # In Starlette, we need to reset the stream state

            if hasattr(request, "_body"):
                request._body = body
                request._stream_consumed = False

            if not body:
                return None

            # Check content type
            content_type = request.headers.get("content-type", "").lower()

            if "application/json" in content_type:
                try:
                    json_data = json.loads(body)

                    # Validate JSON structure
                    if not self.validate_json_structure(json_data):
                        return {"error": "Invalid JSON structure", "code": "INVALID_JSON_STRUCTURE"}

                    # Sanitize string values if XSS protection is enabled
                    if self.enable_xss_protection:
                        self._sanitize_json_strings(json_data)

                except json.JSONDecodeError:
                    return {"error": "Invalid JSON format", "code": "INVALID_JSON_FORMAT"}

            elif "text/" in content_type:
                text = body.decode("utf-8", errors="ignore")

                if self.enable_xss_protection:
                    if self.dangerous_regex.search(text):
                        return {
                            "error": "Potentially dangerous content detected",
                            "code": "DANGEROUS_CONTENT",
                        }

            # Check size limits
            endpoint_limit = self.endpoint_size_limits.get(
                request.url.path, self.endpoint_size_limits["default"]
            )
            if len(body) > endpoint_limit:
                return {
                    "error": f"Request size exceeds limit of {endpoint_limit} bytes",
                    "code": "REQUEST_TOO_LARGE",
                }

        except Exception as e:
            # Log error but allow request to proceed
            self.logger.warning(f"Content validation error: {e}")
            return None

        return None

        # Skip validation for certain methods
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return None

        try:
            # Get request body
            body = await request.body()

            if not body:
                return None

            # Check content type
            content_type = request.headers.get("content-type", "").lower()

            if "application/json" in content_type:
                try:
                    json_data = json.loads(body)

                    # Validate JSON structure
                    if not self.validate_json_structure(json_data):
                        return {"error": "Invalid JSON structure", "code": "INVALID_JSON_STRUCTURE"}

                    # Sanitize string values if XSS protection is enabled
                    if self.enable_xss_protection:
                        self._sanitize_json_strings(json_data)

                except json.JSONDecodeError:
                    return {"error": "Invalid JSON format", "code": "INVALID_JSON_FORMAT"}

            elif "text/" in content_type:
                text = body.decode("utf-8", errors="ignore")

                # Check for dangerous patterns
                if self.dangerous_regex.search(text):
                    return {
                        "error": "Potentially dangerous content detected",
                        "code": "DANGEROUS_CONTENT",
                    }

        except Exception as e:
            self.logger.error(f"Content validation error: {str(e)}")
            return {"error": "Content validation failed", "code": "VALIDATION_ERROR"}

        return None

    def _sanitize_json_strings(self, data: Any) -> None:
        """
        Recursively sanitize strings in JSON data.

        Args:
            data: JSON data to sanitize (modified in place)
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    data[key] = self.sanitize_text_advanced(value)
                else:
                    self._sanitize_json_strings(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    data[i] = self.sanitize_text_advanced(item)
                else:
                    self._sanitize_json_strings(item)

    async def dispatch(self, request: Request, call_next):
        """Process request through enhanced security checks."""
        start_time = time.time()

        try:
            # Skip security checks for CORS preflight OPTIONS requests
            if request.method == "OPTIONS":
                response = await call_next(request)
                return response

            # 1. Validate host (always check)
            if not self.validate_host(request):
                self.logger.warning(f"Untrusted host: {request.headers.get('host')}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Untrusted host",
                        "detail": f"Host {request.headers.get('host')} is not in trusted hosts list",
                    },
                )

            # 2. Validate request size (always check, doesn't consume body)
            if not await self.validate_request_size(request):
                self.logger.warning(
                    f"Request size exceeded: {request.headers.get('content-length')}"
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request entity too large",
                        "detail": "Request size exceeds maximum allowed limit",
                    },
                )

            # 3. Validate content type
            if not self.validate_content_type(request):
                self.logger.warning(f"Invalid content type: {request.headers.get('content-type')}")
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": "Unsupported media type",
                        "detail": f"Content type {request.headers.get('content-type')} is not supported",
                    },
                )

            # 4. CSRF protection
            if not self.validate_csrf_token_from_request(request):
                self.logger.warning("CSRF token validation failed")
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "CSRF token validation failed",
                        "detail": "Cross-site request forgery token is missing or invalid",
                    },
                )

            # 5. Content validation (skip for endpoints that need body access)
            if request.url.path not in ("/public", "/chat"):
                content_validation = await self.validate_request_content(request)
                if content_validation:
                    self.logger.warning(f"Content validation failed: {content_validation}")
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": str(content_validation),
                            "detail": "Request content validation failed",
                        },
                    )

            # Process request
            response = await call_next(request)

            # Add security headers
            response = self.add_security_headers(response)

            # Add processing time header
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)

            # Security audit log
            self.logger.debug(
                f"Security check passed: {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'} "
                f"in {process_time:.3f}s"
            )

            return response

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log unexpected errors
            self.logger.error(f"Security middleware error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal security error")


# Utility functions for manual security checks
def validate_user_input(text: str, max_length: int = 10000, allow_html: bool = False) -> str:
    """
    Validate and sanitize user input.

    Args:
        text: Input text to validate
        max_length: Maximum allowed length
        allow_html: Whether to allow HTML tags

    Returns:
        Sanitized text

    Raises:
        SecurityException: If input is invalid
    """
    if not isinstance(text, str):
        raise SecurityException("Input must be a string", error_code=ErrorCode.INVALID_INPUT)

    if len(text) > max_length:
        raise SecurityException(
            f"Input exceeds maximum length of {max_length}", error_code=ErrorCode.INVALID_INPUT
        )

    if allow_html:
        # Use bleach for comprehensive HTML sanitization
        css_sanitizer = CSSSanitizer(
            allowed_css_properties=[
                "color",
                "background-color",
                "font-size",
                "font-weight",
                "text-align",
                "margin",
                "padding",
            ]
        )

        sanitized = bleach.clean(
            text,
            tags=[
                "p",
                "br",
                "strong",
                "em",
                "u",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "ul",
                "ol",
                "li",
                "blockquote",
                "code",
                "pre",
                "a",
                "img",
            ],
            attributes={"a": ["href", "title"], "img": ["src", "alt", "title"]},
            css_sanitizer=css_sanitizer,
            strip=True,
        )
        return sanitized
    else:
        # Strip all HTML for plain text
        import html

        return html.escape(text.strip())


def check_sql_injection(query: str) -> bool:
    """
    Check if a string contains SQL injection patterns.

    Args:
        query: String to check

    Returns:
        True if potentially dangerous SQL patterns are detected
    """
    sql_patterns = [
        r"\b(?:union|select|insert|update|delete|drop|create|alter|exec|execute)\b",
        r"--\s",
        r"/\*.*?\*/",
        r"\|\|",
        r"@@",
        r";.*(?:union|select|insert|update|delete|drop)",
    ]

    combined_pattern = re.compile("|".join(sql_patterns), re.IGNORECASE)
    return bool(combined_pattern.search(query))


def validate_email_format(email: str) -> bool:
    """
    Validate email format with security considerations.

    Args:
        email: Email address to validate

    Returns:
        True if email format is valid and safe
    """
    if not email or len(email) > 254:  # RFC 5321 limit
        return False

    # Basic email regex that prevents most injection attempts
    email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    return bool(email_pattern.match(email))


def generate_secure_filename(original_filename: str) -> str:
    """
    Generate a secure filename from user input.

    Args:
        original_filename: Original filename

    Returns:
        Sanitized filename
    """
    import os
    import uuid

    # Remove directory traversal attempts
    filename = os.path.basename(original_filename)

    # Remove dangerous characters
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

    # Limit length
    name, ext = os.path.splitext(filename)
    name = name[:100]  # Limit name part
    ext = ext[:10]  # Limit extension part

    # Add unique prefix to prevent conflicts
    unique_prefix = str(uuid.uuid4())[:8]

    return f"{unique_prefix}_{name}{ext}"
