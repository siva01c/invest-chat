"""Tests for enhanced security middleware improvements."""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from assistant.api.middleware.enhanced_security import EnhancedSecurityMiddleware


class TestEnhancedInputSanitization:
    """Test enhanced input sanitization patterns."""

    @pytest.fixture
    def security_middleware(self):
        """Create security middleware instance for testing."""
        app = FastAPI()
        return EnhancedSecurityMiddleware(
            app,
            enable_csrf=False,  # Disable for testing
            enable_content_validation=True,
            max_request_size=1024 * 1024,  # 1MB
        )

    def test_javascript_injection_patterns(self, security_middleware):
        """Test detection of JavaScript injection patterns."""
        dangerous_inputs = [
            "javascript:alert('xss')",
            "<script>alert('xss')</script>",
            "eval('malicious code')",
            "setTimeout(function(){alert('xss')}, 1000)",
            "document.cookie = 'stolen'",
            "window.location = 'http://evil.com'",
            "innerHTML = '<script>alert(1)</script>'",
            "<iframe src='javascript:alert(1)'></iframe>",
            "expression(alert('xss'))",
            "url('javascript:alert(1)')",
        ]

        for malicious_input in dangerous_inputs:
            sanitized = security_middleware.sanitize_text_advanced(malicious_input)
            # Should be empty or stripped of dangerous content
            assert len(sanitized) == 0 or "javascript" not in sanitized.lower()
            assert "<script" not in sanitized.lower()
            assert "eval" not in sanitized.lower()

    def test_sql_injection_patterns(self, security_middleware):
        """Test detection of SQL injection patterns."""
        sql_injections = [
            "'; DROP TABLE users; --",
            "UNION SELECT * FROM passwords",
            "1 OR 1=1",
            "admin'/**/AND/**/1=1",
            "0x41424344",
            "BENCHMARK(1000000,MD5(1))",
            "SLEEP(5)",
            "WAITFOR DELAY '0:0:5'",
        ]

        for injection in sql_injections:
            # Should be detected by dangerous patterns
            assert security_middleware.dangerous_regex.search(injection.lower())

    def test_command_injection_patterns(self, security_middleware):
        """Test detection of command injection patterns."""
        command_injections = [
            "; cat /etc/passwd",
            "| whoami",
            "`rm -rf /`",
            "$(cat /etc/shadow)",
            "${IFS}cat${IFS}/etc/passwd",
            "cat /etc/passwd",
            "sudo rm -rf /",
            "chmod 777 /",
        ]

        for injection in command_injections:
            # Should be detected by dangerous patterns
            assert security_middleware.dangerous_regex.search(injection.lower())

    def test_path_traversal_patterns(self, security_middleware):
        """Test detection of path traversal patterns."""
        path_traversals = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "%2e%2e%2f%2e%2e%2f",
            "../../../../../../../etc/passwd",
            "..%2fconfig.php",
        ]

        for traversal in path_traversals:
            # Should be detected by dangerous patterns
            assert security_middleware.dangerous_regex.search(traversal.lower())

    def test_xxe_injection_patterns(self, security_middleware):
        """Test detection of XXE injection patterns."""
        xxe_payloads = [
            "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
            "<?xml version='1.0' encoding='UTF-8'?>",
            "<!ENTITY xxe SYSTEM 'http://evil.com/evil.dtd'>",
            "PUBLIC '-//B//DTD XXE//EN' 'http://evil.com/evil.dtd'",
        ]

        for payload in xxe_payloads:
            # Should be detected by dangerous patterns
            assert security_middleware.dangerous_regex.search(payload)

    def test_template_injection_patterns(self, security_middleware):
        """Test detection of template injection patterns."""
        template_injections = [
            "{{7*7}}",
            "{%for item in items%}",
            "${7*7}",
            "<%=7*7%>",
            "{{config.items()}}",
        ]

        for injection in template_injections:
            # Should be detected by dangerous patterns
            assert security_middleware.dangerous_regex.search(injection)

    def test_safe_content_passes(self, security_middleware):
        """Test that safe content passes sanitization."""
        safe_inputs = [
            "Hello, this is a normal message.",
            "What are your pricing options?",
            "I need help with my account.",
            "Please contact me at user@example.com",
            "Can you help me understand your services?",
            "Thank you for your assistance!",
        ]

        for safe_input in safe_inputs:
            sanitized = security_middleware.sanitize_text_advanced(safe_input)
            # Safe content should remain largely unchanged
            assert len(sanitized) > 0
            assert sanitized.strip() != ""


class TestComprehensiveRequestLimits:
    """Test comprehensive request size limits."""

    @pytest.fixture
    def app_with_security(self):
        """Create FastAPI app with security middleware."""
        app = FastAPI()

        app.add_middleware(
            EnhancedSecurityMiddleware,
            enable_csrf=False,
            enable_content_validation=True,
            max_request_size=1024 * 1024,  # 1MB
            trusted_hosts=["localhost", "127.0.0.1", "testserver"],  # Add testserver for tests
        )

        @app.post("/chat")
        async def chat_endpoint():
            return {"message": "success"}

        @app.post("/knowledge")
        async def knowledge_endpoint():
            return {"message": "success"}

        @app.get("/health")
        async def health_endpoint():
            return {"status": "ok"}

        return app

    def test_endpoint_specific_limits(self, app_with_security):
        """Test endpoint-specific size limits."""
        client = TestClient(app_with_security)

        # Chat endpoint should have 100KB limit
        large_chat_data = {"message": "x" * (100 * 1024 + 1)}  # Just over 100KB
        response = client.post("/chat", json=large_chat_data)
        assert response.status_code == 413  # Payload Too Large

        # Knowledge endpoint should have 500KB limit
        large_knowledge_data = {"query": "x" * (500 * 1024 + 1)}  # Just over 500KB
        response = client.post("/knowledge", json=large_knowledge_data)
        assert response.status_code == 413  # Payload Too Large

    def test_content_type_limits(self, app_with_security):
        """Test content-type specific limits."""
        client = TestClient(app_with_security)

        # JSON should have 1MB limit
        large_json = {"data": "x" * (1024 * 1024 + 1)}  # Just over 1MB
        response = client.post("/chat", json=large_json)
        assert response.status_code == 413  # Payload Too Large

    def test_valid_requests_pass(self, app_with_security):
        """Test that valid-sized requests pass through."""
        client = TestClient(app_with_security)

        # Small valid requests should pass
        small_data = {"message": "Hello, how can I help you?"}
        response = client.post("/chat", json=small_data)
        assert response.status_code == 200

        # Health check should always work
        response = client.get("/health")
        assert response.status_code == 200

    def test_request_size_categorization(self):
        """Test request size categorization for rate limiting."""
        app = FastAPI()
        middleware = EnhancedSecurityMiddleware(app)

        # Test size categorization
        assert middleware.get_request_size_category(5 * 1024) == "small"  # 5KB
        assert middleware.get_request_size_category(50 * 1024) == "medium"  # 50KB
        assert middleware.get_request_size_category(500 * 1024) == "large"  # 500KB
        assert middleware.get_request_size_category(5 * 1024 * 1024) == "xlarge"  # 5MB


class TestSecurityValidation:
    """Test overall security validation."""

    @pytest.fixture
    def security_middleware(self):
        """Create security middleware for testing."""
        app = FastAPI()
        return EnhancedSecurityMiddleware(
            app, enable_csrf=False, enable_content_validation=True, max_request_size=1024 * 1024
        )

    def test_json_structure_validation(self, security_middleware):
        """Test JSON structure validation."""
        # Valid JSON should pass
        valid_json = {"message": "hello", "data": [1, 2, 3]}
        assert security_middleware.validate_json_structure(valid_json)

        # Deeply nested JSON should fail
        deeply_nested = {"level1": {"level2": {"level3": {}}}}
        for i in range(15):  # Create very deep nesting
            deeply_nested = {"deeper": deeply_nested}
        assert not security_middleware.validate_json_structure(deeply_nested)

        # Large JSON with too many objects should fail
        large_json = {f"key_{i}": f"value_{i}" for i in range(2000)}
        assert not security_middleware.validate_json_structure(large_json)

        # JSON with dangerous patterns should fail
        dangerous_json = {"script": "<script>alert('xss')</script>"}
        assert not security_middleware.validate_json_structure(dangerous_json)

    def test_content_type_validation(self, security_middleware):
        """Test content type validation."""
        # Mock request with valid content type
        valid_request = Mock()
        valid_request.headers = {"content-type": "application/json"}
        assert security_middleware.validate_content_type(valid_request)

        # Mock request with invalid content type
        invalid_request = Mock()
        invalid_request.headers = {"content-type": "application/x-evil"}
        assert not security_middleware.validate_content_type(invalid_request)

        # Mock request with no content type (should pass)
        no_type_request = Mock()
        no_type_request.headers = {}
        assert security_middleware.validate_content_type(no_type_request)

    @pytest.mark.asyncio
    async def test_malicious_request_blocked(self, security_middleware):
        """Test that malicious requests are properly blocked."""
        # Create mock request with malicious content
        malicious_request = Mock()
        malicious_request.headers = {"content-type": "application/json", "content-length": "100"}
        malicious_request.url.path = "/chat"

        # Mock request body with malicious payload
        malicious_body = json.dumps(
            {"message": "<script>alert('xss')</script>", "exploit": "'; DROP TABLE users; --"}
        ).encode()

        with patch.object(malicious_request, "body", return_value=malicious_body):
            with patch.object(malicious_request, "json", return_value=json.loads(malicious_body)):
                result = await security_middleware.validate_request_content(malicious_request)

                # Should either return None (blocked) or sanitized content
                if result:
                    assert "<script>" not in str(result)
                    assert "DROP TABLE" not in str(result)

    def test_polyglot_payload_detection(self, security_middleware):
        """Test detection of polyglot payloads."""
        polyglot_payloads = [
            "jaVasCript:alert(1)",  # Case variation
            "vBsCrIpT:alert(1)",  # VBScript variation
            "J a v a S c r i p t : alert(1)",  # Spaced variation
        ]

        for payload in polyglot_payloads:
            assert security_middleware.dangerous_regex.search(payload.lower())


class TestSecurityHeaders:
    """Test security headers implementation."""

    def test_security_headers_added(self):
        """Test that security headers are properly added."""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        app.add_middleware(
            EnhancedSecurityMiddleware,
            enable_csrf=False,
            enable_security_headers=True,
            trusted_hosts=["localhost", "127.0.0.1", "testserver"],
        )

        client = TestClient(app)
        response = client.get("/test")

        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Strict-Transport-Security" in response.headers
        assert "Content-Security-Policy" in response.headers


def test_comprehensive_security_integration():
    """Integration test for all security features."""
    app = FastAPI()

    @app.post("/protected")
    async def protected_endpoint(data: dict):
        return {"received": data}

    # Add comprehensive security middleware
    app.add_middleware(
        EnhancedSecurityMiddleware,
        enable_csrf=False,  # Disable for testing
        enable_xss_protection=True,
        enable_content_validation=True,
        enable_security_headers=True,
        max_request_size=100 * 1024,  # 100KB
        trusted_hosts=["localhost", "127.0.0.1", "testserver"],
    )

    client = TestClient(app)

    # Test 1: Valid request should pass
    valid_data = {"message": "Hello, this is a valid request"}
    response = client.post("/protected", json=valid_data)
    assert response.status_code == 200

    # Test 2: Malicious request should be blocked
    malicious_data = {"message": "<script>alert('xss')</script>"}
    response = client.post("/protected", json=malicious_data)
    # Should be blocked due to dangerous content
    assert response.status_code == 400

    # Test 3: Large request should be rejected
    large_data = {"message": "x" * (100 * 1024 + 1)}  # Over 100KB
    response = client.post("/protected", json=large_data)
    assert response.status_code == 413

    # Test 4: Security headers should be present on successful responses
    # Check headers from the first successful request
    response = client.post("/protected", json=valid_data)
    assert response.status_code == 200
    assert "X-Content-Type-Options" in response.headers
