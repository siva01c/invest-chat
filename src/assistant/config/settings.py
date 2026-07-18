"""Application settings using Pydantic BaseSettings for environment validation."""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable validation."""

    # API Configuration
    api_title: str = Field(default="Knowledge Base API", description="API title")
    api_description: str = Field(
        default="API for accessing knowledge base data", description="API description"
    )
    api_version: str = Field(default="1.0.0", description="API version")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5000, description="Server port")

    # Security Configuration
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5000",
            "http://localhost:1313",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5000",
            "http://127.0.0.1:1313",
            "http://localhost",
            "http://localhost:8000"
            "https://mcpserver.cz",
            "https://ludekkvapil.cz",
        ],
        description="CORS allowed origins",
        alias="CORS_ORIGINS",
    )

    # Enhanced Security Configuration
    enable_csrf_protection: bool = Field(default=True, description="Enable CSRF protection")
    trusted_hosts: List[str] = Field(
        default=["ludekkvapil.cz", "mcpserver.cz", "localhost:8000", "localhost:1313", "127.0.0.1", "172.22.198.71:5000", "172.22.*:*"],
        description="Trusted hosts for host validation",
    )
    max_request_size: int = Field(
        default=1048576, description="Maximum request size in bytes (1MB)"
    )
    csrf_token_expiry: int = Field(default=3600, description="CSRF token expiry in seconds")
    content_security_policy: str = Field(
        default="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self'; font-src 'self';",
        description="Content Security Policy header",
    )

    # Rate Limiting Configuration
    rate_limit_requests: int = Field(default=10, description="Requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    email_rate_limit: int = Field(default=300, description="Email rate limit in seconds")

    # Message Validation Configuration
    max_message_length: int = Field(default=4000, description="Maximum message length")
    max_messages_per_request: int = Field(default=10, description="Maximum messages per request")
    max_session_id_length: int = Field(default=128, description="Maximum session ID length")

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    openai_temperature: float = Field(default=0.0, description="OpenAI temperature")
    openai_embedding_model: str = Field(
        default="text-embedding-ada-002", description="OpenAI embedding model"
    )

    # AI Service Configuration
    max_history: int = Field(default=5, description="Maximum chat history entries")
    context_window: int = Field(default=3, description="Context window for conversations")

    # ChromaDB Configuration
    chromadb_collection_name: str = Field(
        default="about_me", description="ChromaDB collection name"
    )
    chromadb_database_path: str = Field(default="chromadb", description="ChromaDB database path")
    chromadb_search_results: int = Field(default=3, description="Number of search results")
    chromadb_host: str = Field(
        default="localhost", description="ChromaDB host", alias="CHROMADB_HOST"
    )
    chromadb_port: int = Field(default=8000, description="ChromaDB port", alias="CHROMADB_PORT")
    chromadb_use_http: bool = Field(default=True, description="Use HTTP client for ChromaDB")

    # ChromaDB Connection Pool Configuration
    chromadb_min_connections: int = Field(default=2, description="Minimum connections in pool")
    chromadb_max_connections: int = Field(default=10, description="Maximum connections in pool")
    chromadb_max_idle_time: int = Field(
        default=300, description="Maximum idle time for connections (seconds)"
    )
    chromadb_health_check_interval: int = Field(
        default=60, description="Health check interval (seconds)"
    )
    chromadb_query_retries: int = Field(default=3, description="Maximum query retries")

    # Vector Search Optimization
    enable_query_optimization: bool = Field(default=True, description="Enable query optimization")
    enable_result_caching: bool = Field(default=True, description="Enable result caching")
    search_cache_ttl: int = Field(default=300, description="Search result cache TTL (seconds)")
    batch_insert_size: int = Field(default=100, description="Batch size for bulk inserts")
    score_threshold_optimization: bool = Field(
        default=True, description="Enable dynamic score threshold optimization"
    )

    # Advanced Caching Configuration
    enable_response_caching: bool = Field(default=True, description="Enable HTTP response caching")
    response_cache_ttl: int = Field(default=300, description="Default response cache TTL (seconds)")
    response_cache_size_mb: int = Field(default=200, description="Response cache size limit (MB)")
    enable_cache_compression: bool = Field(default=True, description="Enable cache compression")
    cache_invalidation_enabled: bool = Field(
        default=True, description="Enable intelligent cache invalidation"
    )
    auto_cache_tuning: bool = Field(
        default=True, description="Enable automatic cache performance tuning"
    )

    # Cache TTL Policies
    health_cache_ttl: int = Field(default=30, description="Health endpoint cache TTL (seconds)")
    knowledge_cache_ttl: int = Field(default=600, description="Knowledge base cache TTL (seconds)")
    session_cache_ttl: int = Field(default=1800, description="Session cache TTL (seconds)")
    static_cache_ttl: int = Field(default=86400, description="Static content cache TTL (seconds)")
    metrics_cache_ttl: int = Field(default=15, description="Metrics cache TTL (seconds)")

    # Prometheus Metrics Configuration
    enable_prometheus_metrics: bool = Field(
        default=True, description="Enable Prometheus metrics collection"
    )
    prometheus_metrics_path: str = Field(
        default="/metrics", description="Prometheus metrics endpoint path"
    )
    prometheus_enable_request_size: bool = Field(
        default=True, description="Track HTTP request size metrics"
    )
    prometheus_enable_response_size: bool = Field(
        default=True, description="Track HTTP response size metrics"
    )
    prometheus_track_in_progress: bool = Field(
        default=True, description="Track in-progress requests"
    )
    prometheus_exclude_paths: List[str] = Field(
        default=["/health", "/metrics", "/favicon.ico", "/docs", "/redoc", "/openapi.json"],
        description="Paths to exclude from Prometheus metrics",
    )

    # Observability Configuration
    enable_structured_logging: bool = Field(
        default=True, description="Enable structured logging with correlation IDs"
    )
    enable_distributed_tracing: bool = Field(
        default=False, description="Enable distributed tracing"
    )
    trace_sample_rate: float = Field(
        default=0.1, description="Distributed tracing sample rate (0.0-1.0)"
    )
    correlation_id_header: str = Field(
        default="X-Correlation-ID", description="Header name for correlation ID"
    )
    log_correlation_id: bool = Field(default=True, description="Include correlation ID in logs")

    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL", alias="REDIS_URL"
    )
    redis_max_connections: int = Field(default=20, description="Redis maximum connections")
    redis_timeout: int = Field(default=5, description="Redis connection timeout in seconds")
    enable_redis_rate_limiting: bool = Field(
        default=True, description="Enable Redis-based rate limiting"
    )

    # Email Configuration
    email_sender: str = Field(..., description="Sender email address", alias="EMAIL")
    email_password: str = Field(..., description="Email password", alias="EMAIL_PWD")
    email_receiver: str = Field(..., description="Receiver email address", alias="EMAIL_RECEIVER")
    smtp_server: str = Field(
        default="mail.gigaserver.cz", description="SMTP server", alias="SMTP_SERVER"
    )
    smtp_port: int = Field(default=465, description="SMTP port", alias="SMTP_PORT")
    smtp_use_ssl: bool = Field(default=True, description="Use SSL for SMTP")

    # Drupal Consumer Configuration (Optional)
    drupal_base_url: Optional[str] = Field(
        default="http://drupal.ddev.site", description="Drupal base URL", alias="DRUPAL_BASE_URL"
    )
    drupal_username: Optional[str] = Field(
        default="api", description="Drupal username", alias="DRUPAL_USERNAME"
    )
    drupal_password: Optional[str] = Field(
        default="password", description="Drupal password", alias="DRUPAL_PASSWORD"
    )

    # JWT Configuration (Optional)
    jwt_secret_key: Optional[str] = Field(
        default=None, description="JWT secret key", alias="JWT_SECRET_KEY"
    )

    # MCP (Model Context Protocol) Configuration
    enable_mcp: bool = Field(default=True, description="Enable MCP integration")
    apify_api_key: Optional[str] = Field(
        default=None, description="Apify API key for web crawling", alias="APIFY_API_KEY"
    )
    mcp_timeout: int = Field(default=60, description="MCP request timeout in seconds")
    mcp_enabled_tools: List[str] = Field(
        default=["apify/rag-web-browser", "compass/crawler-google-places"],
        description="List of enabled MCP tools",
    )
    mcp_max_concurrent_requests: int = Field(
        default=5, description="Maximum concurrent MCP requests"
    )
    mcp_enable_caching: bool = Field(default=True, description="Enable MCP response caching")
    mcp_cache_ttl: int = Field(default=3600, description="MCP cache TTL in seconds")

    # File Paths Configuration
    data_directory: str = Field(default="data", description="Data directory path")
    prompts_directory: str = Field(default="data/prompts", description="Prompts directory path")
    translations_directory: str = Field(
        default="data/translations", description="Translations directory path"
    )
    datasources_directory: str = Field(
        default="data/datasources", description="Data sources directory path"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_directory: str = Field(default="logs", description="Log directory path")
    enable_chat_logging: bool = Field(default=True, description="Enable chat history logging")

    # Development Configuration
    debug: bool = Field(default=False, description="Debug mode")
    disable_docs: bool = Field(default=True, description="Disable API documentation endpoints")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra fields from .env
    }

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        """Validate port is in valid range."""
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("rate_limit_requests")
    @classmethod
    def validate_rate_limit_requests(cls, v):
        """Validate rate limit requests is positive."""
        if v <= 0:
            raise ValueError("Rate limit requests must be positive")
        return v

    @field_validator("openai_temperature")
    @classmethod
    def validate_temperature(cls, v):
        """Validate OpenAI temperature is in valid range."""
        if not (0.0 <= v <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_message_length")
    @classmethod
    def validate_max_message_length(cls, v):
        """Validate max message length is reasonable."""
        if not (100 <= v <= 50000):
            raise ValueError("Max message length must be between 100 and 50000")
        return v

    @field_validator("smtp_port")
    @classmethod
    def validate_smtp_port(cls, v):
        """Validate SMTP port is in valid range."""
        if v not in [25, 465, 587, 2525]:
            raise ValueError("SMTP port should be one of: 25, 465, 587, 2525")
        return v

    def get_data_path(self, relative_path: str) -> Path:
        """Get absolute path for data files."""
        base_path = Path(__file__).parent.parent / self.data_directory
        return base_path / relative_path

    def get_prompt_path(self, filename: str) -> Path:
        """Get absolute path for prompt files."""
        return self.get_data_path(f"prompts/{filename}")

    def get_translation_path(self, language_code: str) -> Path:
        """Get absolute path for translation files."""
        return self.get_data_path(f"translations/{language_code}.yml")

    def get_datasource_path(self, filename: str) -> Path:
        """Get absolute path for data source files."""
        return self.get_data_path(f"datasources/{filename}")

    @property
    def database_url(self) -> str:
        """Get database URL for external connections."""
        return f"chromadb://{self.chromadb_database_path}"

    @property
    def cors_config(self) -> dict:
        """Get CORS configuration."""
        # Start from configured allowed origins (can be overridden via CORS_ORIGINS env)
        origins = list(self.allowed_origins)

        # If the environment provides a virtual host (nginx-proxy), include both http and https variants
        virtual_host = os.getenv("VIRTUAL_HOST") or os.getenv("LETSENCRYPT_HOST")
        if virtual_host:
            https_origin = f"https://{virtual_host}"
            http_origin = f"http://{virtual_host}"
            if https_origin not in origins:
                origins.append(https_origin)
            if http_origin not in origins:
                origins.append(http_origin)

        return {
            "allow_origins": origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "Accept",
                "Origin",
                "X-Requested-With",
                "Access-Control-Request-Method",
                "Access-Control-Request-Headers",
                "X-CSRF-Token",
                "X-Correlation-ID",
                "X-HTTP-Method-Override",
            ],
        }

    @property
    def openai_config(self) -> dict:
        """Get OpenAI configuration."""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "temperature": self.openai_temperature,
            "embedding_model": self.openai_embedding_model,
        }

    @property
    def smtp_config(self) -> dict:
        """Get SMTP configuration."""
        return {
            "server": self.smtp_server,
            "port": self.smtp_port,
            "use_ssl": self.smtp_use_ssl,
            "sender": self.email_sender,
            "password": self.email_password,
            "receiver": self.email_receiver,
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
