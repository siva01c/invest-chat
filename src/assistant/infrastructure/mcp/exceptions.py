"""MCP-specific exceptions."""

from assistant.core.exceptions import AssistantException, ErrorCode


class MCPException(AssistantException):
    """Base exception for MCP-related errors."""

    def __init__(self, message: str, server_url: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.server_url = server_url


class MCPConnectionError(MCPException):
    """Exception raised when MCP server connection fails."""

    def __init__(self, message: str, server_url: str = None, **kwargs):
        super().__init__(
            message, server_url=server_url, error_code=ErrorCode.EXTERNAL_API_ERROR, **kwargs
        )


class MCPToolError(MCPException):
    """Exception raised when MCP tool execution fails."""

    def __init__(self, message: str, tool_name: str = None, server_url: str = None, **kwargs):
        super().__init__(
            message, server_url=server_url, error_code=ErrorCode.EXTERNAL_API_ERROR, **kwargs
        )
        self.tool_name = tool_name


class MCPAuthenticationError(MCPException):
    """Exception raised when MCP server authentication fails."""

    def __init__(self, message: str, server_url: str = None, **kwargs):
        super().__init__(message, server_url=server_url, error_code=ErrorCode.AUTH_ERROR, **kwargs)


class MCPProtocolError(MCPException):
    """Exception raised when MCP protocol communication fails."""

    def __init__(self, message: str, method: str = None, server_url: str = None, **kwargs):
        super().__init__(
            message, server_url=server_url, error_code=ErrorCode.PROTOCOL_ERROR, **kwargs
        )
        self.method = method
