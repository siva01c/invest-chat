"""Base MCP client implementation."""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import httpx

from assistant.core.logging import get_logger

from .exceptions import MCPAuthenticationError, MCPConnectionError, MCPProtocolError, MCPToolError


class MCPMessage:
    """Represents an MCP protocol message."""

    def __init__(self, method: str, params: Dict[str, Any] = None, id: Union[str, int] = None):
        self.id = id or str(uuid.uuid4())
        self.method = method
        self.params = params or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON-RPC."""
        return {"jsonrpc": "2.0", "id": self.id, "method": self.method, "params": self.params}


class MCPTool:
    """Represents an MCP tool."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def __repr__(self) -> str:
        return f"MCPTool(name='{self.name}', description='{self.description[:50]}...')"


class MCPResource:
    """Represents an MCP resource."""

    def __init__(self, uri: str, name: str, mime_type: str = None, description: str = None):
        self.uri = uri
        self.name = name
        self.mime_type = mime_type
        self.description = description

    def __repr__(self) -> str:
        return f"MCPResource(uri='{self.uri}', name='{self.name}')"


class BaseMCPClient(ABC):
    """Abstract base class for MCP clients."""

    def __init__(self, server_url: str, auth_token: str = None, timeout: int = 30):
        self.server_url = server_url
        self.auth_token = auth_token
        self.timeout = timeout
        self.logger = get_logger(self.__class__.__name__)
        self._session_id = None
        self._capabilities = {}
        self._tools: List[MCPTool] = []
        self._resources: List[MCPResource] = []
        self._client = None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to MCP server."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server."""

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""

    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the MCP server."""

    @abstractmethod
    async def list_resources(self) -> List[MCPResource]:
        """List available resources from the MCP server."""

    async def get_resource(self, uri: str) -> Dict[str, Any]:
        """Get a resource from the MCP server."""
        message = MCPMessage("resources/read", {"uri": uri})
        return await self._send_message(message)

    async def _send_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Send a message to the MCP server and return the response."""
        if not self._client:
            raise MCPConnectionError("Not connected to MCP server", self.server_url)

        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }

            # Always include Bearer token for Apify MCP
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            # Also include session ID if we have one
            if self._session_id:
                headers["mcp-session-id"] = self._session_id

            response = await self._client.post(
                self.server_url, json=message.to_dict(), headers=headers, timeout=self.timeout
            )

            if response.status_code == 401:
                raise MCPAuthenticationError("Authentication failed", server_url=self.server_url)
            elif response.status_code == 404:
                raise MCPConnectionError("MCP server not found", server_url=self.server_url)
            elif response.status_code >= 400:
                raise MCPProtocolError(
                    f"HTTP {response.status_code}: {response.text}",
                    method=message.method,
                    server_url=self.server_url,
                )

            # Handle different response types
            content_type = response.headers.get("content-type", "").lower()

            if "application/json" in content_type:
                response_data = response.json()
            elif "text/event-stream" in content_type:
                # Handle server-sent events by reading the response text
                response_text = response.text
                self.logger.debug(f"Received SSE response: {response_text}")
                # Extract session information from SSE response
                try:
                    # Look for JSON data and session ID in SSE format
                    lines = response_text.strip().split("\n")
                    session_id = None

                    # Check if there's an 'id:' line that contains session information
                    for line in lines:
                        if line.startswith("id: "):
                            session_id = line[4:].strip()  # Remove 'id: ' prefix
                            break

                    # Extract JSON data
                    for line in lines:
                        if line.startswith("data: "):
                            json_data = line[6:]  # Remove 'data: ' prefix
                            response_data = json.loads(json_data)
                            break
                    else:
                        # No JSON data found, create a basic response
                        response_data = {"result": {"status": "connected"}}

                    # Store session ID if found
                    if session_id and not self._session_id:
                        self._session_id = session_id
                        self.logger.info(f"Extracted session ID from SSE: {session_id}")

                except json.JSONDecodeError:
                    response_data = {"result": {"status": "connected"}}
            else:
                # Try to parse as JSON anyway
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    # If it fails, create a minimal response
                    response_data = {"result": {"status": "connected"}}

            if "error" in response_data:
                error = response_data["error"]
                raise MCPProtocolError(
                    f"MCP Error: {error.get('message', 'Unknown error')}",
                    method=message.method,
                    server_url=self.server_url,
                )

            return response_data.get("result", {})

        except httpx.TimeoutException:
            raise MCPConnectionError(
                f"Timeout connecting to MCP server after {self.timeout}s",
                server_url=self.server_url,
            )
        except httpx.ConnectError:
            raise MCPConnectionError("Failed to connect to MCP server", server_url=self.server_url)
        except json.JSONDecodeError:
            raise MCPProtocolError(
                "Invalid JSON response from MCP server",
                method=message.method,
                server_url=self.server_url,
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in MCP communication: {e}")
            raise MCPProtocolError(
                f"Unexpected error: {str(e)}", method=message.method, server_url=self.server_url
            )

    async def health_check(self) -> bool:
        """Check if the MCP server is healthy."""
        try:
            # Use ping or initialize method for health check
            message = MCPMessage("ping", {})
            await self._send_message(message)
            return True
        except Exception as e:
            self.logger.warning(f"MCP health check failed: {e}")
            return False


class HTTPMCPClient(BaseMCPClient):
    """HTTP-based MCP client implementation."""

    async def connect(self) -> None:
        """Establish HTTP connection to MCP server."""
        # Use persistent cookies to maintain session state
        self._client = httpx.AsyncClient(
            cookies={}, timeout=httpx.Timeout(self.timeout)  # Enable cookie persistence
        )

        try:
            # Initialize MCP session with Bearer token authentication
            message = MCPMessage(
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "Sales Assistant", "version": "1.0.0"},
                },
            )

            result = await self._send_message(message)
            self._capabilities = result.get("capabilities", {})

            # Session ID not needed for Apify MCP - authentication is via Bearer token
            self.logger.info(f"Connected to MCP server: {self.server_url}")
            self.logger.debug(f"Server capabilities: {self._capabilities}")

            # Load available tools and resources
            await self._refresh_tools_and_resources()

        except Exception as e:
            await self.disconnect()
            raise MCPConnectionError(
                f"Failed to initialize MCP session: {str(e)}", server_url=self.server_url
            )

    async def _authenticate(self) -> None:
        """Perform Bearer token authentication to get mcp-session-id."""
        if not self.auth_token:
            return

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.auth_token}",
            }

            # Make authentication request
            response = await self._client.post(
                self.server_url,
                json={"method": "authenticate"},
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 401:
                raise MCPAuthenticationError(
                    "Authentication failed - invalid API key", server_url=self.server_url
                )
            elif response.status_code >= 400:
                raise MCPAuthenticationError(
                    f"Authentication failed with HTTP {response.status_code}",
                    server_url=self.server_url,
                )

            # Extract session ID from response
            auth_data = response.json()
            if "mcp-session-id" in auth_data:
                self._session_id = auth_data["mcp-session-id"]
                self.logger.info(f"Authentication successful, session ID: {self._session_id}")
            elif "sessionId" in auth_data:
                self._session_id = auth_data["sessionId"]
                self.logger.info(f"Authentication successful, session ID: {self._session_id}")
            else:
                self.logger.warning("Authentication response did not include session ID")

        except httpx.TimeoutException:
            raise MCPAuthenticationError(
                f"Authentication timeout after {self.timeout}s", server_url=self.server_url
            )
        except httpx.ConnectError:
            raise MCPAuthenticationError(
                "Failed to connect for authentication", server_url=self.server_url
            )
        except json.JSONDecodeError:
            raise MCPAuthenticationError(
                "Invalid JSON response during authentication", server_url=self.server_url
            )
        except Exception as e:
            raise MCPAuthenticationError(
                f"Authentication error: {str(e)}", server_url=self.server_url
            )

    async def disconnect(self) -> None:
        """Close HTTP connection to MCP server."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._session_id = None
            self.logger.info(f"Disconnected from MCP server: {self.server_url}")

    async def _refresh_tools_and_resources(self) -> None:
        """Refresh the list of available tools and resources."""
        try:
            # List tools
            tools_message = MCPMessage("tools/list", {})
            tools_result = await self._send_message(tools_message)

            self._tools = []
            for tool_data in tools_result.get("tools", []):
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                )
                self._tools.append(tool)

            # List resources if supported
            if "resources" in self._capabilities:
                resources_message = MCPMessage("resources/list", {})
                resources_result = await self._send_message(resources_message)

                self._resources = []
                for resource_data in resources_result.get("resources", []):
                    resource = MCPResource(
                        uri=resource_data["uri"],
                        name=resource_data["name"],
                        mime_type=resource_data.get("mimeType"),
                        description=resource_data.get("description"),
                    )
                    self._resources.append(resource)

            self.logger.info(
                f"Loaded {len(self._tools)} tools and {len(self._resources)} resources"
            )

        except Exception as e:
            self.logger.warning(f"Failed to refresh tools and resources: {e}")

    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the MCP server."""
        return self._tools.copy()

    async def list_resources(self) -> List[MCPResource]:
        """List available resources from the MCP server."""
        return self._resources.copy()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        # Verify tool exists
        tool = next((t for t in self._tools if t.name == tool_name), None)
        if not tool:
            raise MCPToolError(
                f"Tool '{tool_name}' not found", tool_name=tool_name, server_url=self.server_url
            )

        message = MCPMessage("tools/call", {"name": tool_name, "arguments": arguments})

        try:
            result = await self._send_message(message)
            self.logger.info(f"Successfully called tool '{tool_name}'")
            return result

        except Exception as e:
            self.logger.error(f"Tool call failed for '{tool_name}': {e}")
            raise MCPToolError(
                f"Failed to call tool '{tool_name}': {str(e)}",
                tool_name=tool_name,
                server_url=self.server_url,
            )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
