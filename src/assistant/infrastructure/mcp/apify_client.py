"""Apify MCP client implementation."""

from typing import Any, Dict, List
from urllib.parse import urlencode

from assistant.core.logging import get_logger

from .client import HTTPMCPClient
from .exceptions import MCPConnectionError


class ApifyMCPClient(HTTPMCPClient):
    """Specialized MCP client for Apify services."""

    def __init__(self, api_key: str, tools: List[str] = None, timeout: int = 60):
        """
        Initialize Apify MCP client.

        Args:
            api_key: Apify API key for authentication
            tools: List of Apify tools to enable (e.g., ['apify/rag-web-browser', 'compass/crawler-google-places'])
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.enabled_tools = tools or ["apify/rag-web-browser", "compass/crawler-google-places"]

        # Build Apify MCP server URL with tools parameter
        base_url = "https://mcp.apify.com/"
        if self.enabled_tools:
            tools_param = ",".join(self.enabled_tools)
            server_url = f"{base_url}?{urlencode({'tools': tools_param})}"
        else:
            server_url = base_url

        super().__init__(server_url=server_url, auth_token=api_key, timeout=timeout)

        self.logger = get_logger(self.__class__.__name__)

    async def connect(self) -> None:
        """Connect to Apify MCP server with proper authentication."""
        try:
            await super().connect()

            # Log successful connection with tool details
            tool_names = [tool.name for tool in self._tools]
            self.logger.info(f"Connected to Apify MCP with tools: {tool_names}")

        except Exception as e:
            self.logger.error(f"Failed to connect to Apify MCP: {e}")
            raise MCPConnectionError(
                f"Apify MCP connection failed: {str(e)}", server_url=self.server_url
            )

    async def crawl_website(self, url: str, max_pages: int = 10) -> Dict[str, Any]:
        """
        Crawl a website using Apify's web browser tool.

        Args:
            url: Website URL to crawl
            max_pages: Maximum number of pages to crawl

        Returns:
            Crawling results with extracted content
        """
        return await self.call_tool(
            "apify/rag-web-browser", {"url": url, "maxPages": max_pages, "extractContent": True}
        )

    async def search_google_places(
        self, query: str, location: str = None, max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Search Google Places using Apify's crawler.

        Args:
            query: Search query (e.g., "restaurants near me")
            location: Geographic location for search
            max_results: Maximum number of results to return

        Returns:
            Google Places search results
        """
        search_params = {"query": query, "maxResults": max_results}

        if location:
            search_params["location"] = location

        return await self.call_tool("compass/crawler-google-places", search_params)

    async def extract_web_content(self, url: str, selector: str = None) -> Dict[str, Any]:
        """
        Extract specific content from a web page.

        Args:
            url: Website URL to extract content from
            selector: CSS selector for specific content (optional)

        Returns:
            Extracted web content
        """
        extract_params = {"url": url, "extractContent": True}

        if selector:
            extract_params["cssSelector"] = selector

        return await self.call_tool("apify/rag-web-browser", extract_params)

    async def get_website_info(self, url: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a website.

        Args:
            url: Website URL to analyze

        Returns:
            Website information including metadata, content, and structure
        """
        return await self.call_tool(
            "apify/rag-web-browser",
            {
                "url": url,
                "extractContent": True,
                "extractMetadata": True,
                "extractLinks": True,
                "maxPages": 1,
            },
        )

    async def search_and_extract(
        self, search_query: str, location: str = None, extract_website_content: bool = False
    ) -> Dict[str, Any]:
        """
        Combined search and content extraction workflow.

        Args:
            search_query: What to search for
            location: Geographic location for search
            extract_website_content: Whether to extract content from found websites

        Returns:
            Combined search and extraction results
        """
        # First, search for places
        places_results = await self.search_google_places(
            query=search_query, location=location, max_results=10
        )

        results = {
            "search_query": search_query,
            "location": location,
            "places": places_results,
            "website_content": [],
        }

        # Optionally extract content from found websites
        if extract_website_content and "results" in places_results:
            for place in places_results.get("results", [])[:3]:  # Limit to first 3
                if "website" in place and place["website"]:
                    try:
                        content = await self.extract_web_content(place["website"])
                        results["website_content"].append(
                            {
                                "place_name": place.get("name"),
                                "website": place["website"],
                                "content": content,
                            }
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to extract content from {place['website']}: {e}"
                        )

        return results

    def get_available_tools_info(self) -> List[Dict[str, Any]]:
        """
        Get information about available Apify tools.

        Returns:
            List of tool information dictionaries
        """
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in self._tools
        ]

    async def health_check(self) -> bool:
        """
        Check if Apify MCP server is healthy and tools are accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check base connection
            if not await super().health_check():
                return False

            # Verify tools are loaded
            if not self._tools:
                self.logger.warning("No tools loaded from Apify MCP")
                return False

            # Test a simple tool call if possible
            return True

        except Exception as e:
            self.logger.error(f"Apify MCP health check failed: {e}")
            return False
