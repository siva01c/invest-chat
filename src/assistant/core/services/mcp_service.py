"""MCP integration service for enhanced AI capabilities."""

import asyncio
from typing import Any, Dict, List, Optional

from assistant.core.exceptions import ErrorCode, ServiceException
from assistant.core.interfaces.base import BaseService
from assistant.core.logging import get_logger, log_service_method
from assistant.infrastructure.mcp.apify_client import ApifyMCPClient
from assistant.infrastructure.mcp.exceptions import MCPException


class MCPService(BaseService):
    """Service for managing MCP tool integrations."""

    def __init__(self, apify_api_key: str = None, enabled_tools: List[str] = None):
        """
        Initialize MCP service.

        Args:
            apify_api_key: Apify API key for web crawling tools
            enabled_tools: List of MCP tools to enable
        """
        self.logger = get_logger(self.__class__.__name__)
        self.apify_api_key = apify_api_key
        self.enabled_tools = enabled_tools or [
            "apify/rag-web-browser",
            "compass/crawler-google-places",
        ]

        # MCP clients
        self._apify_client: Optional[ApifyMCPClient] = None
        self._connected = False

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "MCPService"

    async def initialize(self) -> None:
        """Initialize MCP connections and tools."""
        try:
            if self.apify_api_key:
                self._apify_client = ApifyMCPClient(
                    api_key=self.apify_api_key, tools=self.enabled_tools, timeout=60
                )
                await self._apify_client.connect()
                self.logger.info("Apify MCP client connected successfully")

            self._connected = True
            self.logger.info("MCP service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP service: {e}")
            raise ServiceException(
                "MCP service initialization failed",
                error_code=ErrorCode.SERVICE_ERROR,
                details={"error": str(e)},
            )

    @log_service_method()
    async def cleanup(self) -> None:
        """Cleanup MCP connections."""
        try:
            if self._apify_client:
                await self._apify_client.disconnect()
                self._apify_client = None

            self._connected = False
            self.logger.info("MCP service cleaned up successfully")

        except Exception as e:
            self.logger.error(f"Error during MCP service cleanup: {e}")

    async def is_healthy(self) -> bool:
        """Check if MCP service is healthy."""
        if not self._connected:
            return False

        try:
            if self._apify_client:
                return await self._apify_client.health_check()
            return True

        except Exception as e:
            self.logger.warning(f"MCP health check failed: {e}")
            return False

    @log_service_method()
    async def search_web_content(
        self, query: str, location: str = None, max_results: int = 5, extract_content: bool = True
    ) -> Dict[str, Any]:
        """
        Search for web content using MCP tools.

        Args:
            query: Search query
            location: Geographic location for search
            max_results: Maximum number of results
            extract_content: Whether to extract content from found pages

        Returns:
            Search results with optional content extraction
        """
        if not self._apify_client:
            raise ServiceException(
                "Apify MCP client not available", error_code=ErrorCode.SERVICE_UNAVAILABLE
            )

        try:
            # Search for places/websites
            search_results = await self._apify_client.search_google_places(
                query=query, location=location, max_results=max_results
            )

            results = {
                "query": query,
                "location": location,
                "search_results": search_results,
                "extracted_content": [],
            }

            # Extract content from found websites if requested
            if extract_content and "results" in search_results:
                content_tasks = []
                for place in search_results.get("results", [])[:3]:  # Limit to top 3
                    if "website" in place and place["website"]:
                        task = self._extract_website_content_safe(
                            place["website"], place.get("name", "Unknown")
                        )
                        content_tasks.append(task)

                if content_tasks:
                    extracted_contents = await asyncio.gather(
                        *content_tasks, return_exceptions=True
                    )
                    for content in extracted_contents:
                        if isinstance(content, dict):
                            results["extracted_content"].append(content)

            self.logger.info(
                f"Web content search completed: {len(results['extracted_content'])} pages extracted"
            )
            return results

        except MCPException as e:
            self.logger.error(f"MCP error during web search: {e}")
            raise ServiceException(
                f"Web search failed: {str(e)}",
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                details={"mcp_error": str(e)},
            )
        except Exception as e:
            self.logger.error(f"Unexpected error during web search: {e}")
            raise ServiceException(
                f"Web search failed: {str(e)}", error_code=ErrorCode.UNKNOWN_ERROR
            )

    async def _extract_website_content_safe(
        self, url: str, site_name: str = "Unknown"
    ) -> Optional[Dict[str, Any]]:
        """
        Safely extract content from a website with error handling.

        Args:
            url: Website URL
            site_name: Name of the site for logging

        Returns:
            Extracted content or None if failed
        """
        try:
            content = await self._apify_client.extract_web_content(url)
            return {"site_name": site_name, "url": url, "content": content, "success": True}
        except Exception as e:
            self.logger.warning(f"Failed to extract content from {url}: {e}")
            return {"site_name": site_name, "url": url, "error": str(e), "success": False}

    @log_service_method()
    async def crawl_website(
        self, url: str, max_pages: int = 5, extract_content: bool = True
    ) -> Dict[str, Any]:
        """
        Crawl a specific website using MCP tools.

        Args:
            url: Website URL to crawl
            max_pages: Maximum number of pages to crawl
            extract_content: Whether to extract content

        Returns:
            Crawling results
        """
        if not self._apify_client:
            raise ServiceException(
                "Apify MCP client not available", error_code=ErrorCode.SERVICE_UNAVAILABLE
            )

        try:
            results = await self._apify_client.crawl_website(url=url, max_pages=max_pages)

            self.logger.info(f"Website crawl completed: {url}")
            return {"url": url, "max_pages": max_pages, "results": results, "success": True}

        except MCPException as e:
            self.logger.error(f"MCP error during website crawl: {e}")
            raise ServiceException(
                f"Website crawl failed: {str(e)}",
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                details={"url": url, "mcp_error": str(e)},
            )

    @log_service_method()
    async def get_website_info(self, url: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a website.

        Args:
            url: Website URL to analyze

        Returns:
            Website information and metadata
        """
        if not self._apify_client:
            raise ServiceException(
                "Apify MCP client not available", error_code=ErrorCode.SERVICE_UNAVAILABLE
            )

        try:
            info = await self._apify_client.get_website_info(url)

            self.logger.info(f"Website info retrieved: {url}")
            return {"url": url, "info": info, "success": True}

        except MCPException as e:
            self.logger.error(f"MCP error getting website info: {e}")
            raise ServiceException(
                f"Failed to get website info: {str(e)}",
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                details={"url": url, "mcp_error": str(e)},
            )

    @log_service_method()
    async def enhanced_search_and_extract(
        self,
        query: str,
        location: str = None,
        include_web_content: bool = False,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Enhanced search that combines multiple MCP tools for comprehensive results.

        Args:
            query: Search query
            location: Geographic location
            include_web_content: Whether to extract web content from results
            max_results: Maximum number of results

        Returns:
            Comprehensive search results
        """
        if not self._apify_client:
            raise ServiceException(
                "Apify MCP client not available", error_code=ErrorCode.SERVICE_UNAVAILABLE
            )

        try:
            results = await self._apify_client.search_and_extract(
                search_query=query, location=location, extract_website_content=include_web_content
            )

            self.logger.info(f"Enhanced search completed for: {query}")
            return {
                "enhanced_search": True,
                "query": query,
                "location": location,
                "results": results,
                "success": True,
            }

        except MCPException as e:
            self.logger.error(f"MCP error during enhanced search: {e}")
            raise ServiceException(
                f"Enhanced search failed: {str(e)}",
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                details={"query": query, "mcp_error": str(e)},
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about available MCP tools.

        Returns:
            List of available tools and their capabilities
        """
        tools = []

        if self._apify_client:
            apify_tools = self._apify_client.get_available_tools_info()
            for tool in apify_tools:
                tool["provider"] = "Apify"
                tools.append(tool)

        return tools

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status.

        Returns:
            Service status information
        """
        return {
            "service_name": self.get_service_name(),
            "connected": self._connected,
            "apify_client": {
                "available": self._apify_client is not None,
                "connected": self._apify_client is not None and self._connected,
            },
            "enabled_tools": self.enabled_tools,
            "available_tools_count": len(self.get_available_tools()),
        }
