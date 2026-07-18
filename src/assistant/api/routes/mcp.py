"""MCP health and status endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from assistant.core.config.services import get_mcp_service
from assistant.core.exceptions import ServiceException
from assistant.core.logging import get_logger

router = APIRouter(prefix="/mcp", tags=["MCP"])
logger = get_logger(__name__)


@router.get("/health")
async def mcp_health_check() -> Dict[str, Any]:
    """
    Check MCP service health status.

    Returns:
        Health status information
    """
    try:
        mcp_service = get_mcp_service()

        if not mcp_service:
            return {"status": "disabled", "message": "MCP service not configured", "healthy": False}

        is_healthy = await mcp_service.is_healthy()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "healthy": is_healthy,
            "service_available": True,
        }

    except Exception as e:
        logger.error(f"MCP health check failed: {e}")
        return {"status": "error", "healthy": False, "error": str(e)}


@router.get("/status")
async def mcp_status() -> Dict[str, Any]:
    """
    Get comprehensive MCP service status.

    Returns:
        Detailed status information
    """
    try:
        mcp_service = get_mcp_service()

        if not mcp_service:
            return {
                "enabled": False,
                "reason": "MCP service not configured or disabled",
                "available_tools": [],
            }

        # Get service status
        status = mcp_service.get_service_status()

        # Get available tools
        tools = mcp_service.get_available_tools()

        # Perform health check
        health_status = await mcp_service.is_healthy()

        return {
            "enabled": True,
            "healthy": health_status,
            "service_status": status,
            "available_tools": tools,
            "tools_count": len(tools),
        }

    except Exception as e:
        logger.error(f"MCP status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get MCP status: {str(e)}")


@router.get("/tools")
async def list_mcp_tools() -> Dict[str, Any]:
    """
    List available MCP tools.

    Returns:
        List of available tools and their capabilities
    """
    try:
        mcp_service = get_mcp_service()

        if not mcp_service:
            return {"tools": [], "message": "MCP service not available"}

        tools = mcp_service.get_available_tools()

        return {"tools": tools, "count": len(tools)}

    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list MCP tools: {str(e)}")


@router.post("/test-search")
async def test_mcp_search(
    query: str, max_results: int = 3, extract_content: bool = False
) -> Dict[str, Any]:
    """
    Test MCP web search functionality.

    Args:
        query: Search query
        max_results: Maximum number of results
        extract_content: Whether to extract content from results

    Returns:
        Search results and performance metrics
    """
    try:
        mcp_service = get_mcp_service()

        if not mcp_service:
            raise HTTPException(status_code=503, detail="MCP service not available")

        # Perform search
        import time

        start_time = time.time()

        results = await mcp_service.search_web_content(
            query=query, max_results=max_results, extract_content=extract_content
        )

        duration = time.time() - start_time

        return {
            "success": True,
            "query": query,
            "results": results,
            "performance": {
                "duration_seconds": round(duration, 3),
                "results_count": len(results.get("search_results", {}).get("results", [])),
                "extracted_content_count": len(results.get("extracted_content", [])),
            },
        }

    except ServiceException as e:
        logger.error(f"MCP search test failed: {e}")
        raise HTTPException(status_code=400, detail=f"Search test failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in MCP search test: {e}")
        raise HTTPException(status_code=500, detail=f"Search test failed: {str(e)}")


@router.post("/test-crawl")
async def test_mcp_crawl(url: str, max_pages: int = 1) -> Dict[str, Any]:
    """
    Test MCP website crawling functionality.

    Args:
        url: Website URL to crawl
        max_pages: Maximum number of pages to crawl

    Returns:
        Crawling results and performance metrics
    """
    try:
        mcp_service = get_mcp_service()

        if not mcp_service:
            raise HTTPException(status_code=503, detail="MCP service not available")

        # Perform crawl
        import time

        start_time = time.time()

        results = await mcp_service.crawl_website(
            url=url, max_pages=max_pages, extract_content=True
        )

        duration = time.time() - start_time

        return {
            "success": True,
            "url": url,
            "results": results,
            "performance": {"duration_seconds": round(duration, 3), "pages_crawled": max_pages},
        }

    except ServiceException as e:
        logger.error(f"MCP crawl test failed: {e}")
        raise HTTPException(status_code=400, detail=f"Crawl test failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in MCP crawl test: {e}")
        raise HTTPException(status_code=500, detail=f"Crawl test failed: {str(e)}")
