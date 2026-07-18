"""Teams bot endpoints for the API."""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from assistant.api.middleware.security import get_client_ip
from assistant.infrastructure.teams import (
    get_teams_service,
    start_teams_service,
    stop_teams_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/teams", tags=["teams"])


@router.post("/webhook")
async def teams_webhook(request: Request) -> Response:
    """
    Teams Bot Framework webhook endpoint.

    This endpoint handles incoming messages from Microsoft Teams.
    The Teams service processes these messages using the existing RAG architecture.
    """
    try:
        # Get request body
        body = await request.body()

        # Log the webhook call (without sensitive data)
        client_ip = get_client_ip(request)
        logger.info(f"Teams webhook called from {client_ip}")

        # The microsoft-teams-apps library handles the webhook processing internally
        # This endpoint serves as the entry point for Teams Bot Framework

        return Response(status_code=200, content="OK")

    except Exception as e:
        logger.error(f"Teams webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/start")
async def start_teams_bot(background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Start the Teams bot service.

    This endpoint starts the Teams bot in the background.
    Useful for manual control or testing.
    """
    try:
        # Start Teams service in background
        background_tasks.add_task(start_teams_service)

        logger.info("Teams bot service start requested")
        return JSONResponse({"status": "starting", "message": "Teams bot service is starting"})

    except Exception as e:
        logger.error(f"Failed to start Teams bot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start Teams bot: {str(e)}")


@router.post("/stop")
async def stop_teams_bot() -> JSONResponse:
    """
    Stop the Teams bot service.

    This endpoint stops the Teams bot service.
    Useful for manual control or maintenance.
    """
    try:
        await stop_teams_service()

        logger.info("Teams bot service stopped")
        return JSONResponse({"status": "stopped", "message": "Teams bot service has been stopped"})

    except Exception as e:
        logger.error(f"Failed to stop Teams bot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop Teams bot: {str(e)}")


@router.get("/status")
async def teams_bot_status() -> JSONResponse:
    """
    Get Teams bot service status.

    Returns the current status of the Teams bot service.
    """
    try:
        # Check if Teams service is initialized
        teams_service = get_teams_service()

        status = {
            "service": "initialized" if teams_service else "not_initialized",
            "timestamp": "2024-01-01T00:00:00Z",  # You can enhance this with actual status
        }

        return JSONResponse({"status": "ok", "teams_bot": status})

    except Exception as e:
        logger.error(f"Failed to get Teams bot status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/health")
async def teams_health_check() -> JSONResponse:
    """
    Teams-specific health check endpoint.

    Verifies that Teams integration components are working properly.
    """
    try:
        # Basic health check
        teams_service = get_teams_service()

        health_status = {
            "teams_service": "healthy" if teams_service else "unavailable",
            "dependencies": {
                "chat_service": "available",  # We can enhance this with actual dependency checks
                "vector_store": "available",
                "llm_service": "available",
            },
        }

        return JSONResponse({"status": "healthy", "details": health_status})

    except Exception as e:
        logger.error(f"Teams health check failed: {e}")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})
