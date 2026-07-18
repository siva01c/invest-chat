#!/usr/bin/env python3
"""Standalone Microsoft Teams bot runner.

This module provides a standalone Teams bot that integrates with the existing
RAG chatbot architecture. It can be run independently or alongside the main
FastAPI application.

Usage:
    python -m assistant.teams_bot
    or
    python src/assistant/teams_bot.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from assistant.config import get_settings
from assistant.core.config.services import cleanup_services, initialize_services
from assistant.infrastructure.teams import start_teams_service

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point for the Teams bot."""
    # Load environment variables
    project_root = Path(__file__).parent.parent.parent
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path)

    # Get settings
    settings = get_settings()

    # Check required environment variables for Teams
    required_env_vars = ["TEAMS_BOT_ID", "TEAMS_BOT_PASSWORD", "OPENAI_API_KEY"]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("Please set the following environment variables:")
        logger.info("- TEAMS_BOT_ID: Your Microsoft Teams Bot ID")
        logger.info("- TEAMS_BOT_PASSWORD: Your Microsoft Teams Bot Password")
        logger.info("- OPENAI_API_KEY: Your OpenAI API key")
        sys.exit(1)

    logger.info("Starting Microsoft Teams bot with RAG capabilities...")

    try:
        # Initialize services (async parts)
        await initialize_services()
        logger.info("Services initialized successfully")

        # Get Teams bot configuration
        teams_host = os.getenv("TEAMS_BOT_HOST", "127.0.0.1")
        teams_port = int(os.getenv("TEAMS_BOT_PORT", "3978"))

        logger.info(f"Starting Teams bot on {teams_host}:{teams_port}")

        # Start the Teams service
        await start_teams_service(host=teams_host, port=teams_port)

    except KeyboardInterrupt:
        logger.info("Teams bot stopped by user")
    except Exception as e:
        logger.error(f"Error starting Teams bot: {e}")
        sys.exit(1)
    finally:
        # Cleanup services
        cleanup_services()
        logger.info("Teams bot shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Teams bot interrupted")
    except Exception as e:
        logger.error(f"Teams bot failed: {e}")
        sys.exit(1)
