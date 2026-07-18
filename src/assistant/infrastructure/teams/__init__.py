"""Microsoft Teams integration infrastructure."""

from .teams_cards import TeamsCardBuilder, TeamsMessageFormatter
from .teams_service import (
    TeamsService,
    get_teams_service,
    start_teams_service,
    stop_teams_service,
)

__all__ = [
    "TeamsService",
    "TeamsCardBuilder",
    "TeamsMessageFormatter",
    "get_teams_service",
    "start_teams_service",
    "stop_teams_service",
]
