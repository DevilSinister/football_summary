"""Team list for the confirmation screen's searchable team picker."""

from fastapi import APIRouter, Depends
from sqlalchemy import exists, select
from sqlalchemy.orm import Session, selectinload

from ..database import get_db
from ..models import Player, Team


router = APIRouter(prefix="/api/teams", tags=["teams"])


@router.get("")
def list_teams(db: Session = Depends(get_db)):
    """Every team a user can pick, with the spellings that should find it.

    Only teams with at least one named player are offered. A row without a
    squad is one a job created from a typed name ("Athletico Madrid" beside the
    seeded "Atletico Madrid"), and offering it would keep the misspelling alive.
    The list is one row per club (about 120), so the client filters it locally.
    """
    has_squad = exists().where(Player.team_id == Team.team_id, Player.name.is_not(None))
    teams = db.scalars(
        select(Team).where(has_squad).options(selectinload(Team.aliases)).order_by(Team.team_name)
    ).all()
    return [
        {
            "team_id": team.team_id,
            "team_name": team.team_name,
            "short_code": team.short_code,
            "aliases": sorted({alias.alias for alias in team.aliases}),
        }
        for team in teams
    ]
