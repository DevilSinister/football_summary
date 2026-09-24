"""Resolving a typed team name to a team row, and loading squads into it.

Two callers share this module: the processing router, which resolves whatever
the user typed at the team-confirmation step, and scripts/seed_roster.py, which
loads data/rosters/*.json. They must agree on resolution or the seeded squad
attaches to a different team row than the one a job writes to, and the roster
lookup silently returns no name.

It deliberately does not live in routers/processing.py - importing that module
pulls in the whole detection stack (torch, ultralytics, httpx), which a seeding
CLI has no use for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Player, Team, TeamAlias
from .team_naming import normalise_team_name

ROSTER_DIR = Path(__file__).resolve().parent.parent / "data" / "rosters"


@dataclass
class SeedStats:
    """What one seeding run changed, for the CLI summary."""

    teams_created: int = 0
    teams_matched: int = 0
    aliases_added: int = 0
    players_inserted: int = 0
    players_named: int = 0
    players_renamed: int = 0
    players_unchanged: int = 0
    codes_set: int = 0
    notes: list[str] = field(default_factory=list)

    def merge(self, other: "SeedStats") -> None:
        self.teams_created += other.teams_created
        self.teams_matched += other.teams_matched
        self.aliases_added += other.aliases_added
        self.players_inserted += other.players_inserted
        self.players_named += other.players_named
        self.players_renamed += other.players_renamed
        self.players_unchanged += other.players_unchanged
        self.codes_set += other.codes_set
        self.notes.extend(other.notes)

    @property
    def wrote_anything(self) -> bool:
        return bool(
            self.teams_created
            or self.aliases_added
            or self.players_inserted
            or self.players_named
            or self.players_renamed
            or self.codes_set
        )


def find_team(db: Session, name: str) -> Team | None:
    """The team a typed name refers to, or None.

    Three chances to match before giving up, cheapest first: the literal name,
    a registered alias, and finally the normalised form of every stored team
    name. Only the third needs a scan, and `teams` holds one row per club, not
    one per match.
    """
    if not name:
        return None

    team = db.scalar(select(Team).where(func.lower(Team.team_name) == name.lower()))
    if team is not None:
        return team

    key = normalise_team_name(name)
    if not key:
        return None

    alias = db.scalar(select(TeamAlias).where(TeamAlias.alias == key))
    if alias is not None:
        return alias.team

    for candidate in db.scalars(select(Team)).all():
        if normalise_team_name(candidate.team_name) == key:
            return candidate
    return None


def add_alias(db: Session, team: Team, raw_alias: str) -> bool:
    """Bind a spelling to a team. True when a row was added.

    An alias already pointing at another team is left alone rather than
    stolen - two clubs can legitimately share a short form, and silently
    repointing it would move a squad out from under the other one.
    """
    key = normalise_team_name(raw_alias)
    if not key:
        return False

    existing = db.scalar(select(TeamAlias).where(TeamAlias.alias == key))
    if existing is not None:
        return False

    db.add(TeamAlias(alias=key, team_id=team.team_id))
    db.flush()
    return True


def resolve_team(db: Session, name: str, color: str | None = None) -> Team:
    """Find or create the team for a typed name, registering its spelling.

    The normalised form of a newly created team is stored as an alias, so the
    next job that spells the name differently lands on this same row instead of
    forking a second one.
    """
    team = find_team(db, name)
    if team is None:
        team = Team(team_name=name, primary_color=color)
        db.add(team)
        db.flush()
        add_alias(db, team, name)
    elif color:
        team.primary_color = color
    return team


def _seed_player(db: Session, team: Team, entry: dict, stats: SeedStats, dry_run: bool) -> None:
    jersey_no = entry.get("jersey_no")
    name = (entry.get("name") or "").strip()
    if jersey_no is None or not name:
        stats.notes.append(f"skipped entry with no number or no name: {entry!r}")
        return

    jersey_no = int(jersey_no)
    player = db.scalar(
        select(Player).where(Player.team_id == team.team_id, Player.jersey_no == jersey_no)
    )

    if player is None:
        stats.players_inserted += 1
        if not dry_run:
            db.add(Player(team_id=team.team_id, jersey_no=jersey_no, name=name))
        return

    if player.name is None:
        # An empty slot the pipeline created when it read this number off a
        # shirt but had no roster to name it from. Adopt it rather than insert
        # a second row - the (team_id, jersey_no) constraint would reject that
        # anyway, and occurrences already pointing at it gain a name for free.
        stats.players_named += 1
        if not dry_run:
            player.name = name
    elif player.name != name:
        stats.players_renamed += 1
        stats.notes.append(f"#{jersey_no} {team.team_name}: {player.name!r} -> {name!r}")
        if not dry_run:
            player.name = name
    else:
        stats.players_unchanged += 1


def seed_roster(db: Session, payload: dict, dry_run: bool = False) -> SeedStats:
    """Load one roster payload into the database. Safe to run repeatedly."""
    stats = SeedStats()

    team_name = (payload.get("team_name") or "").strip()
    if not team_name:
        raise ValueError("roster payload has no team_name")

    team = find_team(db, team_name)
    if team is None:
        stats.teams_created += 1
        team = Team(team_name=team_name)
        db.add(team)
        db.flush()
    else:
        stats.teams_matched += 1

    for raw_alias in [team_name, *(payload.get("aliases") or [])]:
        if add_alias(db, team, raw_alias):
            stats.aliases_added += 1

    # The three-letter scoreboard code. Filled only when empty, so a code
    # corrected by hand in the database is never overwritten by the file.
    short_code = (payload.get("short_code") or "").strip().upper()
    if short_code:
        if len(short_code) != 3 or not short_code.isalpha():
            stats.notes.append(f"{team_name}: short_code {short_code!r} is not three letters; ignored")
        else:
            if team.short_code is None:
                stats.codes_set += 1
                if not dry_run:
                    team.short_code = short_code
            elif team.short_code != short_code:
                stats.notes.append(
                    f"{team_name}: stored short_code {team.short_code!r} kept; file says {short_code!r}"
                )
            if add_alias(db, team, short_code):
                stats.aliases_added += 1

    for entry in payload.get("players") or []:
        _seed_player(db, team, entry, stats, dry_run)

    if payload.get("needs_review"):
        stats.notes.append(f"{team_name}: needs_review is set - squad numbers are unverified")

    return stats


def load_roster_files(
    directory: Path | None = None, team: str | None = None
) -> list[tuple[Path, dict]]:
    """Every roster file, or just the one matching `team`.

    Files beginning with "_" are skipped so _template.json is never seeded.
    """
    directory = directory or ROSTER_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"no roster directory at {directory}")

    loaded: list[tuple[Path, dict]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if team and normalise_team_name(payload.get("team_name")) != normalise_team_name(team):
            continue
        loaded.append((path, payload))
    return loaded
