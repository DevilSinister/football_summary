"""Additive schema changes for tables that already exist.

``Base.metadata.create_all`` creates missing tables but never alters an
existing one, and this project has no migration tool. Each change here is
idempotent - it inspects the live table first - so it is safe to run on every
startup and from the seeder.

Only additive, nullable changes belong here. Anything that rewrites data or
drops a column needs a reviewed, one-off script.

It takes the engine as an argument and does not import backend.database, so
tests can run it against SQLite without a SQL Server connection.
"""

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def _add_team_short_code(engine: Engine) -> list[str]:
    applied: list[str] = []
    inspector = inspect(engine)
    if "teams" not in inspector.get_table_names():
        return applied  # create_all will build it with the column

    columns = {column["name"] for column in inspector.get_columns("teams")}
    with engine.begin() as connection:
        if "short_code" not in columns:
            # Three-letter scoreboard code (RMA, MCI). Nullable: most teams a
            # user types have none until a roster file or a person supplies it.
            connection.execute(text("ALTER TABLE teams ADD short_code VARCHAR(3) NULL"))
            applied.append("teams.short_code")

    indexes = {index["name"] for index in inspect(engine).get_indexes("teams")}
    if "ix_teams_short_code" not in indexes:
        with engine.begin() as connection:
            connection.execute(text("CREATE INDEX ix_teams_short_code ON teams (short_code)"))
        applied.append("ix_teams_short_code")
    return applied


def _add_event_clip_times(engine: Engine) -> list[str]:
    applied: list[str] = []
    inspector = inspect(engine)
    if "events" not in inspector.get_table_names():
        return applied

    columns = {column["name"] for column in inspector.get_columns("events")}
    with engine.begin() as connection:
        # Where the event's highlight clip starts and ends in the source video,
        # in seconds. NULL for passes, which are not clipped, and for rows
        # written before clips were recorded.
        for column in ("clip_start_seconds", "clip_end_seconds"):
            if column not in columns:
                connection.execute(text(f"ALTER TABLE events ADD {column} FLOAT NULL"))
                applied.append(f"events.{column}")
    return applied


def _add_column(engine: Engine, table: str, column: str, definition: str) -> list[str]:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return []  # create_all builds it with the column
    if column in {c["name"] for c in inspector.get_columns(table)}:
        return []
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {table} ADD {column} {definition}"))
    return [f"{table}.{column}"]


def _widen_to_unicode(engine: Engine, table: str, column: str, length: int) -> list[str]:
    """VARCHAR -> NVARCHAR on SQL Server, keeping every stored value.

    The one exception to "additive only": widening is lossless, and the
    squad import (2026-10-15) showed VARCHAR under SQL_Latin1_General_CP1
    turning 84 player names' letters outside Windows-1252 into "?" or
    unaccented letters (Dragusin, Yigit). Only unindexed, nullable name
    columns are widened here.
    """
    if engine.dialect.name != "mssql":
        return []
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return []
    columns = {c["name"]: c for c in inspector.get_columns(table)}
    if column not in columns or "NVARCHAR" in str(columns[column]["type"]).upper():
        return []
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {table} ALTER COLUMN {column} NVARCHAR({length}) NULL"))
    return [f"{table}.{column} nvarchar"]


def ensure_schema(engine: Engine) -> list[str]:
    """Apply every pending additive change; returns what was applied."""
    return (
        _add_team_short_code(engine)
        + _add_event_clip_times(engine)
        # GK/DF/MF/FW from the squad import; saves name the keeper from it.
        + _add_column(engine, "players", "position", "VARCHAR(2) NULL")
        # scorer / assister / fouler ... so a goal's two players are told apart.
        + _add_column(engine, "occurrences", "role", "VARCHAR(20) NULL")
        + _widen_to_unicode(engine, "players", "name", 100)
        + _widen_to_unicode(engine, "occurrences", "detected_player_name", 100)
    )
