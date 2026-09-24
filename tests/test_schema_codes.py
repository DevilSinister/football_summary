"""teams.short_code: the additive migration and the seeder that fills it."""

import sys
import types
import unittest

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class _Base(DeclarativeBase):
    pass


_engine = create_engine("sqlite://", future=True)
_fake_database = types.ModuleType("backend.database")
_fake_database.Base = _Base
_fake_database.engine = _engine
_fake_database.SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False, future=True)
_fake_database.get_db = lambda: iter(())
sys.modules.setdefault("backend.database", _fake_database)
# Adopt whichever stand-in registered first (see tests/test_roster_seed.py).
_fake_database = sys.modules["backend.database"]
_Base = _fake_database.Base
_engine = _fake_database.engine

from backend import models, roster  # noqa: E402
from backend.schema import ensure_schema  # noqa: E402


class EnsureSchemaTests(unittest.TestCase):
    def test_adds_the_column_to_an_existing_table_once(self):
        engine = create_engine("sqlite://", future=True)
        with engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE teams (team_id INTEGER PRIMARY KEY, team_name VARCHAR(100) NOT NULL, "
                "primary_color VARCHAR(7))"
            ))
            connection.execute(text("INSERT INTO teams (team_name) VALUES ('Real Madrid')"))
        self.assertEqual(ensure_schema(engine), ["teams.short_code", "ix_teams_short_code"])
        columns = {column["name"] for column in inspect(engine).get_columns("teams")}
        self.assertIn("short_code", columns)
        self.assertEqual(ensure_schema(engine), [])  # idempotent
        with engine.connect() as connection:
            self.assertEqual(connection.execute(text("SELECT team_name FROM teams")).scalar(), "Real Madrid")

    def test_nothing_to_do_before_the_table_exists(self):
        self.assertEqual(ensure_schema(create_engine("sqlite://", future=True)), [])

    def test_adds_event_clip_times_once_and_keeps_existing_rows(self):
        engine = create_engine("sqlite://", future=True)
        with engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE events (event_id INTEGER PRIMARY KEY, match_id INTEGER NOT NULL, "
                "event_type VARCHAR(50) NOT NULL, time_seconds FLOAT)"
            ))
            connection.execute(text("INSERT INTO events (match_id, event_type, time_seconds) VALUES (1, 'goal', 4.2)"))
        self.assertEqual(ensure_schema(engine), ["events.clip_start_seconds", "events.clip_end_seconds"])
        self.assertEqual(ensure_schema(engine), [])
        with engine.connect() as connection:
            row = connection.execute(text(
                "SELECT event_type, time_seconds, clip_start_seconds, clip_end_seconds FROM events"
            )).one()
        self.assertEqual(tuple(row), ("goal", 4.2, None, None))


class SeedShortCodeTests(unittest.TestCase):
    def setUp(self):
        _Base.metadata.drop_all(_engine)
        _Base.metadata.create_all(_engine)
        self.Session = _fake_database.SessionLocal

    def payload(self, code):
        return {"team_name": "FC Porto", "short_code": code, "aliases": ["porto"], "players": [
            {"jersey_no": 99, "name": "Diogo Costa", "position": "GK"}]}

    def test_code_is_set_and_registered_as_an_alias(self):
        with self.Session() as db:
            stats = roster.seed_roster(db, self.payload("POR"))
            db.commit()
            team = db.scalar(select(models.Team))
            self.assertEqual(team.short_code, "POR")
            self.assertEqual(stats.codes_set, 1)
            self.assertIs(roster.find_team(db, "POR"), team)

    def test_a_stored_code_is_never_overwritten(self):
        with self.Session() as db:
            roster.seed_roster(db, self.payload("FCP"))
            db.commit()
            stats = roster.seed_roster(db, self.payload("POR"))
            db.commit()
            self.assertEqual(db.scalar(select(models.Team)).short_code, "FCP")
            self.assertEqual(stats.codes_set, 0)
            self.assertTrue(any("kept" in note for note in stats.notes))

    def test_an_invalid_code_is_ignored(self):
        with self.Session() as db:
            stats = roster.seed_roster(db, self.payload("PORTO"))
            db.commit()
            self.assertIsNone(db.scalar(select(models.Team)).short_code)
            self.assertTrue(any("three letters" in note for note in stats.notes))

    def test_existing_team_row_is_adopted_under_its_stored_spelling(self):
        # FootballDB already had "Fc Porto" from an earlier job.
        with self.Session() as db:
            db.add(models.Team(team_name="Fc Porto"))
            db.commit()
            stats = roster.seed_roster(db, self.payload("POR"))
            db.commit()
            self.assertEqual(stats.teams_created, 0)
            self.assertEqual(db.scalar(select(models.Team)).short_code, "POR")


if __name__ == "__main__":
    unittest.main()
