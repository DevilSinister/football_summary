"""Roster seeding and team-name resolution, against an in-memory SQLite copy.

backend.database connects to SQL Server at import time, so it is replaced with
a stand-in bound to SQLite, the same way tests/test_persist_match.py does. The
real seeding code runs untouched.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class _Base(DeclarativeBase):
    pass


_engine = create_engine("sqlite://", future=True)
_fake_database = types.ModuleType("backend.database")
_fake_database.Base = _Base
_fake_database.engine = _engine
_fake_database.SessionLocal = sessionmaker(
    bind=_engine, autocommit=False, autoflush=False, future=True
)


def _get_db():
    database = _fake_database.SessionLocal()
    try:
        yield database
    finally:
        database.close()


_fake_database.get_db = _get_db
sys.modules.setdefault("backend.database", _fake_database)

# Another test module may have registered its own stand-in first, and
# setdefault leaves that one in place. The models bind to whichever won, so
# adopt its Base and engine rather than this module's - otherwise create_all
# here builds tables in metadata nothing is mapped to.
_fake_database = sys.modules["backend.database"]
_Base = _fake_database.Base
_engine = _fake_database.engine

from backend import models, roster  # noqa: E402
from backend.team_naming import normalise_team_name  # noqa: E402

REAL_MADRID = {
    "team_name": "Real Madrid",
    "aliases": ["real madrid cf", "rma", "los blancos"],
    "season": "2025-26",
    "players": [
        {"jersey_no": 7, "name": "Vinicius Junior", "position": "FW"},
        {"jersey_no": 9, "name": "Endrick", "position": "FW"},
        {"jersey_no": 10, "name": "Kylian Mbappe", "position": "FW"},
    ],
}


class NormaliseTeamNameTests(unittest.TestCase):
    def test_strips_club_affixes_punctuation_and_case(self):
        for raw, expected in [
            ("Real Madrid C.F.", "real madrid"),
            ("FC Barcelona", "barcelona"),
            ("  liverpool  fc ", "liverpool"),
            ("Paris Saint-Germain", "paris saint germain"),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(normalise_team_name(raw), expected)

    def test_strips_accents_so_an_umlaut_is_optional(self):
        self.assertEqual(
            normalise_team_name("Bayern München"), normalise_team_name("Bayern Munchen")
        )

    def test_keeps_meaningful_tokens_that_distinguish_clubs(self):
        # Dropping these would merge clubs that share a first word.
        self.assertEqual(normalise_team_name("Manchester United"), "manchester united")
        self.assertEqual(normalise_team_name("Manchester City"), "manchester city")
        self.assertNotEqual(normalise_team_name("Atletico"), normalise_team_name("Atletico Madrid"))

    def test_unusable_input_is_not_a_key(self):
        for raw in ["", "   ", "...", None]:
            with self.subTest(raw=raw):
                self.assertEqual(normalise_team_name(raw), "")


class SeedRosterTests(unittest.TestCase):
    def setUp(self):
        _Base.metadata.drop_all(_engine)
        _Base.metadata.create_all(_engine)

    def _seed(self, payload=None, dry_run=False):
        with _fake_database.SessionLocal() as db:
            stats = roster.seed_roster(db, payload or REAL_MADRID, dry_run=dry_run)
            if dry_run:
                db.rollback()
            else:
                db.commit()
            return stats

    def test_first_run_inserts_team_aliases_and_squad(self):
        stats = self._seed()
        self.assertEqual(stats.teams_created, 1)
        self.assertEqual(stats.players_inserted, 3)
        with _fake_database.SessionLocal() as db:
            self.assertEqual(len(db.scalars(select(models.Player)).all()), 3)
            aliases = {row.alias for row in db.scalars(select(models.TeamAlias)).all()}
            self.assertIn("real madrid", aliases)
            self.assertIn("rma", aliases)

    def test_second_run_writes_nothing(self):
        self._seed()
        stats = self._seed()
        self.assertFalse(stats.wrote_anything)
        self.assertEqual(stats.teams_matched, 1)
        self.assertEqual(stats.players_unchanged, 3)
        with _fake_database.SessionLocal() as db:
            self.assertEqual(len(db.scalars(select(models.Player)).all()), 3)

    def test_dry_run_reports_without_writing(self):
        stats = self._seed(dry_run=True)
        self.assertEqual(stats.players_inserted, 3)
        with _fake_database.SessionLocal() as db:
            self.assertEqual(db.scalars(select(models.Player)).all(), [])

    def test_adopts_the_unnamed_slot_the_pipeline_created(self):
        # A number read off a shirt before any roster existed.
        with _fake_database.SessionLocal() as db:
            team = models.Team(team_name="Real Madrid")
            db.add(team)
            db.flush()
            db.add(models.Player(team_id=team.team_id, jersey_no=9, name=None))
            db.commit()
            orphan_id = db.scalar(select(models.Player.player_id))

        stats = self._seed()
        self.assertEqual(stats.players_named, 1)
        with _fake_database.SessionLocal() as db:
            adopted = db.get(models.Player, orphan_id)
            self.assertEqual(adopted.name, "Endrick")
            self.assertEqual(len(db.scalars(select(models.Player)).all()), 3)

    def test_changed_number_overwrites_and_is_reported(self):
        self._seed()
        stats = self._seed({**REAL_MADRID, "players": [{"jersey_no": 9, "name": "Gonzalo Garcia"}]})
        self.assertEqual(stats.players_renamed, 1)
        self.assertTrue(any("Endrick" in note for note in stats.notes))

    def test_entry_without_a_number_or_name_is_skipped(self):
        stats = self._seed(
            {"team_name": "Real Madrid", "players": [{"jersey_no": None, "name": "Unknown"}]}
        )
        self.assertEqual(stats.players_inserted, 0)
        self.assertTrue(stats.notes)


class ResolveTeamTests(unittest.TestCase):
    def setUp(self):
        _Base.metadata.drop_all(_engine)
        _Base.metadata.create_all(_engine)
        with _fake_database.SessionLocal() as db:
            roster.seed_roster(db, REAL_MADRID)
            db.commit()

    def test_spelling_variants_resolve_to_the_seeded_team(self):
        with _fake_database.SessionLocal() as db:
            expected = db.scalar(select(models.Team)).team_id
            for typed in [
                "Real Madrid",
                "real madrid cf",
                "REAL MADRID C.F.",
                "RMA",
                "Los Blancos",
            ]:
                with self.subTest(typed=typed):
                    self.assertEqual(roster.resolve_team(db, typed).team_id, expected)
            self.assertEqual(len(db.scalars(select(models.Team)).all()), 1)

    def test_a_genuinely_new_team_is_created_once_and_aliased(self):
        with _fake_database.SessionLocal() as db:
            first = roster.resolve_team(db, "Rayo Vallecano", "#FF0000")
            db.flush()
            second = roster.resolve_team(db, "rayo vallecano cf")
            self.assertEqual(first.team_id, second.team_id)
            self.assertEqual(len(db.scalars(select(models.Team)).all()), 2)

    def test_an_alias_is_not_stolen_from_another_team(self):
        with _fake_database.SessionLocal() as db:
            other = models.Team(team_name="Rival Club")
            db.add(other)
            db.flush()
            self.assertFalse(roster.add_alias(db, other, "RMA"))
            self.assertEqual(roster.find_team(db, "RMA").team_name, "Real Madrid")


class LoadRosterFilesTests(unittest.TestCase):
    def test_skips_underscore_files_and_filters_by_team(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "_template.json").write_text(
                json.dumps({"team_name": "Template"}), encoding="utf-8"
            )
            (directory / "real_madrid.json").write_text(
                json.dumps(REAL_MADRID), encoding="utf-8"
            )

            everything = roster.load_roster_files(directory)
            self.assertEqual([payload["team_name"] for _, payload in everything], ["Real Madrid"])

            # The filter goes through normalisation, so a variant still matches.
            filtered = roster.load_roster_files(directory, team="real madrid cf")
            self.assertEqual(len(filtered), 1)
            self.assertEqual(roster.load_roster_files(directory, team="Barcelona"), [])

    def test_missing_directory_is_an_error_not_an_empty_result(self):
        with self.assertRaises(FileNotFoundError):
            roster.load_roster_files(Path("no/such/directory"))


class ShippedRosterFilesTests(unittest.TestCase):
    """The files in data/rosters must stay loadable and internally consistent."""

    def test_every_shipped_roster_is_valid(self):
        files = roster.load_roster_files()
        self.assertTrue(files, "no roster files found in data/rosters")
        for path, payload in files:
            with self.subTest(path=path.name):
                self.assertTrue(payload.get("team_name"))
                numbers = [entry["jersey_no"] for entry in payload["players"]]
                self.assertEqual(len(numbers), len(set(numbers)), "duplicate shirt number")
                for entry in payload["players"]:
                    self.assertTrue(entry.get("name"))
                    self.assertIsInstance(entry["jersey_no"], int)

    def test_shipped_rosters_seed_without_collision(self):
        _Base.metadata.drop_all(_engine)
        _Base.metadata.create_all(_engine)
        files = roster.load_roster_files()
        with _fake_database.SessionLocal() as db:
            for _, payload in files:
                roster.seed_roster(db, payload)
            db.commit()
            self.assertEqual(len(db.scalars(select(models.Team)).all()), len(files))


class RosterLookupEndToEndTests(unittest.TestCase):
    """A seeded squad must reach the name that _persist_match writes out."""

    def setUp(self):
        _Base.metadata.drop_all(_engine)
        _Base.metadata.create_all(_engine)
        with _fake_database.SessionLocal() as db:
            roster.seed_roster(db, REAL_MADRID)
            db.commit()

    def test_a_detected_shirt_number_resolves_to_the_roster_name(self):
        from backend.routers import processing

        with _fake_database.SessionLocal() as db:
            # The spelling the user typed, not the spelling that was seeded.
            team = processing._get_or_create_team(db, "real madrid cf", "#FFFFFF")
            player = processing._roster_player(db, team, 9)
            self.assertEqual(player.name, "Endrick")

    def test_an_unknown_number_still_yields_a_slot_but_no_name(self):
        from backend.routers import processing

        with _fake_database.SessionLocal() as db:
            team = processing._get_or_create_team(db, "Real Madrid", "#FFFFFF")
            player = processing._roster_player(db, team, 77)
            self.assertIsNotNone(player)
            self.assertIsNone(player.name)

    def test_no_number_means_no_lookup(self):
        from backend.routers import processing

        with _fake_database.SessionLocal() as db:
            team = processing._get_or_create_team(db, "Real Madrid", "#FFFFFF")
            self.assertIsNone(processing._roster_player(db, team, None))


if __name__ == "__main__":
    unittest.main()
