"""Persistence of multi-participant events, against an in-memory SQLite copy.

backend.database connects to SQL Server at import time. It is replaced here
with a stand-in module that binds the same declarative models to SQLite, so the
test exercises the real _persist_match without touching the local server or
its schema.
"""

import sys
import types
import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class _Base(DeclarativeBase):
    pass


_engine = create_engine("sqlite://", future=True)
_fake_database = types.ModuleType("backend.database")
_fake_database.Base = _Base
_fake_database.engine = _engine
_fake_database.SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False, future=True)


def _get_db():
    database = _fake_database.SessionLocal()
    try:
        yield database
    finally:
        database.close()


_fake_database.get_db = _get_db
sys.modules.setdefault("backend.database", _fake_database)

from backend import models  # noqa: E402
from backend.routers import processing  # noqa: E402


class PersistMatchTests(unittest.TestCase):
    def setUp(self):
        _Base.metadata.drop_all(_engine)
        _Base.metadata.create_all(_engine)
        with _fake_database.SessionLocal() as db:
            team = models.Team(team_name="Real Madrid", primary_color="#FFFFFF")
            db.add(team)
            db.flush()
            db.add(models.Player(team_id=team.team_id, jersey_no=7, name="Vinicius"))
            db.commit()

    @staticmethod
    def _job():
        return {
            "job_id": "11111111-2222-3333-4444-555555555555",
            "video_path": "match.mp4",
            "user_id": None,
            "team_a_name": "Real Madrid",
            "team_a_color": "#FFFFFF",
            "team_b_name": "Atletico",
            "team_b_color": "#D71920",
        }

    def test_pass_stores_both_players_and_uses_roster_names(self):
        events = [
            {
                "event": "pass",
                "frame": 50,
                "end_frame": 60,
                "team_id": 1,
                "team_name": "Real Madrid",
                "participants": [
                    {"role": "passer", "track_id": 3, "team_id": 1, "team_name": "Real Madrid"},
                    {"role": "receiver", "track_id": 8, "team_id": 1, "team_name": "Real Madrid"},
                ],
                "actors": [
                    {"role": "passer", "track_id": 3, "team_id": 1, "team_name": "Real Madrid", "jersey_number": 7, "jersey_confidence": 0.8},
                    {"role": "receiver", "track_id": 8, "team_id": 1, "team_name": "Real Madrid", "jersey_number": 10, "jersey_confidence": 0.7},
                ],
                "scorer": {"role": "passer", "track_id": 3, "team_id": 1, "team_name": "Real Madrid", "jersey_number": 7},
            },
            {
                "event": "goal",
                "frame": 200,
                "scorer": {"track_id": 3, "team_id": 1, "team_name": "Real Madrid", "jersey_number": 7},
                "clip_url": "/api/processing/clips/11111111-2222-3333-4444-555555555555/goal-01.mp4",
                "clip_start_seconds": 3.0,
                "clip_end_seconds": 20.04,
            },
        ]
        match_id = processing._persist_match(self._job(), events, fps=25.0)

        with _fake_database.SessionLocal() as db:
            stored = db.scalars(
                select(models.Event).where(models.Event.match_id == match_id).order_by(models.Event.time_seconds)
            ).all()
            self.assertEqual([e.event_type for e in stored], ["pass", "goal"])
            self.assertIn("#7 Vinicius", stored[0].description)
            self.assertIn("#10", stored[0].description)
            self.assertEqual(stored[0].time_seconds, 2.0)
            self.assertIn("Vinicius", stored[1].description)
            # Clip bounds are stored for clipped events; passes have none.
            self.assertEqual((stored[1].clip_start_seconds, stored[1].clip_end_seconds), (3.0, 20.04))
            self.assertTrue(stored[1].video_path.endswith("/goal-01.mp4"))
            self.assertIsNone(stored[0].clip_start_seconds)

            occurrences = db.scalars(
                select(models.Occurrence).where(models.Occurrence.event_id == stored[0].event_id)
            ).all()
            self.assertEqual(len(occurrences), 2)
            self.assertEqual(occurrences[0].detected_jersey_no, 7)
            self.assertEqual(occurrences[0].detected_player_name, "Vinicius")
            self.assertEqual(occurrences[0].end_time, 2.4)
            self.assertEqual(occurrences[1].detected_jersey_no, 10)
            self.assertIsNone(occurrences[1].detected_player_name)
            # The receiver got an unnamed roster slot, ready for a real line-up.
            slot = db.scalar(select(models.Player).where(models.Player.jersey_no == 10))
            self.assertIsNotNone(slot)
            self.assertIsNone(slot.name)

            plays = {
                play.team.team_name: play.score
                for play in db.scalars(select(models.Play).where(models.Play.match_id == match_id))
            }
            self.assertEqual(plays, {"Real Madrid": 1, "Atletico": 0})

    def test_card_with_unknown_team_is_stored_with_null_team(self):
        events = [
            {
                "event": "yellow_card",
                "frame": 100,
                "end_frame": 110,
                "team_id": 0,
                "team_name": "unknown_team",
                "participants": [{"role": "referee", "track_id": 40, "team_id": 0, "team_name": "unknown_team"}],
                "actors": [{"role": "referee", "track_id": 40, "team_id": 0, "team_name": "unknown_team", "jersey_number": None}],
                "scorer": {"role": "referee", "track_id": 40, "team_id": 0, "team_name": "unknown_team"},
            }
        ]
        match_id = processing._persist_match(self._job(), events, fps=25.0)
        with _fake_database.SessionLocal() as db:
            event = db.scalar(select(models.Event).where(models.Event.match_id == match_id))
            self.assertEqual(event.event_type, "yellow_card")
            self.assertEqual(event.event_name, "Yellow Card")
            occurrence = db.scalar(select(models.Occurrence).where(models.Occurrence.event_id == event.event_id))
            self.assertIsNone(occurrence.detected_team_name)
            self.assertIsNone(occurrence.player_id)


if __name__ == "__main__":
    unittest.main()
