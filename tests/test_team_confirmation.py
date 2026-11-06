"""Team confirmation: alternative kit images, the team list, and stored names.

backend.database connects to SQL Server at import; a SQLite stand-in is
installed instead (the first test module to install one wins, so the tables
are created on whichever module is in sys.modules).
"""

import sys
import types
import unittest

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class _Base(DeclarativeBase):
    pass


_stand_in = types.ModuleType("backend.database")
_stand_in.Base = _Base
_stand_in.engine = create_engine("sqlite://", future=True)
_stand_in.SessionLocal = sessionmaker(bind=_stand_in.engine, autocommit=False, autoflush=False, future=True)


def _get_db():
    session = _stand_in.SessionLocal()
    try:
        yield session
    finally:
        session.close()


_stand_in.get_db = _get_db
database = sys.modules.setdefault("backend.database", _stand_in)

from backend import models  # noqa: E402
from backend.routers import processing, teams  # noqa: E402
from backend.schemas import TeamConfirmationRequest  # noqa: E402
from team_assigner import TeamSample, cluster_team_samples, rank_team_samples  # noqa: E402


def _sample(color, frame, height=80, sharpness=100.0):
    crop = np.zeros((height, height // 2, 3), dtype=np.uint8)
    return TeamSample(
        color=np.asarray(color, dtype=float),
        crop=crop,
        frame=frame,
        bbox=(100, 100, 100 + height // 2, 100 + height),
        sharpness=sharpness,
    )


class RankTeamSamplesTests(unittest.TestCase):
    def setUp(self):
        self.frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(4)]
        self.samples = []
        for position, frame in enumerate(self.frames):
            # Two red and two blue players per frame, sizes varied so ranking is defined.
            self.samples.append(_sample((20, 20, 220), frame, height=80 + position * 4))
            self.samples.append(_sample((22, 18, 225), frame, height=60 + position * 4))
            self.samples.append(_sample((220, 40, 20), frame, height=80 + position * 4))
            self.samples.append(_sample((225, 38, 22), frame, height=60 + position * 4))

    def test_first_choice_is_the_representative_cluster_team_samples_shows(self):
        centers, representatives = cluster_team_samples(self.samples)
        ranked_centers, ranked = rank_team_samples(self.samples, per_team=4)
        self.assertEqual(centers, ranked_centers)
        for representative, choices in zip(representatives, ranked):
            self.assertIs(choices[0], representative)

    def test_alternatives_stay_in_their_kit_and_come_from_different_frames_first(self):
        centers, ranked = rank_team_samples(self.samples, per_team=4)
        for center, choices in zip(centers, ranked):
            self.assertEqual(len(choices), 4)
            self.assertEqual(len({id(choice) for choice in choices}), 4)
            self.assertEqual(len({id(choice.frame) for choice in choices}), 4)
            red_kit = center[0] > 150
            for choice in choices:
                # TeamSample colours are BGR: index 2 is red.
                self.assertEqual(choice.color[2] > 150, red_kit)

    def test_fewer_samples_than_requested_returns_what_exists(self):
        _, ranked = rank_team_samples(self.samples, per_team=50)
        self.assertTrue(all(len(choices) < 50 for choices in ranked))
        self.assertTrue(all(len(choices) >= 1 for choices in ranked))


class TeamListAndConfirmationTests(unittest.TestCase):
    def setUp(self):
        models.Base.metadata.drop_all(database.engine)
        models.Base.metadata.create_all(database.engine)
        with database.SessionLocal() as db:
            atletico = models.Team(team_name="Atletico Madrid", short_code="ATM")
            misspelt = models.Team(team_name="Athletico Madrid")
            madrid = models.Team(team_name="Real Madrid", short_code="RMA")
            db.add_all([atletico, misspelt, madrid])
            db.flush()
            db.add_all(
                [
                    models.Player(team_id=atletico.team_id, jersey_no=19, name="Julian Alvarez"),
                    models.Player(team_id=madrid.team_id, jersey_no=7, name="Vinicius Junior"),
                    # An unnamed slot is not a squad.
                    models.Player(team_id=misspelt.team_id, jersey_no=9, name=None),
                    models.TeamAlias(alias="atleti", team_id=atletico.team_id),
                ]
            )
            db.commit()
            self.ids = {team.team_name: team.team_id for team in db.query(models.Team)}

    def test_team_list_offers_only_teams_with_a_named_squad(self):
        with database.SessionLocal() as db:
            listed = teams.list_teams(db)
        self.assertEqual([team["team_name"] for team in listed], ["Atletico Madrid", "Real Madrid"])
        self.assertEqual(listed[0]["short_code"], "ATM")
        self.assertEqual(listed[0]["aliases"], ["atleti"])

    def test_team_ids_resolve_to_the_stored_names(self):
        payload = TeamConfirmationRequest(
            team_1_id=self.ids["Real Madrid"], team_2_id=self.ids["Atletico Madrid"]
        )
        self.assertEqual(processing._confirmed_team_names(payload), ("Real Madrid", "Atletico Madrid"))

    def test_an_unknown_team_id_is_rejected(self):
        payload = TeamConfirmationRequest(team_1_id=9999, team_2_id=self.ids["Real Madrid"])
        with self.assertRaises(HTTPException) as raised:
            processing._confirmed_team_names(payload)
        self.assertEqual(raised.exception.status_code, 422)

    def test_typed_names_from_older_clients_map_to_existing_teams(self):
        payload = TeamConfirmationRequest(team_1_name="  real madrid ", team_2_name="Atleti")
        self.assertEqual(processing._confirmed_team_names(payload), ("Real Madrid", "Atletico Madrid"))

    def test_a_typed_name_no_team_matches_is_kept_as_typed(self):
        payload = TeamConfirmationRequest(team_1_name="Real Madrid", team_2_name="Sunday League XI")
        self.assertEqual(processing._confirmed_team_names(payload), ("Real Madrid", "Sunday League XI"))

    def test_a_missing_team_is_rejected(self):
        payload = TeamConfirmationRequest(team_1_name="Real Madrid", team_2_name="   ")
        with self.assertRaises(HTTPException) as raised:
            processing._confirmed_team_names(payload)
        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
