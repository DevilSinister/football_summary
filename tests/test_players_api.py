"""Reference-image enrolment API, against an in-memory SQLite copy.

backend.database connects to SQL Server at import; a stand-in module is
installed instead (test_persist_match installs the same kind; whichever is
imported first wins and the models bind to its Base). This test always uses
its own thread-safe engine, because TestClient runs endpoints on a worker
thread and a plain in-memory SQLite connection refuses that.
"""

import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool


class _Base(DeclarativeBase):
    pass


_stand_in = types.ModuleType("backend.database")
_stand_in.Base = _Base
_stand_in.engine = create_engine(
    "sqlite://", future=True, connect_args={"check_same_thread": False}, poolclass=StaticPool
)
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
from backend.routers import players  # noqa: E402
from identity import PlayerGallery  # noqa: E402


def png(width, height):
    ok, data = cv2.imencode(".png", np.full((height, width, 3), 120, np.uint8))
    assert ok
    return data.tobytes()


class _NoDetector:
    @staticmethod
    def crops(images):
        return players.crops_from_images(images, None)


class ReferenceImageApiTests(unittest.TestCase):
    def setUp(self):
        database.Base.metadata.drop_all(_stand_in.engine)
        database.Base.metadata.create_all(_stand_in.engine)
        with _stand_in.SessionLocal() as db:
            team = models.Team(team_name="Porto")
            db.add(team)
            db.flush()
            db.add(models.Player(team_id=team.team_id, jersey_no=7, name="Seven"))
            db.add(models.Player(team_id=team.team_id, jersey_no=None, name="No Number"))
            db.commit()
            self.player_id = db.query(models.Player).filter_by(jersey_no=7).one().player_id
            self.unnumbered_id = db.query(models.Player).filter_by(name="No Number").one().player_id
        self.root = tempfile.mkdtemp()
        app = FastAPI()
        app.include_router(players.router)
        app.dependency_overrides[players.get_db] = _get_db
        self.client = TestClient(app)
        self.patches = [
            mock.patch.object(players, "PlayerGallery", lambda: PlayerGallery(self.root)),
            mock.patch.object(players, "detector", _NoDetector()),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def upload(self, player_id, *sizes):
        files = [("images", (f"p{i}.png", png(w, h), "image/png")) for i, (w, h) in enumerate(sizes)]
        return self.client.post(f"/api/players/{player_id}/reference-images", files=files)

    def test_one_image_is_stored_and_listed(self):
        response = self.upload(self.player_id, (60, 160))
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["stored"], 1)
        listing = self.client.get(f"/api/players/{self.player_id}/reference-images").json()
        self.assertEqual(len(listing["images"]), 1)
        self.assertEqual(listing["jersey_no"], 7)

    def test_an_image_without_a_player_is_rejected(self):
        response = self.upload(self.player_id, (60, 160), (300, 100))
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["stored"], 1)
        self.assertEqual(response.json()["rejected"], ["p1.png"])

    def test_more_than_ten_images_is_refused(self):
        self.assertEqual(self.upload(self.player_id, *[(60, 160)] * 11).status_code, 422)

    def test_a_player_without_a_shirt_number_cannot_be_enrolled(self):
        self.assertEqual(self.upload(self.unnumbered_id, (60, 160)).status_code, 422)

    def test_garbage_is_not_an_image(self):
        files = [("images", ("x.png", b"not an image", "image/png"))]
        response = self.client.post(f"/api/players/{self.player_id}/reference-images", files=files)
        self.assertEqual(response.status_code, 415)

    def test_an_image_can_be_deleted(self):
        self.upload(self.player_id, (60, 160))
        image_id = self.client.get(f"/api/players/{self.player_id}/reference-images").json()["images"][0]
        self.assertEqual(
            self.client.delete(f"/api/players/{self.player_id}/reference-images/{image_id}").status_code, 204
        )
        self.assertEqual(
            self.client.delete(f"/api/players/{self.player_id}/reference-images/{image_id}").status_code, 404
        )


if __name__ == "__main__":
    unittest.main()
