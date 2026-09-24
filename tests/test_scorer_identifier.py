from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from scorer_identifier import (
    IdentityReadResult,
    JerseyReader,
    OcrCandidate,
    PlayerIdentityStore,
    PossessionHistory,
    ScorerIdentifier,
    TrackedCrop,
    format_goal_event,
)


class JerseyConsensusTests(unittest.TestCase):
    def setUp(self):
        self.reader = JerseyReader(min_supporting_frames=2)

    def test_repeated_number_wins_and_single_name_is_omitted(self):
        result = self.reader.fuse_candidates(
            [
                OcrCandidate("7", 0.91, 10, 0.9),
                OcrCandidate("7", 0.86, 13, 0.8),
                OcrCandidate("9", 0.93, 11, 0.7),
                OcrCandidate("RONALDO", 0.94, 13, 0.8),
            ]
        )

        self.assertEqual(result.jersey_number, 7)
        self.assertIsNone(result.player_name)

    def test_shirt_names_are_never_read(self):
        """Names come from the roster, not from the shirt.

        At the resolutions this pipeline sees, shirt text is a few pixels tall,
        and the old free-text path voted sponsor and advertising-board wording
        into the player field.
        """
        result = self.reader.fuse_candidates(
            [
                OcrCandidate("MESSI", 0.88, 20, 0.9),
                OcrCandidate("MESSI", 0.82, 23, 0.8),
                OcrCandidate("FLY EMIRATES", 0.95, 24, 0.9),
            ]
        )

        self.assertIsNone(result.player_name)
        self.assertIsNone(result.jersey_number)

    def test_letters_inside_a_number_do_not_vote(self):
        """A shirt read as R0NALD0 used to cast two votes for jersey #0.

        The number regex ran over the raw string before letters were stripped.
        """
        result = self.reader.fuse_candidates(
            [
                OcrCandidate("R0NALD0", 0.91, 30, 0.9),
                OcrCandidate("R0NALD0", 0.90, 33, 0.9),
            ]
        )

        self.assertIsNone(result.jersey_number)

    def test_common_digit_confusions_are_repaired(self):
        """A 1O reading is shirt 10, but a bare letter must not become a number."""
        result = self.reader.fuse_candidates(
            [
                OcrCandidate("1O", 0.90, 40, 0.9),
                OcrCandidate("1O", 0.88, 43, 0.9),
                OcrCandidate("O", 0.99, 44, 0.9),
                OcrCandidate("S", 0.99, 45, 0.9),
            ]
        )

        self.assertEqual(result.jersey_number, 10)

    def test_close_conflicting_numbers_are_rejected(self):
        result = self.reader.fuse_candidates(
            [
                OcrCandidate("7", 0.88, 10),
                OcrCandidate("7", 0.87, 11),
                OcrCandidate("9", 0.87, 12),
                OcrCandidate("9", 0.86, 13),
            ]
        )

        self.assertIsNone(result.jersey_number)

    def test_paddle_v3_result_shape_is_supported(self):
        class FakePaddleV3:
            def predict(self, input):
                return [{"res": {"rec_texts": ["7"], "rec_scores": [0.91]}}]

        reader = JerseyReader(engine=FakePaddleV3(), min_supporting_frames=2)
        image = np.zeros((48, 30, 3), dtype=np.uint8)
        result = reader.read(
            [TrackedCrop(1, image, 0.9), TrackedCrop(2, image, 0.8)],
            track_id=7,
        )

        self.assertEqual(result.jersey_number, 7)

    def test_paddle_v2_result_shape_is_supported(self):
        class FakePaddleV2:
            def ocr(self, image, cls=True):
                return [[[[0, 0], [1, 0], [1, 1], [0, 1]], ("11", 0.89)]]

        reader = JerseyReader(engine=FakePaddleV2(), min_supporting_frames=2)
        image = np.zeros((48, 30, 3), dtype=np.uint8)
        result = reader.read(
            [TrackedCrop(3, image, 0.9), TrackedCrop(4, image, 0.8)],
            track_id=11,
        )

        self.assertEqual(result.jersey_number, 11)


class PossessionTests(unittest.TestCase):
    def test_latest_player_in_possession_is_selected(self):
        history = PossessionHistory(lookback_frames=60)
        team_one = SimpleNamespace(team_id=1, team_name="Team A")
        team_two = SimpleNamespace(team_id=2, team_name="Team B")
        history.record(80, 4, team_one)
        history.record(88, 7, team_two)
        history.record(91, 7, team_two)

        scorer = history.select_scorer(100)

        self.assertEqual(scorer["track_id"], 7)
        self.assertEqual(scorer["team_name"], "Team B")
        self.assertEqual(scorer["last_touch_frame"], 91)

    def test_old_possession_is_not_used(self):
        history = PossessionHistory(lookback_frames=20)
        identity = SimpleNamespace(team_id=1, team_name="Team A")
        history.record(10, 4, identity)

        self.assertIsNone(history.select_scorer(100))


class IdentityStoreTests(unittest.TestCase):
    def test_track_snapshot_contains_team_and_crop_state(self):
        frame = np.zeros((160, 120, 3), dtype=np.uint8)
        cv2.putText(frame, "7", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        player = {
            "bbox": [25, 15, 95, 145],
            "team": 1,
            "team_name": "Team A",
            "team_confidence": 0.9,
            "rear_facing_score": 1.0,
        }
        store = PlayerIdentityStore(crop_stride=1)

        state = store.update_player(1, frame, 7, player)

        self.assertEqual(state.team_name, "Team A")
        self.assertEqual(len(state.recent_crops), 1)
        self.assertEqual(player["identity"]["track_id"], 7)


class GoalFormattingTests(unittest.TestCase):
    def test_number_only_format_matches_required_fallback(self):
        event = {"event": "goal", "scorer": {"team_name": "Team A", "jersey_number": 7}}
        self.assertEqual(format_goal_event(event), "GOAL — Team A — #7 scored")

    def test_visible_name_is_appended(self):
        event = {
            "event": "goal",
            "scorer": {"team_name": "Team A", "jersey_number": 7, "player_name": "SMITH"},
        }
        self.assertEqual(format_goal_event(event), "GOAL — Team A — #7 SMITH scored")

    def test_end_to_end_identity_uses_last_touch_and_ocr_result(self):
        class FakeReader:
            def read(self, crops, track_id):
                return IdentityReadResult(jersey_number=7, jersey_confidence=0.91)

        frame = np.zeros((160, 120, 3), dtype=np.uint8)
        player = {
            "bbox": [25, 15, 95, 145],
            "team": 1,
            "team_name": "Team A",
            "team_confidence": 0.86,
        }
        identifier = ScorerIdentifier(
            jersey_reader=FakeReader(),
            identity_store=PlayerIdentityStore(crop_stride=1),
            possession_history=PossessionHistory(lookback_frames=30),
        )
        identifier.observe_players(40, frame, {7: player})
        identifier.record_possession(40, 7)

        scorer = identifier.identify_goal(45)

        self.assertEqual(scorer["track_id"], 7)
        self.assertEqual(scorer["team_name"], "Team A")
        self.assertEqual(scorer["jersey_number"], 7)


if __name__ == "__main__":
    unittest.main()
