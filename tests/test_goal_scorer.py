"""Goal scorer fusion: caption names, celebration close-ups, the live read."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from scorer_identifier import (
    ScorerCandidate,
    celebration_candidates,
    format_goal_event,
    fuse_scorer,
    live_candidate,
    match_caption,
    read_caption_candidates,
    squad_for,
)

SQUAD = [
    {"jersey_no": 1, "name": "Jan Oblak", "position": "GK"},
    {"jersey_no": 7, "name": "Antoine Griezmann", "position": "FW"},
    {"jersey_no": 9, "name": "Alexander Sorloth", "position": "FW"},
    {"jersey_no": 11, "name": "Ademola Lookman", "position": "FW"},
    {"jersey_no": 14, "name": "Marcos Llorente", "position": "MF"},
    {"jersey_no": 15, "name": "Marcos Alonso", "position": "DF"},
    {"jersey_no": 24, "name": "Robin Le Normand", "position": "DF"},
]


class CaptionMatchTests(unittest.TestCase):
    def names(self, lines, squad=SQUAD):
        return [player["jersey_no"] for player, _ in match_caption(lines, squad)]

    def test_broadcast_caption_names_the_scorer(self):
        self.assertEqual(self.names(["ADEMOLA LOOKMAN 0-1 MIN. 33"]), [11])

    def test_small_ocr_slips_still_match(self):
        self.assertEqual(self.names(["LOOKMAM 33'"]), [11])
        self.assertEqual(self.names(["GRIEZMAN"]), [7])

    def test_a_shared_first_name_names_nobody(self):
        # Two Marcos in the squad: the first name alone is not evidence.
        self.assertEqual(self.names(["MARCOS"]), [])
        self.assertEqual(self.names(["MARCOS LLORENTE"]), [14])

    def test_substitution_and_booking_graphics_are_ignored(self):
        self.assertEqual(self.names(["SUBSTITUTION", "OFF LOOKMAN", "ON SORLOTH"]), [])
        self.assertEqual(self.names(["YELLOW CARD GRIEZMANN"]), [])

    def test_the_assist_line_is_skipped(self):
        self.assertEqual(self.names(["LOOKMAN 33'", "ASSIST GRIEZMANN"]), [11])

    def test_a_team_sheet_is_not_a_caption(self):
        self.assertEqual(self.names(["OBLAK", "LLORENTE", "LE NORMAND", "GRIEZMANN", "SORLOTH"]), [])

    def test_accents_and_particles_are_folded(self):
        squad = squad_for("Real Madrid")
        self.assertTrue(squad, "data/rosters/real_madrid.json should load")
        self.assertEqual(self.names(["VINÍCIUS JR. 1-0"], squad), [7])


class FusionTests(unittest.TestCase):
    def test_caption_alone_names_the_scorer(self):
        fused = fuse_scorer([ScorerCandidate(11, 3.0 * 0.95, "caption")], SQUAD)
        self.assertEqual(fused["jersey_number"], 11)
        self.assertEqual(fused["player_name"], "Ademola Lookman")
        self.assertFalse(fused["unconfirmed"])

    def test_caption_outweighs_a_wrong_live_read(self):
        candidates = live_candidate({"jersey_number": 24, "jersey_confidence": 0.8}, "latest shot by the scoring team")
        candidates.append(ScorerCandidate(11, 3.0, "caption"))
        fused = fuse_scorer(candidates, SQUAD)
        self.assertEqual(fused["jersey_number"], 11)
        self.assertIn("24", fused["scorer_alternatives"])

    def test_a_weak_lone_live_read_is_unconfirmed_and_unnamed(self):
        fused = fuse_scorer(live_candidate({"jersey_number": 9, "jersey_confidence": 0.6}), SQUAD)
        self.assertEqual(fused["jersey_number"], 9)
        self.assertTrue(fused["unconfirmed"])
        self.assertIsNone(fused["player_name"])

    def test_a_confident_lone_live_read_on_a_forward_is_named(self):
        fused = fuse_scorer(live_candidate({"jersey_number": 9, "jersey_confidence": 0.95}), SQUAD)
        self.assertFalse(fused["unconfirmed"])
        self.assertEqual(fused["player_name"], "Alexander Sorloth")

    def test_live_and_celebration_agreeing_confirm_each_other(self):
        candidates = live_candidate({"jersey_number": 15, "jersey_confidence": 0.7})
        candidates.append(ScorerCandidate(15, 1.5 * 0.9 * 0.8, "celebration"))
        fused = fuse_scorer(candidates, SQUAD)
        self.assertEqual(fused["jersey_number"], 15)
        self.assertEqual(fused["scorer_sources"], ["celebration", "live"])

    def test_a_number_the_squad_lacks_is_discounted(self):
        candidates = [ScorerCandidate(88, 1.0, "live"), ScorerCandidate(9, 0.6, "celebration")]
        self.assertEqual(fuse_scorer(candidates, SQUAD)["jersey_number"], 9)

    def test_keeper_rarely_scores(self):
        candidates = [ScorerCandidate(1, 1.0, "live"), ScorerCandidate(7, 0.4, "celebration")]
        self.assertEqual(fuse_scorer(candidates, SQUAD)["jersey_number"], 7)

    def test_no_evidence_is_none(self):
        self.assertIsNone(fuse_scorer([], SQUAD))

    def test_unconfirmed_goal_text(self):
        text = format_goal_event({"scorer": {"team_name": "Atletico", "jersey_number": 9, "unconfirmed": True}})
        self.assertEqual(text, "GOAL — Atletico — #9 (unconfirmed) scored")


class CelebrationTests(unittest.TestCase):
    def test_the_player_the_camera_follows_wins(self):
        store = SimpleNamespace(
            players={
                # Scorer in close-up for many frames after the goal (frame 1000).
                5: SimpleNamespace(team_id=2, sightings=[(f, 600.0) for f in range(1010, 1400, 3)]),
                # Team-mate briefly, smaller.
                6: SimpleNamespace(team_id=2, sightings=[(f, 200.0) for f in range(1100, 1160, 3)]),
                # Opponent in close-up: another team, ignored.
                7: SimpleNamespace(team_id=1, sightings=[(f, 700.0) for f in range(1010, 1400, 3)]),
                # Scorer's team, but before the goal.
                8: SimpleNamespace(team_id=2, sightings=[(f, 700.0) for f in range(500, 900, 3)]),
            }
        )
        numbers = {5: 11, 6: 7, 7: 4, 8: 9}
        read = lambda track: {"jersey_number": numbers[track], "jersey_confidence": 0.8}
        candidates = celebration_candidates(store, read, 2, 1000, 1750, 1080)
        by_number = {c.jersey_number: c.weight for c in candidates}
        self.assertEqual(set(by_number), {11, 7})
        self.assertGreater(by_number[11], 5 * by_number[7])


class CaptionReaderTests(unittest.TestCase):
    def test_reads_the_caption_that_appears_after_the_goal(self):
        fps = 25.0
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "caption.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
            for index in range(250):
                frame = np.zeros((180, 320, 3), np.uint8)
                if 100 <= index < 200:
                    frame[140:, :] = 255  # the caption bar
                writer.write(frame)
            writer.release()

            calls = []

            def fake_ocr(region):
                calls.append(region.shape)
                # Only the bright lower-third bar "says" anything.
                return [("ADEMOLA LOOKMAN 0-1", 0.93, (0, 0, 1, 1))] if region.mean() > 60 else []

            candidates = read_caption_candidates(path, 50, fps, fake_ocr, SQUAD, seconds=8.0)
        self.assertEqual([c.jersey_number for c in candidates], [11])
        self.assertEqual(candidates[0].player_name, "Ademola Lookman")
        # Stopped once two frames agreed rather than reading all 8 seconds.
        self.assertLess(len(calls), 2 * 6)

    def test_no_squad_means_no_reads(self):
        self.assertEqual(read_caption_candidates("missing.mp4", 0, 25.0, lambda r: [], []), [])


if __name__ == "__main__":
    unittest.main()
