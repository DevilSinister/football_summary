"""Scoreboard goal detection.

Parser inputs are the token shapes PaddleOCR actually returned on the two
reference clips (see the 2026-09-24 entry in Verification Evidence): the UEFA
one-line graphic and the stacked LaLiga graphic.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from scoreboard import ScoreboardReader, ScoreTracker, Token, attribute_goals, map_codes_to_teams, parse_tokens
from scoreboard.parser import as_code, find_codes
from scoreboard.reader import ScoreGoal
from scoreboard.teams import code_spells_name, expected_codes_for


def tok(text, box, confidence=0.95):
    return Token(text, confidence, tuple(float(v) for v in box))


# The stacked LaLiga graphic at 480p, frame ~200 of the Real Madrid clip.
RMA_BOX = (83, 30, 104, 43)
ATM_BOX = (82, 54, 104, 71)


def stacked(top="0", bottom="0"):
    return [
        tok("RMA", RMA_BOX),
        tok(top, (114, 27, 130, 47)),
        tok(bottom, (114, 48, 131, 72)),
        tok("09:59", (44, 55, 79, 70)),
        tok("ATM", ATM_BOX),
    ]


class OneLineParserTests(unittest.TestCase):
    def test_letter_o_read_for_zero(self):
        reading = parse_tokens([tok("46:33", (40, 60, 120, 90)), tok("POR O 0 MCI", (260, 60, 600, 90))])
        self.assertEqual(reading.codes, ("POR", "MCI"))
        self.assertEqual(reading.scores, (0, 0))

    def test_digit_run_into_the_code(self):
        reading = parse_tokens([tok("POR0 1 MCI", (260, 60, 600, 90))])
        self.assertEqual(reading.scores, (0, 1))

    def test_clock_in_the_same_token(self):
        reading = parse_tokens([tok("52:58 POR 0 1 MCI", (40, 60, 600, 90))])
        self.assertEqual((reading.codes, reading.scores), (("POR", "MCI"), (0, 1)))

    def test_expected_codes_reject_other_words(self):
        self.assertIsNone(parse_tokens([tok("BET 0 1 WIN", (0, 0, 300, 30))], expected=["POR", "MCI"]))


class StackedParserTests(unittest.TestCase):
    def test_each_score_goes_to_the_code_on_its_row(self):
        reading = parse_tokens(stacked("1", "0"))
        self.assertEqual(reading.codes, ("RMA", "ATM"))
        self.assertEqual(reading.scores, (1, 0))

    def test_merged_token_covering_both_cells_is_split_top_to_bottom(self):
        tokens = [tok("RMA", RMA_BOX), tok("10", (114, 27, 131, 72)), tok("ATM", ATM_BOX)]
        self.assertEqual(parse_tokens(tokens).scores, (1, 0))

    def test_remembered_layout_reads_digits_when_the_codes_are_missed(self):
        layout = [("RMA", RMA_BOX), ("ATM", ATM_BOX)]
        tokens = [tok("2", (114, 27, 130, 47)), tok("0", (114, 48, 131, 72))]
        reading = parse_tokens(tokens, expected=["RMA", "ATM"], layout=layout)
        self.assertEqual(reading.scores, (2, 0))

    def test_one_edit_of_tolerance_with_expected_codes(self):
        self.assertEqual(as_code("RNA", ["RMA", "ATM"]), "RMA")
        self.assertIsNone(as_code("RIAA", ["RMA", "ATM"]))  # two edits: not trusted

    def test_codes_alone_locate_the_graphic(self):
        found = find_codes([tok("RMA", RMA_BOX), tok("--", (106, 22, 139, 77)), tok("ATM", ATM_BOX)])
        self.assertEqual([code for code, _ in found], ["RMA", "ATM"])

    def test_captions_and_clocks_are_not_scores(self):
        # The lower third on the 480p clip: a scorer caption, not the score bug.
        tokens = [tok("ADEMOLALOOKMAN", (151, 16, 307, 36)), tok("0-1 MIN.33", (150, 40, 233, 62))]
        self.assertIsNone(parse_tokens(tokens))
        self.assertIsNone(parse_tokens([tok("07:58", (44, 55, 78, 70))]))
        self.assertIsNone(as_code("MIN"))


def reading(scores, codes=("POR", "MCI")):
    return SimpleNamespace(codes=codes, scores=scores, code_boxes=None, region=None)


class ScoreTrackerTests(unittest.TestCase):
    def test_a_confirmed_rise_is_one_goal_and_a_single_misread_is_not(self):
        tracker = ScoreTracker(confirm_reads=2)
        for frame, scores in [(0, (0, 0)), (25, (0, 0)), (50, (0, 0)), (75, (0, 1)), (100, (0, 0)),
                              (125, (0, 1)), (150, (0, 1)), (175, (0, 1))]:
            tracker.observe(frame, reading(scores))
        self.assertEqual(len(tracker.goals), 1)
        goal = tracker.goals[0]
        self.assertEqual(goal.code, "MCI")
        self.assertEqual(goal.frame, 125)
        self.assertEqual(goal.window_start, 100)
        self.assertEqual(goal.score_after, {"POR": 0, "MCI": 1})

    def test_implausible_jump_is_logged_not_scored(self):
        tracker = ScoreTracker(confirm_reads=2)
        for frame, scores in [(0, (0, 0)), (25, (0, 0)), (50, (8, 0)), (75, (8, 0))]:
            tracker.observe(frame, reading(scores))
        self.assertEqual(tracker.goals, [])
        self.assertTrue(any("implausible" in reason for _, reason in tracker.rejected))

    def test_a_goal_taken_back_is_withdrawn(self):
        tracker = ScoreTracker(confirm_reads=2)
        for frame, scores in [(0, (0, 0)), (25, (0, 0)), (50, (1, 0)), (75, (1, 0)), (100, (0, 0)), (125, (0, 0))]:
            tracker.observe(frame, reading(scores))
        self.assertEqual(tracker.goals, [])

    def test_a_different_match_graphic_is_ignored(self):
        tracker = ScoreTracker(confirm_reads=2)
        for frame, scores in [(0, (0, 0)), (25, (0, 0))]:
            tracker.observe(frame, reading(scores))
        tracker.observe(50, reading((3, 2), codes=("ARS", "LIV")))
        tracker.observe(75, reading((3, 2), codes=("ARS", "LIV")))
        self.assertEqual(tracker.goals, [])
        self.assertEqual(tracker.state, {"POR": 0, "MCI": 0})


class ReaderTests(unittest.TestCase):
    def test_reader_locks_on_the_graphic_and_reports_the_change(self):
        score = {"value": "0 0"}
        calls = []

        def fake_ocr(image):
            calls.append(image.shape)
            return [(f"POR {score['value']} MCI", 0.93, (60.0, 40.0, 400.0, 80.0))]

        fps = 25.0
        reader = ScoreboardReader(fake_ocr, fps, (1080, 1920), expected_codes=["POR", "MCI"])
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for number in range(0, 300):
            if number == 150:
                score["value"] = "0 1"
            reader.update(number, frame)
        self.assertIsNotNone(reader.region)
        self.assertEqual([goal.code for goal in reader.tracker.goals], ["MCI"])
        goal = reader.tracker.goals[0]
        self.assertGreaterEqual(goal.frame, 150)
        self.assertLess(goal.window_start, 150)
        # One scan, then one small-region read per second - not every frame.
        self.assertLess(len(calls), 20)


class TeamCodeTests(unittest.TestCase):
    def test_spelling_rule(self):
        self.assertTrue(code_spells_name("RMA", "Real Madrid"))
        self.assertTrue(code_spells_name("ATM", "Atletico Madrid"))
        self.assertTrue(code_spells_name("POR", "Porto"))
        self.assertTrue(code_spells_name("MCI", "Man City"))
        self.assertFalse(code_spells_name("MCI", "Porto"))

    def test_roster_codes(self):
        self.assertIn("RMA", expected_codes_for("Real Madrid"))
        self.assertIn("MCI", expected_codes_for("Man City"))
        self.assertIn("POR", expected_codes_for("Porto"))
        self.assertIn("ATM", expected_codes_for("Atletico"))

    def test_mapping_uses_known_codes_then_spelling_then_elimination(self):
        self.assertEqual(
            map_codes_to_teams(["POR", "MCI"], {1: "Porto", 2: "Man City"}, {2: {"MCI"}}),
            {"MCI": 2, "POR": 1},
        )
        # MCI spells both Manchester clubs; MUN only United, which settles City.
        self.assertEqual(
            map_codes_to_teams(["MCI", "MUN"], {1: "Manchester City", 2: "Manchester United"}),
            {"MUN": 2, "MCI": 1},
        )


class AttributionTests(unittest.TestCase):
    CODES = ("POR", "MCI")
    NAMES = {1: "Porto", 2: "Man City"}
    GOAL = ScoreGoal("MCI", 350, 50, {"POR": 0, "MCI": 0}, {"POR": 0, "MCI": 1})

    def shot(self, frame, team_id, track_id):
        return {
            "event": "shot", "frame": frame, "team_id": team_id, "team_name": self.NAMES.get(team_id),
            "participants": [{"role": "shooter", "track_id": track_id, "team_id": team_id}],
            "details": {"outcome": "unknown"},
        }

    def test_the_shot_in_the_window_becomes_the_goal_and_names_the_scorer(self):
        shot = self.shot(174, 2, 31)
        goals, evidence = attribute_goals([self.GOAL], self.CODES, {"MCI": 2, "POR": 1}, self.NAMES, [shot], 25.0)
        self.assertEqual(goals[0]["frame"], 174)
        self.assertEqual(goals[0]["team_id"], 2)
        self.assertEqual(goals[0]["participants"][0]["track_id"], 31)
        self.assertEqual(goals[0]["participants"][0]["role"], "scorer")
        self.assertEqual(goals[0]["details"]["score_after"], "POR 0-1 MCI")
        self.assertEqual(shot["details"]["outcome"], "goal")
        self.assertEqual(evidence, [shot])

    def test_a_shot_outside_the_window_is_not_used(self):
        goals, _ = attribute_goals([self.GOAL], self.CODES, {"MCI": 2}, self.NAMES, [self.shot(10, 2, 31)], 25.0)
        self.assertEqual(goals[0]["participants"], [])
        self.assertEqual(goals[0]["details"]["attribution"], "scoreboard only")

    def test_a_misread_kit_does_not_lose_the_scorer(self):
        goals, _ = attribute_goals([self.GOAL], self.CODES, {"MCI": 2, "POR": 1}, self.NAMES, [self.shot(200, 1, 12)], 25.0)
        self.assertEqual(goals[0]["participants"][0]["track_id"], 12)
        self.assertEqual(goals[0]["participants"][0]["team_id"], 2)

    def test_possession_is_the_fallback(self):
        samples = [SimpleNamespace(frame_number=f, track_id=t, team_id=team) for f, t, team in
                   [(100, 4, 1), (200, 9, 2), (260, 9, 2), (300, 4, 1)]]
        goals, _ = attribute_goals([self.GOAL], self.CODES, {"MCI": 2}, self.NAMES, [], 25.0, samples)
        self.assertEqual(goals[0]["participants"][0]["track_id"], 9)
        self.assertEqual(goals[0]["frame"], 260)


if __name__ == "__main__":
    unittest.main()
