import threading
import time
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.pipeline_runtime import SerialJobQueue
from commentary import Word, find_mentions
from main import (
    _apply_commentary,
    _attach_pass_assisters,
    _first_choice_keeper,
    _name_keepers,
    final_score_by_team,
    goal_output_events,
    should_calibrate_teams,
)
from pipeline_config import PipelineConfig, scheduled_frames
from trackers import Tracker
from scoreboard import ScoreboardReader
from scoreboard.parser import Token, parse_one_line
from track_events import TrackEventDetector


class PipelineConfigTests(unittest.TestCase):
    def test_adaptive_schedule_preserves_source_frame_numbers(self):
        config = PipelineConfig.short_clip()
        self.assertEqual(config.analysis_fps(26), 12.0)
        self.assertEqual(config.analysis_fps(61), 10.0)
        frames = list(scheduled_frames(100, 25.0, 10.0))
        self.assertEqual(frames[:5], [0, 2, 5, 8, 10])
        self.assertEqual([frame / 25.0 for frame in frames[:3]], [0.0, 0.08, 0.2])

    def test_short_profile_detector_sizes(self):
        config = PipelineConfig.short_clip()
        self.assertEqual(config.detector_size(480), 640)
        self.assertEqual(config.detector_size(720), 640)
        self.assertEqual(config.detector_size(1080), 960)

    def test_confirmed_upload_centres_skip_second_calibration(self):
        self.assertFalse(should_calibrate_teams(SimpleNamespace(confirmed_team_centers=True)))
        self.assertTrue(should_calibrate_teams(SimpleNamespace()))

    def test_bundled_short_clip_annotations_are_complete(self):
        fixture_path = Path(__file__).parent / "fixtures" / "short_clips.json"
        fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(len(fixtures), 3)
        for fixture in fixtures:
            self.assertEqual(len(fixture["team_codes"]), 2)
            self.assertEqual(sum(fixture["final_score"]), fixture["goal_count"])
            self.assertEqual(len(fixture["goals"]), fixture["goal_count"])


class RuntimeTests(unittest.TestCase):
    def test_goal_profile_disables_non_goal_event_work(self):
        detector = TrackEventDetector(
            25.0,
            detect_cards=False,
            detect_fouls=False,
            detect_passes=False,
            detect_penalties=False,
        )
        self.assertIsNone(detector.card_detector)
        self.assertIsNone(detector.foul_detector)
        self.assertFalse(detector.detect_passes)
        self.assertFalse(detector.detect_penalties)

    def test_tracker_uses_injected_models_without_reloading(self):
        person, ball = object(), object()
        tracker = Tracker("unused.pt", model=person, ball_model=ball)
        self.assertIs(tracker.model, person)
        self.assertIs(tracker.ball_model, ball)

    def test_serial_queue_never_overlaps_jobs(self):
        queue = SerialJobQueue()
        active = 0
        peak = 0
        lock = threading.Lock()

        def job(value):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return value

        first = queue.submit(job, 1)
        second = queue.submit(job, 2)
        self.assertEqual((first.result(), second.result()), (1, 2))
        self.assertEqual(peak, 1)


class HybridScoreboardTests(unittest.TestCase):
    def test_stray_ocr_parentheses_do_not_hide_a_score(self):
        reading = parse_one_line(
            Token("(POR(1-0)EGY)", 0.9, (0.0, 0.0, 100.0, 20.0)),
            ["POR", "EGY"],
        )
        self.assertEqual(reading.scores, (1, 0))

    def test_shot_burst_temporarily_uses_half_second_cadence(self):
        reader = ScoreboardReader(lambda _image: [], 25.0, (480, 854))
        reader.region = (0, 0, 100, 50)
        reader.layout = []
        reader.locked_at = 0
        reader._read_region = lambda _frame: None
        reader.trigger_burst(100)
        reader.update(100, np.zeros((480, 854, 3), dtype=np.uint8))
        self.assertEqual(reader._next_frame, 112)
        self.assertGreaterEqual(reader.burst_until, 100 + 8 * 25)

    def test_expected_code_filter_relaxes_after_repeated_misses(self):
        calls = {"count": 0}

        def fake_ocr(_image):
            calls["count"] += 1
            return [("XYZ 1 0 QRS", 0.95, (10.0, 10.0, 180.0, 35.0))]

        reader = ScoreboardReader(
            fake_ocr,
            25.0,
            (480, 854),
            expected_codes=["AAA", "BBB"],
            permissive_after_misses=1,
        )
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        first, _ = reader._scan(frame)
        second, _ = reader._scan(frame)
        self.assertIsNone(first)
        self.assertIsNotNone(second)


class GoalOutputTests(unittest.TestCase):
    def test_goal_only_output_keeps_internal_evidence_out(self):
        goal = {"event": "goal", "team_id": 1}
        self.assertEqual(goal_output_events([{"event": "shot"}, goal, {"event": "save"}]), [goal])

    def test_scoreboard_latest_state_is_authoritative(self):
        tracker = SimpleNamespace(state={"AAA": 2, "BBB": 1})
        reader = SimpleNamespace(active=True, tracker=tracker)
        events = [{"event": "goal", "team_id": 2}]
        self.assertEqual(final_score_by_team(events, reader, {"AAA": 1, "BBB": 2}), {1: 2, 2: 1})

    def test_detected_goals_are_the_ocr_fallback(self):
        events = [
            {"event": "goal", "team_id": 1},
            {"event": "goal", "scorer": {"team_id": 2}},
        ]
        self.assertEqual(final_score_by_team(events), {1: 1, 2: 1})


SQUADS = {
    1: [{"jersey_no": 13, "name": "Backup Keeper", "position": "GK"},
        {"jersey_no": 1, "name": "Number One", "position": "GK"},
        {"jersey_no": 9, "name": "Erling Haaland", "position": "FW"},
        {"jersey_no": 17, "name": "Kevin De Bruyne", "position": "MF"}],
    2: [{"jersey_no": 23, "name": "Lone Keeper", "position": "GK"},
        {"jersey_no": 4, "name": "Virgil van Dijk", "position": "DF"}],
}
NAMES = {1: "City", 2: "Reds"}


def person(role, track, team):
    return {"role": role, "track_id": track, "team_id": team, "team_name": NAMES[team]}


class RoleHelperTests(unittest.TestCase):
    def test_the_last_team_mate_pass_to_the_scorer_is_the_assist(self):
        events = [
            {"event": "pass", "frame": 100, "participants": [person("passer", 3, 1), person("receiver", 9, 1)]},
            {"event": "pass", "frame": 200, "participants": [person("passer", 17, 1), person("receiver", 9, 1)]},
            {"event": "pass", "frame": 230, "participants": [person("passer", 4, 2), person("receiver", 9, 1)]},
            {"event": "goal", "frame": 260, "participants": [person("scorer", 9, 1)]},
        ]
        _attach_pass_assisters(events, 25.0)
        assister = next(p for p in events[-1]["participants"] if p["role"] == "assister")
        self.assertEqual(assister["track_id"], 17)

    def test_the_number_one_keeper_is_shirt_one_else_the_lowest_number(self):
        self.assertEqual(_first_choice_keeper(SQUADS[1])["name"], "Number One")
        self.assertEqual(_first_choice_keeper(SQUADS[2])["name"], "Lone Keeper")
        self.assertIsNone(_first_choice_keeper([{"jersey_no": 9, "name": "X", "position": "FW"}]))

    def test_a_save_names_the_roster_keeper_when_no_shirt_was_read(self):
        save = {"event": "save", "frame": 10,
                "actors": [{**person("goalkeeper", 30, 2), "jersey_number": None}, person("shooter", 9, 1)]}
        _name_keepers([save], SQUADS, NAMES)
        self.assertEqual(save["actors"][0]["player_name"], "Lone Keeper")
        self.assertEqual(save["scorer"]["jersey_number"], 23)
        self.assertEqual(save["actors"][0]["identity_source"], "roster keeper")

    WORDS = [Word(10.0 + i * 0.3, 10.2 + i * 0.3, w)
             for i, w in enumerate("De Bruyne crosses and Haaland scores".split())]

    def _goal(self, **scorer_fields):
        scorer = {**person("scorer", 9, 1), "jersey_number": 19, "player_image": "x.jpg", **scorer_fields}
        return {"event": "goal", "frame": int(11.5 * 25), "team_id": 1, "team_name": "City",
                "actors": [dict(scorer)], "scorer": dict(scorer)}

    def test_commentary_names_an_unconfirmed_scorer_and_adds_the_assister(self):
        goal = self._goal(unconfirmed=True)
        _apply_commentary([goal], self.WORDS, find_mentions(self.WORDS, SQUADS), 25.0, NAMES)
        self.assertEqual((goal["scorer"]["jersey_number"], goal["scorer"]["player_name"]), (9, "Erling Haaland"))
        self.assertNotIn("player_image", goal["scorer"])  # the photo showed #19
        assister = next(a for a in goal["actors"] if a["role"] == "assister")
        self.assertEqual(assister["player_name"], "Kevin De Bruyne")
        self.assertEqual(goal["details"]["scorer_from"], "commentary")

    def test_commentary_does_not_overrule_a_confirmed_scorer(self):
        # Levante-Sevilla: the commentary named 1 of 6 scorers wrong, the
        # caption none; a confirmed (fused) scorer stands, the assister is added.
        goal = self._goal(unconfirmed=False, player_name="Read From Caption")
        _apply_commentary([goal], self.WORDS, find_mentions(self.WORDS, SQUADS), 25.0, NAMES)
        self.assertEqual(goal["scorer"]["jersey_number"], 19)
        self.assertTrue(any(a["role"] == "assister" for a in goal["actors"]))

    def test_goal_only_output_can_include_fouls(self):
        events = [{"event": "goal"}, {"event": "foul"}, {"event": "shot"}]
        self.assertEqual(goal_output_events(events, ("goal", "foul")), events[:2])


class BothSquadsScorerTests(unittest.TestCase):
    """main._improve_goal_scorers: a caption from the other squad corrects the team."""

    SQUADS = {
        1: [{"jersey_no": 9, "name": "Iván Romero", "position": "FW"}],
        2: [{"jersey_no": 16, "name": "Isaac Romero", "position": "FW"}],
    }
    NAMES = {1: "Levante", 2: "Sevilla"}

    def _run(self, goal, caption):
        import main

        identifier = SimpleNamespace(
            jersey_reader=SimpleNamespace(max_ocr_calls=0),
            identity_store=SimpleNamespace(players={}),
            identify_track=lambda track_id: {},
        )
        original = main.read_caption_candidates
        main.read_caption_candidates = lambda *args, **kwargs: list(caption(kwargs.get("squads") or {}))
        try:
            main._improve_goal_scorers(
                [goal], identifier, "clip.mp4", 25.0, 1080, ocr_fn=lambda image: [],
                extra_ocr_allowance=0, team_squads=self.SQUADS, team_names=self.NAMES,
            )
        finally:
            main.read_caption_candidates = original
        return goal

    @staticmethod
    def _ivan(squads):
        from scorer_identifier import ScorerCandidate

        if 1 in squads:
            yield ScorerCandidate(9, 6.0, "caption", "Iván Romero", team_id=1)

    def _goal(self, source="tracks"):
        shooter = {"role": "scorer", "track_id": 4, "team_id": 2, "team_name": "Sevilla",
                   "jersey_number": 16, "jersey_confidence": 0.9}
        return {"event": "goal", "frame": 2360, "source": source, "team_id": 2, "team_name": "Sevilla",
                "scorer": dict(shooter), "actors": [dict(shooter)], "details": {}}

    def test_a_caption_naming_the_other_squads_player_corrects_the_team(self):
        goal = self._run(self._goal(), self._ivan)
        self.assertEqual((goal["team_id"], goal["team_name"]), (1, "Levante"))
        self.assertEqual((goal["scorer"]["jersey_number"], goal["scorer"]["player_name"]), (9, "Iván Romero"))
        self.assertEqual(goal["scorer"]["team_name"], "Levante")
        self.assertEqual(goal["details"]["team_corrected"], {"from": "Sevilla", "by": ["caption"]})

    def test_a_goal_without_a_tracked_scorer_gets_a_scorer_actor(self):
        # Levante-Sevilla 167.5 s: a spotted goal, caption Ndukwe; only the
        # commentary assister was in the actors, so the API stored no scorer.
        goal = {"event": "goal", "frame": 2360, "source": "spotting", "team_id": 1, "team_name": "Levante",
                "details": {}, "actors": [{"role": "assister", "team_id": 1, "jersey_number": 24}]}
        goal = self._run(goal, self._ivan)
        self.assertEqual([a["role"] for a in goal["actors"]], ["scorer", "assister"])
        self.assertEqual(goal["actors"][0]["player_name"], "Iván Romero")

    def test_the_score_graphics_team_is_not_corrected(self):
        goal = self._run(self._goal(source="scoreboard"), self._ivan)
        self.assertEqual(goal["team_id"], 2)
        self.assertNotIn("team_corrected", goal["details"])
        self.assertEqual(goal["scorer"]["jersey_number"], 16)


class CaptionLedScorerTests(unittest.TestCase):
    def test_commentary_does_not_overwrite_an_unconfirmed_caption_scorer(self):
        # Levante-Sevilla: the caption said Ndukwe, the commentary De la Fuente.
        scorer = {"role": "scorer", "team_id": 1, "team_name": "City", "jersey_number": 9,
                  "unconfirmed": True, "scorer_sources": ["caption"]}
        goal = {"event": "goal", "frame": int(11.5 * 25), "team_id": 1, "team_name": "City",
                "actors": [dict(scorer)], "scorer": dict(scorer)}
        words = RoleHelperTests.WORDS
        _apply_commentary([goal], words, find_mentions(words, SQUADS), 25.0, NAMES)
        self.assertEqual(goal["scorer"]["jersey_number"], 9)
        self.assertNotIn("scorer_from", goal["details"])

    def test_a_shared_spoken_surname_moves_to_the_captioned_player(self):
        from main import _back_caption_with_shared_name
        from scorer_identifier import ScorerCandidate

        spoken = ScorerCandidate(16, 3.2, "commentary", "Isaac Romero",
                                 detail={"shared_with": [(1, 9)]}, team_id=2)
        caption = ScorerCandidate(9, 6.0, "caption", "Iván Romero", team_id=1)
        _back_caption_with_shared_name([spoken, caption], 2)
        self.assertEqual((spoken.team_id, spoken.jersey_number, spoken.player_name), (1, 9, "Iván Romero"))
        self.assertEqual(spoken.detail["moved_from"], [2, 16])

    def test_a_clear_spoken_name_is_not_moved(self):
        from main import _back_caption_with_shared_name
        from scorer_identifier import ScorerCandidate

        spoken = ScorerCandidate(16, 3.2, "commentary", "Isaac Romero", detail={}, team_id=2)
        _back_caption_with_shared_name([spoken, ScorerCandidate(9, 6.0, "caption", team_id=1)], 2)
        self.assertEqual((spoken.team_id, spoken.jersey_number), (2, 16))


if __name__ == "__main__":
    unittest.main()
