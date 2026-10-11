import threading
import time
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.pipeline_runtime import SerialJobQueue
from main import final_score_by_team, goal_output_events, should_calibrate_teams
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


if __name__ == "__main__":
    unittest.main()
