"""The same play must give the same events whatever the source frame rate.

The API analyses 10-12 frames a second of a 25-60 fps source (commit
99063991). Thresholds that counted source frames or updates then meant a
different amount of time on every video, and a goal found on a 25 fps clip
was missed on a 50 or 60 fps one. These tests describe one play in seconds and
sample it the way the pipeline does.
"""

import unittest
from unittest import mock

from pipeline_config import scheduled_frames
from scene import SceneBoundary, SceneTimeline
from track_events import TrackEventDetector, withdraw_saves_of_goals
from trackers.tracker import BallSelector

from tests.test_goal_from_shot import FRAME, MOUTH, ball_at, scene

SHOT_START = 0.4      # seconds the shooter holds the ball before striking it
FLIGHT = 0.2          # seconds from the boot to the goal line
IN_NET = 0.12         # seconds the ball is visible inside the goal
TOTAL = 4.5


def goal_play(t):
    """Ball position at time t: held, struck, in the net, then lost."""
    if t < SHOT_START:
        return ball_at(210.0, 495.0)
    if t < SHOT_START + FLIGHT:
        progress = (t - SHOT_START) / FLIGHT
        return ball_at(210.0 + (950.0 - 210.0) * progress, 470.0)
    if t < SHOT_START + FLIGHT + IN_NET:
        return ball_at(990.0, 470.0)
    return None


def saved_play(t):
    """Same strike; the keeper holds it."""
    if t < SHOT_START + FLIGHT:
        return goal_play(t)
    return ball_at(905.0, 470.0)


def run_play(play, source_fps, analysis_fps):
    total_frames = int(TOTAL * source_fps)
    detector = TrackEventDetector(source_fps, (1280, 720), detect_cards=False)
    events = []
    with mock.patch("track_events.detector.find_goal_mouth", return_value=MOUTH):
        for frame in scheduled_frames(total_frames, source_fps, analysis_fps):
            events.extend(detector.update(frame, scene(), play(frame / source_fps), frame=FRAME))
        events.extend(detector.finish(total_frames))
    return withdraw_saves_of_goals(events)


RATES = [(25.0, 25.0), (25.0, 12.0), (30.0, 12.0), (50.0, 12.0), (60.0, 12.0), (60.0, 10.0)]


class FrameRateInvarianceTests(unittest.TestCase):
    def test_a_goal_is_found_at_every_source_and_analysis_rate(self):
        for source, analysis in RATES:
            with self.subTest(source_fps=source, analysis_fps=analysis):
                goals = [e for e in run_play(goal_play, source, analysis) if e["event"] == "goal"]
                self.assertEqual(len(goals), 1)
                seconds = goals[0]["frame"] / source
                self.assertAlmostEqual(seconds, SHOT_START, delta=0.3)

    def test_a_held_ball_is_never_a_goal_at_any_rate(self):
        for source, analysis in RATES:
            with self.subTest(source_fps=source, analysis_fps=analysis):
                events = run_play(saved_play, source, analysis)
                self.assertNotIn("goal", [e["event"] for e in events])


class BallGateTests(unittest.TestCase):
    def test_a_fast_shot_passes_the_gate_at_the_analysis_rate(self):
        # 25 player heights a second (a hard shot) seen at 12 analysed frames a
        # second: about 2 heights between analysed frames, past the old 1.6
        # heights-per-frame gate that was tuned at 25 fps.
        selector = BallSelector(frame_rate=12.0)
        height = 100.0
        step = 25.0 * height / 12.0
        chosen = []
        for i in range(4):
            x = 100.0 + step * i
            chosen.append(selector.select([([x - 5, 500, x + 5, 510], 0.4)], height))
        self.assertTrue(all(box is not None for box in chosen))

    def test_a_teleport_is_still_rejected(self):
        selector = BallSelector(frame_rate=12.0)
        selector.select([([100, 500, 110, 510], 0.4)], 100.0)
        selector.select([([110, 500, 120, 510], 0.4)], 100.0)
        self.assertIsNone(selector.select([([1200, 100, 1210, 110], 0.4)], 100.0))


class SceneStartBetweenAnalysedFramesTests(unittest.TestCase):
    def test_a_cut_between_analysed_frames_is_reported_at_the_next_one(self):
        timeline = SceneTimeline([SceneBoundary(51, 51, "cut", 0.7), SceneBoundary(90, 97, "dissolve", 0.5)])
        analysed = list(scheduled_frames(150, 25.0, 12.0))
        self.assertNotIn(51, analysed)  # the exact cut frame is skipped
        handled, resets = -1, []
        for number in analysed:
            if timeline.in_transition(number):
                continue
            start = timeline.latest_start_at_or_before(number)
            if start is not None and start > handled:
                resets.append(number)
                handled = start
        self.assertEqual(len(resets), 2)
        self.assertTrue(51 < resets[0] <= 53)
        self.assertTrue(98 <= resets[1] <= 100)

    def test_no_start_before_the_first_boundary(self):
        timeline = SceneTimeline([SceneBoundary(51, 51, "cut", 0.7)])
        self.assertIsNone(timeline.latest_start_at_or_before(50))
        self.assertEqual(timeline.latest_start_at_or_before(400), 51)


if __name__ == "__main__":
    unittest.main()
