"""Shot boundaries, and what the pipeline forgets at one."""

import unittest

import numpy as np

from scene import SceneBoundary, SceneTimeline, find_boundaries, frame_signature
from scoreboard import attribute_goals
from scoreboard.reader import ScoreGoal
from track_events import TrackEventDetector


RNG = np.random.default_rng(3)


def texture(bgr):
    """A noisy image dominated by one colour, like a pitch or a crowd."""
    base = np.full((90, 160, 3), bgr, dtype=np.float32)
    noise = RNG.normal(0, 25, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def signatures(frames):
    return [frame_signature(frame) for frame in frames]


class FindBoundariesTests(unittest.TestCase):
    def test_hard_cut_and_dissolve_are_found(self):
        pitch, crowd, close_up = texture((40, 150, 40)), texture((120, 60, 160)), texture((30, 30, 200))
        frames = [pitch] * 10 + [crowd] * 10
        for step in range(1, 9):  # an eight-frame crossfade
            alpha = step / 9.0
            frames.append(((1 - alpha) * crowd + alpha * close_up).astype(np.uint8))
        frames += [close_up] * 10
        boundaries = find_boundaries(signatures(frames))
        self.assertEqual(len(boundaries), 2)
        cut, dissolve = boundaries
        self.assertEqual((cut.kind, cut.start), ("cut", 10))
        self.assertEqual(dissolve.kind, "dissolve")
        self.assertLessEqual(dissolve.start, 21)
        self.assertGreaterEqual(dissolve.end, 26)

    def test_a_camera_pan_is_not_a_boundary(self):
        # Slide across a wide image: every frame differs a little, none a lot.
        wide = np.concatenate([texture((40, 150, 40)), texture((60, 130, 50)), texture((45, 140, 45))], axis=1)
        frames = [np.ascontiguousarray(wide[:, offset:offset + 160]) for offset in range(0, 320, 8)]
        self.assertEqual(find_boundaries(signatures(frames)), [])

    def test_timeline_marks_blends_and_scene_starts(self):
        timeline = SceneTimeline([SceneBoundary(52, 52, "cut", 0.7), SceneBoundary(168, 175, "dissolve", 0.57)])
        self.assertTrue(timeline.starts_scene(52))
        self.assertFalse(timeline.in_transition(52))
        self.assertTrue(all(timeline.in_transition(f) for f in range(168, 176)))
        self.assertTrue(timeline.starts_scene(176))
        self.assertFalse(timeline.in_transition(176))
        self.assertEqual(timeline.describe(), "cut 52, dissolve 168-175")


HEIGHT = 100.0


def player(x, team=1):
    return {"bbox": [x - 20.0, 400.0, x + 20.0, 500.0], "team": team, "team_name": f"Team {team}",
            "team_confidence": 0.9, "role": "player"}


def ball(x):
    return [x - 6.0, 489.0, x + 6.0, 501.0]


class ResetSceneTests(unittest.TestCase):
    def run_frames(self, cut_at=None):
        detector = TrackEventDetector(25.0, detect_cards=False)
        events = []
        for frame in range(40):
            if frame == cut_at:
                events.extend(detector.reset_scene(frame))
            if frame < 20:
                players, ball_box = {1: player(100.0)}, ball(110.0)
            else:
                # After the cut a different camera: a team-mate far away on screen.
                players, ball_box = {2: player(700.0)}, ball(690.0)
            events.extend(detector.update(frame, players, ball_box))
        return events

    def test_without_a_cut_the_jump_reads_as_a_pass(self):
        self.assertIn("pass", [e["event"] for e in self.run_frames(cut_at=None)])

    def test_no_pass_is_formed_across_a_cut(self):
        self.assertNotIn("pass", [e["event"] for e in self.run_frames(cut_at=20)])


class ConfidentOpponentIsNotTheScorerTests(unittest.TestCase):
    def test_a_confidently_read_defender_is_not_credited(self):
        goal = ScoreGoal("MCI", 350, 50, {"POR": 0, "MCI": 0}, {"POR": 0, "MCI": 1})
        # The 1080p case: Porto's #20 lunging to block, read as Porto with confidence.
        block = {"event": "shot", "frame": 174, "team_id": 1, "team_name": "Porto",
                 "participants": [{"role": "shooter", "track_id": 50, "team_id": 1, "team_confidence": 0.8}],
                 "details": {}}
        goals, evidence = attribute_goals([goal], ("POR", "MCI"), {"POR": 1, "MCI": 2},
                                          {1: "Porto", 2: "Man City"}, [block], 25.0)
        self.assertEqual(goals[0]["team_id"], 2)
        self.assertEqual(goals[0]["participants"], [])
        self.assertEqual(evidence, [])


if __name__ == "__main__":
    unittest.main()
