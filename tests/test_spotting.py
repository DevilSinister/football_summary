"""Spotting-model plumbing: sampling, windowing, peaks, and conservative fusion."""

import unittest

import numpy as np
import torch

from spotting import SOCCERNET_V2_CLASSES, ActionSpotter, fuse_spotted_events
from spotting.labels import load_labels

FPS = 25.0
LABELS = list(SOCCERNET_V2_CLASSES)
GOAL = LABELS.index("Goal") + 1


class FakeModel(torch.nn.Module):
    """Scores 'Goal' on frames whose mean brightness is high; background otherwise."""

    def forward(self, x):
        batch, length = x.shape[:2]
        logits = torch.zeros(batch, length, len(LABELS) + 1)
        logits[..., 0] = 2.0
        bright = x.mean(dim=(2, 3, 4)) > 1.0
        logits[..., GOAL] = torch.where(bright, torch.tensor(6.0), torch.tensor(-6.0))
        return logits


def frame(bright):
    return np.full((72, 128, 3), 255 if bright else 0, dtype=np.uint8)


class LabelOrderTests(unittest.TestCase):
    def test_labels_are_the_sorted_order_e2e_spot_writes(self):
        # E2E-Spot's parse_soccernet.py writes class.txt as sorted(names).
        self.assertEqual(LABELS, sorted(LABELS))
        self.assertEqual(LABELS[0], "Ball out of play")
        self.assertEqual(LABELS.index("Goal"), 5)
        self.assertEqual(LABELS.index("Shots on target"), 12)
        self.assertEqual(load_labels(None), LABELS)


class SpotterTests(unittest.TestCase):
    def test_samples_at_the_requested_rate_and_finds_the_peak(self):
        spotter = ActionSpotter(FakeModel(), LABELS, FPS, sample_fps=2.0, clip_len=10)
        for number in range(int(30 * FPS)):
            spotter.add_frame(number, frame(bright=12.0 <= number / FPS < 12.6))
        spotter.finish()
        self.assertEqual(len(spotter.sample_frames), 60)
        self.assertTrue(all(count > 0 for count in spotter._score_count))
        goals = [e for e in spotter.events(min_confidence=0.5) if e["event"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertAlmostEqual(goals[0]["frame"] / FPS, 12.0, delta=0.5)

    def test_a_clip_shorter_than_one_window_is_still_scored(self):
        spotter = ActionSpotter(FakeModel(), LABELS, FPS, sample_fps=2.0, clip_len=100)
        for number in range(int(5 * FPS)):
            spotter.add_frame(number, frame(bright=False))
        spotter.finish()
        self.assertEqual(spotter.scores().shape, (10, len(LABELS) + 1))
        self.assertEqual(spotter.events(min_confidence=0.5), [])


def spotted(event, second, confidence):
    return {"event": event, "frame": int(second * FPS), "confidence": confidence, "source": "spotting",
            "participants": [], "details": {}}


def tracked_shot(second, team=1, track=7):
    return {"event": "shot", "frame": int(second * FPS), "source": "tracks", "team_id": team,
            "team_name": f"Team {team}", "details": {},
            "participants": [{"role": "shooter", "track_id": track, "team_id": team}]}


class FusionTests(unittest.TestCase):
    def test_a_weak_spotted_goal_after_a_tracked_shot_becomes_that_shooters_goal(self):
        events = fuse_spotted_events([tracked_shot(10.0)], [spotted("goal", 11.0, 0.22)], FPS)
        goals = [e for e in events if e["event"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["team_id"], 1)
        self.assertEqual(goals[0]["participants"][0]["role"], "scorer")
        self.assertEqual(goals[0]["participants"][0]["track_id"], 7)

    def test_a_weak_spotted_goal_without_a_shot_is_ignored(self):
        events = fuse_spotted_events([], [spotted("goal", 11.0, 0.22)], FPS)
        self.assertEqual([e for e in events if e["event"] == "goal"], [])

    def test_a_confident_spotted_goal_stands_alone(self):
        events = fuse_spotted_events([], [spotted("goal", 11.0, 0.8)], FPS)
        self.assertEqual(len([e for e in events if e["event"] == "goal"]), 1)

    def test_a_spotted_goal_at_a_video_goal_only_adds_confidence(self):
        video = {"event": "goal", "frame": int(10.5 * FPS), "source": "tracks", "details": {}}
        events = fuse_spotted_events([video], [spotted("goal", 11.0, 0.9)], FPS)
        self.assertEqual([e for e in events if e["event"] == "goal"], [video])
        self.assertEqual(video["details"]["spotting_confidence"], 0.9)

    def test_a_spotted_shot_fills_in_where_tracking_saw_none(self):
        events = fuse_spotted_events([], [spotted("shot_on_target", 10.0, 0.42)], FPS)
        shots = [e for e in events if e["event"] == "shot"]
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0]["source"], "spotting")
        self.assertTrue(shots[0]["details"]["on_target"])

    def test_a_spotted_shot_next_to_a_tracked_one_is_not_doubled(self):
        events = fuse_spotted_events([tracked_shot(10.0)], [spotted("shot_on_target", 10.5, 0.9)], FPS)
        self.assertEqual(len([e for e in events if e["event"] == "shot"]), 1)

    def test_on_and_off_target_on_one_strike_make_one_shot(self):
        # Regression (test.mp4 run): both outputs peaked at 10.0 s.
        events = fuse_spotted_events(
            [], [spotted("shot_off_target", 10.0, 0.37), spotted("shot_on_target", 10.0, 0.42)], FPS
        )
        shots = [e for e in events if e["event"] == "shot"]
        self.assertEqual(len(shots), 1)
        self.assertTrue(shots[0]["details"]["on_target"])

    def test_weak_shots_and_kick_offs_are_dropped(self):
        events = fuse_spotted_events(
            [], [spotted("shot_on_target", 10.0, 0.1), spotted("kick_off", 40.0, 0.1)], FPS
        )
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
