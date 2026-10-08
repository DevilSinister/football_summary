"""A shot whose ball ends up in the goal mouth is a goal, without pitch calibration."""

import unittest
from unittest import mock

import numpy as np

from track_events import TrackEventDetector, withdraw_saves_of_goals

FPS = 25.0
HEIGHT = 100.0
FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
# The goal behind the keeper (keeper's feet at x=900, y=500): posts 140 px high.
MOUTH = ((850.0, 380.0), (1050.0, 380.0), (1050.0, 520.0), (850.0, 520.0))


def player(x, y=500.0, team=1, role="player"):
    return {
        "bbox": [x - 20.0, y - HEIGHT, x + 20.0, y],
        "team": team,
        "team_name": f"Team {team}",
        "team_confidence": 0.9,
        "role": role,
    }


def ball_at(x, y):
    return [x - 6.0, y - 6.0, x + 6.0, y + 6.0]


def scene():
    return {1: player(200.0, 500.0, 1), 9: player(900.0, 500.0, 2, role="goalkeeper")}


def shot_then(after):
    """The shooter holds the ball, strikes it at the goal, then `after` frames."""
    frames = [(scene(), ball_at(210.0, 495.0)) for _ in range(10)]
    for step in range(1, 6):
        frames.append((scene(), ball_at(210.0 + (950.0 - 210.0) * step / 5.0, 470.0)))
    return frames + after


def run(frames):
    detector = TrackEventDetector(FPS, (1280, 720), detect_cards=False)
    events = []
    with mock.patch("track_events.detector.find_goal_mouth", return_value=MOUTH):
        for index, (players, ball) in enumerate(frames):
            events.extend(detector.update(index, players, ball, frame=FRAME))
        events.extend(detector.finish(len(frames)))
    return events


class GoalFromShotTests(unittest.TestCase):
    def test_ball_lost_in_the_net_after_a_shot_is_a_goal(self):
        # Regression (FC clip 2026-10-07): the ball passed the keeper into the
        # net, vanished there, and the shot was reported as saved.
        in_net = [(scene(), ball_at(990.0, 470.0)) for _ in range(3)]
        lost = [(scene(), None) for _ in range(90)]
        events = withdraw_saves_of_goals(run(shot_then(in_net + lost)))
        goals = [e for e in events if e["event"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["source"], "tracks")
        self.assertEqual(goals[0]["team_id"], 1)
        scorer = next(p for p in goals[0]["participants"] if p["role"] == "scorer")
        self.assertEqual(scorer["track_id"], 1)
        shot = next(e for e in events if e["event"] == "shot")
        self.assertEqual(shot["details"]["outcome"], "goal")
        self.assertNotIn("save", [e["event"] for e in events])

    def test_a_stray_detection_after_the_ball_vanished_in_the_net_does_not_undo_the_goal(self):
        # Regression (FC clip, 2026-10-08 run): the ball was lost in the net,
        # then the detector picked up something far away about a second later.
        in_net = [(scene(), ball_at(990.0, 470.0)) for _ in range(3)]
        lost = [(scene(), None) for _ in range(20)]
        stray = [(scene(), ball_at(300.0, 480.0)) for _ in range(4)]
        later = [(scene(), None) for _ in range(80)]
        events = run(shot_then(in_net + lost + stray + later))
        self.assertIn("goal", [e["event"] for e in events])

    def test_an_unread_shooter_scores_for_the_side_the_keeper_does_not_play_for(self):
        # Regression (FC clip): the shooter's kit was unread and the goal came
        # out as "unknown_team"; it must never take the keeper's own side.
        def unread_scene():
            players = scene()
            players[1].update(team=0, team_name="unknown_team", team_confidence=0.0)
            players[5] = player(150.0, 300.0, 1)  # a team-mate, so team 1 has a name
            return players

        frames = [(unread_scene(), ball) for _, ball in shot_then(
            [(None, ball_at(990.0, 470.0)) for _ in range(3)] + [(None, None) for _ in range(90)]
        )]
        goals = [e for e in run(frames) if e["event"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["team_id"], 1)

    def test_a_cut_right_after_the_ball_goes_in_is_still_a_goal(self):
        in_net = [(scene(), ball_at(990.0, 470.0)) for _ in range(3)]
        detector = TrackEventDetector(FPS, (1280, 720), detect_cards=False)
        events = []
        with mock.patch("track_events.detector.find_goal_mouth", return_value=MOUTH):
            for index, (players, ball) in enumerate(shot_then(in_net)):
                events.extend(detector.update(index, players, ball, frame=FRAME))
            events.extend(detector.reset_scene(len(shot_then(in_net))))
        self.assertIn("goal", [e["event"] for e in events])

    def test_a_ball_held_by_the_keeper_is_a_save_not_a_goal(self):
        held = [(scene(), ball_at(905.0, 470.0)) for _ in range(40)]
        events = run(shot_then(held))
        self.assertNotIn("goal", [e["event"] for e in events])

    def test_a_ball_that_comes_back_out_is_not_a_goal(self):
        in_mouth = [(scene(), ball_at(990.0, 470.0)) for _ in range(2)]
        out = [(scene(), ball_at(600.0 - 20 * i, 480.0)) for i in range(10)]
        lost = [(scene(), None) for _ in range(60)]
        events = run(shot_then(in_mouth + out + lost))
        self.assertNotIn("goal", [e["event"] for e in events])

    def test_without_a_goal_frame_nothing_is_claimed(self):
        in_net = [(scene(), ball_at(990.0, 470.0)) for _ in range(3)]
        lost = [(scene(), None) for _ in range(90)]
        detector = TrackEventDetector(FPS, (1280, 720), detect_cards=False)
        events = []
        with mock.patch("track_events.detector.find_goal_mouth", return_value=None):
            for index, (players, ball) in enumerate(shot_then(in_net + lost)):
                events.extend(detector.update(index, players, ball, frame=FRAME))
            events.extend(detector.finish(200))
        self.assertNotIn("goal", [e["event"] for e in events])


if __name__ == "__main__":
    unittest.main()
