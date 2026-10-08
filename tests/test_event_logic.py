"""Gate on the learned goal/foul model's proposals."""

import unittest
from types import SimpleNamespace

from event_logic import EventLogic


def goal(frame, confidence=0.6):
    return {"event": "goal", "frame": frame, "confidence": confidence}


NO_BALL = {"ball": [{}], "players": [{}]}


class GoalGateTests(unittest.TestCase):
    def test_a_goal_after_a_shot_is_confirmed_without_the_ball_in_view(self):
        # Regression: the gate required the ball in the judged frame, but at the
        # moment of a goal it is usually in the net or hidden, so true goals
        # were rejected.
        shots = SimpleNamespace(events=[{"event": "shot", "frame": 90}])
        confirmed = EventLogic(fps=25.0).update(goal(120), NO_BALL, 120, track_events=shots)
        self.assertEqual([e["event"] for e in confirmed], ["goal"])
        self.assertEqual(confirmed[0]["source"], "model")

    def test_a_goal_without_a_recent_shot_is_rejected(self):
        old_shot = SimpleNamespace(events=[{"event": "shot", "frame": 10}])
        self.assertEqual(EventLogic(fps=25.0).update(goal(500), NO_BALL, 500, track_events=old_shot), [])

    def test_a_weak_goal_proposal_is_rejected(self):
        shots = SimpleNamespace(events=[{"event": "shot", "frame": 90}])
        self.assertEqual(
            EventLogic(fps=25.0).update(goal(120, confidence=0.3), NO_BALL, 120, track_events=shots), []
        )


if __name__ == "__main__":
    unittest.main()
