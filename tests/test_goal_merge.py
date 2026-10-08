"""Replays of a goal and saves of a shot that went in."""

import unittest

from track_events import drop_replayed_goals, withdraw_saves_of_goals

FPS = 25.0


def goal(frame, source="tracks"):
    return {"event": "goal", "frame": frame, "source": source, "details": {}}


class ReplayTests(unittest.TestCase):
    def test_a_replay_of_a_goal_without_a_scoreboard_is_dropped(self):
        live, replay = goal(40), goal(140)
        kept = drop_replayed_goals([live, replay], FPS)
        self.assertEqual(kept, [live])
        self.assertEqual(live["details"]["replays_dropped"], 1)

    def test_two_goals_more_than_a_minute_apart_are_both_kept(self):
        first, second = goal(100), goal(100 + int(61 * FPS))
        self.assertEqual(drop_replayed_goals([first, second], FPS), [first, second])

    def test_a_goal_near_a_score_change_is_its_replay(self):
        board = goal(300, "scoreboard")
        replay = goal(400)
        self.assertEqual(drop_replayed_goals([board, replay], FPS), [board])

    def test_score_changes_are_never_dropped(self):
        a, b = goal(100, "scoreboard"), goal(200, "scoreboard")
        self.assertEqual(drop_replayed_goals([a, b], FPS), [a, b])

    def test_other_events_pass_through(self):
        shot = {"event": "shot", "frame": 120}
        self.assertEqual(drop_replayed_goals([goal(40), shot, goal(140)], FPS)[1], shot)


class SaveWithdrawalTests(unittest.TestCase):
    def test_the_save_of_a_shot_that_went_in_is_withdrawn(self):
        # Regression (FC clip 2026-10-07): the ball passed the keeper into the
        # net and was reported as "shot (saved)" plus a save.
        shot = {"event": "shot", "frame": 149, "details": {"outcome": "goal"}}
        save = {"event": "save", "frame": 149}
        self.assertEqual(withdraw_saves_of_goals([shot, save]), [shot])

    def test_a_real_save_stays(self):
        shot = {"event": "shot", "frame": 132, "details": {"outcome": "saved"}}
        save = {"event": "save", "frame": 132}
        self.assertEqual(withdraw_saves_of_goals([shot, save]), [shot, save])


if __name__ == "__main__":
    unittest.main()
