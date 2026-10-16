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

    def test_a_kick_off_between_two_goals_makes_the_second_a_new_goal(self):
        first, second = goal(100), goal(100 + int(30 * FPS))
        kick_off = {"event": "kick_off", "frame": 100 + int(20 * FPS), "source": "spotting"}
        kept = drop_replayed_goals([first, kick_off, second], FPS)
        self.assertEqual([e for e in kept if e["event"] == "goal"], [first, second])

    def test_a_kick_off_does_not_split_a_goal_from_its_score_change(self):
        board = goal(300, "scoreboard")
        kick_off = {"event": "kick_off", "frame": 350, "source": "spotting"}
        replay = goal(400)
        kept = drop_replayed_goals([board, kick_off, replay], FPS)
        self.assertEqual([e for e in kept if e["event"] == "goal"], [board])

    def test_a_learned_model_goal_never_drops_a_video_goal(self):
        # Regression (CLI path): an LSTM "goal" before the real one made the
        # real one look like its replay.
        model, real = goal(40, "model"), goal(140)
        self.assertEqual(drop_replayed_goals([model, real], FPS), [real])

    def test_an_earlier_spotted_goal_does_not_drop_a_tracked_goal(self):
        # Review finding: a spotted goal placed on its shot, before the tracked
        # goal of the same move, made the tracked goal look like its replay.
        spotted, tracked = goal(100, "spotting"), goal(100 + int(8 * FPS))
        self.assertEqual(drop_replayed_goals([spotted, tracked], FPS), [tracked])

    def test_other_events_pass_through(self):
        shot = {"event": "shot", "frame": 120}
        self.assertEqual(drop_replayed_goals([goal(40), shot, goal(140)], FPS)[1], shot)

    def test_goals_more_than_twenty_seconds_apart_are_both_kept(self):
        # Levante-Sevilla highlights: goals 29-43 s apart were dropped by the
        # old 60 s window.
        first, second = goal(100), goal(100 + int(22 * FPS))
        self.assertEqual(drop_replayed_goals([first, second], FPS), [first, second])


def named(frame, name, source="caption", confidence=0.9, team_id=1, unconfirmed=False):
    event = goal(frame)
    event["team_id"] = team_id
    event["scorer"] = {"player_name": name, "jersey_number": 7, "team_id": team_id, "unconfirmed": unconfirmed,
                       "scorer_confidence": confidence, "scorer_sources": [source]}
    event["actors"] = [{"role": "scorer", **event["scorer"]}]
    return event


class NamedScorerReplayTests(unittest.TestCase):
    def test_different_named_scorers_close_together_are_two_goals(self):
        # Levante-Sevilla: Isaac Romero and Ndukwe scored 16 s apart in the package.
        first, second = named(100, "Isaac Romero"), named(100 + int(16 * FPS), "Ifeanyi Ndukwe")
        self.assertEqual(drop_replayed_goals([first, second], FPS), [first, second])

    def test_the_same_named_scorer_close_together_is_a_replay(self):
        live, replay = named(100, "Iván Romero"), named(100 + int(8 * FPS), "Ivan Romero", source="commentary")
        self.assertEqual(drop_replayed_goals([live, replay], FPS), [live])

    def test_a_replay_gives_its_name_to_the_unnamed_live_goal(self):
        live, replay = goal(100), named(100 + int(8 * FPS), "Ifeanyi Ndukwe")
        self.assertEqual(drop_replayed_goals([live, replay], FPS), [live])
        self.assertEqual(live["scorer"]["player_name"], "Ifeanyi Ndukwe")
        self.assertEqual([a["player_name"] for a in live["actors"]], ["Ifeanyi Ndukwe"])
        self.assertEqual(live["details"]["scorer_from_replay_frame"], 100 + int(8 * FPS))

    def test_a_name_from_a_shirt_number_alone_does_not_separate_goals(self):
        live, replay = named(100, "Isaac Romero"), named(100 + int(8 * FPS), "Juan Iglesias", source="live")
        self.assertEqual(drop_replayed_goals([live, replay], FPS), [live])

    def test_an_unconfirmed_or_weak_name_does_not_separate_goals(self):
        live = named(100, "Isaac Romero")
        weak = named(100 + int(8 * FPS), "Juan Iglesias", source="commentary", confidence=0.5)
        unconfirmed = named(100 + int(10 * FPS), "Juan Iglesias", unconfirmed=True)
        self.assertEqual(drop_replayed_goals([live, weak, unconfirmed], FPS), [live])

    def test_a_differently_named_goal_near_a_score_change_is_kept(self):
        board = named(300, "Iván Romero")
        board["source"] = "scoreboard"
        other = named(300 + int(10 * FPS), "Adrián de la Fuente")
        self.assertEqual(drop_replayed_goals([board, other], FPS), [board, other])

    def test_kick_off_frames_can_be_passed_in(self):
        # main.py removes kick-off events before the scorers are named.
        first, second = goal(100), goal(100 + int(15 * FPS))
        kept = drop_replayed_goals([first, second], FPS, kick_off_frames=[100 + int(10 * FPS)])
        self.assertEqual(kept, [first, second])


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
