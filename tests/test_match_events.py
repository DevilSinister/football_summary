"""Fouls, cards and set pieces from spotting + other evidence; score jumps; the evaluator."""

import unittest
from types import SimpleNamespace

from scoreboard.reader import ScoreTracker
from scripts.eval_events import same_player, score_clip
from spotting.match_events import (
    commentary_cues,
    fuse_cards,
    fuse_fouls,
    spotted_set_pieces,
    spotting_classes_for,
)

FPS = 25.0


def at(seconds):
    return int(seconds * FPS)


def track_foul(seconds, fouler_team=1):
    return {"event": "foul", "frame": at(seconds), "source": "tracks", "confidence": 0.6, "team_id": fouler_team,
            "participants": [{"role": "fouler", "track_id": 3, "team_id": fouler_team, "team_name": "Levante"},
                             {"role": "fouled", "track_id": 8, "team_id": 2, "team_name": "Sevilla"}]}


def spot(kind, seconds, confidence):
    return {"event": kind, "frame": at(seconds), "confidence": confidence, "details": {"spotting_label": kind}}


def words(*pairs):
    return [SimpleNamespace(start=float(t), text=text) for t, text in pairs]


class FoulFusionTests(unittest.TestCase):
    def fouls(self, events):
        return [e for e in events if e["event"] == "foul"]

    def test_contact_alone_is_not_enough_with_the_double_check(self):
        self.assertEqual(self.fouls(fuse_fouls([track_foul(10)], [], FPS)), [])

    def test_contact_alone_is_kept_without_the_double_check(self):
        self.assertEqual(len(self.fouls(fuse_fouls([track_foul(10)], [], FPS, double_check=False))), 1)

    def test_contact_and_the_commentary_make_a_foul_with_its_players(self):
        kept = self.fouls(fuse_fouls([track_foul(10)], [], FPS, words((12.0, "Foul"), (12.5, "there."))))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["participants"][0]["role"], "fouler")
        self.assertEqual(kept[0]["details"]["evidence"]["commentary"], ["foul"])

    def test_contact_and_a_spotted_foul_nearby_are_one_foul(self):
        kept = self.fouls(fuse_fouls([track_foul(10)], [spot("foul", 11.5, 0.2)], FPS))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["source"], "tracks")
        self.assertEqual(set(kept[0]["details"]["evidence"]), {"tracks", "spotting_foul"})

    def test_a_strong_spotted_foul_stands_alone_without_players(self):
        kept = self.fouls(fuse_fouls([], [spot("foul", 30, 0.7)], FPS))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["source"], "spotting")
        self.assertEqual(kept[0]["participants"], [])

    def test_a_weak_spotted_foul_needs_a_free_kick_after_it(self):
        self.assertEqual(self.fouls(fuse_fouls([], [spot("foul", 30, 0.2)], FPS)), [])
        kept = self.fouls(fuse_fouls([], [spot("foul", 30, 0.2), spot("direct_free_kick", 45, 0.4)], FPS))
        self.assertEqual(len(kept), 1)

    def test_an_indirect_free_kick_does_not_confirm_a_foul(self):
        kept = fuse_fouls([], [spot("foul", 30, 0.2), spot("indirect_free_kick", 40, 0.9)], FPS)
        self.assertEqual(self.fouls(kept), [])

    def test_other_events_pass_through(self):
        goal = {"event": "goal", "frame": at(5)}
        self.assertIn(goal, fuse_fouls([goal, track_foul(10)], [], FPS))


class CardFusionTests(unittest.TestCase):
    def test_a_strong_spotted_card_stands_alone(self):
        cards = [e for e in fuse_cards([], [spot("yellow_card", 50, 0.7)], FPS) if e["event"] == "yellow_card"]
        self.assertEqual(len(cards), 1)

    def test_a_weak_card_needs_commentary_or_a_foul(self):
        self.assertEqual(fuse_cards([], [spot("yellow_card", 50, 0.3)], FPS), [])
        with_words = fuse_cards([], [spot("yellow_card", 50, 0.3)], FPS, words((52, "booked")))
        self.assertEqual([e["event"] for e in with_words], ["yellow_card"])

    def test_a_card_after_a_foul_takes_the_fouler(self):
        foul = track_foul(40)
        out = fuse_cards([foul], [spot("yellow_card", 50, 0.3)], FPS)
        card = next(e for e in out if e["event"] == "yellow_card")
        self.assertEqual(card["participants"][0]["role"], "player")
        self.assertEqual(card["team_id"], 1)

    def test_second_yellow_is_a_red_card(self):
        out = fuse_cards([], [spot("yellow_red_card", 50, 0.8)], FPS)
        self.assertEqual([e["event"] for e in out], ["red_card"])

    def test_a_track_card_needs_a_second_source(self):
        track_card = {"event": "yellow_card", "frame": at(60), "participants": []}
        self.assertEqual(fuse_cards([track_card], [], FPS), [])
        self.assertEqual(len(fuse_cards([dict(track_card)], [], FPS, double_check=False)), 1)


class SetPieceTests(unittest.TestCase):
    def test_only_switched_on_kinds_above_threshold_and_duplicates_merged(self):
        spotted = [spot("corner", 10, 0.6), spot("corner", 13, 0.5), spot("offside", 20, 0.9),
                   spot("indirect_free_kick", 30, 0.2), spot("direct_free_kick", 40, 0.5)]
        out = spotted_set_pieces(spotted, FPS, ["corner", "free_kick"])
        self.assertEqual([(e["event"], e["frame"]) for e in out], [("corner", at(10)), ("free_kick", at(40))])

    def test_classes_follow_the_switches(self):
        self.assertEqual(spotting_classes_for({}), [])
        self.assertIn("Corner", spotting_classes_for({"corner": True}))
        self.assertIn("Foul", spotting_classes_for({"fouls": True}))

    def test_free_kick_said_as_two_words_is_one_cue(self):
        self.assertEqual(commentary_cues(words((1, "free"), (1.2, "kick")), 0, 5, {"freekick"}), ["freekick"])


class ScoreJumpTests(unittest.TestCase):
    def tracker(self, one_each):
        tracker = ScoreTracker(confirm_reads=1, accept_one_each=one_each)
        tracker.codes = ("LEV", "SEV")
        tracker.state = {"LEV": 0, "SEV": 1}
        tracker.last_state_frame = 100
        return tracker

    def test_one_each_jump_is_two_goals_when_switched_on(self):
        tracker = self.tracker(True)
        goals = tracker._accept({"LEV": 1, "SEV": 2}, 500, 500)
        self.assertEqual(sorted(goal.code for goal in goals), ["LEV", "SEV"])

    def test_one_each_jump_is_rejected_when_switched_off(self):
        tracker = self.tracker(False)
        self.assertEqual(tracker._accept({"LEV": 1, "SEV": 2}, 500, 500), [])
        self.assertIn("implausible", tracker.rejected[-1][1])


class OneEachPlacementTests(unittest.TestCase):
    def test_the_second_goal_of_a_jump_does_not_reuse_the_same_attack(self):
        from scoreboard.attribution import attribute_goals
        from scoreboard.reader import ScoreGoal

        before, after = {"LEV": 0, "SEV": 1}, {"LEV": 1, "SEV": 2}
        goals = [ScoreGoal("LEV", at(96), at(30), before, after), ScoreGoal("SEV", at(96), at(30), before, after)]
        tracked_goal = {"event": "goal", "frame": at(94.4), "source": "tracks", "team_id": 2, "participants": []}
        same_attack_shot = {"event": "shot", "frame": at(93.8), "source": "tracks", "team_id": 2, "participants": []}
        earlier_shot = {"event": "shot", "frame": at(57.5), "source": "spotting", "team_id": 0, "participants": []}
        produced, _ = attribute_goals(
            goals, ("LEV", "SEV"), {"LEV": 1, "SEV": 2}, {1: "Levante", 2: "Sevilla"},
            [same_attack_shot, tracked_goal, earlier_shot], FPS,
        )
        by_team = {goal["team_name"]: goal["frame"] for goal in produced}
        self.assertEqual(by_team["Levante"], at(94.4))
        self.assertEqual(by_team["Sevilla"], at(57.5))


class EvaluatorTests(unittest.TestCase):
    def test_same_player(self):
        self.assertTrue(same_player("Iván Romero", "Ivan Romero"))
        self.assertTrue(same_player("Romero", "Isaac Romero"))
        self.assertFalse(same_player("Isaac Romero", "Iván Romero"))
        self.assertFalse(same_player(None, "Ure"))

    def test_score_clip_counts_found_missed_and_false(self):
        clip = {"id": "x", "complete": ["goal"], "events": [
            {"type": "goal", "t": 10.0, "team": "A", "player": "Robbie Ure"},
            {"type": "goal", "t": 60.0, "team": "B"},
        ]}
        result = {"events": [
            {"type": "goal", "t": 12.0, "team": "A", "scorer": {"player_name": "Robbie Ure"}, "actors": []},
            {"type": "goal", "t": 100.0, "team": "B", "scorer": {}, "actors": []},
            {"type": "foul", "t": 20.0, "team": "A", "scorer": {}, "actors": []},
        ]}
        stats = score_clip(clip, result)["goal"]
        self.assertEqual((stats["labels"], stats["predicted"], stats["found"]), (2, 2, 1))
        self.assertEqual((stats["team_right"], stats["player_right"]), (1, 1))
        self.assertEqual(len([d for d in stats["details"] if d.startswith("FALSE")]), 1)


if __name__ == "__main__":
    unittest.main()
