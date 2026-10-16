"""Commentary names: matching to squads, roles around goals and fouls, attachment to tracks."""

import unittest

import numpy as np

from commentary import Word, attach_to_tracks, find_mentions, foul_roles, goal_roles, ruled_out
from identity import IdentityTracker

FPS = 25.0
SQUADS = {
    1: [
        {"jersey_no": 1, "name": "Thibaut Courtois", "position": "GK"},
        {"jersey_no": 5, "name": "Jude Bellingham", "position": "MF"},
        {"jersey_no": 7, "name": "Vinícius Júnior", "position": "FW"},
        {"jersey_no": 8, "name": "Federico Valverde", "position": "MF"},
        {"jersey_no": 10, "name": "Kylian Mbappé", "position": "FW"},
    ],
    2: [
        {"jersey_no": 2, "name": "Ben White", "position": "DF"},
        {"jersey_no": 4, "name": "Virgil van Dijk", "position": "DF"},
        {"jersey_no": 16, "name": "Rodri", "position": "MF"},
        {"jersey_no": 3, "name": "Bernardo Silva", "position": "MF"},
        {"jersey_no": 20, "name": "Thiago Silva", "position": "DF"},
    ],
}


def say(text, start, step=0.35):
    return [Word(start + i * step, start + i * step + 0.3, w) for i, w in enumerate(text.split())]


def numbers(mentions):
    return [[(c.team_id, c.jersey_no) for c in m.candidates] for m in mentions]


class MentionTests(unittest.TestCase):
    def test_names_misspellings_particles_and_single_names(self):
        words = say("Vinisius goes past van Dijk and lays it to Rodri", 0.0)
        self.assertEqual(numbers(find_mentions(words, SQUADS)), [[(1, 7)], [(2, 4)], [(2, 16)]])

    def test_a_surname_that_is_a_word_needs_the_first_name(self):
        self.assertEqual(find_mentions(say("in the white shirt", 0.0), SQUADS), [])
        self.assertEqual(numbers(find_mentions(say("Ben White clears", 0.0), SQUADS)), [[(2, 2)]])

    def test_a_shared_surname_is_only_named_with_its_first_name(self):
        self.assertEqual(find_mentions(say("Silva again", 0.0), SQUADS), [])
        self.assertEqual(numbers(find_mentions(say("Bernardo Silva again", 0.0), SQUADS)), [[(2, 3)]])

    def test_names_spelt_as_heard_and_split_names(self):
        # Real transcript, Levante 3-3 Sevilla (2026-10-16): Whisper wrote
        # "Suasso" for Suazo, "Brugge" for Brugué, "Isac Romero", "for Fana".
        squads = {
            1: [{"jersey_no": 7, "name": "Roger Brugué", "position": "MF"},
                {"jersey_no": 9, "name": "Iván Romero", "position": "FW"}],
            2: [{"jersey_no": 17, "name": "Gabriel Suazo", "position": "DF"},
                {"jersey_no": 16, "name": "Isaac Romero", "position": "FW"},
                {"jersey_no": 24, "name": "Youssouf Fofana", "position": "MF"}],
        }
        words = say("It's Suasso then Brugge and Isac Romero He is for Fana", 0.0)
        self.assertEqual(numbers(find_mentions(words, squads)), [[(2, 17)], [(1, 7)], [(2, 16)], [(2, 24)]])

    def test_two_ordinary_words_do_not_join_into_a_name(self):
        squads = {1: [{"jersey_no": 24, "name": "Youssouf Fofana", "position": "MF"}]}
        self.assertEqual(find_mentions(say("go for it and pass", 0.0), squads), [])

    def test_ruled_out(self):
        words = say("Flicked on and it's in the back of the net", 12.0) + say(
            "although it has been ruled out the linesman has his flag up", 17.6)
        self.assertEqual(ruled_out(words, 10.0, 40.0), "ruled out")
        self.assertIsNone(ruled_out(say("what a goal", 12.0), 10.0, 40.0))


class GoalRoleTests(unittest.TestCase):
    def test_cross_from_the_assister_and_the_scorer_scores(self):
        words = say("a cross from Bellingham and Vinícius scores", 10.0)
        roles = goal_roles(12.0, 1, find_mentions(words, SQUADS), words)
        self.assertEqual(roles["scorer"]["jersey_number"], 7)
        self.assertEqual(roles["assister"]["jersey_number"], 5)
        self.assertEqual(roles["scorer"]["source"], "commentary")

    def test_named_in_turn_the_last_name_before_the_goal_scores(self):
        words = say("Valverde", 8.0) + say("Mbappé", 10.5) + say("goal! Mbappé! Kylian Mbappé", 12.5)
        roles = goal_roles(11.0, 1, find_mentions(words, SQUADS), words)
        self.assertEqual(roles["scorer"]["jersey_number"], 10)
        self.assertEqual(roles["assister"]["jersey_number"], 8)

    def test_the_other_teams_players_are_not_the_scorer(self):
        words = say("Rodri tries to block but Vinícius scores", 10.0)
        roles = goal_roles(12.0, 1, find_mentions(words, SQUADS), words)
        self.assertEqual(roles["scorer"]["jersey_number"], 7)
        self.assertIsNone(roles["assister"])

    def test_no_names_no_roles(self):
        self.assertEqual(goal_roles(12.0, 1, [], []), {"scorer": None, "assister": None, "ruled_out": None})


class FoulRoleTests(unittest.TestCase):
    def test_foul_by_and_foul_on(self):
        words = say("that's a foul by Rodri on Bellingham", 30.0)
        roles = foul_roles(30.0, find_mentions(words, SQUADS))
        self.assertEqual((roles["fouler"]["team_id"], roles["fouler"]["jersey_number"]), (2, 16))
        self.assertEqual((roles["fouled"]["team_id"], roles["fouled"]["jersey_number"]), (1, 5))

    def test_brings_him_down(self):
        roles = foul_roles(30.0, find_mentions(say("van Dijk brings him down", 30.5), SQUADS))
        self.assertEqual(roles["fouler"]["jersey_number"], 4)


class AttachTests(unittest.TestCase):
    def test_a_spoken_name_goes_to_the_player_on_the_ball(self):
        ids = IdentityTracker(SQUADS, {}, embedder=None, fps=FPS)
        frame = np.zeros((10, 10, 3), np.uint8)
        players = {
            11: {"team": 1, "team_confidence": 0.9, "role": "player", "bbox": [0, 0, 5, 9]},
            12: {"team": 1, "team_confidence": 0.9, "role": "player", "bbox": [0, 0, 5, 9]},
        }
        ids.observe(0, frame, players)
        # Track 11 has the ball from 9.0 s to 10.0 s, track 12 from 12.0 s.
        possession = [(int(t * FPS), 11, 1) for t in np.arange(9.0, 10.0, 0.1)]
        possession += [(int(t * FPS), 12, 1) for t in np.arange(12.0, 13.0, 0.1)]
        words = say("Mbappé", 10.2) + say("Mbappé again", 10.9) + say("Bellingham", 12.6)
        attached = attach_to_tracks(find_mentions(words, SQUADS), possession, FPS, ids)
        self.assertEqual([(a.track_id, a.candidate.jersey_no) for a in attached], [(11, 10), (11, 10), (12, 5)])
        self.assertEqual(ids.name_for(11, concurrent=[12])["player_name"], "Kylian Mbappé")

    def test_a_name_of_the_other_team_is_not_attached_to_the_holder(self):
        ids = IdentityTracker(SQUADS, {}, embedder=None, fps=FPS)
        possession = [(int(10.0 * FPS), 11, 1)]
        self.assertEqual(attach_to_tracks(find_mentions(say("Rodri", 10.3), SQUADS), possession, FPS, ids), [])


class OtherSquadGoalRoleTests(unittest.TestCase):
    SQUADS = {
        1: [{"jersey_no": 9, "name": "Iván Romero", "position": "FW"}],
        2: [{"jersey_no": 16, "name": "Isaac Romero", "position": "FW"},
            {"jersey_no": 1, "name": "Odysseas Vlachodimos", "position": "GK"}],
    }

    def test_a_clear_name_from_the_other_squad_can_be_the_scorer(self):
        # The kit said Sevilla (2); the commentator said "Iván Romero scores".
        words = say("and Iván Romero scores", 10.0)
        roles = goal_roles(11.0, 2, find_mentions(words, self.SQUADS), words)
        self.assertEqual((roles["scorer"]["team_id"], roles["scorer"]["jersey_number"]), (1, 9))

    def test_not_when_the_score_graphic_decided_the_team(self):
        words = say("and Iván Romero scores", 10.0)
        roles = goal_roles(11.0, 2, find_mentions(words, self.SQUADS), words, other_team=False)
        self.assertIsNone(roles["scorer"])

    def test_a_shared_surname_stays_with_the_goals_team(self):
        words = say("and Romero scores", 10.0)
        roles = goal_roles(11.0, 2, find_mentions(words, self.SQUADS), words)
        self.assertEqual((roles["scorer"]["team_id"], roles["scorer"]["jersey_number"]), (2, 16))

    def test_the_other_keeper_beaten_is_not_the_scorer(self):
        words = say("it goes past Vlachodimos", 10.0)
        roles = goal_roles(11.0, 1, find_mentions(words, self.SQUADS), words)
        self.assertIsNone(roles["scorer"])


class SharedSurnameRoleTests(unittest.TestCase):
    SQUADS = OtherSquadGoalRoleTests.SQUADS

    def test_a_scorer_named_only_by_a_shared_surname_lists_the_others(self):
        words = say("and Romero scores", 10.0)
        roles = goal_roles(11.0, 2, find_mentions(words, self.SQUADS), words)
        self.assertEqual(roles["scorer"]["shared_with"], [(1, 9)])

    def test_a_full_name_is_not_shared(self):
        words = say("Isaac Romero scores", 10.0)
        roles = goal_roles(11.0, 2, find_mentions(words, self.SQUADS), words)
        self.assertNotIn("shared_with", roles["scorer"])


if __name__ == "__main__":
    unittest.main()
