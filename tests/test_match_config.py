import unittest

from match_config import MatchConfig, parse_hex_color
from team_assigner import TeamAssigner
from team_assigner import cluster_team_samples
import numpy as np


class MatchConfigTests(unittest.TestCase):
    def test_detected_colors_do_not_require_hex_input(self):
        config = MatchConfig.from_detected_colors(
            "North FC", (220, 30, 40), "South FC", (20, 60, 210)
        )

        self.assertEqual(config.team_a.name, "North FC")
        self.assertEqual(config.team_a.primary_color_rgb, (220, 30, 40))

    def test_player_samples_produce_two_rgb_team_centers(self):
        crop = np.zeros((80, 40, 3), dtype=np.uint8)
        centers, representatives = cluster_team_samples(
            [
                (np.array([10, 20, 230]), crop),
                (np.array([12, 18, 225]), crop),
                (np.array([220, 40, 20]), crop),
                (np.array([225, 38, 22]), crop),
            ]
        )

        self.assertEqual(len(centers), 2)
        self.assertEqual(len(representatives), 2)
        self.assertTrue(any(center[0] > 200 for center in centers))
        self.assertTrue(any(center[2] > 200 for center in centers))

    def test_hex_color_is_parsed_as_rgb(self):
        self.assertEqual(parse_hex_color("#12A0ff"), (18, 160, 255))

    def test_invalid_hex_color_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_hex_color("blue")

    def test_player_color_maps_to_configured_team(self):
        config = MatchConfig.from_user_input("Reds", "#FF0000", "Blues", "#0000FF")
        assigner = TeamAssigner(config)

        assignment = assigner.match_color_to_team((4, 3, 248))

        self.assertEqual(assignment["team_id"], 1)
        self.assertEqual(assignment["team_name"], "Reds")
        self.assertGreater(assignment["confidence"], 0.7)

    def test_ambiguous_team_color_is_unknown(self):
        config = MatchConfig.from_user_input("Red One", "#FF0000", "Red Two", "#F50000")
        assigner = TeamAssigner(config, color_ambiguity_margin=10.0)

        assignment = assigner.match_color_to_team((0, 0, 250))

        self.assertEqual(assignment["team_id"], 0)
        self.assertEqual(assignment["team_name"], "unknown_team")

    def test_distant_color_is_unknown(self):
        config = MatchConfig.from_user_input("Reds", "#FF0000", "Blues", "#0000FF")
        assigner = TeamAssigner(config, color_distance_threshold=25.0)

        assignment = assigner.match_color_to_team((128, 128, 128))

        self.assertEqual(assignment["team_id"], 0)


if __name__ == "__main__":
    unittest.main()
