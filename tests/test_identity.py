"""Few-shot player identity: gallery, per-track probabilities, cuts, exclusivity."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from identity import IdentityTracker, PlayerGallery, TeamPrototypes

DIM = 512


def unit(index, noise=0.0, seed=0):
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = 1.0
    if noise:
        vector += np.random.default_rng(seed).normal(0, noise, DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


class ColourEmbedder:
    """Stand-in for OSNet: the dominant colour channel picks the direction."""

    name = "fake"

    def embed(self, crops):
        return np.stack([unit(int(np.argmax(c.reshape(-1, 3).mean(axis=0)))) for c in crops])


def prototypes(jerseys, directions, names=None):
    # No enrolled names: the roster names are used.
    return TeamPrototypes(
        "Team", list(jerseys), list(names or [None] * len(jerseys)),
        np.stack([unit(d) for d in directions]), [1] * len(jerseys),
    )


SQUAD = [
    {"jersey_no": 1, "name": "Keeper", "position": "GK"},
    {"jersey_no": 7, "name": "Seven", "position": "FW"},
    {"jersey_no": 10, "name": "Ten", "position": "MF"},
]
FRAME = np.zeros((200, 200, 3), np.uint8)


def tracker(gallery=None):
    return IdentityTracker({1: SQUAD}, {1: gallery}, embedder=None, fps=25.0)


def player(team=1, confidence=0.9, role="player", bbox=(0, 0, 40, 120)):
    return {"team": team, "team_confidence": confidence, "role": role, "bbox": list(bbox)}


class PosteriorTests(unittest.TestCase):
    def test_one_reference_image_names_a_matching_track(self):
        ids = tracker(prototypes([7, 10], [0, 1]))
        ids.observe(0, FRAME, {5: player()})
        for _ in range(3):
            ids.add_appearance(5, unit(0, noise=0.02))
        named = ids.name_for(5)
        self.assertEqual(named["jersey_number"], 7)
        self.assertEqual(named["player_name"], "Seven")
        self.assertGreaterEqual(named["probability"], 0.7)

    def test_a_stranger_is_not_named(self):
        ids = tracker(prototypes([7, 10], [0, 1]))
        ids.observe(0, FRAME, {5: player()})
        for _ in range(3):
            ids.add_appearance(5, unit(300))
        self.assertIsNone(ids.name_for(5))

    def test_a_shirt_number_read_names_a_player_without_any_image(self):
        ids = tracker()
        ids.observe(0, FRAME, {5: player()})
        ids.add_number_read(5, 10, 0.9)
        ids.add_number_read(5, 10, 0.8)
        self.assertEqual(ids.name_for(5)["jersey_number"], 10)

    def test_the_same_number_read_counts_once(self):
        # Regression (review): a track in a shot, a goal and a save had its one
        # OCR read applied three times.
        ids = tracker()
        ids.observe(0, FRAME, {5: player()})
        ids.add_number_read(5, 10, 0.5)
        once = ids.tracks[5].probabilities().copy()
        ids.add_number_read(5, 10, 0.5)
        ids.add_number_read(5, 10, 0.5)
        np.testing.assert_allclose(ids.tracks[5].probabilities(), once)

    def test_two_players_on_screen_cannot_share_a_name(self):
        ids = tracker(prototypes([7, 10], [0, 1]))
        ids.observe(0, FRAME, {5: player(), 6: player()})
        for _ in range(4):
            ids.add_appearance(5, unit(0, noise=0.02, seed=1))
        ids.add_appearance(6, unit(0, noise=0.2, seed=2))
        answer = ids.resolve([5, 6])
        self.assertEqual(ids.candidates[1][answer[5][0]].jersey_no, 7)
        self.assertNotEqual(answer[6][0], answer[5][0])

    def test_a_track_after_a_cut_inherits_the_matching_earlier_track(self):
        # No reference images: appearance only links tracklets, it names nobody.
        ids = tracker()
        ids.observe(0, FRAME, {5: player(), 6: player()})
        ids.add_appearance(5, unit(2))
        ids.add_appearance(6, unit(3))
        ids.add_number_read(5, 7, 0.9)      # the shirt was read before the cut
        ids.end_scene()
        ids.observe(100, FRAME, {40: player()})
        ids.add_appearance(40, unit(2, noise=0.02))   # the same look after the cut (cos ~0.9)
        self.assertEqual(ids.tracks[40].linked_from, 5)
        self.assertEqual(ids.name_for(40)["jersey_number"], 7)

    def test_a_different_looking_track_after_a_cut_inherits_nothing(self):
        ids = tracker()
        ids.observe(0, FRAME, {5: player()})
        ids.add_appearance(5, unit(2))
        ids.add_number_read(5, 7, 0.9)
        ids.end_scene()
        ids.observe(100, FRAME, {40: player()})
        ids.add_appearance(40, unit(9))
        self.assertIsNone(ids.tracks[40].linked_from)
        self.assertIsNone(ids.name_for(40))

    def test_a_goalkeeper_track_prefers_the_keeper(self):
        ids = tracker(prototypes([7, 10], [0, 1]))
        ids.observe(0, FRAME, {5: player(role="goalkeeper")})
        ids.add_appearance(5, unit(300))
        probs = ids.tracks[5].probabilities()
        keeper = next(i for i, c in enumerate(ids.candidates[1]) if c.jersey_no == 1)
        self.assertEqual(int(np.argmax(probs[:-1])), keeper)

    def test_an_unconfident_team_reading_is_not_tracked(self):
        ids = tracker()
        ids.observe(0, FRAME, {5: player(confidence=0.2)})
        self.assertNotIn(5, ids.tracks)

    def test_positions_are_recorded_at_the_observation_stride(self):
        ids = tracker()
        for number in range(0, 50):
            ids.observe(number, FRAME, {5: player(bbox=(10, 20, 30, 100))})
        positions = ids.tracks[5].positions
        self.assertEqual(len(positions), 4)  # every 0.5 s at 25 fps over 2 s
        self.assertEqual(positions[0][1:3], (20.0, 100.0))


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.gallery = PlayerGallery(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def crop(channel):
        image = np.full((120, 50, 3), 30, np.uint8)
        image[..., channel] = 220
        return image

    def test_one_image_per_player_gives_prototypes(self):
        self.gallery.enroll("Real Madrid", 7, "Seven", [self.crop(2)])
        self.gallery.enroll("real madrid cf", 10, "Ten", [self.crop(1)])
        team = self.gallery.team_prototypes("Real Madrid", ColourEmbedder())
        self.assertEqual(team.jerseys, [7, 10])
        self.assertIsNone(team.classifier)
        self.assertGreater(team.member_probabilities(unit(2), 0.05)[0], 0.9)

    def test_three_images_each_add_a_classifier(self):
        for _ in range(3):
            self.gallery.enroll("Porto", 7, "Seven", [self.crop(2)])
            self.gallery.enroll("Porto", 10, "Ten", [self.crop(0)])
        team = self.gallery.team_prototypes("Porto", ColourEmbedder())
        self.assertIsNotNone(team.classifier)
        self.assertGreater(team.member_probabilities(unit(0), 0.05)[1], 0.5)

    def test_at_most_ten_images_per_player(self):
        self.gallery.enroll("Porto", 7, "Seven", [self.crop(2)] * 10)
        with self.assertRaises(ValueError):
            self.gallery.enroll("Porto", 7, "Seven", [self.crop(2)])

    def test_removing_the_last_image_removes_the_player(self):
        path = self.gallery.enroll("Porto", 7, "Seven", [self.crop(2)])[0]
        image_id = os.path.splitext(os.path.basename(path))[0]
        self.assertFalse(self.gallery.remove_image("Porto", 7, "../../etc"))
        self.assertTrue(self.gallery.remove_image("Porto", 7, image_id))
        self.assertEqual(self.gallery.members("Porto"), [])


if __name__ == "__main__":
    unittest.main()
