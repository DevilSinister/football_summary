import unittest

import numpy as np

from trackers.tracker import (
    BallSelector,
    _is_referee_name,
    choose_image_size,
    is_on_pitch,
)


class ClassNameTests(unittest.TestCase):
    def test_misspelled_referee_class_from_the_trained_weights_is_recognised(self):
        # models/best.pt names the class "refree"; the old lookup for "referee"
        # never matched and dropped every referee.
        self.assertTrue(_is_referee_name("refree"))
        self.assertTrue(_is_referee_name("referee"))
        self.assertFalse(_is_referee_name("player"))


class ImageSizeTests(unittest.TestCase):
    def test_sizes_scale_with_source_resolution(self):
        self.assertEqual(choose_image_size(1920, 1080), 1280)
        self.assertEqual(choose_image_size(1280, 720), 1280)
        self.assertEqual(choose_image_size(854, 480), 960)
        self.assertEqual(choose_image_size(320, 240), 640)


class OnPitchTests(unittest.TestCase):
    def test_feet_on_turf(self):
        frame = np.full((400, 600, 3), (50, 150, 40), dtype=np.uint8)
        self.assertTrue(is_on_pitch(frame, [280, 100, 320, 200]))

    def test_feet_on_grey_concrete(self):
        frame = np.full((400, 600, 3), (120, 120, 120), dtype=np.uint8)
        self.assertFalse(is_on_pitch(frame, [280, 100, 320, 200]))

    def test_box_cut_off_by_frame_edge_is_kept(self):
        frame = np.full((400, 600, 3), (120, 120, 120), dtype=np.uint8)
        self.assertTrue(is_on_pitch(frame, [280, 300, 320, 399]))


class BallSelectorTests(unittest.TestCase):
    def test_rejects_candidates_far_larger_than_a_ball(self):
        selector = BallSelector()
        # Player height 100 -> a 70 px "ball" is a boot or a head.
        self.assertIsNone(selector.select([([0, 0, 70, 70], 0.9)], 100.0))
        self.assertEqual(selector.select([([0, 0, 12, 12], 0.4)], 100.0), [0, 0, 12, 12])

    def test_prefers_the_candidate_near_the_last_ball_over_a_confident_far_one(self):
        selector = BallSelector()
        selector.select([([100, 100, 112, 112], 0.6)], 100.0)
        selector.select([([110, 100, 122, 112], 0.6)], 100.0)
        near = [122, 100, 134, 112]
        far = [900, 700, 912, 712]  # a white mark in the crowd
        self.assertEqual(selector.select([(far, 0.9), (near, 0.3)], 100.0), near)

    def test_far_candidate_is_dropped_while_the_trail_is_warm(self):
        selector = BallSelector()
        selector.select([([100, 100, 112, 112], 0.6)], 100.0)
        self.assertIsNone(selector.select([([900, 700, 912, 712], 0.9)], 100.0))

    def test_any_plausible_candidate_is_accepted_once_the_trail_is_cold(self):
        selector = BallSelector(history_frames=3)
        selector.select([([100, 100, 112, 112], 0.6)], 100.0)
        for _ in range(4):
            selector.select([], 100.0)
        self.assertEqual(
            selector.select([([900, 700, 912, 712], 0.9)], 100.0), [900, 700, 912, 712]
        )


if __name__ == "__main__":
    unittest.main()
