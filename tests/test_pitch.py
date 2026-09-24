"""Pitch calibration gate, pitch zones, and the pitch-aware event rules.

The trained pitch model does not yet produce an acceptable homography on the
reference clips, so these tests build one synthetically: project the 32 layout
points through a known perspective transform, then check the gate recovers it
and the event rules use metres correctly.
"""

import unittest

import cv2
import numpy as np

from pitch import PitchCalibrator, fit_calibration, layout
from track_events import TrackEventDetector
from trackers.tracker import choose_ball_image_size


FRAME_SHAPE = (1080, 1920)
FPS = 25.0
HEIGHT = 100.0


def broadcast_view():
    """Pitch-to-image transform looking at the right half, far side narrower."""
    pitch_pts = np.float32([(60, 0), (120, 0), (120, 70), (60, 70)])
    image_pts = np.float32([(300, 200), (1650, 200), (1900, 1050), (10, 1050)])
    return cv2.getPerspectiveTransform(pitch_pts, image_pts)


def project(transform, points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, transform).reshape(-1, 2)


class CalibrationGateTests(unittest.TestCase):
    def setUp(self):
        self.view = broadcast_view()
        self.xy = project(self.view, layout.VERTICES)
        self.conf = np.full(len(self.xy), 0.9, dtype=np.float32)

    def test_consistent_keypoints_recover_pitch_metres(self):
        calibration, reason = fit_calibration(0, self.xy, self.conf, FRAME_SHAPE)
        self.assertEqual(reason, "ok")
        self.assertGreaterEqual(calibration.inliers, 6)
        spot_image = project(self.view, [layout.penalty_spot("right")])[0]
        x, y = calibration.to_pitch(spot_image)
        self.assertAlmostEqual(x, 109.0, delta=0.05)
        self.assertAlmostEqual(y, 35.0, delta=0.05)

    def test_keypoints_with_scrambled_labels_are_rejected(self):
        # What the current model does: plausible positions, inconsistent labels.
        rng = np.random.default_rng(7)
        scrambled = self.xy.copy()
        visible = np.where(
            (self.xy[:, 0] > 2) & (self.xy[:, 1] > 2)
            & (self.xy[:, 0] < FRAME_SHAPE[1] - 2) & (self.xy[:, 1] < FRAME_SHAPE[0] - 2)
        )[0]
        scrambled[visible] = self.xy[rng.permutation(visible)]
        calibration, reason = fit_calibration(0, scrambled, self.conf, FRAME_SHAPE)
        self.assertIsNone(calibration)
        self.assertNotEqual(reason, "ok")

    def test_four_exact_points_are_not_enough(self):
        # Four points always fit a homography exactly; that proves nothing.
        conf = np.zeros(len(self.xy), dtype=np.float32)
        conf[[13, 16, 24, 29]] = 0.9
        calibration, reason = fit_calibration(0, self.xy, conf, FRAME_SHAPE)
        self.assertIsNone(calibration)
        self.assertEqual(reason, "too_few_keypoints")

    def test_low_confidence_keypoints_are_ignored(self):
        # The trained model's keypoint confidences topped out at 0.35.
        calibration, reason = fit_calibration(0, self.xy, np.full(len(self.xy), 0.35), FRAME_SHAPE)
        self.assertIsNone(calibration)
        self.assertEqual(reason, "too_few_keypoints")

    def test_calibration_is_held_between_keyframes_then_expires(self):
        calibrator = PitchCalibrator(model=None, stride=5)
        self.assertIsNotNone(calibrator.update_from_keypoints(0, self.xy, self.conf, FRAME_SHAPE))
        self.assertIsNotNone(calibrator.valid_for(15))
        self.assertIsNone(calibrator.valid_for(16))
        self.assertEqual(calibrator.summary()["accepted_keyframes"], 1)


class LayoutTests(unittest.TestCase):
    def test_zones(self):
        self.assertEqual(layout.in_penalty_box((110, 35)), "right")
        self.assertEqual(layout.in_penalty_box((10, 20)), "left")
        self.assertIsNone(layout.in_penalty_box((60, 35)))
        self.assertTrue(layout.in_wide_channel((100, 5)))
        self.assertFalse(layout.in_wide_channel((100, 35)))
        self.assertEqual(layout.attacking_third((100, 5)), "right")

    def test_in_goal_needs_the_ball_between_the_posts_and_inside_the_net(self):
        self.assertEqual(layout.in_goal((121, 35)), "right")
        self.assertEqual(layout.in_goal((-1, 33)), "left")
        self.assertIsNone(layout.in_goal((121, 40)))   # wide of the post
        self.assertIsNone(layout.in_goal((124, 35)))   # deeper than the net
        self.assertIsNone(layout.in_goal((119, 35)))   # still on the pitch


class BallImageSizeTests(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(choose_ball_image_size(1920, 1080), 1280)
        self.assertEqual(choose_ball_image_size(1280, 720), 1280)
        self.assertEqual(choose_ball_image_size(854, 480), 960)
        self.assertEqual(choose_ball_image_size(320, 240), 640)


class TenPixelsPerMetre:
    """A stand-in calibration: image pixel / 10 = pitch metre."""

    def to_pitch(self, point):
        return (float(point[0]) / 10.0, float(point[1]) / 10.0)


def player_at(metres, team=1, role="player"):
    x, y = metres[0] * 10.0, metres[1] * 10.0
    return {
        "bbox": [x - 20.0, y - HEIGHT, x + 20.0, y],
        "team": team,
        "team_name": f"Team {team}",
        "team_confidence": 0.9,
        "role": role,
    }


def ball_px(x, y):
    return [x - 6.0, y - 6.0, x + 6.0, y + 6.0]


def run_with_pitch(frames, pitch=None):
    detector = TrackEventDetector(FPS, detect_cards=False)
    pitch = pitch or TenPixelsPerMetre()
    events = []
    for index, (players, ball) in enumerate(frames):
        events.extend(detector.update(index, players, ball, pitch=pitch))
    events.extend(detector.finish(len(frames)))
    return events


def pass_between(origin_m, destination_m, hold=10, flight=12):
    """Holder at origin, receiver at destination, the ball travels between."""
    ox, oy = origin_m[0] * 10.0 + 10.0, origin_m[1] * 10.0 - 5.0
    dx, dy = destination_m[0] * 10.0 - 10.0, destination_m[1] * 10.0 - 5.0
    frames = []

    def scene():
        return {1: player_at(origin_m), 2: player_at(destination_m)}

    for _ in range(hold):
        frames.append((scene(), ball_px(ox, oy)))
    for step in range(1, flight + 1):
        t = step / float(flight + 1)
        frames.append((scene(), ball_px(ox + (dx - ox) * t, oy + (dy - oy) * t)))
    for _ in range(hold):
        frames.append((scene(), ball_px(dx, dy)))
    return frames


class PitchAwareEventTests(unittest.TestCase):
    def test_ball_from_the_wing_into_the_box_is_a_cross(self):
        events = run_with_pitch(pass_between((100, 5), (110, 35)))
        crosses = [e for e in events if e["event"] == "cross"]
        self.assertEqual(len(crosses), 1)
        self.assertTrue(crosses[0]["details"]["pitch"]["is_cross"])
        self.assertNotIn("pass", [e["event"] for e in events])

    def test_central_ball_into_the_box_is_a_pass(self):
        events = run_with_pitch(pass_between((95, 30), (110, 35)))
        self.assertEqual([e["event"] for e in events], ["pass"])
        self.assertFalse(events[0]["details"]["pitch"]["is_cross"])

    def test_ball_behind_the_line_between_the_posts_is_a_goal(self):
        frames = []

        def scene():
            return {1: player_at((112, 35))}

        for _ in range(10):
            frames.append((scene(), ball_px(1130, 345)))
        for x in (1150, 1170, 1190, 1205, 1212, 1215, 1216):
            frames.append((scene(), ball_px(x, 345)))
        events = run_with_pitch(frames)
        goals = [e for e in events if e["event"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["source"], "pitch")
        self.assertEqual(goals[0]["details"]["goal_side"], "right")
        self.assertEqual(goals[0]["participants"][0]["role"], "scorer")
        self.assertEqual(goals[0]["participants"][0]["track_id"], 1)
        self.assertEqual(goals[0]["team_id"], 1)

    def test_ball_behind_the_line_wide_of_the_post_is_not_a_goal(self):
        frames = []

        def scene():
            return {1: player_at((112, 45))}

        for _ in range(10):
            frames.append((scene(), ball_px(1130, 445)))
        for x in (1150, 1170, 1190, 1205, 1212, 1215, 1216):
            frames.append((scene(), ball_px(x, 445)))
        events = run_with_pitch(frames)
        self.assertNotIn("goal", [e["event"] for e in events])

    def _penalty_frames(self, ball_m, keeper=None):
        ball_x, ball_y = ball_m[0] * 10.0, ball_m[1] * 10.0
        taker = player_at((ball_m[0] - 6.0, ball_m[1]))  # 60 px = 0.6 heights
        # Everyone else waits outside the box, over three heights away.
        others = {20 + i: player_at((ball_m[0] - 45.0 + 3 * i, 60.0), team=1 + i % 2) for i in range(6)}

        def scene():
            players = {1: taker, **others}
            if keeper is not None:
                players[9] = keeper
            return players

        frames = [(scene(), ball_px(ball_x, ball_y)) for _ in range(45)]
        for step in range(1, 8):
            frames.append((scene(), ball_px(ball_x + 50.0 * step, ball_y)))
        return frames

    def test_still_ball_on_the_spot_is_a_penalty_without_a_visible_keeper(self):
        events = run_with_pitch(self._penalty_frames((109, 35)))
        penalties = [e for e in events if e["event"] == "penalty"]
        self.assertEqual(len(penalties), 1)
        self.assertEqual(penalties[0]["details"]["penalty_spot"], "right")
        self.assertEqual(penalties[0]["participants"][0]["role"], "taker")

    def test_still_ball_off_the_spot_is_not_a_penalty_even_with_a_keeper_in_place(self):
        # The keeper-distance rule alone would call this a penalty; the pitch
        # position says it is a free kick.
        keeper = player_at((135, 35), team=2, role="goalkeeper")  # 4 heights away
        events = run_with_pitch(self._penalty_frames((95, 35), keeper=keeper))
        self.assertNotIn("penalty", [e["event"] for e in events])

    def test_shot_at_goal_is_recognised_without_a_visible_keeper(self):
        frames = []

        def scene():
            return {1: player_at((100, 35))}

        for _ in range(10):
            frames.append((scene(), ball_px(1010, 345)))
        for x in (1040, 1080, 1120, 1160):
            frames.append((scene(), ball_px(x, 346)))
        # The ball leaves the frame; the possession times out after 3 s.
        for _ in range(90):
            frames.append((scene(), None))
        events = run_with_pitch(frames)
        shots = [e for e in events if e["event"] == "shot"]
        self.assertEqual(len(shots), 1)
        self.assertLessEqual(shots[0]["details"]["angle_to_goal_degrees"], 25.0)
        self.assertEqual(shots[0]["participants"][0]["role"], "shooter")


if __name__ == "__main__":
    unittest.main()
