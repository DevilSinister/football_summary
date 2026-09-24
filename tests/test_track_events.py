import unittest

import numpy as np

from track_events import CardDetector, TrackEventDetector, summarise_passes
from track_events.cards import card_colour_in_region


FPS = 25.0
HEIGHT = 100.0  # player box height in px; every threshold is relative to this


def player(x, y=500.0, team=1, role="player", team_name=None):
    return {
        "bbox": [x - 20.0, y - HEIGHT, x + 20.0, y],
        "team": team,
        "team_name": team_name or (f"Team {team}" if team else "unknown_team"),
        "team_confidence": 0.9 if team else 0.0,
        "role": role,
    }


def ball_at(x, y):
    return [x - 6.0, y - 6.0, x + 6.0, y + 6.0]


def run(detector, frames):
    """frames: iterable of (players, ball_bbox). Returns all emitted events."""
    events = []
    for index, (players, ball) in enumerate(frames):
        events.extend(detector.update(index, players, ball))
    events.extend(detector.finish(len(frames)))
    return events


def possession_then_flight(holder_x, receiver_x, hold_frames=10, flight_frames=10, arc=0.0, y=500.0, extra=None):
    """Ball sits at the holder's feet, travels to the receiver, then sits at theirs."""
    extra = extra or {}
    frames = []
    for _ in range(hold_frames):
        players = {1: player(holder_x, y, 1), 2: player(receiver_x, y, 1), **extra}
        frames.append((players, ball_at(holder_x + 10, y - 5)))
    for step in range(1, flight_frames + 1):
        t = step / float(flight_frames + 1)
        bx = holder_x + 10 + (receiver_x - 10 - holder_x - 10) * t
        by = y - 5 - arc * np.sin(np.pi * t)
        players = {1: player(holder_x, y, 1), 2: player(receiver_x, y, 1), **extra}
        frames.append((players, ball_at(bx, by)))
    for _ in range(hold_frames):
        players = {1: player(holder_x, y, 1), 2: player(receiver_x, y, 1), **extra}
        frames.append((players, ball_at(receiver_x - 10, y - 5)))
    return frames


class PassDetectionTests(unittest.TestCase):
    def test_ground_pass_between_team_mates(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        events = run(detector, possession_then_flight(100.0, 400.0))
        passes = [e for e in events if e["event"] == "pass"]
        self.assertEqual(len(passes), 1)
        roles = {p["role"]: p["track_id"] for p in passes[0]["participants"]}
        self.assertEqual(roles, {"passer": 1, "receiver": 2})
        self.assertEqual(passes[0]["team_id"], 1)
        self.assertFalse(passes[0]["details"]["aerial"])

    def test_no_pass_between_opponents(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        frames = []
        for players, ball in possession_then_flight(100.0, 400.0):
            players[2]["team"] = 2
            players[2]["team_name"] = "Team 2"
            frames.append((players, ball))
        events = run(detector, frames)
        self.assertEqual([e["event"] for e in events], [])

    def test_track_id_switch_on_a_still_ball_is_not_a_pass(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        frames = []
        for index in range(30):
            holder = 1 if index < 15 else 7
            players = {holder: player(100.0, 500.0, 1)}
            frames.append((players, ball_at(110.0, 495.0)))
        events = run(detector, frames)
        self.assertEqual([e["event"] for e in events], [])

    def test_pass_count_summary(self):
        events = [
            {"event": "pass", "team_name": "A"},
            {"event": "cross", "team_name": "A"},
            {"event": "pass", "team_name": "B"},
            {"event": "shot", "team_name": "B"},
        ]
        self.assertEqual(
            summarise_passes(events),
            {"A": {"passes": 2, "crosses": 1}, "B": {"passes": 1, "crosses": 0}},
        )


class CrossDetectionTests(unittest.TestCase):
    def test_long_aerial_ball_landing_by_the_keeper_is_a_cross(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        keeper = {9: player(950.0, 500.0, 2, role="goalkeeper")}
        frames = possession_then_flight(
            100.0, 800.0, flight_frames=20, arc=1.2 * HEIGHT, extra=keeper
        )
        events = run(detector, frames)
        types = [e["event"] for e in events]
        self.assertIn("cross", types)
        self.assertNotIn("pass", types)
        cross = next(e for e in events if e["event"] == "cross")
        self.assertTrue(cross["details"]["aerial"])

    def test_short_ground_pass_near_keeper_is_not_a_cross(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        keeper = {9: player(600.0, 500.0, 2, role="goalkeeper")}
        events = run(detector, possession_then_flight(300.0, 500.0, extra=keeper))
        self.assertEqual([e["event"] for e in events], ["pass"])


class ShotAndSaveTests(unittest.TestCase):
    def test_fast_ball_gathered_by_opposing_keeper_is_a_saved_shot(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        shooter_x, keeper_x = 200.0, 900.0

        def scene():
            # The keeper's kit reads as the shooter's team: keeper kits match
            # neither strip, so the colour vote is noise. The save must still be
            # credited to the opposing side.
            return {1: player(shooter_x, 500.0, 1), 9: player(keeper_x, 500.0, 1, role="goalkeeper")}

        frames = []
        for _ in range(10):
            frames.append((scene(), ball_at(shooter_x + 10, 495.0)))
        # 670 px in 5 frames is 6.7 heights in 0.2 s: about 33 heights/s.
        for step in range(1, 6):
            bx = shooter_x + 10 + (keeper_x - 20 - shooter_x - 10) * step / 5.0
            frames.append((scene(), ball_at(bx, 480.0)))
        for _ in range(6):
            frames.append((scene(), ball_at(keeper_x - 15, 495.0)))
        events = run(detector, frames)
        types = [e["event"] for e in events]
        self.assertIn("shot", types)
        self.assertIn("save", types)
        save = next(e for e in events if e["event"] == "save")
        self.assertEqual(save["participants"][0]["role"], "goalkeeper")
        self.assertEqual(save["participants"][0]["track_id"], 9)
        self.assertEqual(save["participants"][0]["team_id"], 2)
        self.assertEqual(save["team_id"], 2)
        shot = next(e for e in events if e["event"] == "shot")
        self.assertEqual(shot["team_id"], 1)

    def test_slow_back_pass_to_own_keeper_is_a_pass_not_a_save(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        frames = []
        for players, ball in possession_then_flight(300.0, 500.0, flight_frames=20):
            players[2]["role"] = "goalkeeper"
            frames.append((players, ball))
        events = run(detector, frames)
        self.assertEqual([e["event"] for e in events], ["pass"])
        self.assertTrue(events[0]["details"].get("to_goalkeeper"))


class PenaltyDetectionTests(unittest.TestCase):
    def test_still_ball_lone_taker_and_distant_keeper_then_kick(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        ball_x, ball_y = 500.0, 500.0
        keeper = player(900.0, 500.0, 2, role="goalkeeper")  # 4 heights away
        taker = player(440.0, 500.0, 1)  # 0.6 heights away
        # Team-mates and opponents waiting on the edge of the box, well over
        # three heights from the ball.
        others = {index: player(100.0 + 30 * index, 150.0, 1 if index % 2 else 2) for index in range(20, 26)}

        def scene():
            return {1: taker, 9: keeper, **others}

        frames = []
        for _ in range(45):  # 1.8 s stationary
            frames.append((scene(), ball_at(ball_x, ball_y)))
        for step in range(1, 8):  # struck towards the keeper
            frames.append((scene(), ball_at(ball_x + 50.0 * step, ball_y - 5 * step)))
        events = run(detector, frames)
        penalties = [e for e in events if e["event"] == "penalty"]
        self.assertEqual(len(penalties), 1)
        self.assertEqual(penalties[0]["participants"][0]["role"], "taker")
        self.assertEqual(penalties[0]["participants"][0]["track_id"], 1)

    def test_still_ball_with_players_crowding_it_is_not_a_penalty(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        keeper = player(900.0, 500.0, 2, role="goalkeeper")
        crowd = {index: player(460.0 + 25 * index, 500.0, 1) for index in range(1, 5)}
        frames = []
        for _ in range(45):
            frames.append(({9: keeper, **crowd}, ball_at(500.0, 500.0)))
        for step in range(1, 8):
            frames.append(({9: keeper, **crowd}, ball_at(500.0 + 50.0 * step, 500.0)))
        events = run(detector, frames)
        self.assertNotIn("penalty", [e["event"] for e in events])


class CardDetectionTests(unittest.TestCase):
    @staticmethod
    def _frame_with_card(colour_bgr, card_above_head=True):
        frame = np.full((400, 400, 3), (60, 140, 60), dtype=np.uint8)  # turf
        # Referee: 40 px wide, 120 px tall, standing at x=200. The raised arm
        # makes the box start above the head.
        x1, y1, x2, y2 = 180, 200, 220, 320
        frame[y1 + 30:y2, x1:x2] = (30, 30, 30)  # black kit from the shoulders down
        if card_above_head:
            frame[y1 + 2:y1 + 12, 205:213] = colour_bgr  # 8x10 card in the raised hand
        else:
            frame[y1 + 40:y1 + 70, x1:x2] = colour_bgr  # coloured torso, not a card
        return frame, [x1, y1, x2, y2]

    def test_yellow_rectangle_above_the_head_reads_as_yellow(self):
        frame, bbox = self._frame_with_card((0, 220, 255))
        x1, y1, x2, y2 = bbox
        region = frame[y1 - 18 : y1 + 19, x1 - 14 : x2 + 14]
        self.assertEqual(card_colour_in_region(region, (x2 - x1) * (y2 - y1)), "yellow")

    def test_sustained_yellow_card_is_emitted_with_nearest_player(self):
        detector = CardDetector(FPS)
        frame, bbox = self._frame_with_card((0, 220, 255))
        referees = {50: {"bbox": bbox}}
        players = {3: player(260.0, 320.0, 2), 4: player(20.0, 320.0, 1)}
        events = []
        for index in range(12):
            events.extend(detector.update(index, frame, referees, players, HEIGHT))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "yellow_card")
        self.assertEqual(events[0]["participants"][0]["role"], "player")
        self.assertEqual(events[0]["participants"][0]["track_id"], 3)
        self.assertEqual(events[0]["team_id"], 2)
        # Cooldown: the same card is not reported twice.
        for index in range(12, 24):
            events.extend(detector.update(index, frame, referees, players, HEIGHT))
        self.assertEqual(len(events), 1)

    def test_red_card(self):
        detector = CardDetector(FPS)
        frame, bbox = self._frame_with_card((20, 20, 230))
        events = []
        for index in range(12):
            events.extend(detector.update(index, frame, {50: {"bbox": bbox}}, {}, HEIGHT))
        self.assertEqual([e["event"] for e in events], ["red_card"])

    def test_coloured_kit_on_the_torso_is_not_a_card(self):
        detector = CardDetector(FPS)
        frame, bbox = self._frame_with_card((0, 220, 255), card_above_head=False)
        events = []
        for index in range(12):
            events.extend(detector.update(index, frame, {50: {"bbox": bbox}}, {}, HEIGHT))
        self.assertEqual(events, [])

    def test_tiny_referee_box_is_skipped(self):
        # A 35 px tall box (a steward in the far crowd) cannot show a legible card.
        detector = CardDetector(FPS)
        frame, _ = self._frame_with_card((0, 220, 255))
        events = []
        for index in range(12):
            events.extend(detector.update(index, frame, {50: {"bbox": [195, 200, 207, 235]}}, {}, HEIGHT))
        self.assertEqual(events, [])

    def test_one_frame_flash_is_ignored(self):
        detector = CardDetector(FPS)
        frame, bbox = self._frame_with_card((0, 220, 255))
        plain = np.full_like(frame, 60)
        events = detector.update(0, frame, {50: {"bbox": bbox}}, {}, HEIGHT)
        for index in range(1, 12):
            events.extend(detector.update(index, plain, {50: {"bbox": bbox}}, {}, HEIGHT))
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
