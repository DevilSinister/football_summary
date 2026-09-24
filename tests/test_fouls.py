"""Fouls from tracks: contact near the ball, then a player down or play stopped.

Synthetic tracks only - neither reference clip contains a foul.
"""

import unittest

from track_events import TrackEventDetector

FPS = 25.0
HEIGHT = 100.0


def player(x, y=500.0, team=1, lying=False):
    if lying:
        # On the grass: a wide, low box.
        box = [x - 45.0, y - 40.0, x + 45.0, y]
    else:
        box = [x - 20.0, y - HEIGHT, x + 20.0, y]
    return {"bbox": box, "team": team, "team_name": f"Team {team}", "team_confidence": 0.9, "role": "player"}


def ball_at(x, y=495.0):
    return [x - 6.0, y - 6.0, x + 6.0, y + 6.0]


def crowd():
    # Enough other players that the running median height stays a standing player.
    return {10 + i: player(900.0 + 60 * i, team=1 + i % 2) for i in range(6)}


def run(detector, frames, start=0):
    events = []
    for index, (players, ball) in enumerate(frames, start=start):
        events.extend(detector.update(index, players, ball))
    return events


def approach(frames=25):
    """Attacker (team 1, id 1) dribbles; defender (team 2, id 2) closes in."""
    out = []
    for step in range(frames):
        attacker_x = 300.0 + 2 * step
        defender_x = 450.0 - 4 * step
        out.append(({1: player(attacker_x), 2: player(defender_x, team=2), **crowd()}, ball_at(attacker_x + 12)))
    return out


def contact(frames=5, x=350.0):
    return [({1: player(x), 2: player(x + 30, team=2), **crowd()}, ball_at(x + 15)) for _ in range(frames)]


class FoulTests(unittest.TestCase):
    def fouls(self, events):
        return [e for e in events if e["event"] == "foul"]

    def test_player_down_after_contact_is_a_foul_on_him(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        frames = approach() + contact()
        # The attacker goes down beside the defender; the ball rolls on.
        for step in range(40):
            frames.append(
                ({1: player(350.0, lying=True), 2: player(420.0, team=2), **crowd()}, ball_at(380.0 + 6 * step))
            )
        events = run(detector, frames)
        events += detector.finish(len(frames))
        fouls = self.fouls(events)
        self.assertEqual(len(fouls), 1)
        roles = {p["role"]: p["track_id"] for p in fouls[0]["participants"]}
        self.assertEqual(roles, {"fouler": 2, "fouled": 1})
        self.assertEqual(fouls[0]["team_id"], 2)  # the fouling team
        self.assertTrue(fouls[0]["details"]["player_down"])

    def test_play_stopping_at_the_duel_is_a_foul(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        frames = approach() + contact()
        # Both stand; the ball is placed where the duel was and stays there.
        for _ in range(int(FPS * 4)):
            frames.append(({1: player(300.0), 2: player(460.0, team=2), **crowd()}, ball_at(365.0)))
        events = run(detector, frames)
        events += detector.finish(len(frames))
        fouls = self.fouls(events)
        self.assertEqual(len(fouls), 1)
        self.assertTrue(fouls[0]["details"]["play_stopped"])
        # The attacker had the ball, so he was the one fouled.
        roles = {p["role"]: p["track_id"] for p in fouls[0]["participants"]}
        self.assertEqual(roles["fouled"], 1)

    def test_a_clean_duel_is_not_a_foul(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        frames = approach() + contact()
        # Nobody falls and play runs on.
        for step in range(60):
            frames.append(({1: player(350.0 + 3 * step), 2: player(380.0 + 3 * step, team=2), **crowd()},
                           ball_at(400.0 + 8 * step)))
        events = run(detector, frames)
        events += detector.finish(len(frames))
        self.assertEqual(self.fouls(events), [])

    def test_team_mates_touching_is_not_a_foul(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        frames = [({1: player(350.0), 2: player(380.0, team=1), **crowd()}, ball_at(365.0)) for _ in range(10)]
        frames += [({1: player(350.0, lying=True), 2: player(420.0, team=1), **crowd()}, ball_at(365.0))
                   for _ in range(int(FPS * 4))]
        events = run(detector, frames)
        events += detector.finish(len(frames))
        self.assertEqual(self.fouls(events), [])

    def test_a_fall_seen_just_before_a_camera_cut_still_counts(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        frames = approach() + contact()
        frames += [({1: player(350.0, lying=True), 2: player(420.0, team=2), **crowd()}, ball_at(400.0))
                   for _ in range(10)]
        events = run(detector, frames)
        events += detector.reset_scene(len(frames))
        self.assertEqual(len(self.fouls(events)), 1)

    def test_a_card_after_a_contact_names_the_booked_player_as_fouler(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        frames = approach() + contact()
        # Contact, no visible consequence, then a cut: nothing recorded yet.
        events = run(detector, frames)
        events += detector.reset_scene(len(frames))
        self.assertEqual(self.fouls(events), [])
        card = {
            "event": "yellow_card",
            "frame": len(frames) + 200,
            "participants": [{"role": "player", "track_id": 2, "team_id": 2, "team_name": "Team 2"}],
        }
        fouls = detector._foul_for_card(card, card["frame"])
        self.assertEqual(len(fouls), 1)
        self.assertLess(fouls[0]["frame"], len(frames))  # at the contact, not at the card
        roles = {p["role"]: p["track_id"] for p in fouls[0]["participants"]}
        self.assertEqual(roles, {"fouler": 2, "fouled": 1})

    def test_a_card_with_no_contact_still_records_a_foul(self):
        detector = TrackEventDetector(FPS, detect_cards=False)
        card = {
            "event": "red_card",
            "frame": 500,
            "participants": [{"role": "player", "track_id": 7, "team_id": 1, "team_name": "Team 1"}],
        }
        fouls = detector._foul_for_card(card, 500)
        self.assertEqual(len(fouls), 1)
        self.assertEqual(fouls[0]["frame"], 500)
        self.assertEqual(fouls[0]["participants"][0]["role"], "fouler")
        # A second card for the same foul adds nothing.
        self.assertEqual(detector._foul_for_card({**card, "frame": 520}, 520), [])


if __name__ == "__main__":
    unittest.main()
