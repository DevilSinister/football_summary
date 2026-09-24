"""Foul detection from player and ball tracks.

Before this module the only foul source was the learned event model, and
EventLogic (event_logic.py) rejects nearly everything it proposes because
the model's "foul" is noise - on the reference clips it fired on the opening
frames. So in practice no foul was ever recorded.

A foul here is judged the way a viewer sees one:

1. contact - two opponents' feet within CONTACT_HEIGHTS of each other (or
   their boxes overlapping), with the ball close by;
2. then, within OUTCOME_SECONDS of the contact, at least one consequence:
   - a player goes down: one of the two players' box shrinks to under
     DOWN_HEIGHT_RATIO of his own standing height and of the other players'
     median, and turns wide, for DOWN_MIN_SECONDS - a player on the grass;
   - play stops: the ball rests within STOP_TOLERANCE_HEIGHTS for
     STOP_SECONDS, near where the contact happened - the free kick being set.
     A throw-in or goal kick stops play far from the duel, so it does not count.

Contact alone is never a foul: a match has hundreds of clean duels.

Who fouled whom: the player who went down was fouled; otherwise the player who
had the ball was. With neither known the roles are reported as uncertain.

A card also implies a foul. TrackEventDetector links a card to the latest
contact involving the booked player (``contact_for_card``), and records a foul
at the card itself if no contact was seen.

Everything is measured in player heights, like the rest of track_events.
These are heuristics: verified on synthetic tracks only (tests/test_fouls.py),
because neither reference clip contains a foul.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

CONTACT_HEIGHTS = 0.8          # foot-to-foot distance that counts as contact
CONTACT_MIN_IOU = 0.15         # or boxes overlapping this much
BALL_NEAR_HEIGHTS = 2.5        # the ball must be this close to the duel
BALL_RECENT_SECONDS = 0.5      # ...seen within this long
OUTCOME_SECONDS = 3.0          # how long after contact a consequence may start
DOWN_HEIGHT_RATIO = 0.6
DOWN_MIN_ASPECT = 0.9          # box width / height of a player on the ground
DOWN_MIN_SECONDS = 0.3
STOP_SECONDS = 2.0
STOP_TOLERANCE_HEIGHTS = 0.5
STOP_NEAR_CONTACT_HEIGHTS = 4.0
STANDING_HISTORY_SECONDS = 2.0
# Contacts are kept this long so a card shown later can be tied to its foul.
CONTACT_HISTORY_SECONDS = 60.0

CONFIDENCE_BOTH = 0.75
CONFIDENCE_DOWN = 0.6
CONFIDENCE_STOP = 0.5


def _foot(bbox) -> Tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3])


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _snapshot(player: dict) -> dict:
    return {
        "team_id": int(player.get("team", 0) or 0),
        "team_name": str(player.get("team_name", "unknown_team") or "unknown_team"),
        "team_confidence": float(player.get("team_confidence", 0.0) or 0.0),
    }


@dataclass
class Contact:
    frame: int
    last_frame: int
    a: int
    b: int
    players: Dict[int, dict]
    holder: Optional[int]
    point: Tuple[float, float]
    standing: Dict[int, float]
    down_run: Dict[int, int] = field(default_factory=dict)
    down: Optional[int] = None  # the track that went down
    stopped: bool = False
    resolved: bool = False
    emitted: bool = False

    @property
    def pair(self) -> frozenset:
        return frozenset((self.a, self.b))


class FoulDetector:
    def __init__(self, fps: float) -> None:
        self.fps = max(float(fps), 1.0)
        self._boxes: Dict[int, Deque[Tuple[int, float]]] = {}
        self._ball: Deque[Tuple[int, float, float]] = deque(maxlen=int(self.fps * 10))
        self._open: List[Contact] = []
        self.history: Deque[Contact] = deque()

    # ----------------------------------------------------------------- update
    def update(
        self,
        frame_number: int,
        players: Dict[int, dict],
        ball_centre: Optional[Tuple[float, float]],
        holder: int,
        scale: float,
    ) -> List[dict]:
        """Foul proposals resolved on this frame (see ``_proposal``)."""
        if ball_centre is not None:
            self._ball.append((frame_number, float(ball_centre[0]), float(ball_centre[1])))
        self._record_heights(frame_number, players)
        if scale <= 0:
            return []

        proposals: List[dict] = []
        for contact in self._open:
            self._watch_consequences(contact, frame_number, players, scale)
            if frame_number - contact.frame > (OUTCOME_SECONDS + STOP_SECONDS) * self.fps:
                proposals.extend(self._resolve(contact, frame_number))
        self._open = [c for c in self._open if not c.resolved]

        self._find_contacts(frame_number, players, holder, scale)
        self._trim_history(frame_number)
        return proposals

    def reset_scene(self, frame_number: int) -> List[dict]:
        """A camera cut: judge open contacts on what was seen, drop the rest.

        Broadcasters cut to a close-up soon after a foul, so a player seen
        going down before the cut still counts. History survives the cut so a
        card shown in the close-up can find its contact.
        """
        proposals: List[dict] = []
        for contact in self._open:
            proposals.extend(self._resolve(contact, frame_number))
        self._open = []
        self._boxes.clear()
        self._ball.clear()
        return proposals

    def finish(self, frame_number: int) -> List[dict]:
        return self.reset_scene(frame_number)

    def contact_for_card(self, card_frame: int, booked_track: Optional[int]) -> Optional[Contact]:
        """Latest unreported contact before a card, involving the booked player if known."""
        window = CONTACT_HISTORY_SECONDS * self.fps
        candidates = [
            c for c in self.history
            if 0 <= card_frame - c.frame <= window and not c.emitted
            and (booked_track is None or booked_track in (c.a, c.b))
        ]
        return max(candidates, key=lambda c: c.frame) if candidates else None

    def proposal_for_card(self, contact: Contact, booked_track: Optional[int], frame_number: int) -> dict:
        """The foul behind a card: the booked player is the fouler."""
        proposal = self._proposal(contact, frame_number)
        if booked_track in (contact.a, contact.b):
            fouled = contact.b if booked_track == contact.a else contact.a
            proposal["participants"] = [
                ("fouler", booked_track, contact.players[booked_track]),
                ("fouled", fouled, contact.players[fouled]),
            ]
            proposal["details"]["roles_certain"] = True
        proposal["confidence"] = max(proposal["confidence"], CONFIDENCE_BOTH)
        proposal["details"]["rule"] = "card shown after opponents' contact near the ball"
        return proposal

    # --------------------------------------------------------------- helpers
    def _record_heights(self, frame_number: int, players: Dict[int, dict]) -> None:
        keep = int(self.fps * (STANDING_HISTORY_SECONDS + OUTCOME_SECONDS + STOP_SECONDS))
        for track_id, player in players.items():
            bbox = player["bbox"]
            history = self._boxes.setdefault(int(track_id), deque(maxlen=max(keep, 8)))
            history.append((frame_number, float(bbox[3]) - float(bbox[1])))

    def _standing_height(self, track_id: int, before: int) -> float:
        start = before - int(STANDING_HISTORY_SECONDS * self.fps)
        heights = [h for f, h in self._boxes.get(track_id, ()) if start <= f <= before]
        return float(np.median(heights)) if heights else 0.0

    def _ball_near(self, frame_number: int, point: Tuple[float, float], scale: float) -> bool:
        recent = int(BALL_RECENT_SECONDS * self.fps)
        return any(
            frame_number - f <= recent and math.hypot(x - point[0], y - point[1]) <= BALL_NEAR_HEIGHTS * scale
            for f, x, y in self._ball
        )

    def _find_contacts(self, frame_number: int, players: Dict[int, dict], holder: int, scale: float) -> None:
        open_pairs = {c.pair: c for c in self._open}
        # Keepers are left out: their kit matches neither team, so the colour
        # reading that decides "opponents" is noise for them.
        outfield = [
            (int(track_id), player) for track_id, player in players.items()
            if player.get("role") != "goalkeeper" and int(player.get("team", 0) or 0) in (1, 2)
        ]
        for index, (id_a, player_a) in enumerate(outfield):
            for id_b, player_b in outfield[index + 1:]:
                if int(player_a.get("team", 0)) == int(player_b.get("team", 0)):
                    continue
                foot_a, foot_b = _foot(player_a["bbox"]), _foot(player_b["bbox"])
                touching = (
                    math.hypot(foot_a[0] - foot_b[0], foot_a[1] - foot_b[1]) <= CONTACT_HEIGHTS * scale
                    or _iou(player_a["bbox"], player_b["bbox"]) >= CONTACT_MIN_IOU
                )
                if not touching:
                    continue
                point = ((foot_a[0] + foot_b[0]) / 2.0, (foot_a[1] + foot_b[1]) / 2.0)
                if not self._ball_near(frame_number, point, scale):
                    continue
                pair = frozenset((id_a, id_b))
                existing = open_pairs.get(pair)
                if existing is not None:
                    existing.last_frame = frame_number
                    continue
                contact = Contact(
                    frame=frame_number,
                    last_frame=frame_number,
                    a=id_a,
                    b=id_b,
                    players={id_a: _snapshot(player_a), id_b: _snapshot(player_b)},
                    holder=holder if holder in (id_a, id_b) else None,
                    point=point,
                    standing={
                        id_a: self._standing_height(id_a, frame_number),
                        id_b: self._standing_height(id_b, frame_number),
                    },
                )
                self._open.append(contact)
                self.history.append(contact)
                open_pairs[pair] = contact

    def _watch_consequences(self, contact: Contact, frame_number: int, players: Dict[int, dict], scale: float) -> None:
        if contact.down is None and frame_number - contact.frame <= OUTCOME_SECONDS * self.fps:
            for track_id in (contact.a, contact.b):
                player = players.get(track_id)
                if player is None:
                    contact.down_run[track_id] = 0
                    continue
                x1, y1, x2, y2 = [float(v) for v in player["bbox"]]
                height, width = y2 - y1, x2 - x1
                standing = contact.standing.get(track_id) or scale
                # Compared with his own height just before the contact AND with
                # everyone else's now, so a camera zoom-out does not read as a fall.
                lying = (
                    height <= DOWN_HEIGHT_RATIO * standing
                    and height <= DOWN_HEIGHT_RATIO * scale
                    and width >= DOWN_MIN_ASPECT * height
                )
                contact.down_run[track_id] = contact.down_run.get(track_id, 0) + 1 if lying else 0
                if contact.down_run[track_id] >= max(1, int(DOWN_MIN_SECONDS * self.fps)):
                    contact.down = track_id
                    break
        if not contact.stopped:
            contact.stopped = self._ball_stopped(contact, frame_number, scale)

    def _ball_stopped(self, contact: Contact, frame_number: int, scale: float) -> bool:
        """The ball rested near the duel for STOP_SECONDS, starting inside the outcome window."""
        need = int(STOP_SECONDS * self.fps)
        tail = [s for s in self._ball if contact.frame < s[0] and s[0] >= frame_number - need]
        if len(tail) < 0.6 * need or frame_number - tail[0][0] < need - 1:
            return False
        if tail[0][0] > contact.frame + int(OUTCOME_SECONDS * self.fps):
            return False
        _, x0, y0 = tail[0]
        if any(math.hypot(x - x0, y - y0) > STOP_TOLERANCE_HEIGHTS * scale for _, x, y in tail):
            return False
        return math.hypot(x0 - contact.point[0], y0 - contact.point[1]) <= STOP_NEAR_CONTACT_HEIGHTS * scale

    def _resolve(self, contact: Contact, frame_number: int) -> List[dict]:
        contact.resolved = True
        if contact.down is None and not contact.stopped:
            return []
        return [self._proposal(contact, frame_number)]

    def _proposal(self, contact: Contact, frame_number: int) -> dict:
        """A foul: (role, track id, team snapshot) participants, fouler first."""
        certain = True
        if contact.down is not None:
            fouled = contact.down
        elif contact.holder is not None:
            fouled = contact.holder
        else:
            fouled, certain = contact.b, False
        fouler = contact.a if fouled == contact.b else contact.b
        if contact.down is not None and contact.stopped:
            confidence = CONFIDENCE_BOTH
        elif contact.down is not None:
            confidence = CONFIDENCE_DOWN
        else:
            confidence = CONFIDENCE_STOP
        contact.emitted = True
        seen = [
            part
            for part, flag in (("a player down", contact.down is not None), ("play stopped", contact.stopped))
            if flag
        ]
        return {
            "frame": contact.frame,
            "end_frame": max(contact.last_frame, frame_number),
            "confidence": confidence,
            "participants": [
                ("fouler", fouler, contact.players[fouler]),
                ("fouled", fouled, contact.players[fouled]),
            ],
            "details": {
                "rule": "opponents in contact near the ball, then " + " and ".join(seen or ["nothing"]),
                "player_down": contact.down is not None,
                "play_stopped": contact.stopped,
                "roles_certain": certain,
                "contact_seconds": round((contact.last_frame - contact.frame + 1) / self.fps, 2),
            },
        }

    def _trim_history(self, frame_number: int) -> None:
        window = CONTACT_HISTORY_SECONDS * self.fps
        while self.history and frame_number - self.history[0].frame > window:
            self.history.popleft()
