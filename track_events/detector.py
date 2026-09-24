"""Match events derived from player and ball tracks.

Everything here is measured in *player heights* rather than pixels, using a
running median of the tracked player boxes, so the same thresholds hold for a
480p phone clip and a 1080p broadcast. Positions are frame coordinates; the
camera pans, so only short-range comparisons (a second or two) are trusted.

Events and how they are judged:

* pass   - possession moves from one player to a team-mate, with the ball
           actually travelling between them.
* cross  - a pass that is long, lateral, airborne, and lands near the
           goalkeeper (the penalty area cannot be located without pitch
           calibration, so the keeper stands in for the goal).
* shot   - the ball leaves a player fast, heading at the goalkeeper.
* save   - a shot the goalkeeper gathers, or that reverses direction at the
           goalkeeper.
* penalty- a long-stationary ball, one player beside it, the goalkeeper a
           penalty-spot's distance away and everyone else standing off, followed
           by a kick towards the keeper.
* cards  - see track_events.cards.

These are rule-based approximations and are reported with a confidence and
``"source": "tracks"`` so the caller can decide what to store or show.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pitch import layout as pitch_layout
from player_ball_assigner import PlayerBallAssigner

from .cards import CardDetector


# Possession
CONFIRM_FRAMES = 2            # frames a new holder needs before a switch counts
MAX_FLIGHT_SECONDS = 2.5      # longest gap that can still be a pass
LOOSE_BALL_TIMEOUT_SECONDS = 3.0
MIN_PASS_DISTANCE = 0.8       # player heights the ball must travel
ID_SWITCH_MAX_GAP_FRAMES = 3

# Cross
CROSS_MIN_LATERAL = 3.5       # player heights of horizontal travel
CROSS_MIN_SECONDS = 0.45
CROSS_MIN_ARC = 0.4           # how far above the chord the ball rises
CROSS_MAX_KEEPER_DISTANCE = 4.5

# Shot / save
SHOT_MIN_SPEED = 6.0          # player heights per second at release
SHOT_MAX_ANGLE_DEGREES = 35.0
SHOT_KEEPER_REACH = 2.5
KEEPER_MEMORY_SECONDS = 2.0

# Penalty
PENALTY_STILL_SECONDS = 1.5
PENALTY_STILL_TOLERANCE = 0.15
PENALTY_KEEPER_DISTANCE = (2.5, 8.0)
PENALTY_TAKER_DISTANCE = 3.0
PENALTY_OTHERS_DISTANCE = 3.0
PENALTY_MIN_OTHERS = 4
PENALTY_KICK_SPEED = 4.0
PENALTY_KICK_TIMEOUT_SECONDS = 6.0

# Pitch-aware rules, used only on frames with an accepted homography (metres).
PITCH_CROSS_MIN_LATERAL_M = 10.0      # the ball has to come in from the side
PITCH_PENALTY_SPOT_TOLERANCE_M = 1.5
PITCH_SHOT_MAX_DISTANCE_M = 35.0
PITCH_SHOT_MAX_ANGLE_DEGREES = 25.0
PITCH_SHOT_MIN_TRAVEL_M = 2.0
GOAL_CONFIRM_SAMPLES = 2               # consecutive ball positions in the net
GOAL_CONFIRM_WINDOW_FRAMES = 4
GOAL_APPROACH_SECONDS = 2.0
GOAL_APPROACH_DISTANCE_M = 25.0

EVENT_COOLDOWN_SECONDS = {"shot": 1.0, "save": 1.0, "penalty": 10.0, "cross": 0.5, "goal": 30.0}


@dataclass
class Participant:
    role: str
    track_id: int
    team_id: int = 0
    team_name: str = "unknown_team"
    team_confidence: float = 0.0

    @classmethod
    def from_player(cls, role: str, track_id: int, player: dict) -> "Participant":
        return cls(
            role=role,
            track_id=int(track_id),
            team_id=int(player.get("team", 0) or 0),
            team_name=str(player.get("team_name", "unknown_team") or "unknown_team"),
            team_confidence=float(player.get("team_confidence", 0.0) or 0.0),
        )


@dataclass
class _Segment:
    track_id: int
    team_id: int
    team_name: str
    team_confidence: float
    role: str
    start_frame: int
    end_frame: int
    frames: int
    start_ball: Optional[Tuple[float, float]]
    end_ball: Optional[Tuple[float, float]]
    # Where the holder stood when the segment began and ended (foot point).
    start_player: Optional[Tuple[float, float]] = None
    end_player: Optional[Tuple[float, float]] = None
    # Ball position in pitch metres, when the frame had an accepted homography.
    start_ball_pitch: Optional[Tuple[float, float]] = None
    end_ball_pitch: Optional[Tuple[float, float]] = None

    def participant(self, role: str) -> Participant:
        return Participant(role, self.track_id, self.team_id, self.team_name, self.team_confidence)

    def absorb(self, player: dict) -> None:
        # Keep the most confident team reading seen for this holder.
        confidence = float(player.get("team_confidence", 0.0) or 0.0)
        team_id = int(player.get("team", 0) or 0)
        if team_id != 0 and (self.team_id == 0 or confidence >= self.team_confidence):
            self.team_id = team_id
            self.team_name = str(player.get("team_name", "unknown_team") or "unknown_team")
            self.team_confidence = confidence


@dataclass
class _Keeper:
    frame: int
    track_id: int
    x: float
    y: float
    team_id: int
    team_name: str


def _centre(bbox) -> Tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def _foot(bbox) -> Tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3])


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


class TrackEventDetector:
    def __init__(
        self,
        fps: float,
        frame_size: Optional[Tuple[int, int]] = None,
        ball_assigner: Optional[PlayerBallAssigner] = None,
        detect_cards: bool = True,
    ) -> None:
        self.fps = max(float(fps), 1.0)
        self.frame_size = frame_size
        self.ball_assigner = ball_assigner or PlayerBallAssigner()
        self.card_detector = CardDetector(self.fps) if detect_cards else None

        self._heights: Deque[float] = deque(maxlen=400)
        self._scale = 0.0
        self._ball: Deque[Tuple[int, float, float]] = deque(maxlen=int(self.fps * 10))
        self._last_ball_frame = -10**9
        self._keepers: Dict[int, _Keeper] = {}
        # Pitch calibration for the current frame (a PitchCalibration or any
        # object with to_pitch), and the ball trail in metres.
        self._pitch = None
        self._ball_pitch: Deque[Tuple[int, float, float]] = deque(maxlen=int(self.fps * 10))
        self._ball_pitch_now: Optional[Tuple[float, float]] = None
        self._last_segment: Optional[_Segment] = None

        self.current: Optional[_Segment] = None
        self._candidate: Optional[Tuple[int, int, int]] = None  # track, count, first frame
        self._frames_without_holder = 0

        self._still_anchor: Optional[Tuple[int, float, float]] = None
        self._penalty_setup: Optional[dict] = None

        self._last_event_frame: Dict[str, int] = {}
        self._team_names: Dict[int, str] = {}
        self.scene_resets = 0
        self.events: List[dict] = []
        self.pass_count: Dict[int, int] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def scale(self) -> float:
        return self._scale

    def _update_scale(self, players: Dict[int, dict]) -> None:
        for player in players.values():
            bbox = player["bbox"]
            height = float(bbox[3]) - float(bbox[1])
            if height > 4:
                self._heights.append(height)
            team_id = int(player.get("team", 0) or 0)
            if team_id and player.get("role") != "goalkeeper":
                self._team_names[team_id] = str(player.get("team_name") or f"team_{team_id}")
        if self._heights:
            self._scale = float(np.median(self._heights))

    def _opposing_keeper(self, keeper: Participant, shooter: Participant) -> Participant:
        """A keeper's kit matches neither strip, so its colour vote is noise.

        For a shot the keeper is, by definition, on the other side from the
        shooter; that deduction beats the colour reading.
        """
        if shooter.team_id in (1, 2):
            other = 2 if shooter.team_id == 1 else 1
            keeper.team_id = other
            keeper.team_name = self._team_names.get(other, f"team_{other}")
            keeper.team_confidence = max(keeper.team_confidence, shooter.team_confidence)
        return keeper

    def _seconds(self, frames: int) -> float:
        return frames / self.fps

    def _cooldown_ok(self, event_type: str, frame: int) -> bool:
        cooldown = int(EVENT_COOLDOWN_SECONDS.get(event_type, 0.0) * self.fps)
        last = self._last_event_frame.get(event_type)
        return last is None or frame - last >= cooldown

    def _emit(
        self,
        event_type: str,
        frame: int,
        end_frame: int,
        confidence: float,
        participants: Sequence[Participant],
        details: Optional[dict] = None,
    ) -> Optional[dict]:
        if not self._cooldown_ok(event_type, frame):
            return None
        team = next((p for p in participants if p.team_id != 0), None)
        event = {
            "event": event_type,
            "frame": int(frame),
            "end_frame": int(end_frame),
            "confidence": round(float(min(1.0, max(0.0, confidence))), 3),
            "team_id": team.team_id if team else 0,
            "team_name": team.team_name if team else "unknown_team",
            "source": "tracks",
            "participants": [asdict(p) for p in participants],
            "details": details or {},
        }
        self._last_event_frame[event_type] = int(frame)
        self.events.append(event)
        if event_type in ("pass", "cross"):
            self.pass_count[event["team_id"]] = self.pass_count.get(event["team_id"], 0) + 1
        return event

    def _ball_between(self, start: int, end: int) -> List[Tuple[int, float, float]]:
        return [sample for sample in self._ball if start <= sample[0] <= end]

    def _release_velocity(self, release_frame: int) -> Optional[Tuple[float, float, float]]:
        """(speed in heights/s, dx, dy) over the first frames after release."""
        if self._scale <= 0:
            return None
        samples = [s for s in self._ball if release_frame <= s[0] <= release_frame + 6]
        if len(samples) < 2:
            return None
        first, last = samples[0], samples[-1]
        frames = last[0] - first[0]
        if frames <= 0:
            return None
        dx, dy = last[1] - first[1], last[2] - first[2]
        speed = math.hypot(dx, dy) / self._scale / self._seconds(frames)
        return speed, dx, dy

    def _keeper_near(self, frame: int, point: Optional[Tuple[float, float]] = None) -> Optional[_Keeper]:
        memory = int(KEEPER_MEMORY_SECONDS * self.fps)
        candidates = [k for k in self._keepers.values() if 0 <= frame - k.frame <= memory or 0 <= k.frame - frame <= memory]
        if not candidates:
            return None
        if point is None:
            return max(candidates, key=lambda k: k.frame)
        return min(candidates, key=lambda k: _distance((k.x, k.y), point))

    @staticmethod
    def _angle_between(vx: float, vy: float, tx: float, ty: float) -> float:
        norm = math.hypot(vx, vy) * math.hypot(tx, ty)
        if norm <= 1e-9:
            return 180.0
        cosine = max(-1.0, min(1.0, (vx * tx + vy * ty) / norm))
        return math.degrees(math.acos(cosine))

    # ------------------------------------------------------------------ update
    def update(
        self,
        frame_number: int,
        players: Dict[int, dict],
        ball_bbox: Optional[Sequence[float]],
        referees: Optional[Dict[int, dict]] = None,
        frame: Optional[np.ndarray] = None,
        pitch=None,
    ) -> List[dict]:
        emitted: List[dict] = []
        self._update_scale(players)
        self._pitch = pitch
        self._ball_pitch_now = None

        for track_id, player in players.items():
            if player.get("role") == "goalkeeper":
                x, y = _foot(player["bbox"])
                self._keepers[int(track_id)] = _Keeper(
                    frame_number,
                    int(track_id),
                    x,
                    y,
                    int(player.get("team", 0) or 0),
                    str(player.get("team_name", "unknown_team") or "unknown_team"),
                )

        ball_centre = None
        holder = -1
        if ball_bbox is not None:
            ball_centre = _centre(ball_bbox)
            self._ball.append((frame_number, ball_centre[0], ball_centre[1]))
            self._last_ball_frame = frame_number
            if pitch is not None:
                mapped = pitch.to_pitch(ball_centre)
                if mapped is not None:
                    self._ball_pitch_now = (float(mapped[0]), float(mapped[1]))
                    self._ball_pitch.append((frame_number, self._ball_pitch_now[0], self._ball_pitch_now[1]))
            if players:
                holder = int(self.ball_assigner.assign_ball_to_player(players, ball_bbox))
                if holder >= 0:
                    players[holder]["has_ball"] = True

        emitted.extend(self._update_possession(frame_number, players, holder, ball_centre))
        emitted.extend(self._update_penalty(frame_number, players, ball_centre))
        emitted.extend(self._update_goal_line(frame_number))

        if self.card_detector is not None and frame is not None and referees:
            emitted.extend(
                self._collect(
                    self.card_detector.update(frame_number, frame, referees, players, self._scale)
                )
            )
        return emitted

    def _collect(self, events: List[dict]) -> List[dict]:
        for event in events:
            self._last_event_frame[event["event"]] = event["frame"]
            self.events.append(event)
        return events

    # -------------------------------------------------------------- possession
    def _update_possession(
        self,
        frame_number: int,
        players: Dict[int, dict],
        holder: int,
        ball_centre: Optional[Tuple[float, float]],
    ) -> List[dict]:
        emitted: List[dict] = []
        current = self.current

        if holder >= 0 and current is not None and holder == current.track_id:
            current.end_frame = frame_number
            current.frames += 1
            current.end_ball = ball_centre
            current.end_player = _foot(players[holder]["bbox"])
            if self._ball_pitch_now is not None:
                current.end_ball_pitch = self._ball_pitch_now
            current.absorb(players[holder])
            self._candidate = None
            self._frames_without_holder = 0
            return emitted

        if holder >= 0:
            if self._candidate is not None and self._candidate[0] == holder:
                track, count, first = self._candidate
                self._candidate = (track, count + 1, first)
            else:
                self._candidate = (holder, 1, frame_number)
            track, count, first = self._candidate
            if count >= CONFIRM_FRAMES:
                player = players[holder]
                start_ball = next((s for s in self._ball if s[0] == first), None)
                start_point = (start_ball[1], start_ball[2]) if start_ball else ball_centre
                new_segment = _Segment(
                    track_id=holder,
                    team_id=int(player.get("team", 0) or 0),
                    team_name=str(player.get("team_name", "unknown_team") or "unknown_team"),
                    team_confidence=float(player.get("team_confidence", 0.0) or 0.0),
                    role=str(player.get("role", "player")),
                    start_frame=first,
                    end_frame=frame_number,
                    frames=count,
                    start_ball=start_point,
                    end_ball=ball_centre,
                    start_player=_foot(player["bbox"]),
                    end_player=_foot(player["bbox"]),
                    start_ball_pitch=self._ball_pitch_at(first) or self._ball_pitch_now,
                    end_ball_pitch=self._ball_pitch_now,
                )
                if current is not None:
                    emitted.extend(self._on_transition(current, new_segment))
                    self._last_segment = current
                self.current = new_segment
                self._candidate = None
                self._frames_without_holder = 0
            return emitted

        # Nobody has the ball this frame.
        self._frames_without_holder += 1
        if current is not None and self._frames_without_holder > LOOSE_BALL_TIMEOUT_SECONDS * self.fps:
            emitted.extend(self._on_release_without_receiver(current, frame_number))
            self._last_segment = current
            self.current = None
            self._candidate = None
        return emitted

    def _ball_pitch_at(self, frame: int) -> Optional[Tuple[float, float]]:
        for sample in reversed(self._ball_pitch):
            if sample[0] == frame:
                return (sample[1], sample[2])
            if sample[0] < frame:
                break
        return None

    def _on_transition(self, prev: _Segment, new: _Segment) -> List[dict]:
        emitted: List[dict] = []
        if self._scale <= 0 or prev.end_ball is None or new.start_ball is None:
            return emitted
        gap_frames = new.start_frame - prev.end_frame
        ball_travel = _distance(prev.end_ball, new.start_ball) / self._scale
        # Possession is judged by reach, so on a slow pass the ball leaves the
        # passer's reach and enters the receiver's almost at once and the ball
        # itself barely "travels" between the two segments. The distance between
        # the two players is the honest measure of how far the pass went.
        if prev.end_player is not None and new.start_player is not None:
            displacement = max(ball_travel, _distance(prev.end_player, new.start_player) / self._scale)
        else:
            displacement = ball_travel

        # Same player under a new id, or a duel for a ball that never moved.
        if displacement < MIN_PASS_DISTANCE and gap_frames <= ID_SWITCH_MAX_GAP_FRAMES:
            return emitted

        release = self._release_velocity(prev.end_frame)
        speed = release[0] if release else 0.0
        same_team = prev.team_id != 0 and prev.team_id == new.team_id
        conflicting = prev.team_id != 0 and new.team_id != 0 and prev.team_id != new.team_id
        unknown = prev.team_id == 0 or new.team_id == 0
        if new.role == "goalkeeper" and same_team and speed >= SHOT_MIN_SPEED:
            # A fast ball into a keeper wearing "the same" colour: the colour is
            # the unreliable part. Only a slow ball is a genuine back-pass.
            same_team = False

        # Keeper gathers an opponent's fast ball: a shot that was saved.
        if new.role == "goalkeeper" and not same_team and speed >= SHOT_MIN_SPEED:
            shooter = prev.participant("shooter")
            keeper = self._opposing_keeper(new.participant("goalkeeper"), shooter)
            shot = self._emit(
                "shot",
                prev.end_frame,
                new.start_frame,
                min(0.9, 0.5 + speed / 40.0),
                [shooter, keeper],
                {"release_speed_heights_per_second": round(speed, 2), "outcome": "saved"},
            )
            if shot:
                emitted.append(shot)
            save = self._emit(
                "save",
                prev.end_frame,
                new.start_frame,
                min(0.9, 0.5 + speed / 40.0),
                [keeper, shooter],
                {"release_speed_heights_per_second": round(speed, 2)},
            )
            if save:
                emitted.append(save)
            return emitted

        if conflicting:
            # Interception or tackle. Not a requested event.
            return emitted

        if (
            (same_team or unknown)
            and displacement >= MIN_PASS_DISTANCE
            and gap_frames <= MAX_FLIGHT_SECONDS * self.fps
            and prev.track_id != new.track_id
        ):
            passer = prev.participant("passer")
            receiver = new.participant("receiver")
            if passer.team_id == 0 and receiver.team_id != 0:
                passer.team_id, passer.team_name = receiver.team_id, receiver.team_name
            elif receiver.team_id == 0 and passer.team_id != 0:
                receiver.team_id, receiver.team_name = passer.team_id, passer.team_name
            flight = self._ball_between(prev.end_frame, new.start_frame)
            details = {
                "distance_player_heights": round(displacement, 2),
                "ball_travel_player_heights": round(ball_travel, 2),
                "duration_seconds": round(self._seconds(gap_frames), 2),
                "release_speed_heights_per_second": round(speed, 2),
                "aerial": self._is_aerial(prev.end_ball, new.start_ball, flight),
            }
            confidence = 0.55 + (0.2 if same_team else 0.0) + min(0.2, prev.frames / 40.0)
            if new.role == "goalkeeper":
                details["to_goalkeeper"] = True
            event_type = "pass"
            pitch_cross = self._pitch_cross(prev, new)
            if pitch_cross is not None:
                # Real pitch zones decide when both ends were calibrated.
                details["pitch"] = pitch_cross
                if pitch_cross["is_cross"]:
                    event_type = "cross"
            else:
                keeper = self._keeper_near(new.start_frame, new.start_ball)
                if self._is_cross(prev, new, displacement, gap_frames, details["aerial"], keeper):
                    event_type = "cross"
                    details["keeper_distance_heights"] = round(
                        _distance((keeper.x, keeper.y), new.start_ball) / self._scale, 2
                    )
            event = self._emit(
                event_type, prev.end_frame, new.start_frame, confidence, [passer, receiver], details
            )
            if event:
                emitted.append(event)
        return emitted

    def _pitch_cross(self, prev: _Segment, new: _Segment) -> Optional[dict]:
        """Cross test in metres: from a wide channel in the attacking third into
        that end's penalty area, travelling at least 10 m across the pitch."""
        origin, destination = prev.end_ball_pitch, new.start_ball_pitch
        if origin is None or destination is None:
            return None
        box = pitch_layout.in_penalty_box(destination)
        is_cross = (
            box is not None
            and pitch_layout.in_wide_channel(origin)
            and pitch_layout.attacking_third(origin) == box
            and abs(destination[1] - origin[1]) >= PITCH_CROSS_MIN_LATERAL_M
        )
        return {
            "is_cross": bool(is_cross),
            "origin_m": [round(origin[0], 1), round(origin[1], 1)],
            "destination_m": [round(destination[0], 1), round(destination[1], 1)],
            "distance_m": round(pitch_layout.distance(origin, destination), 1),
        }

    def _is_aerial(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        flight: List[Tuple[int, float, float]],
    ) -> bool:
        """Did the ball rise above the straight line between release and receipt?"""
        inner = flight[1:-1]
        if len(inner) < 2 or self._scale <= 0:
            return False
        sx, sy = start
        ex, ey = end
        length = math.hypot(ex - sx, ey - sy)
        if length < 1e-6:
            return False
        best = 0.0
        for _, x, y in inner:
            t = max(0.0, min(1.0, ((x - sx) * (ex - sx) + (y - sy) * (ey - sy)) / (length * length)))
            chord_y = sy + t * (ey - sy)
            # Image y grows downwards, so "above the chord" is smaller y.
            best = max(best, (chord_y - y) / self._scale)
        return best >= CROSS_MIN_ARC

    def _is_cross(
        self,
        prev: _Segment,
        new: _Segment,
        displacement: float,
        gap_frames: int,
        aerial: bool,
        keeper: Optional[_Keeper],
    ) -> bool:
        if keeper is None or not aerial or self._scale <= 0:
            return False
        lateral = abs(new.start_ball[0] - prev.end_ball[0]) / self._scale
        if lateral < CROSS_MIN_LATERAL or gap_frames < CROSS_MIN_SECONDS * self.fps:
            return False
        if keeper.team_id != 0 and keeper.team_id == prev.team_id:
            return False
        keeper_distance = _distance((keeper.x, keeper.y), new.start_ball) / self._scale
        return keeper_distance <= CROSS_MAX_KEEPER_DISTANCE

    # ------------------------------------------------------------- shots/saves
    def _on_release_without_receiver(self, prev: _Segment, frame_number: int) -> List[dict]:
        emitted: List[dict] = []
        if self._scale <= 0 or prev.end_ball is None:
            return emitted
        release = self._release_velocity(prev.end_frame)
        if release is None or release[0] < SHOT_MIN_SPEED:
            return emitted
        speed, dx, dy = release
        keeper = self._keeper_near(prev.end_frame)
        if keeper is None:
            # No keeper in view: with pitch calibration the goal itself is the
            # target, so a shot can still be recognised.
            return self._pitch_shot_without_keeper(prev, frame_number, speed)
        angle = self._angle_between(dx, dy, keeper.x - prev.end_ball[0], keeper.y - prev.end_ball[1])
        if angle > SHOT_MAX_ANGLE_DEGREES:
            return emitted

        flight = self._ball_between(prev.end_frame, frame_number)
        nearest = min(
            (_distance((x, y), (keeper.x, keeper.y)) / self._scale for _, x, y in flight),
            default=float("inf"),
        )
        vanished = frame_number - self._last_ball_frame > 0.6 * self.fps
        if nearest > SHOT_KEEPER_REACH and not vanished:
            return emitted

        shooter = prev.participant("shooter")
        keeper_participant = self._opposing_keeper(
            Participant("goalkeeper", keeper.track_id, keeper.team_id, keeper.team_name), shooter
        )
        saved = self._direction_reversed(flight, keeper)
        outcome = "saved" if saved else ("ball_lost_from_view" if vanished else "unknown")
        shot = self._emit(
            "shot",
            prev.end_frame,
            frame_number,
            min(0.85, 0.45 + speed / 40.0),
            [shooter, keeper_participant],
            {
                "release_speed_heights_per_second": round(speed, 2),
                "angle_to_keeper_degrees": round(angle, 1),
                "outcome": outcome,
            },
        )
        if shot:
            emitted.append(shot)
        if saved:
            save = self._emit(
                "save",
                prev.end_frame,
                frame_number,
                min(0.8, 0.4 + speed / 40.0),
                [keeper_participant, shooter],
                {"release_speed_heights_per_second": round(speed, 2)},
            )
            if save:
                emitted.append(save)
        return emitted

    def _pitch_heading(self, start_frame: int, window: int = 8):
        """(origin, displacement, goal side) of the ball trail in metres."""
        samples = [s for s in self._ball_pitch if start_frame <= s[0] <= start_frame + window]
        if len(samples) < 2:
            return None
        origin = (samples[0][1], samples[0][2])
        dx, dy = samples[-1][1] - origin[0], samples[-1][2] - origin[1]
        side = "left" if origin[0] < pitch_layout.PITCH_LENGTH / 2.0 else "right"
        return origin, (dx, dy), side

    def _pitch_shot_without_keeper(self, prev: _Segment, frame_number: int, speed: float) -> List[dict]:
        heading = self._pitch_heading(prev.end_frame)
        if heading is None:
            return []
        origin, (dx, dy), side = heading
        goal = pitch_layout.goal_centre(side)
        if pitch_layout.distance(origin, goal) > PITCH_SHOT_MAX_DISTANCE_M:
            return []
        if math.hypot(dx, dy) < PITCH_SHOT_MIN_TRAVEL_M:
            return []
        angle = self._angle_between(dx, dy, goal[0] - origin[0], goal[1] - origin[1])
        if angle > PITCH_SHOT_MAX_ANGLE_DEGREES:
            return []
        shot = self._emit(
            "shot",
            prev.end_frame,
            frame_number,
            min(0.8, 0.4 + speed / 40.0),
            [prev.participant("shooter")],
            {
                "release_speed_heights_per_second": round(speed, 2),
                "angle_to_goal_degrees": round(angle, 1),
                "distance_to_goal_m": round(pitch_layout.distance(origin, goal), 1),
                "outcome": "unknown",
            },
        )
        return [shot] if shot else []

    def _update_goal_line(self, frame_number: int) -> List[dict]:
        """A goal: the ball, in metres, sits behind a goal line between the posts
        for consecutive frames, having come from the field of play near that goal.

        Positions are ground projections, so a ball in the air over the bar can
        project behind the line too; the net-depth limit in ``in_goal`` removes
        most of those, and the confidence is kept moderate.
        """
        if self._pitch is None or not self._ball_pitch:
            return []
        recent = [s for s in self._ball_pitch if frame_number - s[0] <= GOAL_CONFIRM_WINDOW_FRAMES]
        if len(recent) < GOAL_CONFIRM_SAMPLES:
            return []
        confirm = recent[-GOAL_CONFIRM_SAMPLES:]
        sides = {pitch_layout.in_goal((x, y)) for _, x, y in confirm}
        if len(sides) != 1 or None in sides:
            return []
        side = next(iter(sides))
        first_in_goal = confirm[0][0]
        window = int(GOAL_APPROACH_SECONDS * self.fps)
        goal = pitch_layout.goal_centre(side)
        approach = [
            s for s in self._ball_pitch
            if 0 < first_in_goal - s[0] <= window
            and 0.0 <= s[1] <= pitch_layout.PITCH_LENGTH
            and pitch_layout.distance((s[1], s[2]), goal) <= GOAL_APPROACH_DISTANCE_M
        ]
        if not approach:
            return []
        segment = self.current or self._last_segment
        participants = [segment.participant("scorer")] if segment is not None else []
        event = self._emit(
            "goal",
            first_in_goal,
            frame_number,
            0.6,
            participants,
            {"goal_side": side, "rule": "ball behind the goal line between the posts"},
        )
        if event is None:
            return []
        event["source"] = "pitch"
        return [event]

    def _direction_reversed(self, flight: List[Tuple[int, float, float]], keeper: _Keeper) -> bool:
        """Ball approached the keeper, then moved away: a parry or block."""
        if len(flight) < 4 or self._scale <= 0:
            return False
        distances = [_distance((x, y), (keeper.x, keeper.y)) for _, x, y in flight]
        closest = int(np.argmin(distances))
        if distances[closest] / self._scale > SHOT_KEEPER_REACH:
            return False
        if closest == 0 or closest >= len(flight) - 1:
            return False
        approach = distances[0] - distances[closest]
        retreat = distances[-1] - distances[closest]
        return approach > 0.5 * self._scale and retreat > 0.5 * self._scale

    # ----------------------------------------------------------------- penalty
    def _update_penalty(
        self,
        frame_number: int,
        players: Dict[int, dict],
        ball_centre: Optional[Tuple[float, float]],
    ) -> List[dict]:
        emitted: List[dict] = []
        if self._scale <= 0:
            return emitted

        setup = self._penalty_setup
        if setup is not None:
            if frame_number - setup["frame"] > PENALTY_KICK_TIMEOUT_SECONDS * self.fps:
                self._penalty_setup = None
            elif ball_centre is not None:
                moved = _distance(ball_centre, setup["ball"]) / self._scale
                if moved >= 1.0:
                    release = self._release_velocity(setup["last_still_frame"])
                    keeper = setup["keeper"]
                    if release and release[0] >= PENALTY_KICK_SPEED:
                        if keeper is not None:
                            angle = self._angle_between(
                                release[1], release[2], keeper.x - setup["ball"][0], keeper.y - setup["ball"][1]
                            )
                        else:
                            heading = self._pitch_heading(setup["last_still_frame"])
                            angle = 180.0
                            if heading is not None and setup.get("spot_side"):
                                origin, (dx, dy), _ = heading
                                goal = pitch_layout.goal_centre(setup["spot_side"])
                                angle = self._angle_between(dx, dy, goal[0] - origin[0], goal[1] - origin[1])
                        if angle <= 50.0:
                            participants = [setup["taker"]]
                            if keeper is not None:
                                participants.append(
                                    self._opposing_keeper(
                                        Participant("goalkeeper", keeper.track_id, keeper.team_id, keeper.team_name),
                                        setup["taker"],
                                    )
                                )
                            details = {
                                "still_seconds": round(setup["still_seconds"], 2),
                                "kick_speed_heights_per_second": round(release[0], 2),
                            }
                            if setup["keeper_distance"] is not None:
                                details["keeper_distance_heights"] = round(setup["keeper_distance"], 2)
                            if setup.get("spot_side"):
                                details["penalty_spot"] = setup["spot_side"]
                            event = self._emit(
                                "penalty",
                                setup["last_still_frame"],
                                frame_number,
                                0.7 if setup.get("spot_side") else 0.55,
                                participants,
                                details,
                            )
                            if event:
                                emitted.append(event)
                    self._penalty_setup = None
                else:
                    setup["last_still_frame"] = frame_number
            return emitted

        if ball_centre is None:
            return emitted
        anchor = self._still_anchor
        if anchor is None or _distance(ball_centre, (anchor[1], anchor[2])) / self._scale > PENALTY_STILL_TOLERANCE:
            self._still_anchor = (frame_number, ball_centre[0], ball_centre[1])
            return emitted
        still_frames = frame_number - anchor[0]
        if still_frames < PENALTY_STILL_SECONDS * self.fps:
            return emitted

        keeper = self._keeper_near(frame_number, ball_centre)
        spot_side = None
        if self._ball_pitch_now is not None:
            # Calibrated: the ball has to be on a penalty spot. A still ball
            # anywhere else is a free kick, however the players stand.
            for side in ("left", "right"):
                if pitch_layout.distance(self._ball_pitch_now, pitch_layout.penalty_spot(side)) <= PITCH_PENALTY_SPOT_TOLERANCE_M:
                    spot_side = side
            if spot_side is None:
                return emitted
        elif keeper is None:
            return emitted
        keeper_distance = None
        if keeper is not None:
            keeper_distance = _distance((keeper.x, keeper.y), ball_centre) / self._scale
            if spot_side is None and not (PENALTY_KEEPER_DISTANCE[0] <= keeper_distance <= PENALTY_KEEPER_DISTANCE[1]):
                return emitted

        near: List[Tuple[int, dict]] = []
        far = 0
        for track_id, player in players.items():
            if (keeper is not None and int(track_id) == keeper.track_id) or player.get("role") == "goalkeeper":
                continue
            distance = _distance(_foot(player["bbox"]), ball_centre) / self._scale
            if distance <= PENALTY_TAKER_DISTANCE:
                near.append((int(track_id), player))
            elif distance >= PENALTY_OTHERS_DISTANCE:
                far += 1
        if len(near) != 1 or far < PENALTY_MIN_OTHERS:
            return emitted

        taker_id, taker_player = near[0]
        if keeper is not None and spot_side is None and keeper.team_id != 0 and int(taker_player.get("team", 0) or 0) == keeper.team_id:
            return emitted
        self._penalty_setup = {
            "frame": frame_number,
            "last_still_frame": frame_number,
            "ball": ball_centre,
            "keeper": keeper,
            "keeper_distance": keeper_distance,
            "spot_side": spot_side,
            "taker": Participant.from_player("taker", taker_id, taker_player),
            "still_seconds": self._seconds(still_frames),
        }
        self._still_anchor = None
        return emitted

    # ------------------------------------------------------------ shot cuts
    def reset_scene(self, frame_number: int) -> List[dict]:
        """A new camera shot begins at ``frame_number``.

        The open possession is judged as a possible shot first - a broadcast
        often cuts away while the ball is in flight - then everything measured
        in the old shot is dropped: the ball trail, the player scale (a close-up
        player is 5x taller), keeper positions, a half-built penalty. No pass
        is ever formed across a cut. The last holder is kept for goal
        attribution: whoever last had the ball live is still the last to touch it.
        """
        emitted: List[dict] = []
        if self.current is not None:
            emitted.extend(self._on_release_without_receiver(self.current, frame_number))
            self._last_segment = self.current
            self.current = None
        self._candidate = None
        self._frames_without_holder = 0
        self._ball.clear()
        self._ball_pitch.clear()
        self._ball_pitch_now = None
        self._keepers.clear()
        self._heights.clear()
        self._scale = 0.0
        self._still_anchor = None
        self._penalty_setup = None
        if self.card_detector is not None:
            self.card_detector.reset()
        self.scene_resets += 1
        return emitted

    # ----------------------------------------------------------------- summary
    def finish(self, frame_number: int) -> List[dict]:
        """Flush a possession that was still open when the video ended."""
        emitted: List[dict] = []
        if self.current is not None:
            emitted.extend(self._on_release_without_receiver(self.current, frame_number))
            self.current = None
        return emitted


def summarise_passes(events: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    """Pass and cross counts per team name, for the match summary."""
    summary: Dict[str, Dict[str, int]] = {}
    for event in events:
        if event.get("event") not in ("pass", "cross"):
            continue
        team = str(event.get("team_name") or "unknown_team")
        entry = summary.setdefault(team, {"passes": 0, "crosses": 0})
        entry["passes"] += 1
        if event.get("event") == "cross":
            entry["crosses"] += 1
    return summary
