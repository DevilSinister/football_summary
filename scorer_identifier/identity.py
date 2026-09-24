from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional

import cv2
import numpy as np

from .jersey_reader import IdentityReadResult, JerseyReader, OcrCandidate, TrackedCrop


@dataclass
class PlayerIdentity:
    track_id: int
    team_id: int = 0
    team_name: str = "unknown_team"
    team_confidence: float = 0.0
    recent_crops: List[TrackedCrop] = field(default_factory=list)
    jersey_candidates: List[OcrCandidate] = field(default_factory=list)
    name_candidates: List[OcrCandidate] = field(default_factory=list)
    jersey_number: Optional[int] = None
    jersey_confidence: float = 0.0
    player_name: Optional[str] = None
    name_confidence: float = 0.0

    def snapshot(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "team_confidence": self.team_confidence,
            "jersey_number": self.jersey_number,
            "jersey_confidence": self.jersey_confidence,
            "player_name": self.player_name,
            "name_confidence": self.name_confidence,
            "crop_count": len(self.recent_crops),
            "jersey_candidates": [candidate.text for candidate in self.jersey_candidates],
            "name_candidates": [candidate.text for candidate in self.name_candidates],
        }


class PlayerIdentityStore:
    def __init__(
        self,
        max_crops_per_player: int = 8,
        crop_stride: int = 3,
        min_box_height: int = 0,
    ) -> None:
        self.max_crops_per_player = max_crops_per_player
        self.crop_stride = max(1, crop_stride)
        # Below this box height the shirt number is not resolvable at all, so
        # keeping the crop only buys a wasted OCR pass. 0 disables the gate.
        self.min_box_height = max(0, int(min_box_height))
        self.players: Dict[int, PlayerIdentity] = {}

    @staticmethod
    def _crop_jersey_region(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        """The band of the shirt that carries the number.

        The previous window ran from 8% to 67% of the box at 84% of its width -
        head to hip, arms included - so the digits were a small minority of the
        pixels and competed with the face, the shorts, and the background.
        """
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = x2 - x1
        height = y2 - y1
        left = max(0, min(frame_width, int(x1 + width * 0.25)))
        right = max(0, min(frame_width, int(x2 - width * 0.25)))
        top = max(0, min(frame_height, int(y1 + height * 0.20)))
        bottom = max(0, min(frame_height, int(y1 + height * 0.52)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0 or crop.shape[0] < 14 or crop.shape[1] < 10:
            return None
        return crop.copy()

    @staticmethod
    def _rear_facing_score(frame: np.ndarray, bbox) -> float:
        """Rough guess at whether the player has their back to the camera.

        Nothing in the pipeline ever set this, so the "prefer rear-facing crops"
        ranking was a constant and most crops sent to OCR showed a chest rather
        than a number. Skin in the head region means the face is turned towards
        the camera, so little skin implies a rear view. Cheap and approximate,
        but far better than a fixed 0.5.
        """
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = x2 - x1
        height = y2 - y1
        left = max(0, min(frame_width, int(x1 + width * 0.30)))
        right = max(0, min(frame_width, int(x2 - width * 0.30)))
        top = max(0, min(frame_height, int(y1)))
        bottom = max(0, min(frame_height, int(y1 + height * 0.20)))
        head = frame[top:bottom, left:right]
        if head.size == 0:
            return 0.5

        ycrcb = cv2.cvtColor(head, cv2.COLOR_BGR2YCrCb)
        chroma_red = ycrcb[:, :, 1]
        chroma_blue = ycrcb[:, :, 2]
        skin = (
            (chroma_red >= 133)
            & (chroma_red <= 173)
            & (chroma_blue >= 77)
            & (chroma_blue <= 127)
        )
        skin_fraction = float(skin.mean())
        # A face fills a good part of the head box; 25% skin is treated as
        # unambiguously front-on.
        return round(max(0.0, min(1.0, 1.0 - skin_fraction / 0.25)), 3)

    @staticmethod
    def _crop_quality(crop: np.ndarray, rear_facing_score: float) -> float:
        height, width = crop.shape[:2]
        size_score = min(1.0, (height * width) / 12000.0)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] > 96:
            scale = 96.0 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # CV_32F on a downscaled image. The CV_64F full-size path ran for every
        # tracked player on every strided frame.
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        sharpness_score = min(1.0, sharpness / 180.0)
        score = 0.45 * size_score + 0.35 * sharpness_score + 0.20 * rear_facing_score
        return round(max(0.0, min(1.0, score)), 3)

    def update_player(self, frame_number: int, frame: np.ndarray, track_id: int, player: dict) -> PlayerIdentity:
        state = self.players.setdefault(track_id, PlayerIdentity(track_id=track_id))
        state.team_id = int(player.get("team", state.team_id) or 0)
        state.team_name = str(player.get("team_name", state.team_name) or "unknown_team")
        state.team_confidence = float(player.get("team_confidence", state.team_confidence) or 0.0)

        bbox = player["bbox"]
        box_height = float(bbox[3]) - float(bbox[1])
        if frame_number % self.crop_stride == 0 and box_height >= self.min_box_height:
            crop = self._crop_jersey_region(frame, bbox)
            if crop is not None:
                rear_score = player.get("rear_facing_score")
                if rear_score is None:
                    rear_score = self._rear_facing_score(frame, bbox)
                rear_score = float(rear_score)
                quality = self._crop_quality(crop, rear_score)
                state.recent_crops.append(
                    TrackedCrop(frame_number, crop, quality, rear_score)
                )
                state.recent_crops.sort(
                    key=lambda item: (item.quality + 0.25 * item.rear_facing_score, item.frame_number),
                    reverse=True,
                )
                del state.recent_crops[self.max_crops_per_player :]

        player["identity"] = state.snapshot()
        return state

    def update_frame(self, frame_number: int, frame: np.ndarray, players: Dict[int, dict]) -> None:
        for track_id, player in players.items():
            self.update_player(frame_number, frame, int(track_id), player)

    def get(self, track_id: int) -> Optional[PlayerIdentity]:
        return self.players.get(track_id)


@dataclass(frozen=True)
class PossessionSample:
    frame_number: int
    track_id: int
    team_id: int
    team_name: str


class PossessionHistory:
    def __init__(self, max_samples: int = 600, lookback_frames: int = 180) -> None:
        self.samples: Deque[PossessionSample] = deque(maxlen=max_samples)
        self.lookback_frames = lookback_frames

    def record(self, frame_number: int, track_id: int, identity: Optional[PlayerIdentity]) -> None:
        if track_id is None or int(track_id) < 0:
            return
        self.samples.append(
            PossessionSample(
                frame_number=int(frame_number),
                track_id=int(track_id),
                team_id=identity.team_id if identity else 0,
                team_name=identity.team_name if identity else "unknown_team",
            )
        )

    def select_scorer(self, goal_frame: int) -> Optional[Dict[str, object]]:
        eligible = [
            sample
            for sample in self.samples
            if goal_frame - self.lookback_frames <= sample.frame_number <= goal_frame
        ]
        if not eligible:
            return None

        latest = max(eligible, key=lambda sample: sample.frame_number)
        recent = sorted(eligible, key=lambda sample: sample.frame_number, reverse=True)[:12]
        matching = sum(sample.track_id == latest.track_id for sample in recent)
        return {
            "track_id": latest.track_id,
            "team_id": latest.team_id,
            "team_name": latest.team_name,
            "last_touch_frame": latest.frame_number,
            "possession_confidence": round(matching / len(recent), 3),
        }


def min_readable_box_height(frame_height: int) -> int:
    """Smallest player box whose shirt number has any chance of being read.

    Measured on 854x480 source, the median player box is about 27x60 px, which
    leaves a number roughly 12x15 px - below what any OCR model can resolve. At
    720p and 1080p a useful share of boxes clear this bar, so the gate scales
    with the source rather than being a fixed pixel count.
    """
    return max(100, int(frame_height * 0.14))


class ScorerIdentifier:
    def __init__(
        self,
        jersey_reader: Optional[JerseyReader] = None,
        identity_store: Optional[PlayerIdentityStore] = None,
        possession_history: Optional[PossessionHistory] = None,
        frame_height: Optional[int] = None,
    ) -> None:
        self.jersey_reader = jersey_reader or JerseyReader()
        if identity_store is None:
            identity_store = PlayerIdentityStore(
                min_box_height=(
                    min_readable_box_height(frame_height) if frame_height else 0
                )
            )
        self.identity_store = identity_store
        self.possession_history = possession_history or PossessionHistory()
        # One OCR pass per track for the whole run. Every event a player takes
        # part in - a goal, a dozen passes, a card - reuses the same reading,
        # which keeps the OCR budget for tracks that were never read.
        self._track_reads: Dict[int, Dict[str, object]] = {}

    def observe_players(self, frame_number: int, frame: np.ndarray, players: Dict[int, dict]) -> None:
        self.identity_store.update_frame(frame_number, frame, players)

    def record_possession(self, frame_number: int, track_id: int) -> None:
        self.possession_history.record(frame_number, track_id, self.identity_store.get(track_id))

    def identify_goal(self, goal_frame: int) -> Optional[Dict[str, object]]:
        scorer = self.possession_history.select_scorer(goal_frame)
        if scorer is None:
            return None
        scorer.update(self.identify_track(int(scorer["track_id"])))
        return scorer

    def identify_track(self, track_id: int) -> Dict[str, object]:
        """Team and shirt number for one track, read once and cached.

        The name is deliberately absent: it comes from the roster lookup on
        (team, shirt number) in the persistence layer, never from the shirt.
        """
        track_id = int(track_id)
        cached = self._track_reads.get(track_id)
        if cached is not None:
            return dict(cached)

        state = self.identity_store.get(track_id)
        if state is None:
            identity: Dict[str, object] = {
                "track_id": track_id,
                "team_id": 0,
                "team_name": "unknown_team",
                "team_confidence": 0.0,
                "jersey_number": None,
                "jersey_confidence": 0.0,
                "player_name": None,
                "name_confidence": 0.0,
                "ocr_error": "track was never observed",
            }
        else:
            result = self.jersey_reader.read(state.recent_crops, state.track_id)
            self._apply_read_result(state, result)
            identity = {
                "track_id": track_id,
                "team_id": state.team_id,
                "team_name": state.team_name,
                "team_confidence": state.team_confidence,
                "jersey_number": result.jersey_number,
                "jersey_confidence": result.jersey_confidence,
                "player_name": result.player_name,
                "name_confidence": result.name_confidence,
                "ocr_error": result.error,
            }
        self._track_reads[track_id] = identity
        return dict(identity)

    def identify_participants(self, participants: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        """Resolve every participant of a track-derived event.

        The participant's own team reading wins when the identity store never
        settled one, so a pass keeps the team the detector saw at the time.
        """
        resolved = []
        for participant in participants:
            identity = self.identify_track(int(participant.get("track_id", -1)))
            # A goalkeeper's kit matches neither team, so the colour-based team
            # in the identity store is noise; the event detector's deduction
            # (the side opposite the shooter) is the reliable reading.
            keeper_override = participant.get("role") == "goalkeeper" and participant.get("team_id", 0)
            if (identity.get("team_id", 0) == 0 and participant.get("team_id", 0)) or keeper_override:
                identity["team_id"] = int(participant["team_id"])
                identity["team_name"] = str(participant.get("team_name", "unknown_team"))
                identity["team_confidence"] = float(participant.get("team_confidence", 0.0) or 0.0)
            identity["role"] = participant.get("role", "player")
            resolved.append(identity)
        return resolved

    @staticmethod
    def _apply_read_result(state: PlayerIdentity, result: IdentityReadResult) -> None:
        state.jersey_number = result.jersey_number
        state.jersey_confidence = result.jersey_confidence
        state.player_name = result.player_name
        state.name_confidence = result.name_confidence
        state.jersey_candidates = [
            candidate
            for candidate in result.candidates
            if any(character.isdigit() for character in candidate.text)
        ]
        # Shirt names are no longer read; player_name is filled from the roster.
        state.name_candidates = []


def format_goal_event(event: Dict[str, object]) -> str:
    scorer = event.get("scorer") or {}
    team_name = scorer.get("team_name") or "unknown_team"
    jersey_number = scorer.get("jersey_number")
    player_name = scorer.get("player_name")
    if jersey_number is None:
        return f"GOAL — {team_name} — scorer number unavailable"
    if player_name:
        return f"GOAL — {team_name} — #{jersey_number} {player_name} scored"
    return f"GOAL — {team_name} — #{jersey_number} scored"
