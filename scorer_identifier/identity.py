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
    def __init__(self, max_crops_per_player: int = 16, crop_stride: int = 3) -> None:
        self.max_crops_per_player = max_crops_per_player
        self.crop_stride = max(1, crop_stride)
        self.players: Dict[int, PlayerIdentity] = {}

    @staticmethod
    def _crop_jersey_region(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = x2 - x1
        height = y2 - y1
        left = max(0, min(frame_width, int(x1 + width * 0.08)))
        right = max(0, min(frame_width, int(x2 - width * 0.08)))
        top = max(0, min(frame_height, int(y1 + height * 0.08)))
        bottom = max(0, min(frame_height, int(y1 + height * 0.67)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 16:
            return None
        return crop.copy()

    @staticmethod
    def _crop_quality(crop: np.ndarray, rear_facing_score: float) -> float:
        height, width = crop.shape[:2]
        size_score = min(1.0, (height * width) / 12000.0)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, sharpness / 180.0)
        score = 0.45 * size_score + 0.35 * sharpness_score + 0.20 * rear_facing_score
        return round(max(0.0, min(1.0, score)), 3)

    def update_player(self, frame_number: int, frame: np.ndarray, track_id: int, player: dict) -> PlayerIdentity:
        state = self.players.setdefault(track_id, PlayerIdentity(track_id=track_id))
        state.team_id = int(player.get("team", state.team_id) or 0)
        state.team_name = str(player.get("team_name", state.team_name) or "unknown_team")
        state.team_confidence = float(player.get("team_confidence", state.team_confidence) or 0.0)

        if frame_number % self.crop_stride == 0:
            crop = self._crop_jersey_region(frame, player["bbox"])
            if crop is not None:
                rear_score = float(player.get("rear_facing_score", 0.5))
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


class ScorerIdentifier:
    def __init__(
        self,
        jersey_reader: Optional[JerseyReader] = None,
        identity_store: Optional[PlayerIdentityStore] = None,
        possession_history: Optional[PossessionHistory] = None,
    ) -> None:
        self.jersey_reader = jersey_reader or JerseyReader()
        self.identity_store = identity_store or PlayerIdentityStore()
        self.possession_history = possession_history or PossessionHistory()

    def observe_players(self, frame_number: int, frame: np.ndarray, players: Dict[int, dict]) -> None:
        self.identity_store.update_frame(frame_number, frame, players)

    def record_possession(self, frame_number: int, track_id: int) -> None:
        self.possession_history.record(frame_number, track_id, self.identity_store.get(track_id))

    def identify_goal(self, goal_frame: int) -> Optional[Dict[str, object]]:
        scorer = self.possession_history.select_scorer(goal_frame)
        if scorer is None:
            return None

        state = self.identity_store.get(int(scorer["track_id"]))
        if state is None:
            return scorer

        result = self.jersey_reader.read(state.recent_crops, state.track_id)
        self._apply_read_result(state, result)
        scorer.update(
            {
                "team_id": state.team_id,
                "team_name": state.team_name,
                "team_confidence": state.team_confidence,
                "jersey_number": result.jersey_number,
                "jersey_confidence": result.jersey_confidence,
                "player_name": result.player_name,
                "name_confidence": result.name_confidence,
                "ocr_error": result.error,
            }
        )
        return scorer

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
        state.name_candidates = [
            candidate
            for candidate in result.candidates
            if any(character.isalpha() for character in candidate.text)
        ]


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
