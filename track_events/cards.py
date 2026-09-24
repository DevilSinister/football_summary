"""Yellow and red card detection from the referee's raised hand.

A card is a small, strongly coloured rectangle held above the shoulder. The
detector looks only at the band around and above a tracked referee's head, so a
fluorescent kit, a steward's vest or the pitch itself cannot count: those fill
the box from the shoulders down, never the space above the head.

This is a heuristic. It fires when a card-sized blob of card colour sits above a
referee for a sustained fraction of a second, and it names the nearest player as
the recipient. It cannot see a card shown while the referee is out of frame or
mislabelled by the detector, and it does not read the broadcast graphic.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np


# OpenCV hue is 0-179. Card yellow is a saturated warm yellow; card red wraps
# around zero.
YELLOW_HUE = (18, 36)
RED_HUE_LOW = 8
RED_HUE_HIGH = 168
CARD_MIN_SATURATION = 110
YELLOW_MIN_VALUE = 140
RED_MIN_VALUE = 90

# Card area relative to the referee's box. A 12x9 cm card on a 1.8 m person
# whose box is ~0.6 m wide is about 1% of the box.
CARD_MIN_AREA_RATIO = 0.0015
CARD_MAX_AREA_RATIO = 0.06
CARD_MIN_ASPECT = 0.35
CARD_MAX_ASPECT = 2.8
CARD_MIN_FILL = 0.45

# How long the card has to stay up, and how long to wait before the same
# referee can show another.
CARD_MIN_SECONDS = 0.32
CARD_WINDOW_SECONDS = 1.5
CARD_COOLDOWN_SECONDS = 6.0
# Nearest player within this many player heights is booked.
RECIPIENT_MAX_HEIGHTS = 3.0
# A referee box shorter than this cannot show a card the detector could see
# (the card would be a pixel or two), and a blob smaller than this is noise.
CARD_MIN_BOX_HEIGHT = 48
CARD_MIN_AREA_PIXELS = 16
# How far below the top of the box the search strip reaches: head and raised
# hand, never the shoulders.
REGION_BOTTOM_FRACTION = 0.16


@dataclass
class _Sighting:
    frame: int
    colour: str
    nearest_track: Optional[int]


def card_colour_in_region(region: np.ndarray, reference_area: float) -> Optional[str]:
    """'yellow', 'red' or None for one crop above a referee's shoulders."""
    if region is None or region.size == 0 or reference_area <= 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    masks = {
        "yellow": (
            (hue >= YELLOW_HUE[0])
            & (hue <= YELLOW_HUE[1])
            & (saturation >= CARD_MIN_SATURATION)
            & (value >= YELLOW_MIN_VALUE)
        ),
        "red": (
            ((hue <= RED_HUE_LOW) | (hue >= RED_HUE_HIGH))
            & (saturation >= CARD_MIN_SATURATION)
            & (value >= RED_MIN_VALUE)
        ),
    }
    min_area = max(CARD_MIN_AREA_PIXELS, CARD_MIN_AREA_RATIO * reference_area)
    max_area = CARD_MAX_AREA_RATIO * reference_area
    region_height = region.shape[0]
    best: Optional[str] = None
    best_area = 0.0
    for colour, mask in masks.items():
        mask_u8 = mask.astype(np.uint8)
        if int(mask_u8.sum()) < min_area:
            continue
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area < min_area or area > max_area or width == 0 or height == 0:
                continue
            # A blob running off the bottom of the strip is clothing coming up
            # from the shoulders, not something held in the air.
            if y + height >= region_height:
                continue
            aspect = width / float(height)
            if aspect < CARD_MIN_ASPECT or aspect > CARD_MAX_ASPECT:
                continue
            if area / float(width * height) < CARD_MIN_FILL:
                continue
            if area > best_area:
                best, best_area = colour, float(area)
    return best


def _card_region(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """The strip above the shoulders where a raised card appears.

    A person box grows to include a raised arm, so the card sits just inside
    the top of the box with the head below it. The strip stops well above the
    shoulders: on the reference clip the collar of a yellow referee kit, and the
    shoulders of a steward's vest, read as card-sized yellow blobs when the
    region reached that far down.
    """
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if height < CARD_MIN_BOX_HEIGHT:
        return None
    left = int(max(0, x1 - 0.35 * width))
    right = int(min(frame_width, x2 + 0.35 * width))
    top = int(max(0, y1 - 0.15 * height))
    bottom = int(min(frame_height, y1 + REGION_BOTTOM_FRACTION * height))
    if right - left < 4 or bottom - top < 4:
        return None
    return frame[top:bottom, left:right]


class CardDetector:
    def __init__(self, fps: float) -> None:
        self.fps = max(float(fps), 1.0)
        self.min_hits = max(3, int(round(CARD_MIN_SECONDS * self.fps)))
        self.window_frames = int(round(CARD_WINDOW_SECONDS * self.fps))
        self.cooldown_frames = int(round(CARD_COOLDOWN_SECONDS * self.fps))
        self._sightings: Dict[int, Deque[_Sighting]] = {}
        self._last_card_frame: Dict[int, int] = {}

    def reset(self) -> None:
        """A new shot: a card half-seen in the last one is not carried over."""
        self._sightings.clear()

    @staticmethod
    def _nearest_player(referee_bbox, players: Dict[int, dict], scale: float) -> Optional[int]:
        rx = (float(referee_bbox[0]) + float(referee_bbox[2])) / 2.0
        ry = float(referee_bbox[3])
        best_id, best_distance = None, float("inf")
        for track_id, player in players.items():
            bbox = player["bbox"]
            px = (float(bbox[0]) + float(bbox[2])) / 2.0
            py = float(bbox[3])
            distance = float(np.hypot(px - rx, py - ry))
            if distance < best_distance:
                best_id, best_distance = int(track_id), distance
        if best_id is None or scale <= 0 or best_distance > RECIPIENT_MAX_HEIGHTS * scale:
            return None
        return best_id

    def update(
        self,
        frame_number: int,
        frame: np.ndarray,
        referees: Dict[int, dict],
        players: Dict[int, dict],
        scale: float,
    ) -> List[dict]:
        events: List[dict] = []
        for referee_id, referee in referees.items():
            referee_id = int(referee_id)
            bbox = referee["bbox"]
            if frame_number - self._last_card_frame.get(referee_id, -10**9) < self.cooldown_frames:
                continue
            region = _card_region(frame, bbox)
            box_area = max(1.0, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))
            colour = card_colour_in_region(region, box_area)
            history = self._sightings.setdefault(referee_id, deque())
            while history and frame_number - history[0].frame > self.window_frames:
                history.popleft()
            if colour is None:
                continue
            history.append(
                _Sighting(frame_number, colour, self._nearest_player(bbox, players, scale))
            )
            hits = [item for item in history if item.colour == colour]
            if len(hits) < self.min_hits:
                continue

            recipients = [item.nearest_track for item in hits if item.nearest_track is not None]
            recipient = max(set(recipients), key=recipients.count) if recipients else None
            participants = [{"role": "referee", "track_id": referee_id, "team_id": 0, "team_name": "unknown_team", "team_confidence": 0.0}]
            team_id, team_name, team_confidence = 0, "unknown_team", 0.0
            if recipient is not None and recipient in players:
                player = players[recipient]
                team_id = int(player.get("team", 0) or 0)
                team_name = str(player.get("team_name", "unknown_team") or "unknown_team")
                team_confidence = float(player.get("team_confidence", 0.0) or 0.0)
                participants.insert(
                    0,
                    {
                        "role": "player",
                        "track_id": recipient,
                        "team_id": team_id,
                        "team_name": team_name,
                        "team_confidence": team_confidence,
                    },
                )
            confidence = min(0.9, 0.45 + 0.05 * len(hits) + (0.1 if recipient is not None else 0.0))
            events.append(
                {
                    "event": f"{colour}_card",
                    "frame": hits[0].frame,
                    "end_frame": frame_number,
                    "confidence": round(confidence, 3),
                    "team_id": team_id,
                    "team_name": team_name,
                    "source": "tracks",
                    "participants": participants,
                    "details": {"sightings": len(hits), "referee_track_id": referee_id},
                }
            )
            self._last_card_frame[referee_id] = frame_number
            history.clear()
        return events
