"""Shot boundaries in broadcast footage: hard cuts and dissolves.

Highlight packages cut between the live wide shot, close-ups and replays,
often through a crossfade. Across a boundary nothing carries over - player
identities, ball trail, player scale, pitch calibration - and inside a
crossfade two shots are blended, so kit colours and shirt crops are garbage.
On the 1080p reference clip that blend put Porto's #20 on Man City and made
him the "scorer" of Man City's goal.

Method: twin-comparison (Zhang, Kankanhalli and Smoliar, 1993) on HSV colour
histograms. A frame-to-frame distance above ``low`` opens a candidate; when it
closes, the frames either side are compared, and a total change above
``high`` is a boundary - a single frame is a cut, several are a dissolve.
Camera pans move the histogram a little every frame and never accumulate past
``high``; a broadcast caption animating in changes too few pixels.

Measured on the reference clips (Bhattacharyya distance): pans and motion
0.03-0.10 per frame; hard cuts 0.41-0.72 in one frame; dissolves 8-10 frames
above 0.08 with 0.40-0.56 between the frames either side.
"""

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

LOW_THRESHOLD = 0.08
HIGH_THRESHOLD = 0.28
MAX_GAP_FRAMES = 1       # a dissolve may dip under `low` for this many frames


@dataclass(frozen=True)
class SceneBoundary:
    start: int      # first frame of the transition (the first frame of the new shot, for a cut)
    end: int        # last frame of the transition
    kind: str       # "cut" or "dissolve"
    strength: float


def frame_signature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 1, 0, cv2.NORM_L1)
    return histogram


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def find_boundaries(
    signatures: Sequence[np.ndarray],
    low: float = LOW_THRESHOLD,
    high: float = HIGH_THRESHOLD,
    max_gap: int = MAX_GAP_FRAMES,
) -> List[SceneBoundary]:
    boundaries: List[SceneBoundary] = []
    count = len(signatures)
    if count < 2:
        return boundaries
    step = [0.0] + [_distance(signatures[i - 1], signatures[i]) for i in range(1, count)]

    i = 1
    while i < count:
        if step[i] < low:
            i += 1
            continue
        start, end, gap = i, i, 0
        j = i + 1
        while j < count:
            if step[j] >= low:
                end, gap = j, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            j += 1
        change = _distance(signatures[start - 1], signatures[end])
        if change >= high:
            kind = "cut" if end == start else "dissolve"
            boundaries.append(SceneBoundary(start, end, kind, round(change, 3)))
        i = end + 1
    return boundaries


def detect_scene_boundaries(video_path: str) -> List[SceneBoundary]:
    """Decode the video once and return its shot boundaries."""
    capture = cv2.VideoCapture(video_path)
    signatures = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            signatures.append(frame_signature(frame))
    finally:
        capture.release()
    return find_boundaries(signatures)


class SceneTimeline:
    """Per-frame questions about a list of boundaries."""

    def __init__(self, boundaries: Sequence[SceneBoundary]) -> None:
        self.boundaries = list(boundaries)
        self._transition = set()
        self._scene_starts = set()
        for boundary in self.boundaries:
            if boundary.kind == "cut":
                self._scene_starts.add(boundary.start)
            else:
                self._transition.update(range(boundary.start, boundary.end + 1))
                self._scene_starts.add(boundary.end + 1)

    def in_transition(self, frame: int) -> bool:
        """Frame is part of a blend of two shots: use nothing from it."""
        return frame in self._transition

    def starts_scene(self, frame: int) -> bool:
        """First usable frame of a new shot: forget everything tracked so far."""
        return frame in self._scene_starts

    def describe(self) -> str:
        if not self.boundaries:
            return "no shot boundaries"
        return ", ".join(
            f"{b.kind} {b.start}" + (f"-{b.end}" if b.end != b.start else "") for b in self.boundaries
        )
