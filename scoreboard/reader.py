"""Follow the broadcast score graphic through a video and report score changes.

Reading a whole strip of the frame costs 2-4 s per OCR call on this CPU at
480p, so the graphic is located once (``scan``) and afterwards only its own
small region is read (``locked``). If it stays unreadable for a minute - a
long replay, a different graphic - the reader goes back to scanning.

A new score is believed only after ``confirm_reads`` consecutive reads agree,
and only a rise of one or two goals for one team counts as goals. Anything
else (a 0 read as 8, both scores moving at once) is logged, not reported.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .parser import Box, ScoreReading, Token, find_codes, parse_tokens

OcrFn = Callable[[np.ndarray], List[Tuple[str, float, Box]]]

MIN_TOKEN_CONFIDENCE = 0.5
# A region locked on team codes alone is given this long to yield a score.
CODES_ONLY_GRACE_SECONDS = 10.0
TOP_STRIP = 0.18      # share of the frame height scanned from the top
BOTTOM_STRIP = 0.22   # and from the bottom


def paddle_ocr_fn(engine) -> OcrFn:
    """Adapt a PaddleOCR 3.x engine to (text, confidence, box) tokens."""

    def run(image: np.ndarray):
        tokens = []
        for result in engine.predict(input=image):
            data = result.json if hasattr(result, "json") else result
            data = data.get("res", data) if isinstance(data, dict) else {}
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or []
            boxes = data.get("rec_boxes")
            if boxes is None or len(boxes) == 0:
                boxes = data.get("rec_polys") or []
            for text, score, box in zip(texts, scores, boxes):
                flat = np.asarray(box, dtype=float).reshape(-1)
                if flat.size == 4:
                    x1, y1, x2, y2 = flat
                else:
                    points = flat.reshape(-1, 2)
                    x1, y1 = points.min(axis=0)
                    x2, y2 = points.max(axis=0)
                tokens.append((str(text), float(score), (float(x1), float(y1), float(x2), float(y2))))
        return tokens

    return run


def upscale_for(frame_height: int) -> float:
    """Scale that brings small graphic text to a size the recogniser reads."""
    if frame_height <= 540:
        return 3.0
    if frame_height <= 800:
        return 2.0
    return 1.0


@dataclass
class ScoreGoal:
    code: str
    frame: int                       # first frame the new score was read
    window_start: Optional[int]      # last frame the previous score was read
    score_before: Dict[str, int]
    score_after: Dict[str, int]


@dataclass
class ScoreTracker:
    confirm_reads: int = 2
    max_goals_per_change: int = 2
    codes: Optional[Tuple[str, str]] = None
    state: Optional[Dict[str, int]] = None
    last_state_frame: Optional[int] = None
    goals: List[ScoreGoal] = field(default_factory=list)
    timeline: List[Tuple[int, Dict[str, int]]] = field(default_factory=list)
    rejected: List[Tuple[int, str]] = field(default_factory=list)
    readings: int = 0
    _pending: Optional[Tuple[Tuple[int, int], int, int]] = None  # scores, first frame, count

    def observe(self, frame: int, reading: ScoreReading) -> Optional[List[ScoreGoal]]:
        if self.codes is None:
            self.codes = reading.codes
        elif set(reading.codes) != set(self.codes):
            self.rejected.append((frame, f"other codes {reading.codes}"))
            return None
        self.readings += 1
        by_code = dict(zip(reading.codes, reading.scores))
        key = tuple(by_code[code] for code in self.codes)

        if self.state is not None and key == tuple(self.state[code] for code in self.codes):
            self.last_state_frame = frame
            self._pending = None
            return None
        if self._pending is not None and self._pending[0] == key:
            self._pending = (key, self._pending[1], self._pending[2] + 1)
        else:
            self._pending = (key, frame, 1)
        if self._pending[2] < self.confirm_reads:
            return None
        first_frame = self._pending[1]
        self._pending = None
        return self._accept(dict(zip(self.codes, key)), first_frame, frame)

    def _accept(self, new: Dict[str, int], first_frame: int, frame: int) -> List[ScoreGoal]:
        goals: List[ScoreGoal] = []
        if self.state is None:
            self.timeline.append((first_frame, dict(new)))
        else:
            deltas = {code: new[code] - self.state[code] for code in self.codes}
            rises = [code for code, delta in deltas.items() if delta > 0]
            drops = [code for code, delta in deltas.items() if delta < 0]
            if len(rises) == 1 and not drops and deltas[rises[0]] <= self.max_goals_per_change:
                code = rises[0]
                for _ in range(deltas[code]):
                    goal = ScoreGoal(code, first_frame, self.last_state_frame, dict(self.state), dict(new))
                    goals.append(goal)
                    self.goals.append(goal)
            elif len(drops) == 1 and not rises:
                # A goal taken back (VAR). Withdraw the most recent goal for
                # that team rather than report a score that no longer stands.
                code = drops[0]
                for goal in reversed(self.goals):
                    if goal.code == code:
                        self.goals.remove(goal)
                        break
                self.rejected.append((first_frame, f"{code} score fell to {new[code]}; last goal withdrawn"))
            else:
                self.rejected.append((first_frame, f"implausible change {self.state} -> {new}; re-based"))
            self.timeline.append((first_frame, dict(new)))
        self.state = new
        self.last_state_frame = frame
        return goals


class ScoreboardReader:
    def __init__(
        self,
        ocr_fn: OcrFn,
        fps: float,
        frame_shape: Tuple[int, int],
        expected_codes: Optional[Iterable[str]] = None,
        read_every_seconds: float = 1.0,
        scan_every_seconds: float = 2.0,
        lost_after_seconds: float = 60.0,
        confirm_reads: int = 2,
    ) -> None:
        self.ocr_fn = ocr_fn
        self.fps = max(float(fps), 1.0)
        self.height, self.width = int(frame_shape[0]), int(frame_shape[1])
        self.expected = sorted({code.upper() for code in expected_codes}) if expected_codes else None
        self.read_every = max(1, int(round(read_every_seconds * self.fps)))
        self.scan_every = max(1, int(round(scan_every_seconds * self.fps)))
        self.lost_after = int(round(lost_after_seconds * self.fps))
        self.scale = upscale_for(self.height)
        self.tracker = ScoreTracker(confirm_reads=confirm_reads)

        self.region: Optional[Tuple[int, int, int, int]] = None
        self.layout: Optional[List[Tuple[str, Box]]] = None
        self.last_seen: Optional[int] = None
        self.locked_at: Optional[int] = None
        self._next_frame = 0
        self.ocr_calls = 0
        self.scans = 0

    # ------------------------------------------------------------------ OCR
    def _tokens(self, image: np.ndarray, offset: Tuple[int, int], scale: float) -> List[Token]:
        if scale != 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        self.ocr_calls += 1
        tokens = []
        for text, confidence, (x1, y1, x2, y2) in self.ocr_fn(image):
            if confidence < MIN_TOKEN_CONFIDENCE:
                continue
            tokens.append(
                Token(
                    text,
                    confidence,
                    (x1 / scale + offset[0], y1 / scale + offset[1], x2 / scale + offset[0], y2 / scale + offset[1]),
                )
            )
        return tokens

    def _scan(self, frame: np.ndarray):
        """(reading, codes) from the top and bottom strips; either may be None."""
        self.scans += 1
        strips = (
            (0, int(self.height * TOP_STRIP)),
            (int(self.height * (1.0 - BOTTOM_STRIP)), self.height),
        )
        codes_only = None
        for top, bottom in strips:
            tokens = self._tokens(frame[top:bottom], (0, top), self.scale)
            reading = parse_tokens(tokens, self.expected)
            if reading is not None:
                return reading, None
            if codes_only is None:
                codes_only = find_codes(tokens, self.expected)
        return None, codes_only

    def _read_region(self, frame: np.ndarray) -> Optional[ScoreReading]:
        x1, y1, x2, y2 = self.region
        tokens = self._tokens(frame[y1:y2, x1:x2], (x1, y1), max(self.scale + 1.0, 2.0))
        return parse_tokens(tokens, self.expected or [code for code, _ in self.layout], self.layout)

    def _lock(self, region: Box, layout: List[Tuple[str, Box]], frame_number: int, codes_only: bool = False) -> None:
        x1, y1, x2, y2 = region
        code_width = max(b[2] - b[0] for _, b in layout)
        # Locked on codes alone, the digits are not in the box yet: a score
        # cell sits beside or between the codes, so widen generously.
        pad_x = (2.0 * code_width if codes_only else 0.35 * (x2 - x1)) + 8
        pad_y = 0.5 * (y2 - y1) + 6
        self.region = (
            int(max(0, x1 - pad_x)),
            int(max(0, y1 - pad_y)),
            int(min(self.width, x2 + pad_x)),
            int(min(self.height, y2 + pad_y)),
        )
        self.layout = list(layout)
        self.locked_at = frame_number
        self.last_seen = None

    # --------------------------------------------------------------- update
    def update(self, frame_number: int, frame: np.ndarray) -> Optional[List[ScoreGoal]]:
        if frame_number < self._next_frame:
            return None
        if self.region is None:
            self._next_frame = frame_number + self.scan_every
            reading, codes = self._scan(frame)
            if reading is not None:
                self._lock(reading.region, list(zip(reading.codes, reading.code_boxes)), frame_number)
            elif codes is not None:
                region = (
                    min(b[0] for _, b in codes), min(b[1] for _, b in codes),
                    max(b[2] for _, b in codes), max(b[3] for _, b in codes),
                )
                self._lock(region, codes, frame_number, codes_only=True)
                self._next_frame = frame_number + 1
                return None
            else:
                return None
        else:
            self._next_frame = frame_number + self.read_every
            reading = self._read_region(frame)
            if reading is None:
                if self.last_seen is None:
                    if frame_number - self.locked_at > CODES_ONLY_GRACE_SECONDS * self.fps:
                        self.region, self.layout = None, None
                elif frame_number - self.last_seen > self.lost_after:
                    self.region, self.layout = None, None
                return None
        self.last_seen = frame_number
        return self.tracker.observe(frame_number, reading)

    @property
    def active(self) -> bool:
        """True once a score has been read and confirmed at least once."""
        return self.tracker.state is not None

    def summary(self) -> dict:
        return {
            "codes": list(self.tracker.codes) if self.tracker.codes else None,
            "confirmed_readings": self.tracker.readings,
            "ocr_calls": self.ocr_calls,
            "scans": self.scans,
            "timeline": [(frame, dict(score)) for frame, score in self.tracker.timeline],
            "goals": [
                {"code": goal.code, "frame": goal.frame, "window_start": goal.window_start}
                for goal in self.tracker.goals
            ],
            "rejected": list(self.tracker.rejected),
        }
