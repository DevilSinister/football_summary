"""Run the spotting model over a video as it is decoded.

Frames are taken from the pipeline's own sequential decode (no second pass),
sampled at ``sample_fps``, resized to 224 px high (the SoccerNet frames the
model was trained on are 398x224) and normalised with ImageNet statistics.
They are scored in windows of ``clip_len`` frames that overlap by half; each
sample's scores are the mean over the windows that contain it.

Events are per-class peaks: a sample whose score is the highest within
``nms_seconds`` either side and at least ``min_confidence``.
"""

from collections import deque
from time import perf_counter
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

from .labels import event_name

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FRAME_HEIGHT = 224
# E2E-Spot extracts SoccerNet-v2 frames at 2 fps (its data/soccernetv2
# README); clip_len 100 is therefore 50 s of context. Measured 2026-10-14 on the
# fixture clips: at 12.5 or 25 fps every output stays weak (best 0.6), at 2 fps
# the model is confident (0.85-0.96 peaks).
DEFAULT_SAMPLE_FPS = 2.0


def prepare_frame(frame_bgr: np.ndarray, height: int = FRAME_HEIGHT) -> np.ndarray:
    """BGR uint8 frame -> RGB uint8, ``height`` px high, aspect kept (even width)."""
    h, w = frame_bgr.shape[:2]
    width = max(2, int(round(w * height / float(h) / 2.0)) * 2)
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def to_tensor(frames: Sequence[np.ndarray]) -> torch.Tensor:
    """List of RGB uint8 frames -> (1, T, 3, H, W) normalised float tensor."""
    stack = np.stack(frames).astype(np.float32) / 255.0
    stack = (stack - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(stack).permute(0, 3, 1, 2).unsqueeze(0).contiguous()


class ActionSpotter:
    def __init__(
        self,
        model,
        labels: Sequence[str],
        source_fps: float,
        sample_fps: float = DEFAULT_SAMPLE_FPS,
        clip_len: int = 100,
        overlap: float = 0.5,
    ) -> None:
        self.model = model
        self.labels = list(labels)
        self.source_fps = max(float(source_fps), 1.0)
        self.sample_fps = min(float(sample_fps), self.source_fps)
        self.clip_len = int(clip_len)
        self.stride = max(1, int(round(self.clip_len * (1.0 - overlap))))
        self._step = self.source_fps / self.sample_fps
        self._next_sample = 0.0
        self._buffer: deque = deque(maxlen=self.clip_len)
        self._buffer_start = 0  # sample index of _buffer[0]
        self.sample_frames: List[int] = []
        self._score_sum: List[np.ndarray] = []
        self._score_count: List[int] = []
        self._new_since_run = 0
        self.inference_seconds = 0.0

    def wants(self, frame_number: int) -> bool:
        return frame_number >= self._next_sample - 1e-6

    def add_frame(self, frame_number: int, frame_bgr: np.ndarray) -> None:
        """Offer every decoded frame; only the sampled ones are kept."""
        if not self.wants(frame_number):
            return
        self._next_sample += self._step
        if len(self._buffer) == self.clip_len:
            self._buffer_start += 1
        self._buffer.append(prepare_frame(frame_bgr))
        self.sample_frames.append(int(frame_number))
        self._score_sum.append(np.zeros(len(self.labels) + 1, dtype=np.float64))
        self._score_count.append(0)
        self._new_since_run += 1
        if len(self._buffer) == self.clip_len and (
            self._new_since_run >= self.stride or len(self.sample_frames) == self.clip_len
        ):
            self._run_window()

    def _run_window(self) -> None:
        started = perf_counter()
        with torch.no_grad():
            logits = self.model(to_tensor(list(self._buffer)))[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        self.inference_seconds += perf_counter() - started
        for offset, row in enumerate(probs):
            index = self._buffer_start + offset
            self._score_sum[index] += row
            self._score_count[index] += 1
        self._new_since_run = 0

    def finish(self) -> None:
        """Score the samples no window has covered yet."""
        if self.sample_frames and (self._new_since_run > 0 or 0 in self._score_count):
            self._run_window()

    def scores(self) -> np.ndarray:
        """(samples, classes + 1) mean probabilities; column 0 is background."""
        if not self.sample_frames:
            return np.zeros((0, len(self.labels) + 1))
        counts = np.maximum(np.asarray(self._score_count, dtype=np.float64), 1.0)[:, None]
        return np.stack(self._score_sum) / counts

    def events(
        self,
        min_confidence: float = 0.2,
        nms_seconds: float = 1.0,
        classes: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        scores = self.scores()
        if not len(scores):
            return []
        radius = max(1, int(round(nms_seconds * self.sample_fps)))
        wanted = set(classes) if classes is not None else None
        found: List[Dict] = []
        for column, label in enumerate(self.labels, start=1):
            if wanted is not None and label not in wanted:
                continue
            curve = scores[:, column]
            for index in np.flatnonzero(curve >= min_confidence):
                low, high = max(0, index - radius), min(len(curve), index + radius + 1)
                if curve[index] < curve[low:high].max() or (curve[low:index] == curve[index]).any():
                    continue  # not the peak (ties keep the first)
                found.append(
                    {
                        "event": event_name(label),
                        "frame": int(self.sample_frames[index]),
                        "end_frame": int(self.sample_frames[index]),
                        "confidence": round(float(curve[index]), 4),
                        "source": "spotting",
                        "participants": [],
                        "details": {"spotting_label": label},
                    }
                )
        found.sort(key=lambda event: event["frame"])
        return found
