"""Appearance embeddings for player crops (OSNet, 512-d, L2-normalised)."""

import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

from .osnet import INPUT_SIZE, load_osnet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REID_WEIGHTS = os.path.join(ROOT, "models", "reid", "osnet_x1_0_msmt17.pt")
MODEL_NAME = "osnet_x1_0_msmt17"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Below this box height a crop holds a kit colour and little else: appearance
# cannot tell team-mates apart there, so it is not used as evidence.
MIN_CROP_HEIGHT = 40


def body_crop(frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
    """The player's full-body crop, or None when too small to identify."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if y2 - y1 < MIN_CROP_HEIGHT or x2 - x1 < MIN_CROP_HEIGHT // 4:
        return None
    return frame[y1:y2, x1:x2].copy()


def crop_quality(crop: np.ndarray) -> float:
    """0.2-1.0: taller crops carry more identity than 40 px silhouettes."""
    return float(np.clip((crop.shape[0] - MIN_CROP_HEIGHT) / 120.0, 0.2, 1.0))


def augment(crop: np.ndarray) -> List[np.ndarray]:
    """Extra views of one enrolment image: mirror, tighter/looser framing, light."""
    h, w = crop.shape[:2]
    views = [crop, cv2.flip(crop, 1)]
    dy, dx = max(1, h // 10), max(1, w // 10)
    views.append(crop[dy:h - dy, dx:w - dx])
    views.append(cv2.copyMakeBorder(crop, dy, dy, dx, dx, cv2.BORDER_REPLICATE))
    views.append(cv2.convertScaleAbs(crop, alpha=1.15, beta=-15))
    views.append(cv2.convertScaleAbs(crop, alpha=0.85, beta=15))
    return views


class ReidEmbedder:
    def __init__(self, model, name: str = MODEL_NAME) -> None:
        self.model = model
        self.name = name
        self.calls = 0

    @classmethod
    def load(cls, weights_path: str = DEFAULT_REID_WEIGHTS) -> "ReidEmbedder":
        return cls(load_osnet(weights_path))

    @staticmethod
    def _prepare(crop: np.ndarray) -> np.ndarray:
        height, width = INPUT_SIZE
        rgb = cv2.cvtColor(cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        return ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)

    def embed(self, crops: Sequence[np.ndarray], batch_size: int = 32) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        out = []
        with torch.no_grad():
            for start in range(0, len(crops), batch_size):
                batch = np.stack([self._prepare(c) for c in crops[start:start + batch_size]])
                out.append(self.model(torch.from_numpy(batch)).numpy())
        self.calls += len(crops)
        features = np.concatenate(out).astype(np.float32)
        return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-6)
