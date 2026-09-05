from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans


PlayerSample = tuple[np.ndarray, np.ndarray]


def safe_player_crop(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 20:
        return None
    return crop.copy()


def cluster_team_samples(
    samples: list[PlayerSample],
) -> tuple[list[list[int]], list[np.ndarray]]:
    """Cluster BGR kit colors and choose one representative player per cluster."""
    if len(samples) < 2:
        raise ValueError("Could not find enough clear player images to identify both teams.")
    colors = np.asarray([color for color, _ in samples], dtype=np.float32)
    model = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=7)
    labels = model.fit_predict(colors)
    if len(set(int(label) for label in labels)) < 2:
        raise ValueError("The two team kits could not be separated in this video.")

    centers_rgb: list[list[int]] = []
    representative_crops: list[np.ndarray] = []
    for label in range(2):
        indices = [index for index, value in enumerate(labels) if int(value) == label]
        center = model.cluster_centers_[label]
        best_index = min(
            indices,
            key=lambda index: (
                float(np.linalg.norm(colors[index] - center)),
                -samples[index][1].shape[0] * samples[index][1].shape[1],
            ),
        )
        blue, green, red = np.clip(center, 0, 255).astype(int).tolist()
        centers_rgb.append([red, green, blue])
        representative_crops.append(samples[best_index][1])
    return centers_rgb, representative_crops
