"""Kit-colour sampling, clustering, and representative-image rendering.

The colour of a kit is read as a masked median over a torso region rather than a
per-crop KMeans fit. That is deterministic (the old fit used ``n_init=1`` with no
seed, so the same frame produced a different colour between runs) and roughly two
orders of magnitude cheaper.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans


# Torso window, as a fraction of the player box. Excludes the head, the lower arms,
# the shorts, and the background wedges a full-width slice picks up at the shoulders.
TORSO_TOP = 0.18
TORSO_BOTTOM = 0.55
TORSO_INSET = 0.22

# Pitch green in OpenCV's 0-179 hue scale, plus washed-out and crushed/blown pixels.
GRASS_HUE_LOW = 35
GRASS_HUE_HIGH = 85
GRASS_MIN_SATURATION = 55
MIN_SATURATION = 40
MIN_VALUE = 40
MAX_VALUE = 235

# Above this saturation a kit is treated as genuinely coloured rather than a
# white/black/grey strip, and only its chromatic pixels define the kit colour.
CHROMA_SATURATION = 60


@dataclass
class TeamSample:
    """One observed player: kit colour plus enough context to render an image."""

    color: np.ndarray
    crop: np.ndarray
    frame: Optional[np.ndarray] = None
    bbox: Optional[Sequence[float]] = None
    sharpness: float = 0.0

    @property
    def area(self) -> int:
        return int(self.crop.shape[0]) * int(self.crop.shape[1])

    @property
    def presentation_score(self) -> float:
        """How good this sample is as evidence to show a person.

        Size and sharpness alone pick bad images: a 38x245 sliver of a player
        clipped by the frame edge scores higher than a whole player in the clear.
        Sharpness is capped so it cannot dominate, and implausible aspect ratios
        and frame-edge boxes - both signs of a clipped or merged detection - are
        penalised.
        """
        height, width = self.crop.shape[:2]
        if height <= 0 or width <= 0:
            return 0.0

        aspect = width / float(height)
        # A standing player is roughly 0.3-0.9 as wide as they are tall.
        aspect_penalty = 1.0 if 0.28 <= aspect <= 0.95 else 0.25

        edge_penalty = 1.0
        if self.frame is not None and self.bbox is not None:
            frame_height, frame_width = self.frame.shape[:2]
            x1, y1, x2, y2 = [int(value) for value in self.bbox]
            if x1 <= 2 or y1 <= 2 or x2 >= frame_width - 2 or y2 >= frame_height - 2:
                edge_penalty = 0.2

        sharpness_bonus = 1.0 + min(float(self.sharpness), 1500.0) / 1500.0
        return float(height * width) * sharpness_bonus * aspect_penalty * edge_penalty


def _clamp_box(frame: np.ndarray, bbox: Sequence[float]) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    return x1, y1, x2, y2


def safe_player_crop(
    frame: np.ndarray,
    bbox: Sequence[float],
    min_height: int = 40,
    min_width: int = 20,
) -> np.ndarray | None:
    x1, y1, x2, y2 = _clamp_box(frame, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < min_height or crop.shape[1] < min_width:
        return None
    return crop.copy()


def crop_sharpness(crop: np.ndarray) -> float:
    """Variance of the Laplacian - higher is sharper. Uses the fast 32-bit path."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] > 128:
        scale = 128.0 / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def kit_color_bgr(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
    """Median BGR of the torso, ignoring pitch green, shadow, and blown highlights."""
    x1, y1, x2, y2 = _clamp_box(frame, bbox)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width < 4 or box_height < 8:
        return None

    tx1 = x1 + int(box_width * TORSO_INSET)
    tx2 = x2 - int(box_width * TORSO_INSET)
    ty1 = y1 + int(box_height * TORSO_TOP)
    ty2 = y1 + int(box_height * TORSO_BOTTOM)
    if tx2 - tx1 < 2 or ty2 - ty1 < 2:
        tx1, tx2, ty1, ty2 = x1, x2, y1, y2

    roi = frame[ty1:ty2, tx1:tx2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Hue is meaningless for near-grey pixels - OpenCV hands back an essentially
    # arbitrary value - so testing hue alone classifies a white or grey shirt as
    # grass at random. Real turf is strongly saturated green, so require that too.
    is_grass = (
        (hue >= GRASS_HUE_LOW)
        & (hue <= GRASS_HUE_HIGH)
        & (saturation >= GRASS_MIN_SATURATION)
    )
    not_grass = ~is_grass
    in_range = (value >= MIN_VALUE) & (value <= MAX_VALUE)

    total = roi.shape[0] * roi.shape[1]
    strict = not_grass & in_range & (saturation >= MIN_SATURATION)
    relaxed = not_grass & in_range

    if int(strict.sum()) >= max(12, int(total * 0.12)):
        mask, allow_chroma = strict, True
    elif int(relaxed.sum()) >= max(6, int(total * 0.05)):
        # White and black strips are legitimately low-saturation; keep them, but
        # do not let the chroma step run, or it would latch onto stray turf.
        mask, allow_chroma = relaxed, False
    elif int(in_range.sum()) > 0:
        mask, allow_chroma = in_range, False
    else:
        mask, allow_chroma = np.ones_like(not_grass, dtype=bool), False

    if int(mask.sum()) == 0:
        return None
    pixels = roi[mask].reshape(-1, 3).astype(np.float32)

    # A striped or accented kit - red-and-white, say - medians out to a washed
    # grey, leaving the two teams separable only by lightness. When the kit
    # genuinely carries chroma, keep just the chromatic pixels so the accent
    # colour survives. Plain strips have no such pixels and keep their lightness.
    if allow_chroma:
        kept_saturation = saturation[mask].reshape(-1).astype(np.float32)
        if (
            kept_saturation.size >= 8
            and float(np.percentile(kept_saturation, 75)) >= CHROMA_SATURATION
        ):
            chromatic = kept_saturation >= float(np.percentile(kept_saturation, 60))
            if int(chromatic.sum()) >= 6:
                pixels = pixels[chromatic]

    return np.median(pixels, axis=0)


def bgr_to_lab(color: Sequence[float]) -> np.ndarray:
    pixel = np.clip(np.asarray(color, dtype=np.float32), 0, 255).reshape(1, 1, 3)
    return cv2.cvtColor(pixel / 255.0, cv2.COLOR_BGR2LAB).reshape(3)


def _as_sample(item: Any) -> TeamSample:
    """Accept either a TeamSample or the legacy ``(color, crop)`` tuple."""
    if isinstance(item, TeamSample):
        return item
    color, crop = item
    return TeamSample(
        color=np.asarray(color, dtype=np.float32),
        crop=crop,
        sharpness=crop_sharpness(crop),
    )


def cluster_team_samples(
    samples: list,
    min_samples: int = 4,
    min_per_cluster: int = 1,
    min_separation: float = 0.0,
) -> tuple[list[list[int]], list]:
    """Cluster kit colours into two teams and pick one representative each.

    ``min_samples`` / ``min_per_cluster`` / ``min_separation`` let the caller demand
    a well-supported result; the defaults stay permissive so this remains a pure,
    independently testable transform.
    """
    normalized = [_as_sample(item) for item in samples]
    if len(normalized) < max(2, min_samples):
        raise ValueError(
            "Could not find enough clear player images to identify both teams."
        )

    colors = np.asarray([sample.color for sample in normalized], dtype=np.float32)
    model = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=7)
    labels = model.fit_predict(colors)
    if len(set(int(label) for label in labels)) < 2:
        raise ValueError("The two team kits could not be separated in this video.")

    grouped = {
        label: [index for index, value in enumerate(labels) if int(value) == label]
        for label in range(2)
    }
    if min(len(indices) for indices in grouped.values()) < min_per_cluster:
        raise ValueError(
            "One of the two kits was seen too few times to be identified reliably."
        )

    if min_separation > 0.0:
        separation = float(
            np.linalg.norm(
                bgr_to_lab(model.cluster_centers_[0])
                - bgr_to_lab(model.cluster_centers_[1])
            )
        )
        if separation < min_separation:
            raise ValueError(
                "The two team kits look too similar to tell apart in this video."
            )

    centers_rgb: list[list[int]] = []
    representatives: list[TeamSample] = []
    for label in range(2):
        center = model.cluster_centers_[label]
        candidates = grouped[label]
        distances = {
            index: float(np.linalg.norm(colors[index] - center)) for index in candidates
        }
        # Stay reasonably near the cluster centre, then prefer the crop that
        # shows the player best. Ranking on colour distance alone is how a
        # 22x52 px crop used to win.
        cutoff = float(np.percentile(list(distances.values()), 70))
        pool = [index for index in candidates if distances[index] <= cutoff] or candidates
        best_index = max(
            pool,
            key=lambda index: (
                normalized[index].presentation_score,
                -distances[index],
            ),
        )
        blue, green, red = np.clip(center, 0, 255).astype(int).tolist()
        centers_rgb.append([red, green, blue])
        representatives.append(normalized[best_index])
    return centers_rgb, representatives


def render_team_sample(
    sample: Any,
    output_size: tuple[int, int] = (640, 360),
    box_color: Sequence[int] = (0, 235, 255),
    context: float = 1.9,
) -> np.ndarray:
    """Render a zoomed-out view of the player with their box drawn on top.

    The old code wrote the raw bounding box straight to disk - observed sizes were
    28x46 and 22x52 px, which the app then upscaled ~25x into a 560 px card. Here
    the view is widened to a 16:9 region around the player so there is visible
    context, the player is outlined, and the result is a fixed, readable size.
    """
    sample = _as_sample(sample)
    out_width, out_height = output_size

    if sample.frame is None or sample.bbox is None:
        return cv2.resize(sample.crop, output_size, interpolation=cv2.INTER_CUBIC)

    frame = sample.frame
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_box(frame, sample.bbox)
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)

    centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    aspect = out_width / float(out_height)
    region_height = min(float(frame_height), box_height * context)
    region_width = region_height * aspect
    if region_width > frame_width:
        region_width = float(frame_width)
        region_height = min(float(frame_height), region_width / aspect)
    # Never crop tighter than the player themselves.
    if region_width < box_width:
        region_width = float(min(frame_width, box_width))
        region_height = min(float(frame_height), region_width / aspect)

    left = int(round(min(max(0.0, centre_x - region_width / 2.0), frame_width - region_width)))
    top = int(round(min(max(0.0, centre_y - region_height / 2.0), frame_height - region_height)))
    right = int(round(min(float(frame_width), left + region_width)))
    bottom = int(round(min(float(frame_height), top + region_height)))

    region = frame[top:bottom, left:right]
    if region.size == 0:
        return cv2.resize(sample.crop, output_size, interpolation=cv2.INTER_CUBIC)

    interpolation = cv2.INTER_AREA if region.shape[0] > out_height else cv2.INTER_CUBIC
    canvas = cv2.resize(region, output_size, interpolation=interpolation)

    scale_x = out_width / float(region.shape[1])
    scale_y = out_height / float(region.shape[0])
    box = (
        int(round((x1 - left) * scale_x)),
        int(round((y1 - top) * scale_y)),
        int(round((x2 - left) * scale_x)),
        int(round((y2 - top) * scale_y)),
    )
    cv2.rectangle(canvas, box[:2], box[2:], tuple(int(c) for c in box_color), 3)
    return canvas
