"""Turn uploaded photos of a player into the full-body crops the gallery stores."""

from typing import List, Sequence, Tuple

import numpy as np

from .embedder import MIN_CROP_HEIGHT

PERSON_CLASS_NAMES = ("player", "goalkeeper")


def largest_person_crop(image: np.ndarray, detector, confidence: float = 0.25):
    """The biggest player/goalkeeper box in a photo, or None."""
    names = getattr(detector, "names", {}) or {}
    wanted = {i for i, n in names.items() if str(n).lower() in PERSON_CLASS_NAMES}
    result = detector.predict(image, conf=confidence, verbose=False)[0]
    best, best_area = None, 0.0
    for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        if wanted and int(cls) not in wanted:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best, best_area = box, area
    if best is None:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in best]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if y2 - y1 < MIN_CROP_HEIGHT:
        return None
    return image[y1:y2, x1:x2].copy()


def crops_from_images(images: Sequence[np.ndarray], detector=None) -> Tuple[List[np.ndarray], List[int]]:
    """(crops, indices of rejected images).

    With a detector, each photo is cut to its largest player box. A photo with
    no detected player is kept as it is only if it already looks like one
    person (taller than wide, at least MIN_CROP_HEIGHT px); otherwise rejected.
    """
    crops, rejected = [], []
    for index, image in enumerate(images):
        crop = largest_person_crop(image, detector) if detector is not None else None
        if crop is None:
            h, w = image.shape[:2]
            crop = image if (h >= MIN_CROP_HEIGHT and h > w) else None
        if crop is None:
            rejected.append(index)
        else:
            crops.append(crop)
    return crops, rejected
