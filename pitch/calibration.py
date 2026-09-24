"""Image-to-pitch homography from the pitch keypoint model.

A homography fitted to four points always fits them exactly, so a small
reprojection error proves nothing on its own. A frame is accepted only when
several keypoints independently agree:

* at least ``min_inliers`` keypoints survive RANSAC at 1 m tolerance,
* they are at least ``min_inlier_share`` of the confident keypoints,
* the refit on those inliers leaves a median residual under ``max_error_m``,
* the bottom-centre of the image (where the camera looks) lands on the pitch.

Measured on 2026-09-23 with ``models/football_field_best.pt`` (trained with
mosaic on): keypoint confidences never exceeded 0.35 and RANSAC kept only the
minimal four points, so no frame passes. The gate is what keeps those fits from
steering the event rules; a model trained without mosaic should clear it.

The model runs every ``stride`` frames and the last accepted homography is held
for ``max_age_frames``. A broadcast camera pans slowly enough for that.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .layout import VERTICES, within_pitch


@dataclass
class PitchCalibration:
    """An accepted image-to-pitch mapping for one keyframe."""

    frame: int
    homography: np.ndarray  # image px -> pitch metres
    inliers: int
    median_error_m: float

    def to_pitch(self, point: Sequence[float]) -> Optional[Tuple[float, float]]:
        pts = np.asarray([[point]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pts, self.homography)
        x, y = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (x, y)

    def to_image(self, point: Sequence[float]) -> Optional[Tuple[float, float]]:
        inverse = np.linalg.inv(self.homography)
        pts = np.asarray([[point]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pts, inverse)
        x, y = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (x, y)


def fit_calibration(
    frame_number: int,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    frame_shape: Tuple[int, int],
    keypoint_conf: float = 0.5,
    min_inliers: int = 6,
    min_inlier_share: float = 0.6,
    max_error_m: float = 1.0,
    vertices: np.ndarray = VERTICES,
) -> Tuple[Optional[PitchCalibration], str]:
    """Fit and judge one frame. Returns (calibration or None, reason)."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    xy = np.asarray(keypoints_xy, dtype=np.float32).reshape(-1, 2)
    conf = np.asarray(keypoints_conf, dtype=np.float32).reshape(-1)
    count = min(len(xy), len(vertices), len(conf))
    xy, conf, targets = xy[:count], conf[:count], np.asarray(vertices, dtype=np.float32)[:count]

    # Off-screen keypoints come back clamped to the frame edge; they are guesses.
    inside = (xy[:, 0] > 2) & (xy[:, 1] > 2) & (xy[:, 0] < width - 2) & (xy[:, 1] < height - 2)
    mask = (conf >= keypoint_conf) & inside
    if int(mask.sum()) < min_inliers:
        return None, "too_few_keypoints"

    source, target = xy[mask], targets[mask]
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 1.0)
    if homography is None or inlier_mask is None:
        return None, "no_homography"
    inlier_mask = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < min_inliers:
        return None, "too_few_inliers"
    if inliers / float(len(source)) < min_inlier_share:
        return None, "keypoints_disagree"

    refit, _ = cv2.findHomography(source[inlier_mask], target[inlier_mask], 0)
    if refit is None or abs(float(np.linalg.det(refit))) < 1e-12:
        return None, "degenerate"
    projected = cv2.perspectiveTransform(source[inlier_mask].reshape(-1, 1, 2), refit).reshape(-1, 2)
    residuals = np.linalg.norm(projected - target[inlier_mask], axis=1)
    median_error = float(np.median(residuals))
    if median_error > max_error_m:
        return None, "residual_too_large"

    calibration = PitchCalibration(frame_number, refit, inliers, round(median_error, 3))
    looked_at = calibration.to_pitch((width / 2.0, height * 0.9))
    if looked_at is None or not within_pitch(looked_at, margin=10.0):
        return None, "camera_off_pitch"
    return calibration, "ok"


class PitchCalibrator:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model=None,
        stride: int = 5,
        max_age_frames: Optional[int] = None,
        image_size: int = 640,
        detection_conf: float = 0.25,
        keypoint_conf: float = 0.5,
        min_inliers: int = 6,
        min_inlier_share: float = 0.6,
        max_error_m: float = 1.0,
    ) -> None:
        if model is None and model_path is not None:
            from ultralytics import YOLO

            model = YOLO(model_path)
        self.model = model
        self.stride = max(1, int(stride))
        self.max_age_frames = int(max_age_frames) if max_age_frames is not None else self.stride * 3
        self.image_size = int(image_size)
        self.detection_conf = float(detection_conf)
        self.keypoint_conf = float(keypoint_conf)
        self.min_inliers = int(min_inliers)
        self.min_inlier_share = float(min_inlier_share)
        self.max_error_m = float(max_error_m)

        self.current: Optional[PitchCalibration] = None
        self.keyframes = 0
        self.accepted = 0
        self.frames_calibrated = 0
        self.frames_seen = 0
        self.reasons: Counter = Counter()

    def _keypoints(self, frame: np.ndarray):
        result = self.model.predict(frame, imgsz=self.image_size, conf=self.detection_conf, verbose=False)[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return None, None
        # The pitch is one object; a model trained with mosaic returns several
        # "pitches" per frame. Keep the most confident.
        best = int(np.argmax(result.boxes.conf.cpu().numpy()))
        xy = result.keypoints.xy[best].cpu().numpy()
        conf = result.keypoints.conf
        conf = conf[best].cpu().numpy() if conf is not None else np.ones(len(xy), dtype=np.float32)
        return xy, conf

    def update_from_keypoints(
        self,
        frame_number: int,
        keypoints_xy,
        keypoints_conf,
        frame_shape: Tuple[int, int],
    ) -> Optional[PitchCalibration]:
        self.keyframes += 1
        if keypoints_xy is None:
            self.reasons["no_pitch_detected"] += 1
        else:
            calibration, reason = fit_calibration(
                frame_number,
                keypoints_xy,
                keypoints_conf,
                frame_shape,
                keypoint_conf=self.keypoint_conf,
                min_inliers=self.min_inliers,
                min_inlier_share=self.min_inlier_share,
                max_error_m=self.max_error_m,
            )
            self.reasons[reason] += 1
            if calibration is not None:
                self.current = calibration
                self.accepted += 1
        return self.valid_for(frame_number)

    def reset(self) -> None:
        """The camera cut to another shot: the last homography is meaningless."""
        self.current = None

    def valid_for(self, frame_number: int) -> Optional[PitchCalibration]:
        if self.current is None or frame_number - self.current.frame > self.max_age_frames:
            return None
        return self.current

    def update(self, frame_number: int, frame: np.ndarray) -> Optional[PitchCalibration]:
        self.frames_seen += 1
        if frame_number % self.stride == 0 and self.model is not None:
            xy, conf = self._keypoints(frame)
            calibration = self.update_from_keypoints(frame_number, xy, conf, frame.shape[:2])
        else:
            calibration = self.valid_for(frame_number)
        if calibration is not None:
            self.frames_calibrated += 1
        return calibration

    def summary(self) -> dict:
        return {
            "keyframes": self.keyframes,
            "accepted_keyframes": self.accepted,
            "frames_seen": self.frames_seen,
            "frames_calibrated": self.frames_calibrated,
            "rejections": dict(self.reasons),
        }
