"""Find the goal frame (two posts and the crossbar) near a goalkeeper.

No model in this repository has a "goal" class, so this is classical image
work: thin bright low-saturation structures, joined into near-vertical posts,
paired by a white crossbar between their tops. Searched only around the
goalkeeper - the goal is behind him - with posts sized against his height
(a goal is 2.44 m high, a keeper about 1.9 m), at least one post standing on
grass, and a mouth that is not a solid white board.

Measured 2026-10-08 on the three local clips (EA FC Mobile goal, 1080p
Porto v Man City, 480p Real Madrid v Atletico): a goal frame was returned on
about two thirds of the frames with a keeper in view, usually in the right
place but often too wide (stretched onto an advertising board); close-ups
showing only one post find nothing. It is therefore used only as supporting
evidence for "the ball went in", with a margin, never alone.
"""

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
Quad = Tuple[Point, Point, Point, Point]  # top-left, top-right, bottom-right, bottom-left

POST_MIN_KEEPER_HEIGHTS = 0.7
POST_MAX_KEEPER_HEIGHTS = 2.6
CROSSBAR_MIN_COVERAGE = 0.8


def _masks(image: np.ndarray, frame_height: int) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    k = max(7, int(frame_height / 40)) | 1
    background = cv2.morphologyEx(value, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    white = (
        (value.astype(np.int16) - background > 35) & (hsv[:, :, 1] < 80) & (value > 140)
    ).astype(np.uint8) * 255
    thin = max(2, int(round(frame_height / 360)))
    vertical = cv2.morphologyEx(
        white, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (thin, max(3, int(frame_height * 0.02)))),
    )
    return white, vertical


def _coverage(mask: np.ndarray, a: Point, b: Point, samples: int = 40, radius: int = 2) -> float:
    height, width = mask.shape
    hits = 0
    for s in np.linspace(0.0, 1.0, samples):
        x = int(round(a[0] + s * (b[0] - a[0])))
        y = int(round(a[1] + s * (b[1] - a[1])))
        patch = mask[max(0, y - radius):min(height, y + radius + 1), max(0, x - radius):min(width, x + radius + 1)]
        hits += bool(patch.any())
    return hits / float(samples)


def _posts(vertical: np.ndarray, frame_height: int) -> List[Tuple[Point, Point, float]]:
    lines = cv2.HoughLinesP(
        vertical, 1, np.pi / 180,
        threshold=int(frame_height * 0.02),
        minLineLength=int(frame_height * 0.035),
        maxLineGap=int(frame_height * 0.03),
    )
    segments = []
    for line in [] if lines is None else lines:
        x1, y1, x2, y2 = map(float, line[0])
        if abs(x2 - x1) > 0.27 * abs(y2 - y1):  # more than ~15 degrees off vertical
            continue
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        segments.append([x1, y1, x2, y2])
    # A post broken by a body or the net comes back in pieces: join collinear ones.
    segments.sort(key=lambda s: s[1])
    groups: List[List[float]] = []
    for s in segments:
        for g in groups:
            mid = (s[1] + s[3]) / 2.0
            gx = g[0] + (g[2] - g[0]) * ((mid - g[1]) / max(1.0, g[3] - g[1]))
            if abs(gx - (s[0] + s[2]) / 2.0) <= max(3.0, 0.01 * frame_height) and s[1] <= g[3] + 0.25 * frame_height:
                if s[3] > g[3]:
                    g[2], g[3] = s[2], s[3]
                if s[1] < g[1]:
                    g[0], g[1] = s[0], s[1]
                break
        else:
            groups.append(list(s))
    return [((g[0], g[1]), (g[2], g[3]), g[3] - g[1]) for g in groups]


def _on_grass(grass: np.ndarray, point: Point, radius: int) -> bool:
    x, y = int(point[0]), int(point[1])
    height, width = grass.shape
    patch = grass[max(0, y):min(height, y + radius), max(0, x - radius):min(width, x + radius)]
    return patch.size > 0 and patch.mean() / 255.0 >= 0.35


def find_goal_mouth(frame: np.ndarray, keeper_bbox: Sequence[float]) -> Optional[Quad]:
    """The goal frame near this keeper, in frame coordinates, or None."""
    frame_height, frame_width = frame.shape[:2]
    kx1, ky1, kx2, ky2 = (float(v) for v in keeper_bbox)
    keeper_height = max(8.0, ky2 - ky1)
    centre_x = (kx1 + kx2) / 2.0
    x0 = int(max(0, centre_x - 7 * keeper_height))
    x1 = int(min(frame_width, centre_x + 7 * keeper_height))
    y0 = int(max(0, ky1 - 2.5 * keeper_height))
    y1 = int(min(frame_height, ky2 + 1.5 * keeper_height))
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    white, vertical = _masks(roi, frame_height)
    grass = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), (30, 50, 40), (90, 255, 255))
    widened = cv2.dilate(white, np.ones((3, 3), np.uint8))
    posts = _posts(vertical, frame_height)
    radius = max(3, int(0.15 * keeper_height))
    keeper_x = centre_x - x0

    best = None
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            (t1, b1, l1), (t2, b2, l2) = posts[i], posts[j]
            if t1[0] > t2[0]:
                (t1, b1, l1), (t2, b2, l2) = (t2, b2, l2), (t1, b1, l1)
            longest = max(l1, l2)
            if not (POST_MIN_KEEPER_HEIGHTS * keeper_height <= longest <= POST_MAX_KEEPER_HEIGHTS * keeper_height):
                continue
            if min(l1, l2) < 0.4 * longest:
                continue
            span = math.dist(t1, t2)
            if span < 0.6 * longest or span > 6 * longest:
                continue
            if abs(math.degrees(math.atan2(t2[1] - t1[1], t2[0] - t1[0]))) > 60:
                continue
            if not (_on_grass(grass, b1, radius) or _on_grass(grass, b2, radius)):
                continue
            bar = _coverage(widened, t1, t2)
            if bar < CROSSBAR_MIN_COVERAGE:
                continue
            top_mid = ((t1[0] + t2[0]) / 2.0, (t1[1] + t2[1]) / 2.0 + 0.25 * longest)
            bottom_mid = ((b1[0] + b2[0]) / 2.0, (b1[1] + b2[1]) / 2.0 - 0.1 * longest)
            if _coverage(white, top_mid, bottom_mid, radius=0) > 0.8:
                continue  # a white board, not a goal mouth
            if not (min(t1[0], b1[0]) - 1.5 * keeper_height <= keeper_x <= max(t2[0], b2[0]) + 1.5 * keeper_height):
                continue  # the keeper stands in or in front of his goal
            score = bar * (l1 + l2) * min(span, 3 * longest)
            if best is None or score > best[0]:
                best = (score, t1, t2, b2, b1)
    if best is None:
        return None
    return tuple((float(p[0]) + x0, float(p[1]) + y0) for p in best[1:])  # type: ignore[return-value]


def mouth_height(quad: Quad) -> float:
    (tl, tr, br, bl) = quad
    return max(1.0, (math.dist(tl, bl) + math.dist(tr, br)) / 2.0)


def in_mouth(quad: Quad, point: Point, margin_heights: float = 0.15) -> bool:
    """Is the point inside the goal frame, grown by a margin of post heights?"""
    margin = margin_heights * mouth_height(quad)
    contour = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True) >= -margin
