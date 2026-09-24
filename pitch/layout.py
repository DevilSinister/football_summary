"""The 32-keypoint pitch layout and the zones the event rules ask about.

Coordinates are metres on a 120 x 70 pitch: x runs goal line to goal line, y
touchline to touchline, (0, 0) is a corner. The keypoint order is the Roboflow
sports ``SoccerPitchConfiguration`` used by the football-field keypoint
datasets: points 1-13 belong to the left end, 14-17 and 31-32 to the halfway
line and centre circle, 18-30 to the right end.

If a dataset numbers its keypoints differently, only ``VERTICES`` has to change;
everything else reads positions in metres.
"""

from typing import Optional, Sequence, Tuple

import numpy as np


PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0
PENALTY_BOX_LENGTH = 20.15
PENALTY_BOX_WIDTH = 41.0
GOAL_BOX_LENGTH = 5.5
GOAL_BOX_WIDTH = 18.32
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0
GOAL_WIDTH = 7.32
# How deep behind the goal line a ball can sit and still be in the net.
GOAL_DEPTH = 2.5

_L, _W = PITCH_LENGTH, PITCH_WIDTH
_PBL, _PBW = PENALTY_BOX_LENGTH, PENALTY_BOX_WIDTH
_GBL, _GBW = GOAL_BOX_LENGTH, GOAL_BOX_WIDTH
_CCR, _PSD = CENTRE_CIRCLE_RADIUS, PENALTY_SPOT_DISTANCE

VERTICES = np.array(
    [
        (0, 0), (0, (_W - _PBW) / 2), (0, (_W - _GBW) / 2), (0, (_W + _GBW) / 2), (0, (_W + _PBW) / 2), (0, _W),
        (_GBL, (_W - _GBW) / 2), (_GBL, (_W + _GBW) / 2), (_PSD, _W / 2),
        (_PBL, (_W - _PBW) / 2), (_PBL, (_W - _GBW) / 2), (_PBL, (_W + _GBW) / 2), (_PBL, (_W + _PBW) / 2),
        (_L / 2, 0), (_L / 2, _W / 2 - _CCR), (_L / 2, _W / 2 + _CCR), (_L / 2, _W),
        (_L - _PBL, (_W - _PBW) / 2), (_L - _PBL, (_W - _GBW) / 2),
        (_L - _PBL, (_W + _GBW) / 2), (_L - _PBL, (_W + _PBW) / 2),
        (_L - _PSD, _W / 2), (_L - _GBL, (_W - _GBW) / 2), (_L - _GBL, (_W + _GBW) / 2),
        (_L, 0), (_L, (_W - _PBW) / 2), (_L, (_W - _GBW) / 2), (_L, (_W + _GBW) / 2), (_L, (_W + _PBW) / 2), (_L, _W),
        (_L / 2 - _CCR, _W / 2), (_L / 2 + _CCR, _W / 2),
    ],
    dtype=np.float32,
)

# Line segments between keypoints (1-based), for drawing a calibration overlay.
EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
    (16, 17), (18, 19), (19, 20), (20, 21), (23, 24), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17), (14, 25), (18, 26), (23, 27), (24, 28),
    (21, 29), (17, 30),
]

Point = Tuple[float, float]


def end_of(point: Point) -> str:
    """Which half a point is in: 'left' (towards x=0) or 'right'."""
    return "left" if float(point[0]) < PITCH_LENGTH / 2.0 else "right"


def goal_centre(side: str) -> Point:
    return (0.0 if side == "left" else PITCH_LENGTH, PITCH_WIDTH / 2.0)


def penalty_spot(side: str) -> Point:
    x = PENALTY_SPOT_DISTANCE if side == "left" else PITCH_LENGTH - PENALTY_SPOT_DISTANCE
    return (x, PITCH_WIDTH / 2.0)


def in_penalty_box(point: Point) -> Optional[str]:
    """'left' or 'right' when the point is inside that penalty area, else None."""
    x, y = float(point[0]), float(point[1])
    if abs(y - PITCH_WIDTH / 2.0) > PENALTY_BOX_WIDTH / 2.0:
        return None
    if 0.0 <= x <= PENALTY_BOX_LENGTH:
        return "left"
    if PITCH_LENGTH - PENALTY_BOX_LENGTH <= x <= PITCH_LENGTH:
        return "right"
    return None


def in_wide_channel(point: Point) -> bool:
    """Between the touchline and the side of the penalty area."""
    return abs(float(point[1]) - PITCH_WIDTH / 2.0) > PENALTY_BOX_WIDTH / 2.0


def attacking_third(point: Point) -> Optional[str]:
    """'left' or 'right' when the point is within 40 m of that goal line."""
    x = float(point[0])
    if x <= PITCH_LENGTH / 3.0:
        return "left"
    if x >= PITCH_LENGTH * 2.0 / 3.0:
        return "right"
    return None


def in_goal(point: Point, post_margin: float = 0.3) -> Optional[str]:
    """'left' or 'right' when a ground position lies behind that goal line
    between the posts, and no deeper than the net."""
    x, y = float(point[0]), float(point[1])
    if abs(y - PITCH_WIDTH / 2.0) > GOAL_WIDTH / 2.0 + post_margin:
        return None
    if -GOAL_DEPTH <= x < 0.0:
        return "left"
    if PITCH_LENGTH < x <= PITCH_LENGTH + GOAL_DEPTH:
        return "right"
    return None


def within_pitch(point: Point, margin: float = 3.0) -> bool:
    x, y = float(point[0]), float(point[1])
    return -margin <= x <= PITCH_LENGTH + margin and -margin <= y <= PITCH_WIDTH + margin


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))
