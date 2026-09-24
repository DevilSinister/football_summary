"""Plan highlight clips: one clip per passage of play, not one per event.

A goal, the shot before it and the save before that used to be three files of
the same ten seconds, each a fixed +-5 s around its event that started and
stopped mid-shot. Here every clip-worthy event proposes a window, the window
is snapped to the camera shots around it, and windows that overlap or nearly
touch become one clip.

Windows, in frames:

- start: the first frame of the latest camera shot that began between
  LEAD_MAX and LEAD_MIN seconds before the event - the start of the play as
  the broadcaster framed it. With no cut in that range (one continuous live
  shot, or amateur footage) the window starts LEAD_DEFAULT before.
- end: for most events, the last frame of the first shot that closes between
  TAIL_MIN and TAIL_MAX after the event, else TAIL_DEFAULT after it. For a
  goal, the last frame of the last shot that closes within GOAL_TAIL_MAX, so
  the celebration and the replay that follow the live goal stay in the clip.

A merge that would run past MAX_CLIP_SECONDS is refused; the later window then
starts where the earlier clip ends, so no footage is written twice.
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

# Events that earn a highlight clip. Anything not listed here is still
# detected, stored in the events table and counted in the job result - it just
# gets no clip:
#   - pass: a match has hundreds.
#   - save: recorded only, by the owner's decision (2026-09-24). A save is
#     almost always the end of a shot, and the shot's clip already shows it.
# To clip saves again, add "save" back to this set and to EVENT_PRIORITY.
CLIP_EVENT_TYPES = frozenset(
    {"goal", "foul", "shot", "cross", "penalty", "yellow_card", "red_card"}
)

# Highest first. A merged clip is named after its most important event.
EVENT_PRIORITY = ("goal", "penalty", "red_card", "yellow_card", "shot", "cross", "foul")

LEAD_MIN_SECONDS = 2.0
LEAD_DEFAULT_SECONDS = 5.0
LEAD_MAX_SECONDS = 10.0
TAIL_MIN_SECONDS = 1.5
TAIL_DEFAULT_SECONDS = 4.0
TAIL_MAX_SECONDS = 8.0
GOAL_TAIL_MIN_SECONDS = 8.0
GOAL_TAIL_DEFAULT_SECONDS = 12.0
GOAL_TAIL_MAX_SECONDS = 25.0
MERGE_GAP_SECONDS = 1.0
MAX_CLIP_SECONDS = 60.0


@dataclass
class ClipPlan:
    start: int  # first frame, inclusive
    end: int  # last frame, inclusive
    event_indices: List[int] = field(default_factory=list)
    event_types: List[str] = field(default_factory=list)

    def add(self, index: int, event_type: str) -> None:
        self.event_indices.append(index)
        self.event_types.append(event_type)

    @property
    def primary_type(self) -> str:
        return min(
            self.event_types,
            key=lambda kind: EVENT_PRIORITY.index(kind) if kind in EVENT_PRIORITY else len(EVENT_PRIORITY),
        )

    @property
    def frame_count(self) -> int:
        return self.end - self.start + 1


def shot_starts(boundaries: Iterable) -> List[int]:
    """First clean frame of every shot after a boundary.

    A cut's new shot starts on the boundary frame; a dissolve's starts after
    the blend, so a clip never opens on two shots mixed together.
    """
    return sorted(b.start if b.kind == "cut" else b.end + 1 for b in boundaries)


def shot_ends(boundaries: Iterable) -> List[int]:
    """Last clean frame of every shot before a boundary."""
    return sorted(b.start - 1 for b in boundaries)


def _latest_in(values: Sequence[int], low: int, high: int) -> Optional[int]:
    found = None
    for value in values:
        if low <= value <= high:
            found = value
    return found


def _earliest_in(values: Sequence[int], low: int, high: int) -> Optional[int]:
    for value in values:
        if low <= value <= high:
            return value
    return None


def event_window(
    event: dict,
    fps: float,
    total_frames: int = 0,
    starts: Sequence[int] = (),
    ends: Sequence[int] = (),
) -> Tuple[int, int]:
    """(first frame, last frame) of the footage one event deserves."""
    fps = max(float(fps), 1.0)

    def frames(seconds: float) -> int:
        return int(round(seconds * fps))

    frame = max(0, int(event.get("frame", 0) or 0))
    end_frame = max(frame, int(event.get("end_frame", frame) or frame))

    start = _latest_in(starts, frame - frames(LEAD_MAX_SECONDS), frame - frames(LEAD_MIN_SECONDS))
    if start is None:
        start = frame - frames(LEAD_DEFAULT_SECONDS)

    if str(event.get("event", "")).lower() == "goal":
        end = _latest_in(ends, frame + frames(GOAL_TAIL_MIN_SECONDS), frame + frames(GOAL_TAIL_MAX_SECONDS))
        if end is None:
            end = frame + frames(GOAL_TAIL_DEFAULT_SECONDS)
        end = max(end, end_frame + frames(TAIL_MIN_SECONDS))
    else:
        end = _earliest_in(ends, end_frame + frames(TAIL_MIN_SECONDS), end_frame + frames(TAIL_MAX_SECONDS))
        if end is None:
            end = end_frame + frames(TAIL_DEFAULT_SECONDS)

    start = max(0, start)
    if total_frames > 0:
        end = min(total_frames - 1, end)
    return start, end


def plan_clips(
    events: Sequence[dict],
    fps: float,
    total_frames: int = 0,
    boundaries: Iterable = (),
) -> List[ClipPlan]:
    """Merged, non-overlapping clips covering every clip-worthy event."""
    fps = max(float(fps), 1.0)
    boundaries = list(boundaries)
    starts, ends = shot_starts(boundaries), shot_ends(boundaries)
    gap = int(round(MERGE_GAP_SECONDS * fps))
    max_frames = int(round(MAX_CLIP_SECONDS * fps))

    windows = []
    for index, event in enumerate(events):
        event_type = str(event.get("event", "")).lower()
        if event_type not in CLIP_EVENT_TYPES:
            continue
        start, end = event_window(event, fps, total_frames, starts, ends)
        if end <= start:
            continue
        windows.append((start, end, int(event.get("frame", 0) or 0), index, event_type))
    windows.sort()

    clips: List[ClipPlan] = []
    for start, end, frame, index, event_type in windows:
        current = clips[-1] if clips else None
        if current is not None and start <= current.end + gap:
            if max(end, current.end) - current.start + 1 <= max_frames:
                current.end = max(current.end, end)
                current.add(index, event_type)
                continue
            if frame <= current.end:
                # Too long to merge, but the moment itself is already in the
                # current clip; only its tail is lost.
                current.add(index, event_type)
                continue
            start = current.end + 1
        clips.append(ClipPlan(start, end, [index], [event_type]))
    return clips
