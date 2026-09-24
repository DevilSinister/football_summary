"""Plan highlight clips: one clip per passage of play, never longer than 10 s.

A goal, the shot before it and the save before that used to be three files of
the same ten seconds, each a fixed +-5 s around its event that started and
stopped mid-shot. Here every clip-worthy event proposes a window, the window
is snapped to the camera shots around it, and windows that overlap or nearly
touch become one clip - as long as the result still fits in MAX_CLIP_SECONDS.

Owner rule (2026-09-24): no clip is longer than 10 seconds. Before this cap a
chain of shots, crosses and fouls that each touched the next merged into one
clip of up to 60 s, and a goal ran on for up to 25 s through the replay.

Windows, in frames:

- start: the first frame of the latest camera shot that began between
  LEAD_MAX and LEAD_MIN seconds before the event - the start of the play as
  the broadcaster framed it. With no cut in that range (one continuous live
  shot, or amateur footage) the window starts LEAD_DEFAULT before.
- end: for most events, the last frame of the first shot that closes between
  TAIL_MIN and TAIL_MAX after the event, else TAIL_DEFAULT after it. For a
  goal, the last frame of the last shot that closes between GOAL_TAIL_MIN and
  GOAL_TAIL_MAX, so the first seconds of the celebration stay in.
- a window still longer than MAX_CLIP_SECONDS is trimmed: the lead first
  (never below LEAD_MIN), then the tail.

Placement, most important event first (EVENT_PRIORITY), so a goal always gets
its full window and lesser events fit around it:

1. an event whose moment already lies inside a placed clip joins that clip;
2. else, if its window overlaps or nearly touches a placed clip and the union
   fits in MAX_CLIP_SECONDS, that clip grows to the union;
3. else it gets its own clip, cut back so it never overlaps a placed one - no
   footage is written twice.
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

# Highest first. A merged clip is named after its most important event, and
# the most important events are placed first.
EVENT_PRIORITY = ("goal", "penalty", "red_card", "yellow_card", "foul", "shot", "cross")

LEAD_MIN_SECONDS = 2.0
LEAD_DEFAULT_SECONDS = 5.0
LEAD_MAX_SECONDS = 6.0
TAIL_MIN_SECONDS = 1.5
TAIL_DEFAULT_SECONDS = 3.0
TAIL_MAX_SECONDS = 4.0
GOAL_TAIL_MIN_SECONDS = 3.0
GOAL_TAIL_DEFAULT_SECONDS = 4.0
GOAL_TAIL_MAX_SECONDS = 5.0
MERGE_GAP_SECONDS = 1.0
# Owner rule: a highlight clip is at most 10 seconds.
MAX_CLIP_SECONDS = 10.0


def _priority(event_type: str) -> int:
    return EVENT_PRIORITY.index(event_type) if event_type in EVENT_PRIORITY else len(EVENT_PRIORITY)


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
        return min(self.event_types, key=_priority)

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


def _fit(start: int, end: int, frame: int, max_frames: int, min_lead: int) -> Tuple[int, int]:
    """Trim a window to max_frames: the lead first (keeping min_lead), then the tail."""
    excess = (end - start + 1) - max_frames
    if excess > 0:
        cut = min(excess, max(0, (frame - start) - min_lead))
        start += cut
        excess -= cut
    if excess > 0:
        end -= excess
    return start, end


def event_window(
    event: dict,
    fps: float,
    total_frames: int = 0,
    starts: Sequence[int] = (),
    ends: Sequence[int] = (),
) -> Tuple[int, int]:
    """(first frame, last frame) of the footage one event deserves, at most 10 s."""
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
    else:
        end = _earliest_in(ends, end_frame + frames(TAIL_MIN_SECONDS), end_frame + frames(TAIL_MAX_SECONDS))
        if end is None:
            end = end_frame + frames(TAIL_DEFAULT_SECONDS)

    start = max(0, start)
    if total_frames > 0:
        end = min(total_frames - 1, end)
    # An event that spans seconds (a shot whose ball stays loose, a penalty
    # set-up) can still propose more than the cap.
    return _fit(start, end, frame, frames(MAX_CLIP_SECONDS), frames(LEAD_MIN_SECONDS))


def _free_piece(start: int, end: int, frame: int, clips: Sequence[ClipPlan]) -> Tuple[int, int]:
    """The part of [start, end] around ``frame`` that no placed clip covers."""
    for clip in clips:
        if clip.end < frame:
            start = max(start, clip.end + 1)
        elif clip.start > frame:
            end = min(end, clip.start - 1)
    return start, end


def plan_clips(
    events: Sequence[dict],
    fps: float,
    total_frames: int = 0,
    boundaries: Iterable = (),
) -> List[ClipPlan]:
    """Non-overlapping clips of at most MAX_CLIP_SECONDS covering every clip-worthy event."""
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
        frame = min(max(int(event.get("frame", 0) or 0), start), end)
        windows.append((_priority(event_type), frame, start, end, index, event_type))
    windows.sort()

    clips: List[ClipPlan] = []
    for _, frame, start, end, index, event_type in windows:
        # 1. The moment is already on film.
        home = next((clip for clip in clips if clip.start <= frame <= clip.end), None)
        if home is not None:
            home.add(index, event_type)
            continue

        # 2. Grow the nearest touching clip, if the union fits and runs into no other clip.
        grown = False
        for clip in sorted(clips, key=lambda c: abs(c.start - frame)):
            if start > clip.end + gap or end < clip.start - gap:
                continue
            union_start, union_end = min(start, clip.start), max(end, clip.end)
            if union_end - union_start + 1 > max_frames:
                continue
            if any(
                c is not clip and union_start <= c.end and c.start <= union_end for c in clips
            ):
                continue
            clip.start, clip.end = union_start, union_end
            clip.add(index, event_type)
            grown = True
            break
        if grown:
            continue

        # 3. A clip of its own, cut back to the free footage around the moment.
        start, end = _free_piece(start, end, frame, clips)
        if end < start:
            continue
        clips.append(ClipPlan(start, end, [index], [event_type]))

    clips.sort(key=lambda clip: clip.start)
    for clip in clips:
        # Member events in time order, as the files and the result list them.
        pairs = sorted(zip(clip.event_indices, clip.event_types))
        clip.event_indices = [index for index, _ in pairs]
        clip.event_types = [kind for _, kind in pairs]
    return clips
