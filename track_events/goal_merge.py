"""Final clean-up of the goal list once every source has spoken.

* A broadcast shows a goal live and then again in one or two replays, usually
  with the score graphic hidden. Each showing can be detected as a goal. After
  a goal the restart alone (celebration, kick-off) takes about a minute, so a
  second goal within ``GOAL_REPLAY_SECONDS`` of the first is a replay of it.
  The earliest showing is kept: the live goal comes before its replays. Goals
  from the score graphic are separate score changes and always kept; any other
  goal that close to one of them is its replay.
* A shot that turned out to be a goal was not saved: its paired save (same
  frame, emitted with it when the ball passed close to the keeper) goes.
"""

from typing import List, Sequence

GOAL_REPLAY_SECONDS = 60.0


def drop_replayed_goals(
    events: Sequence[dict], fps: float, window_seconds: float = GOAL_REPLAY_SECONDS
) -> List[dict]:
    window = window_seconds * max(float(fps), 1.0)
    board_frames = [
        int(e.get("frame", 0)) for e in events if e.get("event") == "goal" and e.get("source") == "scoreboard"
    ]
    kept_goals: List[dict] = []
    replays = set()
    others = sorted(
        (e for e in events if e.get("event") == "goal" and e.get("source") != "scoreboard"),
        key=lambda e: int(e.get("frame", 0)),
    )
    for goal in others:
        frame = int(goal.get("frame", 0))
        original = next(
            (k for k in kept_goals if 0 <= frame - int(k.get("frame", 0)) <= window), None
        )
        near_board = any(abs(frame - b) <= window for b in board_frames)
        if original is not None or near_board:
            replays.add(id(goal))
            if original is not None:
                details = original.setdefault("details", {})
                details["replays_dropped"] = int(details.get("replays_dropped", 0)) + 1
            continue
        kept_goals.append(goal)
    return [e for e in events if id(e) not in replays]


def withdraw_saves_of_goals(events: Sequence[dict]) -> List[dict]:
    goal_shot_frames = {
        int(e.get("frame", 0))
        for e in events
        if e.get("event") == "shot" and (e.get("details") or {}).get("outcome") == "goal"
    }
    return [
        e for e in events
        if not (e.get("event") == "save" and int(e.get("frame", 0)) in goal_shot_frames)
    ]
