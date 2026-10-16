"""Turn spotting-model detections into pipeline events, conservatively.

Measured on the fixture clips (2026-10-14, scripts/eval_spotting.py), the
model is good at shots and weak at goals: "Shots on target" peaked at 0.85 on a
saved shot and 0.42 on a goal's shot, while "Goal" reached 0.22 on one real goal
and ~0 on another, and never fired on a clip without a goal. So:

* A spotted goal at the same moment as a video goal only adds its confidence
  to that goal.
* Otherwise it becomes a goal on its own only at ``GOAL_ALONE`` or more; at
  ``GOAL_WITH_SHOT`` or more it needs a shot just before it (tracked, or
  spotted), whose shooter becomes the scorer.
* A spotted shot with no tracked shot nearby is added as a shot without a
  shooter, so a score change can still be placed on it (scoreboard.attribution
  adopts the latest shot in its window). This covers the goals tracking misses
  because no keeper or shooter was tracked.
* Spotted kick-offs are kept: track_events.goal_merge uses one between two
  goals to tell a second goal from a replay. They are not shown to users.
"""

from typing import Dict, List, Optional, Sequence

GOAL_ALONE = 0.5
GOAL_WITH_SHOT = 0.15
SAME_GOAL_SECONDS = 5.0
SHOT_BEFORE_GOAL_SECONDS = 5.0
SHOT_AFTER_GOAL_SECONDS = 1.0
SHOT_MIN = 0.3
SAME_SHOT_SECONDS = 2.0
KICK_OFF_MIN = 0.3


def _frame(event: dict) -> int:
    return int(event.get("frame", 0))


def _nearest(events: Sequence[dict], frame: int, before: float, after: float) -> Optional[dict]:
    window = [e for e in events if frame - before <= _frame(e) <= frame + after]
    return min(window, key=lambda e: abs(_frame(e) - frame)) if window else None


def fuse_spotted_events(events: Sequence[dict], spotted: Sequence[dict], fps: float) -> List[dict]:
    fps = max(float(fps), 1.0)
    out = list(events)
    video_goals = [e for e in out if e.get("event") == "goal"]
    tracked_shots = [e for e in out if e.get("event") == "shot"]

    added_shots: List[dict] = []
    # Strongest first: "on target" and "off target" often both peak on one
    # strike, and only the more confident reading of it is kept.
    candidates = sorted(
        (s for s in spotted if s.get("event") in ("shot_on_target", "shot_off_target")),
        key=lambda s: -float(s.get("confidence", 0.0)),
    )
    for shot in candidates:
        if float(shot.get("confidence", 0.0)) < SHOT_MIN:
            continue
        window = SAME_SHOT_SECONDS * fps
        if _nearest(tracked_shots, _frame(shot), window, window) or _nearest(added_shots, _frame(shot), window, window):
            continue
        added_shots.append(
            {
                "event": "shot",
                "frame": _frame(shot),
                "end_frame": _frame(shot),
                "confidence": float(shot["confidence"]),
                "source": "spotting",
                "team_id": 0,
                "team_name": "unknown_team",
                "participants": [],
                "details": {
                    "on_target": shot["event"] == "shot_on_target",
                    "spotting_label": (shot.get("details") or {}).get("spotting_label"),
                },
            }
        )
    out.extend(added_shots)
    all_shots = tracked_shots + added_shots

    for goal in (s for s in spotted if s.get("event") == "goal"):
        frame, confidence = _frame(goal), float(goal.get("confidence", 0.0))
        same = _nearest(video_goals, frame, SAME_GOAL_SECONDS * fps, SAME_GOAL_SECONDS * fps)
        if same is not None:
            details = same.setdefault("details", {})
            details["spotting_confidence"] = max(confidence, float(details.get("spotting_confidence", 0.0)))
            continue
        shot = _nearest(all_shots, frame, SHOT_BEFORE_GOAL_SECONDS * fps, SHOT_AFTER_GOAL_SECONDS * fps)
        if confidence < GOAL_ALONE and (confidence < GOAL_WITH_SHOT or shot is None):
            continue
        out.append(_goal_from(goal, shot))

    out.extend(
        {**k, "confidence": float(k["confidence"])}
        for k in spotted
        if k.get("event") == "kick_off" and float(k.get("confidence", 0.0)) >= KICK_OFF_MIN
    )
    out.sort(key=_frame)
    return out


def _goal_from(goal: dict, shot: Optional[dict]) -> Dict:
    participants: List[dict] = []
    team_id, team_name = 0, "unknown_team"
    if shot is not None:
        team_id = int(shot.get("team_id", 0) or 0)
        team_name = str(shot.get("team_name", "unknown_team") or "unknown_team")
        shooter = next((p for p in shot.get("participants") or [] if p.get("role") == "shooter"), None)
        if shooter is not None:
            participants = [{**shooter, "role": "scorer"}]
        shot.setdefault("details", {})["outcome"] = "goal"
    return {
        "event": "goal",
        "frame": _frame(shot) if shot is not None else _frame(goal),
        "end_frame": _frame(goal),
        "confidence": float(goal["confidence"]),
        "source": "spotting",
        "team_id": team_id,
        "team_name": team_name,
        "participants": participants,
        "details": {
            "spotting_confidence": float(goal["confidence"]),
            "spotting_frame": _frame(goal),
            "attribution": "spotted goal after a shot" if shot is not None else "spotted goal",
        },
    }
