"""The score graphic as confirmation of goals, and as the safety net for a goal
nothing else saw.

The graphic tells us that a goal happened and for whom, but only when the new
score is first shown - often after the replay, seconds after the ball went in.
The goal happened between the last read of the old score and the first read of
the new one. Inside that window, in order of preference:

1. a goal already detected from the video (pitch goal line, ball in the goal
   mouth after a shot, the learned model) - the earliest one, because the live
   goal comes before its replays. The score change confirms it;
2. the latest shot by the scoring team (or by a player whose team is unknown or
   was read weakly - kit colours are unreliable, the scoreboard is not);
3. the latest shot by anyone: a score change after a shot means that shot was
   the goal. The shooter is named only if his kit was not confidently read as
   the other team (that would be a defender's block or deflection);
4. the last player of the scoring team seen in possession;
5. nothing - the goal is still reported, with no scorer.

Cases 2-5 mean the score changed but no goal was detected from the video; the
event records it (``details.detected_goal`` is False) so the gap stays visible.

A goal detected from the video that no score change confirms is kept, marked
``details.scoreboard = "not confirmed"``: the graphic misses changes it cannot
see (hidden during replays, other codes, a single unconfirmed read).
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .reader import ScoreGoal

# If the old score was never read (the clip starts mid-replay), look back this far.
DEFAULT_LOOKBACK_SECONDS = 60.0
WINDOW_PAD_SECONDS = 1.0
# A shooter read as the other team is credited to the scoring team only when
# that kit reading was weak. A confident reading means a defender - a block, a
# deflection - and on the 1080p reference clip crediting him named the wrong
# player for Man City's goal.
OTHER_TEAM_MAX_CONFIDENCE = 0.5


def _window(goal_event: dict, fps: float) -> Tuple[float, float]:
    details = goal_event.get("details") or {}
    end = int(details.get("scoreboard_frame", goal_event.get("frame", 0)))
    start = details.get("window_start_frame")
    start = end - DEFAULT_LOOKBACK_SECONDS * fps if start is None else int(start)
    return start - WINDOW_PAD_SECONDS * fps, end + WINDOW_PAD_SECONDS * fps


def merge_board_goals(
    events: Sequence[dict], board_goals: Sequence[dict], fps: float
) -> Tuple[List[dict], List[dict]]:
    """Add the score graphic's goals to the other goal proposals.

    A goal from another source inside a board goal's window is that same goal:
    the board's version (which adopted its moment and scorer in
    ``attribute_goals``) replaces it. Any other detected goal is kept and marked
    as not confirmed by the graphic.

    Returns (events with the board goals added, proposals dropped as duplicates).
    """
    fps = max(float(fps), 1.0)
    windows = [_window(goal, fps) for goal in board_goals]
    kept: List[dict] = []
    duplicates: List[dict] = []
    for event in events:
        if event.get("event") == "goal" and event.get("source") != "scoreboard":
            frame = int(event.get("frame", 0))
            if any(start <= frame <= end for start, end in windows):
                duplicates.append(event)
                continue
            event.setdefault("details", {})["scoreboard"] = "not confirmed"
        kept.append(event)
    return kept + list(board_goals), duplicates


def _score_line(codes: Sequence[str], score: Dict[str, int]) -> str:
    return f"{codes[0]} {score[codes[0]]}-{score[codes[1]]} {codes[1]}"


def _team_of(event: dict) -> int:
    return int(event.get("team_id", 0) or 0)


def _shooter(event: dict) -> dict:
    return next(
        (p for p in event.get("participants") or [] if p.get("role") in ("shooter", "scorer")),
        {},
    )


def _confidently_other(person: dict, team_id: int) -> bool:
    their_team = int(person.get("team_id", 0) or 0)
    return (
        bool(team_id)
        and their_team not in (0, team_id)
        and float(person.get("team_confidence", 0.0) or 0.0) >= OTHER_TEAM_MAX_CONFIDENCE
    )


def attribute_goals(
    goals: Sequence[ScoreGoal],
    codes: Sequence[str],
    code_to_team: Dict[str, int],
    team_names: Dict[int, str],
    events: Sequence[dict],
    fps: float,
    possession_samples: Iterable = (),
) -> Tuple[List[dict], List[dict]]:
    """Returns (goal events, events used as evidence)."""
    fps = max(float(fps), 1.0)
    samples = list(possession_samples)
    produced: List[dict] = []
    evidence: List[dict] = []

    for goal in goals:
        team_id = code_to_team.get(goal.code, 0)
        team_name = team_names.get(team_id) or goal.code
        if goal.window_start is not None:
            start = goal.window_start - WINDOW_PAD_SECONDS * fps
        else:
            start = goal.frame - DEFAULT_LOOKBACK_SECONDS * fps
        end = goal.frame

        def in_window(event: dict) -> bool:
            return start <= int(event.get("frame", 0)) <= end

        moment, participants, attribution, source_event = goal.frame, [], "scoreboard only", None
        detected_by = None

        detected = [
            e for e in events
            if e.get("event") == "goal" and e.get("source") != "scoreboard" and in_window(e)
        ]
        shots = [e for e in events if e.get("event") == "shot" and in_window(e)]
        own_shots = [e for e in shots if _team_of(e) in (team_id, 0)]
        weak_other_shots = [
            e for e in shots
            if e not in own_shots and not _confidently_other(_shooter(e), team_id)
        ]

        if detected:
            source_event = min(detected, key=lambda e: int(e.get("frame", 0)))
            detected_by = source_event.get("source", "unknown")
            attribution = f"goal detected ({detected_by}), confirmed by the score graphic"
        elif own_shots:
            source_event = max(own_shots, key=lambda e: int(e.get("frame", 0)))
            attribution = "no goal detected; latest shot by the scoring team"
        elif weak_other_shots and team_id:
            # The shooter's kit was read weakly as the other team; trust the scoreboard.
            source_event = max(weak_other_shots, key=lambda e: int(e.get("frame", 0)))
            attribution = "no goal detected; latest shot (shooter's team re-read from the scoreboard)"
        elif shots:
            source_event = max(shots, key=lambda e: int(e.get("frame", 0)))
            attribution = "no goal detected; latest shot in the window (shooter read as the other team, not named)"

        if source_event is not None:
            moment = int(source_event.get("frame", goal.frame))
            shooter = _shooter(source_event) or None
            if shooter is not None and not _confidently_other(shooter, team_id):
                participants = [
                    {**shooter, "role": "scorer", "team_id": team_id or shooter.get("team_id", 0),
                     "team_name": team_name if team_id else shooter.get("team_name", "unknown_team")}
                ]
            if source_event.get("event") == "shot":
                source_event.setdefault("details", {})["outcome"] = "goal"
            evidence.append(source_event)
        elif team_id:
            holders = [s for s in samples if start <= s.frame_number <= end and s.team_id == team_id]
            if holders:
                last = max(holders, key=lambda s: s.frame_number)
                moment = last.frame_number
                participants = [
                    {"role": "scorer", "track_id": last.track_id, "team_id": team_id,
                     "team_name": team_name, "team_confidence": 0.0}
                ]
                attribution = "no goal detected; last player of the scoring team in possession"

        produced.append(
            {
                "event": "goal",
                "frame": int(moment),
                "end_frame": int(moment),
                "confidence": 0.95 if source_event is not None else 0.85,
                "team_id": team_id,
                "team_name": team_name,
                "source": "scoreboard",
                "participants": participants,
                "details": {
                    "score_before": _score_line(codes, goal.score_before),
                    "score_after": _score_line(codes, goal.score_after),
                    "scoreboard_code": goal.code,
                    "scoreboard_frame": int(goal.frame),
                    "window_start_frame": None if goal.window_start is None else int(goal.window_start),
                    "attribution": attribution,
                    "detected_goal": detected_by is not None,
                    "detected_by": detected_by,
                },
            }
        )
    return produced, evidence
