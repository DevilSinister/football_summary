"""Turn a score change into a goal event with a moment and, if possible, a scorer.

The graphic tells us that a goal happened and for whom, but only when the new
score is first shown - often after the replay, seconds after the ball went in.
The goal happened between the last read of the old score and the first read of
the new one. Inside that window, in order of preference:

1. a pitch-geometry goal (ball behind the line) - the exact moment;
2. the latest shot by the scoring team (or by a player whose team is unknown or
   was misread - kit colours are unreliable, the scoreboard is not);
3. the last player of the scoring team seen in possession;
4. nothing - the goal is still reported, with no scorer.

The latest shot is chosen because a saved attempt early in the window is more
common than a goal nobody shot; a replay of the scoring shot names the same
player, so choosing it costs only the timestamp.
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


def _score_line(codes: Sequence[str], score: Dict[str, int]) -> str:
    return f"{codes[0]} {score[codes[0]]}-{score[codes[1]]} {codes[1]}"


def _team_of(event: dict) -> int:
    return int(event.get("team_id", 0) or 0)


def _shooter(event: dict) -> dict:
    return next(
        (p for p in event.get("participants") or [] if p.get("role") in ("shooter", "scorer")),
        {},
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

        pitch_goals = [e for e in events if e.get("event") == "goal" and e.get("source") == "pitch" and in_window(e)]
        shots = [
            e for e in events
            if e.get("event") == "shot" and in_window(e) and _team_of(e) in (team_id, 0)
        ]
        other_shots = [
            e for e in events
            if e.get("event") == "shot" and in_window(e) and e not in shots
            and float(_shooter(e).get("team_confidence", 0.0) or 0.0) < OTHER_TEAM_MAX_CONFIDENCE
        ]

        if pitch_goals:
            source_event = max(pitch_goals, key=lambda e: int(e.get("frame", 0)))
            attribution = "ball behind the goal line"
        elif shots:
            source_event = max(shots, key=lambda e: int(e.get("frame", 0)))
            attribution = "latest shot by the scoring team"
        elif other_shots and team_id:
            # The shooter's kit was read as the other team; trust the scoreboard.
            source_event = max(other_shots, key=lambda e: int(e.get("frame", 0)))
            attribution = "latest shot (shooter's team re-read from the scoreboard)"

        if source_event is not None:
            moment = int(source_event.get("frame", goal.frame))
            shooter = next(
                (p for p in source_event.get("participants") or [] if p.get("role") in ("shooter", "scorer")),
                None,
            )
            if shooter is not None:
                participants = [
                    {**shooter, "role": "scorer", "team_id": team_id or shooter.get("team_id", 0),
                     "team_name": team_name if team_id else shooter.get("team_name", "unknown_team")}
                ]
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
                attribution = "last player of the scoring team in possession"

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
                },
            }
        )
    return produced, evidence
