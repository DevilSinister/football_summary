"""Final clean-up of the goal list once every source has spoken.

* A broadcast shows a goal live and then again in one or two replays, usually
  with the score graphic hidden. Each showing can be detected as a goal. A
  second goal within ``GOAL_REPLAY_SECONDS`` of the first is a replay of it.
  The earliest showing is kept: the live goal comes before its replays. Goals
  from the score graphic are separate score changes and always kept; any other
  goal that close to one of them is its replay.
  The window was 60 s (a restart takes about a minute in a full match), but
  highlight packages put goals 16-43 s apart (Levante 3-3 Sevilla,
  2026-10-16) and the 60 s rule dropped two of its six goals. Replays in a
  package follow within seconds, so the window is now 20 s.
* Two goals whose scorers were named, differently, by a caption or the
  commentary are two goals however close: one player's goal is not replayed
  as another player's. A name looked up from a shirt number alone does not
  count - one misread would turn a replay into a second goal. When a replay
  names the scorer and the goal it repeats does not, the name moves over.
* A kick-off seen between two goals (from the action-spotting model) means the
  game restarted: the later goal is a new goal, not a replay, however close.
  It does not separate a goal from a score change, because the graphic is
  often only re-read after the restart.
* Goals are judged strongest source first: track and pitch goals, then the
  spotting model's (often without a scorer), then the legacy learned model's.
  A weaker goal never causes a stronger one to be dropped; within the window
  of a stronger goal, before or after, it goes.
* A shot that turned out to be a goal was not saved: its paired save (same
  frame, emitted with it when the ball passed close to the keeper) goes.
"""

import unicodedata
from typing import Iterable, List, Optional, Sequence

GOAL_REPLAY_SECONDS = 20.0
# Lower rank = stronger evidence; unlisted sources (tracks, pitch) rank 0.
SOURCE_RANK = {"spotting": 1, "model": 2}
# Scorer evidence that names a player rather than reading a shirt number.
NAME_SOURCES = frozenset({"caption", "commentary"})
CLEAR_NAME_CONFIDENCE = 0.7


def named_scorer(goal: dict) -> Optional[str]:
    """The scorer's name, folded for comparison, when a caption or the
    commentary named him confidently; otherwise None."""
    scorer = goal.get("scorer") or {}
    name = scorer.get("player_name")
    if not name or scorer.get("unconfirmed"):
        return None
    sources = set(scorer.get("scorer_sources") or []) | {scorer.get("identity_source")}
    if not sources & NAME_SOURCES:
        return None
    if float(scorer.get("scorer_confidence") or 0.0) < CLEAR_NAME_CONFIDENCE:
        return None
    folded = unicodedata.normalize("NFKD", str(name))
    return "".join(ch for ch in folded if not unicodedata.combining(ch)).casefold().strip()


def _different_scorers(a: dict, b: dict) -> bool:
    name_a, name_b = named_scorer(a), named_scorer(b)
    return name_a is not None and name_b is not None and name_a != name_b


def _adopt_scorer(original: dict, replay: dict) -> None:
    """Give ``original`` the scorer its replay named, if it has none of its own."""
    if named_scorer(original) is not None or named_scorer(replay) is None:
        return
    scorer = replay["scorer"]
    original_team = int(original.get("team_id") or 0)
    scorer_team = int(scorer.get("team_id") or 0)
    if original.get("source") == "scoreboard" and scorer_team not in (0, original_team):
        return  # the graphic's team stands; a name from the other side is a misread
    original["scorer"] = dict(scorer)
    actors = [a for a in original.get("actors") or [] if a.get("role") != "scorer"]
    actors.extend(dict(a) for a in replay.get("actors") or [] if a.get("role") == "scorer")
    original["actors"] = actors
    if not original_team and scorer_team:
        original["team_id"], original["team_name"] = scorer_team, scorer.get("team_name")
    original.setdefault("details", {})["scorer_from_replay_frame"] = int(replay.get("frame", 0))


def drop_replayed_goals(
    events: Sequence[dict],
    fps: float,
    window_seconds: float = GOAL_REPLAY_SECONDS,
    kick_off_frames: Optional[Iterable[int]] = None,
) -> List[dict]:
    """``kick_off_frames`` defaults to the kick-off events in ``events``."""
    window = window_seconds * max(float(fps), 1.0)
    board_goals = [e for e in events if e.get("event") == "goal" and e.get("source") == "scoreboard"]
    if kick_off_frames is None:
        kick_off_frames = [int(e.get("frame", 0)) for e in events if e.get("event") == "kick_off"]
    kick_offs = [int(frame) for frame in kick_off_frames]

    def restarted_between(a: int, b: int) -> bool:
        low, high = min(a, b), max(a, b)
        return any(low < k < high for k in kick_offs)

    kept_goals: List[dict] = []
    replays = set()
    # Strongest source first; within a source, time order (the live showing
    # before its replays).
    others = sorted(
        (e for e in events if e.get("event") == "goal" and e.get("source") != "scoreboard"),
        key=lambda e: (SOURCE_RANK.get(str(e.get("source")), 0), int(e.get("frame", 0))),
    )
    for goal in others:
        frame = int(goal.get("frame", 0))
        original = next(
            (
                k for k in kept_goals
                if abs(frame - int(k.get("frame", 0))) <= window
                and not restarted_between(frame, int(k.get("frame", 0)))
                and not _different_scorers(goal, k)
            ),
            None,
        )
        board = next(
            (
                b for b in board_goals
                if abs(frame - int(b.get("frame", 0))) <= window and not _different_scorers(goal, b)
            ),
            None,
        )
        if original is not None or board is not None:
            replays.add(id(goal))
            for target in (original, board):
                if target is not None:
                    _adopt_scorer(target, goal)
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
