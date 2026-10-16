"""Spoken names -> tracks and event roles.

Tracks: a commentator names the player on the ball, a moment after it reaches
him. A mention at time t goes to the player holding the ball between
``HOLDER_BEFORE_SECONDS`` before and ``HOLDER_AFTER_SECONDS`` after t - the one
nearest t minus ``SPEECH_LAG_SECONDS`` - provided the named player plays for the
holder's team. It is added to that track's identity probabilities.

Goals: in the window around the goal, the scorer is the scoring team's player
named with a scoring word ("... Vinícius scores"), named most after the goal
(commentators repeat the scorer's name), and named last just before it. The
assister is a different team-mate named with a passing word ("cross from X",
"X picks out ...") or named just before the scorer.

Fouls: "foul by X", "yellow card for X", "X brings him down" name the fouler;
"foul on X", "brings down X" the fouled player. They must be opponents.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .mentions import Candidate, Mention, ruled_out
from .transcribe import Word

HOLDER_BEFORE_SECONDS = 2.5
HOLDER_AFTER_SECONDS = 0.5
SPEECH_LAG_SECONDS = 0.7
NAME_EVIDENCE = 0.9

GOAL_BEFORE_SECONDS = 10.0
# Highlight packages move to the next attack within seconds, so names said
# long after a goal belong to the next play (Levante-Sevilla, 2026-10-16: the
# scorer of goal 1 came out as the player named 15 s later). Mentions after
# the goal count less the later they are, and never past the next goal.
GOAL_AFTER_SECONDS = 12.0
RULED_OUT_AFTER_SECONDS = 30.0
LAST_NAME_BEFORE_SECONDS = 10.0
LAST_NAME_AFTER_SECONDS = 4.0  # the detected goal moment can lead the spoken name
FOUL_BEFORE_SECONDS = 3.0
FOUL_AFTER_SECONDS = 15.0
ROLE_MIN_SCORE = 1.0
# A name only the other squad has (the kit colours that set the goal's team
# are often wrong) still counts, at this weight: the conceding side is named
# around a goal too ("past Ryan", "beats Kounde"). Their keeper never counts.
OTHER_TEAM_WEIGHT = 0.6


@dataclass
class Attachment:
    mention: Mention
    candidate: Candidate
    track_id: int
    frame: int


def attach_to_tracks(
    mentions: Sequence[Mention],
    possession: Sequence[Tuple[int, int, int]],
    fps: float,
    identity_tracker=None,
) -> List[Attachment]:
    """``possession``: (frame, track_id, team_id) for every analysed frame with a holder."""
    fps = max(float(fps), 1.0)
    attached: List[Attachment] = []
    for mention in mentions:
        target = (mention.start - SPEECH_LAG_SECONDS) * fps
        low, high = (mention.start - HOLDER_BEFORE_SECONDS) * fps, (mention.start + HOLDER_AFTER_SECONDS) * fps
        holders = sorted(
            (p for p in possession if low <= p[0] <= high), key=lambda p: abs(p[0] - target)
        )
        for frame, track_id, team_id in holders:
            candidate = mention.best_for_team(int(team_id)) if team_id else None
            if candidate is None:
                continue
            attached.append(Attachment(mention, candidate, int(track_id), int(frame)))
            if identity_tracker is not None:
                identity_tracker.add_name_evidence(
                    int(track_id), candidate.team_id, candidate.jersey_no, candidate.score * NAME_EVIDENCE
                )
            break
    return attached


def _role(candidate: Candidate, score: float, mention: Mention, total: float, shared_with=None) -> dict:
    role = {
        "team_id": candidate.team_id,
        "jersey_number": candidate.jersey_no,
        "player_name": candidate.name,
        "confidence": round(min(1.0, score / max(total, 1e-6)) * min(1.0, score / 2.0), 3),
        "source": "commentary",
        "said": mention.text,
        "said_at_seconds": round(mention.start, 2),
    }
    if shared_with:
        # Only ever named by words other players share ("Romero"): any of
        # them may be meant; the goal's team picked this one.
        role["shared_with"] = sorted(shared_with)
    return role


def _goal_candidate(mention: Mention, team_id: int, other_team: bool) -> Tuple[Optional[Candidate], float]:
    """The player a mention names for a goal credited to ``team_id``, and a weight factor.

    A name the goal's team has wins (a name both teams share, "Romero", goes
    to the goal's team). With ``other_team``, a name that only one player of
    the other squad has - a clear name - is kept at ``OTHER_TEAM_WEIGHT``.
    """
    own = mention.best_for_team(team_id)
    if own is not None or not team_id:
        return own, 1.0
    if other_team and len(mention.candidates) == 1:
        candidate = mention.candidates[0]
        if str(candidate.position or "").upper() != "GK":
            return candidate, OTHER_TEAM_WEIGHT
    return None, 0.0


def goal_roles(
    goal_seconds: float,
    team_id: int,
    mentions: Sequence[Mention],
    words: Sequence[Word] = (),
    next_goal_seconds: Optional[float] = None,
    previous_goal_seconds: Optional[float] = None,
    other_team: bool = True,
) -> Dict[str, Optional[dict]]:
    """Scorer, assister and a ruled-out phrase for a goal from the commentary.

    ``other_team`` False keeps to ``team_id``'s squad (the score graphic said
    who scored); otherwise a clear name from the other squad can be the
    scorer, and the caller may correct the goal's team from it.
    """
    after_end = goal_seconds + GOAL_AFTER_SECONDS
    if next_goal_seconds is not None:
        after_end = min(after_end, next_goal_seconds - 2.0)
    before_start = goal_seconds - GOAL_BEFORE_SECONDS
    if previous_goal_seconds is not None:
        before_start = max(before_start, previous_goal_seconds + 2.0)
    window = [m for m in mentions if before_start <= m.start <= after_end]
    scorer_scores: Dict[Tuple[int, int], float] = {}
    assist_scores: Dict[Tuple[int, int], float] = {}
    said: Dict[Tuple[int, int], Tuple[Candidate, Mention]] = {}
    shared: Dict[Tuple[int, int], set] = {}
    named_clearly: set = set()
    before = [
        m for m in window
        if goal_seconds - LAST_NAME_BEFORE_SECONDS <= m.start <= goal_seconds + LAST_NAME_AFTER_SECONDS
    ]
    last_before = before[-1] if before else None
    for mention in window:
        candidate, factor = _goal_candidate(mention, team_id, other_team)
        if candidate is None:
            continue
        key = (candidate.team_id, candidate.jersey_no)
        said.setdefault(key, (candidate, mention))
        others = {(c.team_id, c.jersey_no) for c in mention.candidates} - {key}
        if others:
            shared.setdefault(key, set()).update(others)
        else:
            named_clearly.add(key)
        credited_with_pass = ("passing" in mention.cues_before and "agent" in mention.cues_before) or (
            "passing" in mention.cues_after and "scoring" not in mention.cues_after
        )
        if credited_with_pass:
            assist_scores[key] = assist_scores.get(key, 0.0) + (candidate.score + 1.5) * factor
            continue
        weight = candidate.score
        if "scoring" in mention.cues_before or "scoring" in mention.cues_after:
            weight += 1.5
        if mention.start > goal_seconds + LAST_NAME_AFTER_SECONDS:
            # Repeated after the goal: still evidence, fading with time.
            weight *= max(0.3, 1.0 - (mention.start - goal_seconds) / (2 * GOAL_AFTER_SECONDS))
        if mention is last_before:
            weight += 1.0
        scorer_scores[key] = scorer_scores.get(key, 0.0) + weight * factor
    result: Dict[str, Optional[dict]] = {"scorer": None, "assister": None, "ruled_out": None}
    if words:
        ruled_end = goal_seconds + RULED_OUT_AFTER_SECONDS
        if next_goal_seconds is not None:
            ruled_end = min(ruled_end, next_goal_seconds - 2.0)
        result["ruled_out"] = ruled_out(words, goal_seconds - 2.0, ruled_end)
    if scorer_scores:
        key, score = max(scorer_scores.items(), key=lambda item: item[1])
        if score >= ROLE_MIN_SCORE:
            result["scorer"] = _role(
                said[key][0], score, said[key][1], sum(scorer_scores.values()),
                shared_with=None if key in named_clearly else shared.get(key),
            )
    scorer_key = (
        (result["scorer"]["team_id"], result["scorer"]["jersey_number"]) if result["scorer"] else None
    )
    # Named just before the scorer, by a team-mate: the pass that made it.
    if scorer_key is not None:
        keys = []
        for mention in before:
            # The scorer's team: a clear other-squad scorer corrects the team.
            candidate = mention.best_for_team(scorer_key[0])
            if candidate is not None:
                keys.append((candidate.team_id, candidate.jersey_no))
        for previous, current in zip(keys, keys[1:]):
            if current == scorer_key and previous != scorer_key and previous[0] == scorer_key[0]:
                assist_scores[previous] = assist_scores.get(previous, 0.0) + 1.0
    assist_scores.pop(scorer_key, None)
    if scorer_key is not None:
        assist_scores = {k: v for k, v in assist_scores.items() if k[0] == scorer_key[0]}
    if assist_scores:
        key, score = max(assist_scores.items(), key=lambda item: item[1])
        if score >= ROLE_MIN_SCORE:
            result["assister"] = _role(said[key][0], score, said[key][1], sum(assist_scores.values()))
    return result


def foul_roles(foul_seconds: float, mentions: Sequence[Mention]) -> Dict[str, Optional[dict]]:
    window = [m for m in mentions if foul_seconds - FOUL_BEFORE_SECONDS <= m.start <= foul_seconds + FOUL_AFTER_SECONDS]
    fouler: Dict[Tuple[int, int], Tuple[float, Candidate, Mention]] = {}
    fouled: Dict[Tuple[int, int], Tuple[float, Candidate, Mention]] = {}
    for mention in window:
        candidate = mention.best_for_team(0)
        if candidate is None:
            continue
        key = (candidate.team_id, candidate.jersey_no)
        victim = "victim" in mention.cues_before and "foul" in mention.cues_before + mention.cues_after
        agent = (
            ("agent" in mention.cues_before or "for" in mention.cues_before) and "foul" in mention.cues_before
        ) or "foul" in mention.cues_after
        target = fouled if victim else fouler if agent else None
        if target is not None:
            previous = target.get(key, (0.0, candidate, mention))[0]
            target[key] = (previous + candidate.score + 1.0, candidate, mention)
    result: Dict[str, Optional[dict]] = {"fouler": None, "fouled": None}
    for role, scores in (("fouler", fouler), ("fouled", fouled)):
        if scores:
            score, candidate, mention = max(scores.values(), key=lambda item: item[0])
            if score >= ROLE_MIN_SCORE:
                result[role] = _role(candidate, score, mention, sum(s for s, _, _ in scores.values()))
    if result["fouler"] and result["fouled"] and result["fouler"]["team_id"] == result["fouled"]["team_id"]:
        result["fouled"] = None  # team-mates: the weaker reading is wrong
    return result
