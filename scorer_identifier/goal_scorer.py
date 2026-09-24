"""Who scored: several weak sources fused into one answer, with a confidence.

Steps 2, 4 and 5 of the Scorer Accuracy Plan (vault: Football Summary /
20 - Functionality / Scorer Accuracy Plan). Nothing here downloads a model;
it reuses the PaddleOCR engine the shirt and scoreboard readers already load.

Sources, strongest first:

1. The broadcast caption. After a goal most broadcasters print the scorer's
   name ("ADEMOLA LOOKMAN 0-1 MIN. 33") in the lower third, or beside the
   score bug. ``read_caption_candidates`` reads those regions from the goal to
   CAPTION_SECONDS after it and matches the words against the scoring team's
   squad (data/rosters). A roster match turns a name into a shirt number.
2. The celebration. The camera follows the scorer after a goal, in close-up,
   where his number is readable even when it was not in the wide shot.
   ``celebration_candidates`` votes the shirt numbers of the scoring team's
   players seen large on screen in the CELEBRATION_SECONDS after the goal,
   weighted by how long and how large each was shown.
3. The live read: the shirt number of the shooter (or the last player in
   possession) that scoreboard/attribution.py picked.

``fuse_scorer`` adds the weights per shirt number, discounts a number the
squad does not have (an OCR misread) and a goalkeeper or defender (they seldom
score), and returns the winner with a confidence. Below UNCONFIRMED_BELOW the
number is still reported, flagged ``unconfirmed``, and no name is attached -
the app shows "#9" rather than a confident wrong name.

This is unmeasured: two reference clips with one goal between them cannot give
an accuracy figure. The plan's step 1 (a labelled evaluation set) is still
owed and needs footage from the owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROSTER_DIR = Path(__file__).resolve().parent.parent / "data" / "rosters"

# --- captions
CAPTION_SECONDS = 45.0         # how long after the goal to look for the caption
CAPTION_EVERY_SECONDS = 1.5
CAPTION_MAX_READS = 30         # frames per goal (two regions each)
CAPTION_AGREEING_READS = 2     # stop early once this many frames agree
CAPTION_MIN_RATIO = 0.84       # difflib similarity for a surname to match
# Graphics that name players who did not score.
CAPTION_SKIP_WORDS = (
    "SUBSTITUT", "YELLOW", "BOOKED", "RED CARD", "LINE UP", "LINEUP", "STARTING",
    "BENCH", "MAN OF THE MATCH", "PLAYER OF THE MATCH", "REPLACED", "INJUR",
)
CAPTION_ASSIST_WORDS = ("ASSIST", "ASSISTED")
NAME_PARTICLES = {
    "DA", "DE", "DI", "DO", "DOS", "DAS", "DEL", "DER", "DEN", "VAN", "VON",
    "LA", "LE", "EL", "AL", "JR", "JUNIOR", "SR", "BIN", "IBN", "MC", "MAC",
}

# --- celebration
CELEBRATION_SECONDS = 30.0
CELEBRATION_MAX_TRACKS = 4

# --- fusion
WEIGHT_CAPTION = 3.0
WEIGHT_CELEBRATION = 1.5
WEIGHT_LIVE_SHOT = 1.0
WEIGHT_LIVE_POSSESSION = 0.6
NOT_IN_SQUAD_FACTOR = 0.3
POSITION_PRIOR = {"FW": 1.2, "MF": 1.0, "DF": 0.7, "GK": 0.1}
# Total weight at which the winner's share is trusted fully. Tuned so a lone
# live shirt read names a forward only when OCR was >= ~0.9 sure; a weaker
# lone read, or one on a defender, shows as "#N (unconfirmed)".
STRENGTH_FOR_FULL_CONFIDENCE = 2.5
UNCONFIRMED_BELOW = 0.7


@dataclass
class ScorerCandidate:
    jersey_number: Optional[int]
    weight: float
    source: str  # "caption", "celebration" or "live"
    player_name: Optional[str] = None
    detail: dict = field(default_factory=dict)


# ------------------------------------------------------------------ roster
_SQUADS: Dict[Path, Dict[str, List[dict]]] = {}


def _normalise_team(name: Optional[str]) -> str:
    from backend.team_naming import normalise_team_name

    return normalise_team_name(name)


def _load_squads(roster_dir: Path) -> Dict[str, List[dict]]:
    squads: Dict[str, List[dict]] = {}
    for path in sorted(roster_dir.glob("*.json")) if roster_dir.is_dir() else []:
        if path.name.startswith("_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        players = [
            p for p in payload.get("players") or []
            if p.get("jersey_no") is not None and p.get("name")
        ]
        keys = {_normalise_team(payload.get("team_name"))}
        keys.update(_normalise_team(alias) for alias in payload.get("aliases") or [])
        for key in keys - {""}:
            squads[key] = players
    return squads


def squad_for(team_name: Optional[str], roster_dir: Path = ROSTER_DIR) -> List[dict]:
    """``[{jersey_no, name, position}]`` for a team from data/rosters, or []."""
    if roster_dir not in _SQUADS:
        _SQUADS[roster_dir] = _load_squads(roster_dir)
    return _SQUADS[roster_dir].get(_normalise_team(team_name), [])


# ----------------------------------------------------------------- captions
def _plain_words(text: str) -> List[str]:
    """Upper-case ASCII words: 'Vinícius Júnior' -> ['VINICIUS', 'JUNIOR']."""
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return [word for word in re.split(r"[^A-Z]+", folded.upper()) if word]


def _name_tokens(name: str) -> List[str]:
    return [w for w in _plain_words(name) if len(w) >= 3 and w not in NAME_PARTICLES]


def _distinctive_tokens(squad: Sequence[dict]) -> Dict[int, List[str]]:
    """Per player, the name words no team-mate shares ('GABRIEL' names nobody at Arsenal)."""
    counts: Dict[str, int] = {}
    tokens = {}
    for index, player in enumerate(squad):
        words = set(_name_tokens(player["name"]))
        tokens[index] = words
        for word in words:
            counts[word] = counts.get(word, 0) + 1
    return {index: sorted(w for w in words if counts[w] == 1) for index, words in tokens.items()}


def _word_matches(name_word: str, ocr_word: str) -> float:
    if name_word == ocr_word:
        return 1.0
    if len(name_word) < 4 or len(ocr_word) < 4:
        return 0.0
    return SequenceMatcher(None, name_word, ocr_word).ratio()


def match_caption(lines: Sequence[str], squad: Sequence[dict]) -> List[Tuple[dict, float]]:
    """Squad players a caption names, with a match score in [0, 1].

    A frame showing a substitution, booking or line-up graphic yields nothing,
    a line naming the assist is skipped, and a frame naming three or more
    players is a team sheet rather than a scorer caption.
    """
    if not squad:
        return []
    distinctive = _distinctive_tokens(squad)
    words: List[str] = []
    for line in lines:
        upper = " ".join(_plain_words(line))
        if any(skip in upper for skip in CAPTION_SKIP_WORDS):
            return []
        if any(word in upper.split() for word in CAPTION_ASSIST_WORDS):
            continue
        words.extend(w for w in upper.split() if len(w) >= 3)
    if not words:
        return []

    found: List[Tuple[dict, float]] = []
    for index, player in enumerate(squad):
        best = 0.0
        for token in distinctive[index]:
            for word in words:
                best = max(best, _word_matches(token, word))
        full = _name_tokens(player["name"])
        if len(full) >= 2 and all(max(_word_matches(t, w) for w in words) >= 0.8 for t in full):
            best = 1.0
        if best >= CAPTION_MIN_RATIO:
            found.append((player, round(best, 3)))
    if len(found) >= 3:
        return []
    return found


OcrFn = Callable[[np.ndarray], Iterable[Tuple[str, float, tuple]]]


def _caption_regions(frame: np.ndarray) -> List[np.ndarray]:
    """Lower third (name captions) and the top band beside the score bug."""
    height = frame.shape[0]
    regions = [frame[int(height * 0.62):, :], frame[: int(height * 0.22), :]]
    if height <= 540:
        regions = [cv2.resize(r, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC) for r in regions]
    return regions


def read_caption_candidates(
    video_path: str,
    goal_frame: int,
    fps: float,
    ocr_fn: Optional[OcrFn],
    squad: Sequence[dict],
    seconds: float = CAPTION_SECONDS,
    every_seconds: float = CAPTION_EVERY_SECONDS,
    max_reads: int = CAPTION_MAX_READS,
) -> List[ScorerCandidate]:
    """Scorer candidates from name captions shown after the goal."""
    if not squad or ocr_fn is None:
        return []
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return []
    fps = max(float(fps), 1.0)
    step = max(1, int(round(every_seconds * fps)))
    last = int(goal_frame + seconds * fps)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        last = min(last, total - 1)

    votes: Dict[int, List[Tuple[int, float]]] = {}
    players: Dict[int, dict] = {}
    reads = 0
    try:
        for position in range(max(0, int(goal_frame)), last + 1, step):
            if reads >= max_reads:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                break
            reads += 1
            lines: List[str] = []
            for region in _caption_regions(frame):
                try:
                    lines.extend(text for text, score, _ in ocr_fn(region) if score >= 0.5)
                except Exception:
                    continue
            matches = match_caption(lines, squad)
            for player, score in matches:
                number = int(player["jersey_no"])
                players[number] = player
                # Two names in one frame (scorer and assist without the word
                # "assist"): each gets half the weight.
                votes.setdefault(number, []).append((position, score / len(matches)))
            # One name, seen on enough frames: the caption is settled.
            if len(votes) == 1 and any(len(v) >= CAPTION_AGREEING_READS for v in votes.values()):
                break
    finally:
        capture.release()

    candidates = []
    for number, seen in votes.items():
        # Repeated sightings of one caption add up, but only to two frames' worth.
        weight = WEIGHT_CAPTION * sum(sorted((s for _, s in seen), reverse=True)[:CAPTION_AGREEING_READS])
        candidates.append(
            ScorerCandidate(
                number,
                weight,
                "caption",
                player_name=players[number]["name"],
                detail={"frames": [f for f, _ in seen], "best_match": max(s for _, s in seen)},
            )
        )
    return candidates


# -------------------------------------------------------------- celebration
def celebration_candidates(
    identity_store,
    identify_track: Callable[[int], dict],
    team_id: int,
    start_frame: int,
    end_frame: int,
    frame_height: int,
    max_tracks: int = CELEBRATION_MAX_TRACKS,
) -> List[ScorerCandidate]:
    """Shirt numbers of the scoring team's players shown large after the goal.

    Screen time is weighted by box height, so the player the camera follows
    in close-up outweighs a team-mate running past in the background. Track
    ids restart at every cut; the shirt number is what joins them up.
    """
    frame_height = max(int(frame_height or 0), 1)
    weights: List[Tuple[float, int]] = []
    for track_id, state in identity_store.players.items():
        if team_id and state.team_id not in (team_id, 0):
            continue
        seen = sum(h / frame_height for f, h in state.sightings if start_frame <= f <= end_frame)
        if seen <= 0:
            continue
        if team_id and state.team_id == 0:
            seen *= 0.5
        weights.append((seen, int(track_id)))
    weights.sort(reverse=True)
    weights = weights[:max_tracks]
    total = sum(w for w, _ in weights)
    if total <= 0:
        return []

    by_number: Dict[int, ScorerCandidate] = {}
    for weight, track_id in weights:
        identity = identify_track(track_id)
        number = identity.get("jersey_number")
        if number is None:
            continue
        share = weight / total
        confidence = float(identity.get("jersey_confidence") or 0.0) or 0.5
        candidate = by_number.setdefault(
            int(number), ScorerCandidate(int(number), 0.0, "celebration", detail={"tracks": []})
        )
        candidate.weight += WEIGHT_CELEBRATION * share * confidence
        candidate.detail["tracks"].append(track_id)
    return list(by_number.values())


# ------------------------------------------------------------------- fusion
def live_candidate(scorer: Optional[dict], attribution: str = "") -> List[ScorerCandidate]:
    """The shirt number read off the shooter / last holder during play."""
    if not scorer or scorer.get("jersey_number") is None:
        return []
    base = WEIGHT_LIVE_POSSESSION if "possession" in (attribution or "") else WEIGHT_LIVE_SHOT
    confidence = float(scorer.get("jersey_confidence") or 0.0) or 0.5
    return [
        ScorerCandidate(
            int(scorer["jersey_number"]),
            base * confidence,
            "live",
            detail={"track_id": scorer.get("track_id"), "attribution": attribution or None},
        )
    ]


def fuse_scorer(candidates: Sequence[ScorerCandidate], squad: Sequence[dict] = ()) -> Optional[dict]:
    """The best-supported shirt number, its roster name, and a confidence.

    ``None`` when there is no evidence at all.
    """
    by_number = {int(p["jersey_no"]): p for p in squad or []}
    totals: Dict[int, float] = {}
    sources: Dict[int, List[str]] = {}
    for candidate in candidates:
        if candidate.jersey_number is None or candidate.weight <= 0:
            continue
        number = int(candidate.jersey_number)
        weight = candidate.weight
        player = by_number.get(number)
        if by_number and player is None:
            weight *= NOT_IN_SQUAD_FACTOR
        elif player is not None:
            weight *= POSITION_PRIOR.get(str(player.get("position") or "").upper(), 1.0)
        totals[number] = totals.get(number, 0.0) + weight
        sources.setdefault(number, []).append(candidate.source)
    if not totals:
        return None

    number, best = max(totals.items(), key=lambda item: item[1])
    share = best / sum(totals.values())
    strength = min(1.0, best / STRENGTH_FOR_FULL_CONFIDENCE)
    confidence = round(share * (0.5 + 0.5 * strength), 3)
    player = by_number.get(number)
    unconfirmed = confidence < UNCONFIRMED_BELOW
    return {
        "jersey_number": number,
        "player_name": None if unconfirmed else (player or {}).get("name"),
        "scorer_confidence": confidence,
        "unconfirmed": unconfirmed,
        "scorer_sources": sorted(set(sources[number])),
        "scorer_alternatives": {
            str(n): round(w, 3) for n, w in sorted(totals.items(), key=lambda item: -item[1]) if n != number
        },
    }
