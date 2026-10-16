"""Player names in the commentary: which squad player was named, and when.

Each player gets name keys from the roster: the full name, the surname (with
its particle - "van Dijk", "De Bruyne"), and the first name when it is unique
across both squads and long enough to be a name rather than a word
("Vinícius", "Rodrygo"). Single-name players ("Rodri", "Pedri") are their
full name. Keys and transcript words are compared casefolded and without
accents; long keys tolerate a speech-to-text misspelling (difflib >= 0.84).

A surname that is also an ordinary English word ("White", "Rice", "Young")
counts only together with the first name - otherwise "a white shirt" names Ben
White. A key shared by several players (two Silvas) gives a mention with
several candidates; the caller decides by team or by who had the ball.

Each mention also records the cue words around it (scoring, passing, foul,
being fouled), which ``commentary.assign`` turns into roles.
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .transcribe import Word

FUZZY_MIN_RATIO = 0.84
JOINED_MIN_RATIO = 0.9
FUZZY_MIN_LENGTH = 5
FIRST_NAME_MIN_LENGTH = 5
PARTICLES = {
    "da", "de", "di", "do", "dos", "das", "del", "der", "den", "van", "von", "la", "le",
    "el", "al", "ter", "ten", "mac", "mc", "st", "bin", "ibn", "ben",
}
SUFFIXES = {"jr", "junior", "sr", "ii", "iii"}
COMMON_WORDS = {
    "white", "rice", "young", "gray", "grey", "king", "brown", "green", "hall", "mount",
    "stones", "walker", "may", "page", "long", "little", "best", "cash", "field", "wood",
    "ward", "james", "rose", "bell", "hill", "wall", "mane", "lens", "neto", "gil", "can",
    "alba", "marc", "silva", "santos", "costa", "lima", "pereira", "alves", "dias", "sole",
    "said", "will", "max", "mark", "son", "kim", "lee", "park", "ali", "hoe", "back",
}
# Cue words, compared on normalised text, looked for within CUE_WINDOW words.
CUE_WINDOW = 4
SCORING_CUES = {
    "scores", "scored", "score", "goal", "goals", "finishes", "finish", "finished", "header",
    "heads", "headed", "volley", "volleys", "fires", "fired", "slots", "slotted", "taps",
    "tapped", "tucks", "tucked", "nets", "netted", "converts", "converted", "strikes", "smashes",
    "curls", "curled", "buries", "lashes", "rifles", "nods", "chips", "lobs", "brace", "hattrick",
}
PASSING_CUES = {
    "cross", "crosses", "crossed", "pass", "passes", "passed", "assist", "assists", "assisted",
    "delivery", "cutback", "squares", "squared", "lays", "laid", "picks", "picked", "finds", "found",
    "feeds", "fed", "threads", "corner", "setup", "set", "ball", "through",
    "finding", "picking", "playing", "plays", "looking", "slides", "whips", "whipped", "floats",
}
FOUL_CUES = {
    "foul", "fouls", "fouled", "booked", "booking", "yellow", "red", "card", "caution",
    "brings", "brought", "trips", "tripped", "clips", "clipped", "challenge", "tackle", "late",
    "hacks", "upends", "catches",
}
VICTIM_MARKERS = {"on", "down"}   # "foul on X", "brings down X"
AGENT_MARKERS = {"by", "from"}    # "foul by X", "cross from X"
# "for" names who a card is for, but after a pass it names the receiver
# ("looking for Romero" - Romero scored), so it is its own label.
FOR_MARKERS = {"for"}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    text = text.replace("'s", "").replace("’s", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


@dataclass(frozen=True)
class Candidate:
    team_id: int
    jersey_no: int
    name: str
    position: Optional[str]
    score: float  # 0-1: how well and how specifically the words matched
    raw_score: float = 0.0  # before sharing with other players of the same spoken name


@dataclass
class Mention:
    start: float
    end: float
    text: str
    candidates: List[Candidate]
    word_index: int
    cues_before: List[str] = field(default_factory=list)
    cues_after: List[str] = field(default_factory=list)

    def best_for_team(self, team_id: int) -> Optional[Candidate]:
        options = [c for c in self.candidates if team_id in (0, c.team_id)]
        if not options:
            return None
        if team_id and len(options) == 1 and len(self.candidates) > 1:
            # "Romero" was shared between Levante's and Sevilla's Romero; the
            # team settles it, so this player gets the full match back.
            return replace(options[0], score=options[0].raw_score or options[0].score)
        return max(options, key=lambda c: c.score)


@dataclass
class _Key:
    tokens: Tuple[str, ...]
    player: dict
    team_id: int
    weight: float


def _name_tokens(name: str) -> List[str]:
    return [t for t in normalise(name).split() if t not in SUFFIXES]


def build_keys(squads: Dict[int, Sequence[dict]]) -> List[_Key]:
    keys: List[_Key] = []
    first_names: Dict[str, int] = {}
    surnames: Dict[str, int] = {}
    for squad in squads.values():
        for player in squad:
            tokens = _name_tokens(player.get("name") or "")
            if len(tokens) >= 2:
                first_names[tokens[0]] = first_names.get(tokens[0], 0) + 1
            if tokens:
                surnames[tokens[-1]] = surnames.get(tokens[-1], 0) + 1
    for team_id, squad in squads.items():
        for player in squad:
            if not player.get("name") or player.get("jersey_no") is None:
                continue
            tokens = _name_tokens(player["name"])
            if not tokens:
                continue
            keys.append(_Key(tuple(tokens), player, team_id, 1.0))  # full name
            if len(tokens) == 1:
                continue
            surname = tokens[-1]
            if len(tokens) >= 3 and tokens[-2] in PARTICLES:
                keys.append(_Key(tuple(tokens[-2:]), player, team_id, 0.95))
            elif len(tokens) >= 3:
                keys.append(_Key(tuple(tokens[-2:]), player, team_id, 0.9))  # double surname
            if surname not in COMMON_WORDS and surname not in PARTICLES and len(surname) >= 3:
                keys.append(_Key((surname,), player, team_id, 0.85))
            first = tokens[0]
            if (
                first_names.get(first, 0) == 1
                and surnames.get(first, 0) == 0
                and len(first) >= FIRST_NAME_MIN_LENGTH
                and first not in COMMON_WORDS
            ):
                keys.append(_Key((first,), player, team_id, 0.7))
    return keys


_SOUNDS = (
    ("ph", "f"), ("th", "t"), ("qu", "k"), ("gue", "ge"), ("gui", "gi"), ("ck", "k"),
    ("ce", "se"), ("ci", "si"), ("z", "s"), ("v", "b"), ("y", "i"), ("w", "u"),
)


def phonetic(token: str) -> str:
    """A rough sound-alike form: speech-to-text spells names as it hears them
    ("Suasso" for Suazo, "Brugge" for Brugué, "Isac" for Isaac)."""
    text = token
    for written, sound in _SOUNDS:
        text = text.replace(written, sound)
    text = re.sub(r"(?<![cs])h", "", text)        # silent h, keep ch/sh
    text = re.sub(r"(.)\1+", r"\1", text)          # doubled letters
    return text


def _similar(a: str, b: str) -> float:
    """``a`` spoken, ``b`` a name key token."""
    if a == b:
        return 1.0
    if len(b) < FUZZY_MIN_LENGTH or len(a) < FUZZY_MIN_LENGTH - 1:
        return 0.0
    ratio = max(
        difflib.SequenceMatcher(None, a, b).ratio(),
        difflib.SequenceMatcher(None, phonetic(a), phonetic(b)).ratio(),
    )
    return ratio if ratio >= FUZZY_MIN_RATIO else 0.0


def _cues(tokens: Sequence[str], markers_from: int = 0) -> List[str]:
    """Cue labels in ``tokens``. "by/for/from/on/down" describe only the name
    right after them, so they count only from index ``markers_from`` on
    ("a cross from Bellingham and Vinícius": "from" is Bellingham's)."""
    found = []
    for index, token in enumerate(tokens):
        if token in SCORING_CUES:
            found.append("scoring")
        if token in PASSING_CUES:
            found.append("passing")
        if token in FOUL_CUES:
            found.append("foul")
        if index >= markers_from and token in VICTIM_MARKERS:
            found.append("victim")
        if index >= markers_from and token in AGENT_MARKERS:
            found.append("agent")
        if index >= markers_from and token in FOR_MARKERS:
            found.append("for")
    return found


def find_mentions(words: Sequence[Word], squads: Dict[int, Sequence[dict]]) -> List[Mention]:
    """Every place a squad player is named, longest match first, no overlaps."""
    keys = build_keys(squads)
    if not keys or not words:
        return []
    flat = [(normalise(w.text).split() or [""])[0] for w in words]
    max_len = max(len(k.tokens) for k in keys)
    # Speech-to-text misspells inside a name, rarely its first letter: only
    # keys sharing the first letter are compared (a full match has ~15k words).
    by_letter: Dict[str, List[_Key]] = {}
    for key in keys:
        by_letter.setdefault(key.tokens[0][:1], []).append(key)
    mentions: List[Mention] = []
    i = 0
    while i < len(flat):
        best: Dict[Tuple[int, int], Candidate] = {}
        best_len = 0
        for length in range(min(max_len, len(flat) - i), 0, -1):
            span = flat[i:i + length]
            if not all(span):
                continue
            for key in by_letter.get(span[0][:1], ()):
                if len(key.tokens) != length or abs(len(key.tokens[0]) - len(span[0])) > 2:
                    continue
                similarity = min(_similar(a, b) for a, b in zip(span, key.tokens))
                if similarity <= 0:
                    continue
                score = similarity * key.weight
                ident = (key.team_id, int(key.player["jersey_no"]))
                if ident not in best or best[ident].score < score:
                    best[ident] = Candidate(
                        key.team_id, int(key.player["jersey_no"]), key.player["name"],
                        key.player.get("position"), round(score, 3),
                    )
            if best:
                best_len = length
                break
        if not best and i + 1 < len(flat) and flat[i] and flat[i + 1]:
            # A name split in two by the transcript ("for Fana" for Fofana):
            # two adjacent words joined, against one-word keys, held to a
            # stricter similarity because two ordinary words can join into
            # something name-like.
            joined = flat[i] + flat[i + 1]
            for key in by_letter.get(joined[:1], ()):
                if len(key.tokens) != 1 or len(key.tokens[0]) < FUZZY_MIN_LENGTH + 1:
                    continue
                similarity = _similar(joined, key.tokens[0])
                if similarity < JOINED_MIN_RATIO:
                    continue
                ident = (key.team_id, int(key.player["jersey_no"]))
                score = round(similarity * key.weight, 3)
                if ident not in best or best[ident].score < score:
                    best[ident] = Candidate(
                        key.team_id, int(key.player["jersey_no"]), key.player["name"],
                        key.player.get("position"), score,
                    )
            if best:
                best_len = 2
        if not best:
            i += 1
            continue
        candidates = sorted(best.values(), key=lambda c: -c.score)
        if len(candidates) > 1:
            # One spoken name, several players: share the confidence.
            candidates = [
                Candidate(c.team_id, c.jersey_no, c.name, c.position, round(c.score / len(candidates), 3), c.score)
                for c in candidates
            ]
        end = i + best_len
        mentions.append(Mention(
            start=words[i].start,
            end=words[end - 1].end,
            text=" ".join(w.text for w in words[i:end]),
            candidates=candidates,
            word_index=i,
            cues_before=_cues(flat[max(0, i - CUE_WINDOW):i], markers_from=min(i, CUE_WINDOW) - 1),
            cues_after=_cues(flat[end:end + CUE_WINDOW], markers_from=CUE_WINDOW),
        ))
        i = end
    return mentions


RULED_OUT_PHRASES = (
    "ruled out", "disallowed", "flag up", "flag is up", "offside", "no goal", "chalked off",
    "overturned", "wont stand", "won t stand", "will not stand", "not given",
)


# Said after a flag or a check: the goal counts after all. Levante-Sevilla
# (2026-10-16): "the offside flag is up ... And the goal he's given".
STANDS_PHRASES = (  # compared after normalise(), which drops "'s" ("he's" -> "he")
    "goal is given", "goal he given", "goal given", "has been given", "it given", "it stands",
    "goal stands", "allowed to stand", "will stand", "it counts", "onside",
)


def ruled_out(words: Sequence[Word], start: float, end: float) -> Optional[str]:
    """The phrase, when the commentary says a goal does not count (and does
    not say afterwards that it stands)."""
    text = " " + " ".join(normalise(w.text) for w in words if start <= w.start <= end) + " "
    hits = [(text.find(f" {p} "), p) for p in RULED_OUT_PHRASES if f" {p} " in text]
    if not hits:
        return None
    position, phrase = min(hits)
    later = text[position:]
    if any(f" {s} " in later for s in STANDS_PHRASES):
        return None
    return phrase
