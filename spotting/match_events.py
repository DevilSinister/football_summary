"""Fouls, cards and set pieces from the spotting model, checked against other evidence.

The spotting model (E2E-Spot, SoccerNet-v2) scores 17 classes. Goals and shots
are fused in spotting/fusion.py; this module uses the rest:

* Fouls. A foul has up to five independent kinds of evidence:
    - tracks: the contact-and-consequence rule in track_events/fouls.py;
    - spotting "Foul";
    - a "Direct free-kick" or "Penalty" spotted soon after (both are only
      given for a foul - an indirect free kick is not used: offside gives one);
    - a card spotted soon after;
    - the commentary saying "foul", "booked", "free kick" ... around it.
  With ``double_check`` (features.yaml foul_double_check) a foul is kept only
  with at least two kinds, or with one very strong spotting reading. The track
  rule alone was measured wrong on the only checked foul (Porto v Man City,
  2026-09-24), which is why it no longer counts on its own.
* Cards. A spotted "Yellow card" / "Red card" / "Yellow->red card", kept with
  commentary or a foul before it, or alone when strong. Track cards (the
  referee's raised card, track_events/cards.py) need the same second source.
* Set pieces: corners, offsides, free kicks and substitutions, from spotting
  alone above a threshold. They carry no player.

Thresholds are starting points, not tuned: no labelled clip with fouls or
cards exists yet (docs/ACCURACY.md). Measure with scripts/eval_events.py.
"""

from typing import Dict, Iterable, List, Sequence

FOUL_ALONE = 0.5            # spotting "Foul" strong enough on its own
FOUL_SUPPORT = 0.12         # a weaker "Foul" still counts as one kind of evidence
KICK_SUPPORT = 0.25         # "Direct free-kick" / "Penalty" after a foul
CARD_ALONE = 0.5
CARD_SUPPORT = 0.15
SAME_FOUL_SECONDS = 4.0     # readings this close are the same foul
KICK_AFTER_SECONDS = 20.0   # a free kick is taken up to this long after its foul
CARD_AFTER_FOUL_SECONDS = 60.0
WORDS_BEFORE_SECONDS = 3.0
WORDS_AFTER_SECONDS = 12.0
SAME_CARD_SECONDS = 10.0

SET_PIECE_MIN = {"corner": 0.3, "offside": 0.3, "free_kick": 0.35, "substitution": 0.3}
SET_PIECE_SPACING_SECONDS = 8.0

# spotting.labels.event_name of each card class -> pipeline event type.
CARD_EVENTS = {"yellow_card": "yellow_card", "red_card": "red_card", "yellow_red_card": "red_card"}
FOUL_WORDS = {
    "foul", "fouled", "fouls", "booked", "booking", "caution", "cautioned", "tripped", "trips",
    "penalty", "yellow", "card", "whistle", "freekick",
}
CARD_WORDS = {"booked", "booking", "yellow", "caution", "cautioned", "red", "sent", "dismissed", "card"}

# Spotting labels each switch needs the spotter to keep.
SPOTTING_CLASSES = {
    "fouls": ("Foul", "Direct free-kick", "Penalty", "Yellow card", "Red card", "Yellow->red card"),
    "cards": ("Yellow card", "Red card", "Yellow->red card", "Foul"),
    "corner": ("Corner",),
    "offside": ("Offside",),
    "free_kick": ("Direct free-kick", "Indirect free-kick"),
    "substitution": ("Substitution",),
}


def _frame(event: dict) -> int:
    return int(event.get("frame", 0))


def _clean(token: str) -> str:
    return "".join(c for c in str(token).lower() if c.isalnum())


def commentary_cues(words: Sequence, start: float, end: float, vocabulary: Iterable[str]) -> List[str]:
    """Cue words said between ``start`` and ``end`` seconds ("free kick" counts as "freekick")."""
    vocabulary = set(vocabulary)
    found: List[str] = []
    previous = ""
    for word in words or ():
        at = float(getattr(word, "start", 0.0))
        token = _clean(getattr(word, "text", ""))
        if start <= at <= end:
            if token in vocabulary:
                found.append(token)
            elif previous == "free" and token in ("kick", "kicks") and "freekick" in vocabulary:
                found.append("freekick")
        previous = token
    return found


def _strongest(spotted: Sequence[dict], kinds: Iterable[str], low: int, high: int, minimum: float) -> float:
    kinds = set(kinds)
    values = [
        float(s.get("confidence", 0.0)) for s in spotted
        if s.get("event") in kinds and low <= _frame(s) <= high and float(s.get("confidence", 0.0)) >= minimum
    ]
    return max(values, default=0.0)


def fuse_fouls(events: Sequence[dict], spotted: Sequence[dict], fps: float,
               words: Sequence = (), double_check: bool = True) -> List[dict]:
    """Every event, with fouls re-judged on all their evidence (see module doc)."""
    fps = max(float(fps), 1.0)
    out = [e for e in events if e.get("event") != "foul"]
    candidates: List[Dict] = [{"frame": _frame(e), "track": e, "spot": 0.0}
                              for e in events if e.get("event") == "foul"]
    for spot in sorted((s for s in spotted if s.get("event") == "foul"),
                       key=lambda s: -float(s.get("confidence", 0.0))):
        confidence = float(spot.get("confidence", 0.0))
        if confidence < FOUL_SUPPORT:
            continue
        near = min(
            (c for c in candidates if abs(c["frame"] - _frame(spot)) <= SAME_FOUL_SECONDS * fps),
            key=lambda c: abs(c["frame"] - _frame(spot)), default=None,
        )
        if near is not None:
            near["spot"] = max(near["spot"], confidence)
        else:
            candidates.append({"frame": _frame(spot), "track": None, "spot": confidence})

    for candidate in sorted(candidates, key=lambda c: c["frame"]):
        frame = candidate["frame"]
        seconds = frame / fps
        evidence: Dict[str, object] = {}
        if candidate["track"] is not None:
            evidence["tracks"] = round(float(candidate["track"].get("confidence", 0.5) or 0.5), 3)
        if candidate["spot"]:
            evidence["spotting_foul"] = round(candidate["spot"], 3)
        kick = _strongest(spotted, ("direct_free_kick", "penalty"), frame - int(SAME_FOUL_SECONDS * fps),
                          frame + int(KICK_AFTER_SECONDS * fps), KICK_SUPPORT)
        if kick:
            evidence["free_kick_or_penalty"] = round(kick, 3)
        card = _strongest(spotted, CARD_EVENTS, frame - int(SAME_FOUL_SECONDS * fps),
                          frame + int(CARD_AFTER_FOUL_SECONDS * fps), CARD_SUPPORT)
        if card:
            evidence["card"] = round(card, 3)
        cues = commentary_cues(words, seconds - WORDS_BEFORE_SECONDS, seconds + WORDS_AFTER_SECONDS, FOUL_WORDS)
        if cues:
            evidence["commentary"] = sorted(set(cues))
        strong = candidate["spot"] >= FOUL_ALONE or card >= CARD_ALONE
        if double_check:
            accept = len(evidence) >= 2 or strong
        else:
            accept = candidate["track"] is not None or strong or len(evidence) >= 2
        if not accept:
            continue
        if candidate["track"] is not None:
            foul = candidate["track"]
        else:
            foul = {
                "event": "foul", "frame": frame, "end_frame": frame, "source": "spotting",
                "team_id": 0, "team_name": "unknown_team", "participants": [],
                "confidence": round(candidate["spot"], 3),
            }
        details = foul.setdefault("details", {})
        details["evidence"] = evidence
        details["evidence_kinds"] = len(evidence)
        out.append(foul)
    out.sort(key=_frame)
    return out


def fuse_cards(events: Sequence[dict], spotted: Sequence[dict], fps: float,
               words: Sequence = (), double_check: bool = True) -> List[dict]:
    """Every event, with cards re-judged and spotted cards added (see module doc)."""
    fps = max(float(fps), 1.0)
    out = [e for e in events if e.get("event") not in ("yellow_card", "red_card")]
    fouls = [e for e in out if e.get("event") == "foul"]
    kept: List[dict] = []

    def words_near(frame: int) -> List[str]:
        return sorted(set(commentary_cues(
            words, frame / fps - WORDS_BEFORE_SECONDS, frame / fps + WORDS_AFTER_SECONDS, CARD_WORDS)))

    for card in (e for e in events if e.get("event") in ("yellow_card", "red_card")):
        frame = _frame(card)
        evidence: Dict[str, object] = {"tracks": True}
        spot = _strongest(spotted, CARD_EVENTS, frame - int(SAME_CARD_SECONDS * fps),
                          frame + int(SAME_CARD_SECONDS * fps), CARD_SUPPORT)
        if spot:
            evidence["spotting_card"] = round(spot, 3)
        cues = words_near(frame)
        if cues:
            evidence["commentary"] = cues
        if double_check and len(evidence) < 2:
            continue
        card.setdefault("details", {})["evidence"] = evidence
        kept.append(card)

    for spot in sorted((s for s in spotted if s.get("event") in CARD_EVENTS
                        and float(s.get("confidence", 0.0)) >= CARD_SUPPORT),
                       key=lambda s: -float(s["confidence"])):
        frame, confidence = _frame(spot), float(spot["confidence"])
        if any(abs(_frame(k) - frame) <= SAME_CARD_SECONDS * fps for k in kept):
            continue
        evidence = {"spotting_card": round(confidence, 3)}
        cues = words_near(frame)
        if cues:
            evidence["commentary"] = cues
        foul = max((f for f in fouls if 0 <= frame - _frame(f) <= CARD_AFTER_FOUL_SECONDS * fps),
                   key=_frame, default=None)
        if foul is not None:
            evidence["foul_at_frame"] = _frame(foul)
        if double_check:
            accept = confidence >= CARD_ALONE or len(evidence) >= 2
        else:
            accept = confidence >= 2 * CARD_SUPPORT or len(evidence) >= 2
        if not accept:
            continue
        participants: List[dict] = []
        team_id, team_name = 0, "unknown_team"
        if foul is not None:
            fouler = next((p for p in foul.get("participants") or [] if p.get("role") == "fouler"), None)
            if fouler is not None:
                participants = [{**fouler, "role": "player"}]
                team_id = int(fouler.get("team_id", 0) or 0)
                team_name = fouler.get("team_name", "unknown_team")
        kept.append({
            "event": CARD_EVENTS[spot["event"]], "frame": frame, "end_frame": frame, "source": "spotting",
            "team_id": team_id, "team_name": team_name, "participants": participants,
            "confidence": round(confidence, 3),
            "details": {"evidence": evidence, "spotting_label": (spot.get("details") or {}).get("spotting_label")},
        })
    out.extend(kept)
    out.sort(key=_frame)
    return out


def spotted_set_pieces(spotted: Sequence[dict], fps: float, kinds: Iterable[str]) -> List[dict]:
    """Corners, offsides, free kicks and substitutions above their thresholds.

    ``kinds`` names the switched-on ones. Readings of one kind closer than
    SET_PIECE_SPACING_SECONDS are one event (the model often fires twice).
    """
    fps = max(float(fps), 1.0)
    label_kind = {"corner": "corner", "offside": "offside", "substitution": "substitution",
                  "direct_free_kick": "free_kick", "indirect_free_kick": "free_kick"}
    wanted = set(kinds)
    produced: List[dict] = []
    for spot in sorted(spotted, key=lambda s: -float(s.get("confidence", 0.0))):
        kind = label_kind.get(spot.get("event"))
        if kind not in wanted or float(spot.get("confidence", 0.0)) < SET_PIECE_MIN[kind]:
            continue
        if any(p["event"] == kind and abs(_frame(p) - _frame(spot)) <= SET_PIECE_SPACING_SECONDS * fps
               for p in produced):
            continue
        produced.append({
            "event": kind, "frame": _frame(spot), "end_frame": _frame(spot), "source": "spotting",
            "team_id": 0, "team_name": "unknown_team", "participants": [],
            "confidence": round(float(spot["confidence"]), 3),
            "details": {"spotting_label": (spot.get("details") or {}).get("spotting_label")},
        })
    return sorted(produced, key=_frame)


def spotting_classes_for(switches: Dict[str, bool]) -> List[str]:
    """Extra spotting labels to keep for the switched-on features."""
    labels: List[str] = []
    for key, classes in SPOTTING_CLASSES.items():
        if switches.get(key):
            labels.extend(c for c in classes if c not in labels)
    return labels
