"""Who is each tracked player? A running probability over the team's squad.

For each track the tracker keeps a log-probability over the players of the
team its kit was read as (the roster plus anyone enrolled in the gallery) and
an "unknown" option for a player the gallery and roster cannot name. Evidence
is added as it arrives, each piece a likelihood:

* appearance, every ``observe_every_seconds`` while the track is seen, from a
  full-body crop's embedding against the enrolled reference images, weighted
  by crop size (a 40 px figure says little);
* a shirt-number read, with the OCR's confidence;
* role: a track the detector calls a goalkeeper favours the squad's keepers.

Two constraints tie tracks together:

* one name per player on screen - when several tracks seen together would take
  the same name, a Hungarian assignment over their probabilities settles it;
* a camera cut ends every track (ids restart). The ended track is kept; a new
  track of the same team whose first appearance closely matches it inherits
  half its evidence. This re-links the fragments a highlight package creates.

Positions: each appearance observation stores the foot point in pixels and,
when a pitch calibration is available, in metres; the match result lists them
per track. Position is not yet used as evidence - that needs a working pitch
model and a position history per player kept across matches.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .embedder import body_crop, crop_quality

UNKNOWN = -1
# Measured 2026-10-14 with OSNet x1.0 (MSMT17 weights, trained on pedestrians)
# on the 1080p fixture clip, 14 tracklets of >= 80 px players, enrolment from
# a tracklet's start and tests >= 2 s later in the same camera shot:
#   cosine to own prototype, median 0.80 (1 image) - 0.84 (2-10 images);
#   cosine to a team-mate's prototype, median 0.73-0.77;
#   top-1 among all 14: 0.31 / 0.34 / 0.38 / 0.38 with 1 / 2 / 5 / 10 images;
#   top-1 among the same kit: 0.32 / 0.40 / 0.40 / 0.40 (chance ~0.15-0.2).
# Appearance therefore only nudges the belief; shirt numbers decide. A
# football-trained re-ID model (SoccerNet-ReID) is the way to raise this.
UNKNOWN_SIMILARITY = 0.82      # below this, "someone not enrolled" is more likely
SIMILARITY_SCALE = 0.1         # temperature of the similarity softmax / logistic
APPEARANCE_FLOOR = 0.5         # share of each appearance likelihood spread uniformly
LINK_SIMILARITY = 0.88         # a new tracklet continues an ended one...
LINK_MARGIN = 0.04             # ...and matches it clearly better than any other
LINK_WEIGHT = 0.5
LINK_MAX_GAP_SECONDS = 120.0
NAME_MIN_PROBABILITY = 0.7     # below this no name is attached
GOALKEEPER_PRIOR = math.log(4.0)
TEAM_MIN_CONFIDENCE = 0.5


@dataclass
class _Candidate:
    jersey_no: int
    name: Optional[str]
    position: Optional[str]


@dataclass
class _Track:
    track_id: int
    team_id: int
    log_post: np.ndarray
    first_frame: int
    last_frame: int
    last_observed: int = -10**9
    observations: int = 0
    embedding_sum: Optional[np.ndarray] = None
    linked_from: Optional[int] = None
    role: str = "player"
    role_applied: bool = False
    named_by_commentary: int = 0
    positions: List[Tuple[int, float, float, Optional[float], Optional[float]]] = field(default_factory=list)

    def mean_embedding(self) -> Optional[np.ndarray]:
        if self.embedding_sum is None:
            return None
        return self.embedding_sum / max(np.linalg.norm(self.embedding_sum), 1e-6)

    def probabilities(self) -> np.ndarray:
        weights = np.exp(self.log_post - self.log_post.max())
        return weights / weights.sum()


class IdentityTracker:
    def __init__(
        self,
        squads: Dict[int, Sequence[dict]],
        galleries: Dict[int, object],
        embedder,
        fps: float,
        observe_every_seconds: float = 0.5,
    ) -> None:
        """``squads``: team id -> [{jersey_no, name, position}] (roster).
        ``galleries``: team id -> TeamPrototypes (enrolled images) or None."""
        self.fps = max(float(fps), 1.0)
        self.embedder = embedder
        self.galleries = {team: g for team, g in galleries.items() if g is not None}
        self.observe_every = observe_every_seconds * self.fps
        self.candidates: Dict[int, List[_Candidate]] = {}
        for team_id in set(squads) | set(self.galleries):
            by_number: Dict[int, _Candidate] = {}
            for player in squads.get(team_id) or []:
                if player.get("jersey_no") is not None:
                    number = int(player["jersey_no"])
                    by_number[number] = _Candidate(number, player.get("name"), player.get("position"))
            gallery = self.galleries.get(team_id)
            if gallery is not None:
                for number, name in zip(gallery.jerseys, gallery.names):
                    known = by_number.get(number)
                    by_number[number] = _Candidate(
                        number, name or (known.name if known else None), known.position if known else None
                    )
            self.candidates[team_id] = [by_number[n] for n in sorted(by_number)]
        self.tracks: Dict[int, _Track] = {}
        self.ended: List[_Track] = []
        self._number_reads: set = set()

    # -- evidence ---------------------------------------------------------
    def _track(self, track_id: int, team_id: int, frame: int) -> Optional[_Track]:
        if team_id not in self.candidates:
            return None
        track = self.tracks.get(track_id)
        if track is None or track.team_id != team_id:
            # A new track, or a confident team flip: names of the other squad
            # are impossible, so the belief starts again over the new squad.
            size = len(self.candidates[team_id]) + 1  # + unknown
            track = _Track(track_id, team_id, np.full(size, -math.log(size)), frame, frame)
            self.tracks[track_id] = track
        return track

    def appearance_loglik(self, team_id: int, embedding: np.ndarray) -> np.ndarray:
        candidates = self.candidates[team_id]
        size = len(candidates) + 1
        gallery = self.galleries.get(team_id)
        if gallery is None:
            return np.zeros(size)
        enrolled = {number: i for i, number in enumerate(gallery.jerseys)}
        member_probs = gallery.member_probabilities(embedding, SIMILARITY_SCALE)
        margin = (gallery.best_similarity(embedding) - UNKNOWN_SIMILARITY) / SIMILARITY_SCALE
        p_unknown = 1.0 / (1.0 + math.exp(min(max(margin, -50.0), 50.0)))
        not_enrolled = sum(1 for c in candidates if c.jersey_no not in enrolled) + 1  # + unknown
        likelihood = np.empty(size)
        for i, candidate in enumerate(candidates):
            if candidate.jersey_no in enrolled:
                likelihood[i] = (1.0 - p_unknown) * member_probs[enrolled[candidate.jersey_no]]
            else:
                likelihood[i] = p_unknown / not_enrolled
        likelihood[-1] = p_unknown / not_enrolled
        # Tempered: one crop can shift the odds by at most about 1/APPEARANCE_FLOOR,
        # so a single bad embedding (motion blur, occlusion, a team-mate in the
        # box) cannot wipe out a shirt-number read or earlier observations.
        likelihood = (1.0 - APPEARANCE_FLOOR) * likelihood + APPEARANCE_FLOOR / size
        return np.log(likelihood)

    def observe(self, frame_number: int, frame: np.ndarray, players: Dict[int, dict], calibration=None) -> None:
        """Called on analysed, non-transition frames; works at its own stride."""
        due: List[Tuple[_Track, np.ndarray, float]] = []
        for track_id, player in players.items():
            if float(player.get("team_confidence", 0.0) or 0.0) < TEAM_MIN_CONFIDENCE:
                continue
            track = self._track(int(track_id), int(player.get("team", 0) or 0), frame_number)
            if track is None:
                continue
            track.last_frame = frame_number
            track.role = str(player.get("role", track.role))
            if frame_number - track.last_observed < self.observe_every:
                continue
            track.last_observed = frame_number
            x1, _, x2, y2 = player["bbox"]
            foot = ((x1 + x2) / 2.0, float(y2))
            metres = calibration.to_pitch(foot) if calibration is not None else None
            track.positions.append((
                frame_number, round(foot[0], 1), round(foot[1], 1),
                None if metres is None else round(float(metres[0]), 2),
                None if metres is None else round(float(metres[1]), 2),
            ))
            if track.team_id not in self.galleries or self.embedder is None:
                continue
            crop = body_crop(frame, player["bbox"])
            if crop is not None:
                due.append((track, crop, crop_quality(crop)))
        if due:
            embeddings = self.embedder.embed([crop for _, crop, _ in due])
            for (track, _, quality), embedding in zip(due, embeddings):
                self.add_appearance(track.track_id, embedding, quality)

    def add_appearance(self, track_id: int, embedding: np.ndarray, quality: float = 1.0) -> None:
        track = self.tracks[track_id]
        first = track.embedding_sum is None
        track.embedding_sum = embedding.copy() if first else track.embedding_sum + embedding
        track.observations += 1
        track.log_post = track.log_post + quality * self.appearance_loglik(track.team_id, embedding)
        if first:
            self._link_to_ended(track, embedding)
        self._apply_role(track)

    def add_number_read(self, track_id: int, jersey_no: int, confidence: float) -> None:
        """One OCR read is one piece of evidence, however many events show the track."""
        track = self._find(int(track_id))
        key = (int(track_id), int(jersey_no))
        if track is None or key in self._number_reads:
            return
        self._number_reads.add(key)
        candidates = self.candidates[track.team_id]
        size = len(candidates) + 1
        c = float(np.clip(confidence, 0.05, 0.95))
        likelihood = np.full(size, (1.0 - c) / size)
        matches = [i for i, cand in enumerate(candidates) if cand.jersey_no == int(jersey_no)]
        for i in matches or [size - 1]:  # a number nobody in the squad wears: someone unknown
            likelihood[i] += c
        track.log_post = track.log_post + np.log(likelihood)

    def add_name_evidence(self, track_id: int, team_id: int, jersey_no: int, strength: float) -> None:
        """A commentator named this player while this track had the ball.

        Each mention is its own piece of evidence (a player named three times
        on the ball is likelier than one named once). Ignored when the named
        player is not in this track's team.
        """
        track = self._find(int(track_id))
        if track is None or track.team_id != int(team_id):
            return
        candidates = self.candidates[track.team_id]
        size = len(candidates) + 1
        c = float(np.clip(strength, 0.05, 0.95))
        likelihood = np.full(size, (1.0 - c) / size)
        for i, cand in enumerate(candidates):
            if cand.jersey_no == int(jersey_no):
                likelihood[i] += c
        track.log_post = track.log_post + np.log(likelihood)
        track.named_by_commentary += 1

    def _apply_role(self, track: _Track) -> None:
        if track.role_applied:
            return
        track.role_applied = True
        keeper = track.role == "goalkeeper"
        for i, candidate in enumerate(self.candidates[track.team_id]):
            if candidate.position and (candidate.position == "GK") != keeper:
                track.log_post[i] -= GOALKEEPER_PRIOR

    # -- cuts -------------------------------------------------------------
    def end_scene(self) -> None:
        """A camera cut: every live track ends (ids restart after it)."""
        self.ended.extend(self.tracks.values())
        self.tracks = {}

    def _link_to_ended(self, track: _Track, embedding: np.ndarray) -> None:
        scored = []
        for old in self.ended:
            mean = old.mean_embedding()
            if mean is None or old.team_id != track.team_id:
                continue
            if track.first_frame - old.last_frame > LINK_MAX_GAP_SECONDS * self.fps:
                continue
            scored.append((float(mean @ embedding), old))
        scored.sort(key=lambda item: -item[0])
        best = None
        if scored and scored[0][0] >= LINK_SIMILARITY:
            runner_up = scored[1][0] if len(scored) > 1 else -1.0
            if scored[0][0] - runner_up >= LINK_MARGIN:
                best = scored[0][1]
        if best is not None:
            track.log_post = track.log_post + LINK_WEIGHT * (best.log_post - best.log_post.max())
            track.linked_from = best.track_id

    # -- answers ----------------------------------------------------------
    def _find(self, track_id: int) -> Optional[_Track]:
        track = self.tracks.get(track_id)
        if track is not None:
            return track
        return next((t for t in reversed(self.ended) if t.track_id == track_id), None)

    def resolve(self, track_ids: Iterable[int]) -> Dict[int, Tuple[int, float]]:
        """Candidate index and probability per track, one name per player.

        Tracks of one team seen together cannot be the same person; "unknown"
        can repeat.
        """
        from scipy.optimize import linear_sum_assignment

        answer: Dict[int, Tuple[int, float]] = {}
        by_team: Dict[int, List[_Track]] = {}
        for track_id in dict.fromkeys(int(t) for t in track_ids):
            track = self._find(track_id)
            if track is not None:
                by_team.setdefault(track.team_id, []).append(track)
        for team_id, tracks in by_team.items():
            named = len(self.candidates[team_id])
            probs = np.stack([t.probabilities() for t in tracks])
            # Columns: each named candidate once, then one "unknown" per track.
            cost = np.hstack([
                -np.log(np.maximum(probs[:, :named], 1e-9)),
                np.tile(-np.log(np.maximum(probs[:, -1:], 1e-9)), (1, len(tracks))),
            ])
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if col < named:
                    answer[tracks[row].track_id] = (int(col), float(probs[row, col]))
                else:
                    answer[tracks[row].track_id] = (UNKNOWN, float(probs[row, -1]))
        return answer

    def name_for(self, track_id: int, concurrent: Iterable[int] = ()) -> Optional[dict]:
        """Name for a track when it is probable enough, else None."""
        track = self._find(int(track_id))
        if track is None:
            return None
        index, probability = self.resolve([int(track_id), *concurrent])[int(track_id)]
        if index == UNKNOWN or probability < NAME_MIN_PROBABILITY:
            return None
        candidate = self.candidates[track.team_id][index]
        return {
            "jersey_number": candidate.jersey_no,
            "player_name": candidate.name,
            "probability": round(probability, 3),
            "observations": track.observations,
        }

    def summary(self) -> List[dict]:
        latest: Dict[int, _Track] = {}
        for track in [*self.ended, *self.tracks.values()]:
            latest[track.track_id] = track
        out = []
        for track in sorted(latest.values(), key=lambda t: t.first_frame):
            probs = track.probabilities()
            best = int(np.argmax(probs))
            candidate = None if best == len(probs) - 1 else self.candidates[track.team_id][best]
            step = max(1, len(track.positions) // 50)
            out.append({
                "track_id": track.track_id,
                "team_id": track.team_id,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "appearance_observations": track.observations,
                "commentary_mentions": track.named_by_commentary,
                "linked_from_track": track.linked_from,
                "jersey_number": candidate.jersey_no if candidate else None,
                "player_name": candidate.name if candidate else None,
                "probability": round(float(probs[best]), 3),
                "positions": [list(p) for p in track.positions[::step]],
            })
        return out
