from typing import Dict, Optional, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans

from match_config import MatchConfig
from .calibration import bgr_to_lab, kit_color_bgr


# Lab channel weights for kit comparison. Lightness swings by 40 units between
# a player in sun and one in shadow, so unweighted KMeans split the players into
# "lit" and "shaded" rather than into the two strips. Measured on the 480p
# reference clip (202 crops): white shirts read a = -7 (grass tint) while the
# blurred red-and-white stripes read a = +12 to +17, but their b channel swung
# from +17 (stripes) to -27 (blue shorts in the box) between crops. The red-green
# axis is therefore the one that decides, lightness and blue-yellow only assist.
KIT_FEATURE_WEIGHTS = np.asarray([0.5, 3.0, 1.0], dtype=float)


def kit_feature(color_bgr) -> np.ndarray:
    """Weighted Lab vector used for every kit distance in this module."""
    return bgr_to_lab(color_bgr) * KIT_FEATURE_WEIGHTS


def _feature_to_bgr(feature) -> np.ndarray:
    lab = np.asarray(feature, dtype=np.float32) / KIT_FEATURE_WEIGHTS.astype(np.float32)
    pixel = lab.reshape(1, 1, 3).astype(np.float32)
    bgr = cv2.cvtColor(pixel, cv2.COLOR_Lab2BGR).reshape(3)
    return np.clip(bgr * 255.0, 0, 255)


class TeamAssigner:
    """Assigns tracked players to one of two teams by kit colour.

    Two properties matter here and were previously absent:

    * The per-crop colour is deterministic. It used to come from an unseeded
      KMeans(n_init=1) fit whose background cluster was voted from four corner
      pixels, two of which sit at mid-torso - so a tight crop could invert shirt
      and background, and the same frame could answer differently between runs.
    * A track is decided by votes across many frames, not by whichever frame it
      happened to appear in first. The old cache froze the first non-zero answer
      forever, which is why one bad entry frame put a player on the wrong team
      for the whole match.
    """

    def __init__(
        self,
        match_config: Optional[MatchConfig] = None,
        color_distance_threshold: float = 62.0,
        color_ambiguity_margin: float = 10.0,
        vote_weight_target: float = 4.0,
    ):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_assignment_dict = {}
        self.player_votes: Dict[int, Dict[int, float]] = {}
        self.match_config = match_config
        self.color_distance_threshold = color_distance_threshold
        self.color_ambiguity_margin = color_ambiguity_margin
        self.vote_weight_target = vote_weight_target
        self.kmeans = None
        self._team_lab = {}
        self.calibrated = False

        if match_config is not None:
            for team in match_config.teams:
                self.team_colors[team.team_id] = np.asarray(
                    team.primary_color_bgr, dtype=float
                )
                self._team_lab[team.team_id] = kit_feature(team.primary_color_bgr)

    def calibrate_from_samples(
        self,
        colors_bgr,
        min_samples: int = 12,
        min_separation: float = 15.0,
        min_mapping_margin: float = 8.0,
    ) -> bool:
        """Fit the two kit colours actually on screen and name them.

        Matching each crop straight against the configured hex colours fails
        on any kit that is not a flat block of colour: at 480p Atletico's red
        and white stripes blur to a grey-brown that is 50-70 Lab units from
        white and 80-100 from red, so every striped player was either unknown
        or handed to the white team. The observed colours are clustered into
        two kits first, and the clusters are then mapped onto the configured
        teams by which pairing is closer overall. The hex colours only have to
        be *relatively* right.

        Returns False, leaving direct matching in place, when the samples do
        not clearly show two kits (one team on screen, too few crops, or an
        ambiguous pairing).
        """
        if self.match_config is None:
            return False
        samples = [np.asarray(color, dtype=float) for color in colors_bgr if color is not None]
        if len(samples) < min_samples:
            return False
        labs = np.asarray([kit_feature(color) for color in samples], dtype=float)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=7)
        labels = kmeans.fit_predict(labs)
        centres = kmeans.cluster_centers_
        counts = np.bincount(labels, minlength=2)
        if counts.min() < max(4, int(0.1 * len(samples))):
            return False
        separation = float(np.linalg.norm(centres[0] - centres[1]))
        if separation < min_separation:
            return False

        teams = list(self.match_config.teams)
        team_labs = [kit_feature(team.primary_color_bgr) for team in teams]
        distance = [
            [float(np.linalg.norm(centres[cluster] - team_labs[index])) for index in range(2)]
            for cluster in range(2)
        ]
        straight = distance[0][0] + distance[1][1]
        crossed = distance[0][1] + distance[1][0]
        if abs(straight - crossed) < min_mapping_margin:
            return False
        if straight <= crossed:
            mapping = {teams[0].team_id: 0, teams[1].team_id: 1}
        else:
            mapping = {teams[0].team_id: 1, teams[1].team_id: 0}
        for team in teams:
            self._team_lab[team.team_id] = centres[mapping[team.team_id]]
            centre_bgr = _feature_to_bgr(centres[mapping[team.team_id]])
            self.team_colors[team.team_id] = centre_bgr
        # Two kits that sit close together need a tighter ambiguity margin or
        # nothing between them is ever decided.
        self.color_ambiguity_margin = min(self.color_ambiguity_margin, max(4.0, 0.2 * separation))
        self.kmeans = kmeans
        self.calibrated = True
        return True

    def get_player_color(self, frame, bbox):
        """Median kit colour of the torso, in BGR."""
        color = kit_color_bgr(frame, bbox)
        if color is None:
            raise ValueError("player bounding box does not contain a usable crop")
        return color

    def assign_team_color(self, frame, player_detections):
        """Legacy single-frame calibration, kept for the archived main v1 and v2."""
        if self.match_config is not None:
            return self.team_colors

        player_colors = []
        for _, player_detection in player_detections.items():
            try:
                player_colors.append(
                    self.get_player_color(frame, player_detection["bbox"])
                )
            except ValueError:
                continue

        if len(player_colors) < 2:
            raise ValueError(
                "at least two player crops are required for team calibration"
            )

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=7)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    @staticmethod
    def _bgr_to_lab(color: Sequence[float]) -> np.ndarray:
        return kit_feature(color)

    def match_color_to_team(self, player_color) -> Dict[str, object]:
        """Map a BGR player color to a configured team, rejecting weak matches."""
        if self.match_config is None:
            raise RuntimeError("match_color_to_team requires a MatchConfig")

        player_lab = self._bgr_to_lab(player_color)
        distances = []
        for team in self.match_config.teams:
            team_lab = self._team_lab.get(team.team_id)
            if team_lab is None:
                team_lab = kit_feature(team.primary_color_bgr)
                self._team_lab[team.team_id] = team_lab
            distances.append((float(np.linalg.norm(player_lab - team_lab)), team))
        distances.sort(key=lambda item: item[0])

        best_distance, best_team = distances[0]
        second_distance = distances[1][0]
        separation = second_distance - best_distance
        distance_confidence = max(
            0.0, 1.0 - best_distance / self.color_distance_threshold
        )
        separation_confidence = min(
            1.0, separation / max(self.color_ambiguity_margin * 2.0, 1.0)
        )
        confidence = round(
            (0.7 * distance_confidence) + (0.3 * separation_confidence), 3
        )

        if (
            best_distance > self.color_distance_threshold
            or separation < self.color_ambiguity_margin
        ):
            return {
                "team_id": 0,
                "team_name": "unknown_team",
                "confidence": confidence,
                "color_distance": round(best_distance, 3),
            }

        return {
            "team_id": best_team.team_id,
            "team_name": best_team.name,
            "confidence": confidence,
            "color_distance": round(best_distance, 3),
        }

    def _observe(self, frame, player_bbox) -> Dict[str, object]:
        """What a single frame thinks this player wears."""
        player_color = self.get_player_color(frame, player_bbox)
        if self.match_config is not None:
            return self.match_color_to_team(player_color)

        if self.kmeans is None:
            raise RuntimeError("team colors have not been calibrated")
        team_id = int(self.kmeans.predict(np.asarray(player_color).reshape(1, -1))[0]) + 1
        return {
            "team_id": team_id,
            "team_name": f"team_{team_id}",
            "confidence": 1.0,
            "color_distance": 0.0,
        }

    def _team_name(self, team_id: int) -> str:
        if team_id == 0:
            return "unknown_team"
        if self.match_config is not None:
            for team in self.match_config.teams:
                if team.team_id == team_id:
                    return team.name
        return f"team_{team_id}"

    @staticmethod
    def _unknown(vote_weight: float = 0.0) -> Dict[str, object]:
        return {
            "team_id": 0,
            "team_name": "unknown_team",
            "confidence": 0.0,
            "color_distance": 0.0,
            "vote_weight": vote_weight,
        }

    def get_player_assignment(self, frame, player_bbox, player_id) -> Dict[str, object]:
        player_id = int(player_id)
        settled = self.player_assignment_dict.get(player_id)

        # Stop re-measuring once a track has accumulated enough agreeing evidence.
        # This bounds the per-frame cost without freezing on one early observation.
        if (
            settled is not None
            and float(settled.get("vote_weight", 0.0)) >= self.vote_weight_target
        ):
            return settled

        try:
            observation = self._observe(frame, player_bbox)
        except (ValueError, RuntimeError):
            return settled or self._unknown()

        votes = self.player_votes.setdefault(player_id, {})
        observed_team = int(observation["team_id"])
        if observed_team != 0:
            # Weight each vote by how confident that frame was, with a small floor
            # so a run of weak but consistent frames can still settle a track.
            votes[observed_team] = votes.get(observed_team, 0.0) + max(
                float(observation["confidence"]), 0.15
            )

        if not votes:
            assignment = self._unknown()
            self.player_assignment_dict[player_id] = assignment
            self.player_team_dict[player_id] = 0
            return assignment

        ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        winner_id, winner_weight = ranked[0]
        runner_up_weight = ranked[1][1] if len(ranked) > 1 else 0.0
        total_weight = sum(votes.values())
        margin = (winner_weight - runner_up_weight) / max(total_weight, 1e-9)
        share = winner_weight / max(total_weight, 1e-9)

        assignment = {
            "team_id": int(winner_id),
            "team_name": self._team_name(int(winner_id)),
            "confidence": round(min(1.0, margin * share), 3),
            "color_distance": observation.get("color_distance", 0.0),
            "vote_weight": round(float(winner_weight), 3),
        }
        self.player_assignment_dict[player_id] = assignment
        self.player_team_dict[player_id] = assignment["team_id"]
        return assignment

    def get_player_team(self, frame, player_bbox, player_id):
        return self.get_player_assignment(frame, player_bbox, player_id)["team_id"]

    def forget_players(self, active_ids) -> None:
        """Drop per-track state for tracks that are no longer on screen."""
        active = {int(value) for value in active_ids}
        for store in (
            self.player_votes,
            self.player_assignment_dict,
            self.player_team_dict,
        ):
            for stale in [key for key in store if key not in active]:
                del store[stale]
