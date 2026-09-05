from typing import Dict, Optional, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans

from match_config import MatchConfig, TeamConfig

class TeamAssigner:
    def __init__(
        self,
        match_config: Optional[MatchConfig] = None,
        color_distance_threshold: float = 62.0,
        color_ambiguity_margin: float = 10.0,
    ):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_assignment_dict = {}
        self.match_config = match_config
        self.color_distance_threshold = color_distance_threshold
        self.color_ambiguity_margin = color_ambiguity_margin
        self.kmeans = None

        if match_config is not None:
            for team in match_config.teams:
                self.team_colors[team.team_id] = np.asarray(team.primary_color_bgr, dtype=float)
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(frame_width, int(bbox[0])))
        y1 = max(0, min(frame_height, int(bbox[1])))
        x2 = max(0, min(frame_width, int(bbox[2])))
        y2 = max(0, min(frame_height, int(bbox[3])))
        image = frame[y1:y2, x1:x2]

        if image.size == 0 or image.shape[0] < 4 or image.shape[1] < 4:
            raise ValueError("player bounding box does not contain a usable crop")

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        if self.match_config is not None:
            return self.team_colors

        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        if len(player_colors) < 2:
            raise ValueError("at least two player crops are required for team calibration")

        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    @staticmethod
    def _bgr_to_lab(color: Sequence[float]) -> np.ndarray:
        pixel = np.clip(np.asarray(color, dtype=np.float32), 0, 255).reshape(1, 1, 3)
        return cv2.cvtColor(pixel / 255.0, cv2.COLOR_BGR2LAB).reshape(3)

    def match_color_to_team(self, player_color) -> Dict[str, object]:
        """Map a BGR player color to a configured team, rejecting weak matches."""
        if self.match_config is None:
            raise RuntimeError("match_color_to_team requires a MatchConfig")

        player_lab = self._bgr_to_lab(player_color)
        distances = []
        for team in self.match_config.teams:
            team_lab = self._bgr_to_lab(team.primary_color_bgr)
            distances.append((float(np.linalg.norm(player_lab - team_lab)), team))
        distances.sort(key=lambda item: item[0])

        best_distance, best_team = distances[0]
        second_distance = distances[1][0]
        separation = second_distance - best_distance
        distance_confidence = max(0.0, 1.0 - best_distance / self.color_distance_threshold)
        separation_confidence = min(1.0, separation / max(self.color_ambiguity_margin * 2.0, 1.0))
        confidence = round((0.7 * distance_confidence) + (0.3 * separation_confidence), 3)

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

    def get_player_assignment(self, frame, player_bbox, player_id) -> Dict[str, object]:
        if (
            player_id in self.player_assignment_dict
            and self.player_assignment_dict[player_id]["team_id"] != 0
        ):
            return self.player_assignment_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        if self.match_config is not None:
            assignment = self.match_color_to_team(player_color)
        else:
            if self.kmeans is None:
                raise RuntimeError("team colors have not been calibrated")
            team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
            assignment = {
                "team_id": team_id,
                "team_name": f"team_{team_id}",
                "confidence": 1.0,
                "color_distance": 0.0,
            }

        self.player_assignment_dict[player_id] = assignment
        self.player_team_dict[player_id] = assignment["team_id"]
        return assignment

    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        return self.get_player_assignment(frame, player_bbox, player_id)["team_id"]
