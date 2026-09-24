import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner():
    """Which tracked player, if any, has the ball at their feet.

    The reach used to be a fixed 70 px. On the 1080p source the tutorial was
    written against that is under half a player's height; on 480p footage it is
    more than a whole player, so the ball was handed to whoever stood nearest even
    when it was clearly loose. The reach now scales with each player's own box.
    """

    def __init__(self, reach_ratio=0.7, min_reach_px=18.0, max_reach_px=140.0):
        self.reach_ratio = reach_ratio
        self.min_reach_px = min_reach_px
        self.max_reach_px = max_reach_px
        # Kept for callers that still read the old attribute.
        self.max_player_ball_distance = max_reach_px

    def reach_for(self, player_bbox):
        height = float(player_bbox[3]) - float(player_bbox[1])
        return max(self.min_reach_px, min(self.max_reach_px, self.reach_ratio * height))

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        best_score = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)
            reach = self.reach_for(player_bbox)

            if distance < reach:
                # Compare as a fraction of reach so a small, distant player does
                # not lose to a large, near one purely on pixel count.
                score = distance / reach
                if score < best_score:
                    best_score = score
                    assigned_player = player_id

        return assigned_player
