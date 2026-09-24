from collections import deque
import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


PLAYER_CLASS = "player"
GOALKEEPER_CLASS = "goalkeeper"
BALL_CLASS = "ball"
# models/best.pt was trained with the class spelled "refree". The old lookup asked
# for "referee", never matched, and referees were silently dropped from every
# frame - which is why nothing downstream could ever see a card being shown.
REFEREE_CLASS_NAMES = ("referee", "refree", "ref")

DETECT_CONFIDENCE = 0.25
# The ball is small, blurred, and often half-hidden by a boot; it clears 0.25 in
# well under half of the frames. It is admitted at a lower score and then judged
# on size and motion instead.
BALL_CONFIDENCE = 0.12
DETECT_IMAGE_SIZE = 640
DETECT_BATCH_SIZE = 20

# Ball diameter as a fraction of the median player height in the same frame. A
# real ball (22 cm) against a player (~1.8 m) is about 0.12; boxes are looser
# than the ball, and perspective stretches this, so the band is wide.
BALL_MIN_SIZE_RATIO = 0.05
BALL_MAX_SIZE_RATIO = 0.50
# How far the ball may jump between consecutive frames, in player heights. A
# 30 m/s shot at 25 fps covers roughly 0.7 heights per frame.
BALL_GATE_HEIGHTS_PER_FRAME = 1.6
BALL_HISTORY_FRAMES = 12

# Share of the ground patch under a person's feet that must read as turf for
# them to count as on the pitch. Stewards, photographers and bench staff stand on
# track, boards or concrete, and the detector labels several of them as
# referees or players in every broadcast clip.
ON_PITCH_MIN_GRASS = 0.30
GRASS_HUE_LOW = 35
GRASS_HUE_HIGH = 85
GRASS_MIN_SATURATION = 45
GRASS_MIN_VALUE = 40

# A track is treated as the goalkeeper once this share of its observations came
# back as the goalkeeper class. The old code relabelled every goalkeeper box as
# a player before tracking, so the role was lost and a save could not be told
# from an outfield clearance.
GOALKEEPER_VOTE_SHARE = 0.5


def _is_referee_name(name):
    return str(name).lower() in REFEREE_CLASS_NAMES


def choose_image_size(frame_width, frame_height):
    """Detector input size for a source resolution.

    Measured on the two reference clips: at 640 the ball was found in 4/12
    sampled 480p frames and 7/12 1080p frames; at 1280 it was 7/12 and 11/12.
    Players barely change. 1080p sources get the larger input; smaller sources
    get an intermediate size so the ball is still upsampled without paying the
    full 1280 cost on a CPU.
    """
    longest = max(int(frame_width), int(frame_height))
    if longest >= 1200:
        return 1280
    if longest >= 640:
        return 960
    return DETECT_IMAGE_SIZE


# The dedicated ball detector (models/ball_detector_best.pt) is far better
# calibrated than the ball class of models/best.pt: on 30 sampled frames of the
# 480p reference clip its best box was a real ball 29/29 times, where the old
# model's best box was the broadcast club crest in 6 of 28. It is admitted at a
# slightly higher score than the old class needed.
BALL_MODEL_CONFIDENCE = 0.15


def choose_ball_image_size(frame_width, frame_height):
    """Ball-detector input size for a source resolution.

    Measured on 30 sampled frames: 1080p found the ball in 24/30 at 640, 27/30
    at 960 and 28/30 at 1280; 480p reached 29-30/30 from 640 upwards.
    """
    longest = max(int(frame_width), int(frame_height))
    if longest >= 1200:
        return 1280
    if longest >= 640:
        return 960
    return DETECT_IMAGE_SIZE


def grass_fraction(frame, x1, y1, x2, y2):
    """Share of a region that reads as pitch turf. Returns None if empty."""
    height, width = frame.shape[:2]
    x1, x2 = max(0, min(width, int(x1))), max(0, min(width, int(x2)))
    y1, y2 = max(0, min(height, int(y1))), max(0, min(height, int(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grass = (
        (hue >= GRASS_HUE_LOW)
        & (hue <= GRASS_HUE_HIGH)
        & (saturation >= GRASS_MIN_SATURATION)
        & (value >= GRASS_MIN_VALUE)
    )
    return float(grass.mean())


def is_on_pitch(frame, bbox, min_grass=ON_PITCH_MIN_GRASS):
    """Whether a person's feet stand on turf.

    Looks at a band just below and around the feet. When that band is cut off by
    the frame edge the answer is unknown and the person is kept.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_height = max(1.0, y2 - y1)
    box_width = max(1.0, x2 - x1)
    top = y2 - box_height * 0.08
    bottom = y2 + box_height * 0.18
    left = x1 - box_width * 0.5
    right = x2 + box_width * 0.5
    if bottom > height - 1 or left < 0 or right > width - 1:
        return True
    fraction = grass_fraction(frame, left, top, right, bottom)
    if fraction is None:
        return True
    return fraction >= min_grass


class BallSelector:
    """Pick at most one ball per frame and reject implausible candidates.

    The detector routinely offers two or three "balls" per frame: the real one,
    a white spot in the crowd, a boot, a mark on an advertising board. Previously
    the last one in detection order won. Candidates are now scored on size
    against the players in the frame and on distance from where the ball was
    last seen, and a single box is kept.
    """

    def __init__(self, history_frames=BALL_HISTORY_FRAMES):
        self.history = deque(maxlen=history_frames)
        self.frames_since_seen = 10**6
        self._frame_index = 0

    def reset(self):
        self.history.clear()
        self.frames_since_seen = 10**6
        self._frame_index = 0

    def _predicted_position(self):
        if not self.history:
            return None
        if len(self.history) == 1:
            return np.asarray(self.history[-1][1], dtype=float)
        (frame_a, pos_a), (frame_b, pos_b) = self.history[-2], self.history[-1]
        pos_a = np.asarray(pos_a, dtype=float)
        pos_b = np.asarray(pos_b, dtype=float)
        gap = max(1, frame_b - frame_a)
        velocity = (pos_b - pos_a) / gap
        steps = self._frame_index - frame_b
        # Damp the extrapolation: a long gap means the ball was probably kicked
        # or hidden, and a straight-line guess drifts off the pitch.
        return pos_b + velocity * min(steps, 3)

    def select(self, candidates, median_player_height):
        """candidates: list of (bbox, confidence). Returns the chosen bbox or None."""
        self._frame_index += 1
        self.frames_since_seen += 1
        if not candidates:
            return None

        reference = float(median_player_height) if median_player_height else 0.0
        predicted = self._predicted_position()
        scored = []
        for bbox, confidence in candidates:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            size = max(x2 - x1, y2 - y1)
            if size <= 0:
                continue
            if reference > 0:
                ratio = size / reference
                if ratio < BALL_MIN_SIZE_RATIO or ratio > BALL_MAX_SIZE_RATIO:
                    continue
            score = float(confidence)
            if predicted is not None and reference > 0:
                centre = np.asarray(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
                distance = float(np.linalg.norm(centre - predicted))
                gate = BALL_GATE_HEIGHTS_PER_FRAME * reference * self.frames_since_seen
                if distance > gate:
                    # Too far from the last ball to be the same object, unless the
                    # trail has gone cold and anything plausible is welcome.
                    if self.frames_since_seen < self.history.maxlen:
                        continue
                else:
                    score += 0.5 * (1.0 - distance / max(gate, 1e-6))
            scored.append((score, bbox))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = [float(value) for value in scored[0][1]]
        centre = ((chosen[0] + chosen[2]) / 2.0, (chosen[1] + chosen[3]) / 2.0)
        self.history.append((self._frame_index, centre))
        self.frames_since_seen = 0
        return chosen


class Tracker:
    def __init__(
        self,
        model_path,
        frame_rate=25.0,
        image_size=None,
        ball_model_path=None,
        ball_image_size=None,
    ):
        self.model = YOLO(model_path)
        # Optional single-class ball detector. When present it is the only
        # source of ball candidates; the main model then detects people only.
        self.ball_model = YOLO(ball_model_path) if ball_model_path else None
        self.frame_rate = float(frame_rate) if frame_rate else 25.0
        self.image_size = int(image_size) if image_size else None
        self.ball_image_size = int(ball_image_size) if ball_image_size else None
        # A lost player is kept for two seconds so a brief occlusion behind a
        # team-mate does not mint a new id and break pass attribution.
        # minimum_consecutive_frames=2 discards the one-frame junk tracks that used
        # to be treated as players.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=DETECT_CONFIDENCE,
            lost_track_buffer=int(round(self.frame_rate * 2)),
            minimum_matching_threshold=0.8,
            frame_rate=int(round(self.frame_rate)),
            minimum_consecutive_frames=2,
        )
        self.ball_selector = BallSelector()
        # Per track: [goalkeeper observations, total observations].
        self._role_votes = {}
        # ByteTrack.reset() restarts ids at 1. Ids handed out after a shot
        # boundary are offset so a new person never inherits an old person's
        # crops, team votes or possession history.
        self._id_offset = 0
        self._max_track_id = 0

    def reset_tracking(self):
        """Forget every track at a shot boundary."""
        self.tracker.reset()
        self.ball_selector.reset()
        self._id_offset = self._max_track_id + 1

    def _image_size_for(self, frame):
        if self.image_size:
            return self.image_size
        height, width = frame.shape[:2]
        self.image_size = choose_image_size(width, height)
        return self.image_size

    def detect_outfield_players(
        self,
        frames,
        conf=DETECT_CONFIDENCE,
        imgsz=None,
    ):
        """Raw per-frame outfield-player boxes, for team-kit calibration.

        Deliberately skips ByteTrack. Calibration frames are seconds apart, so
        association fails and most detections would be discarded before they
        could become colour samples - which is how the two-team model ended up
        being fitted on a handful of crops.

        Goalkeepers and referees are excluded: their kits are chosen to differ
        from both outfield strips, so including them drags a cluster centre away
        from the real kit colour. People standing off the pitch are excluded for
        the same reason.
        """
        per_frame = []
        for start in range(0, len(frames), DETECT_BATCH_SIZE):
            batch = frames[start:start + DETECT_BATCH_SIZE]
            size = imgsz or self._image_size_for(batch[0])
            results = self.model.predict(batch, conf=conf, imgsz=size, verbose=False)
            for frame, result in zip(batch, results):
                names = result.names
                boxes = []
                for box, class_id in zip(
                    result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()
                ):
                    if names.get(int(class_id)) != PLAYER_CLASS:
                        continue
                    bbox = [float(value) for value in box]
                    if not is_on_pitch(frame, bbox):
                        continue
                    boxes.append(bbox)
                per_frame.append(boxes)
        return per_frame

    def add_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']

                    if object_name == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)

                    tracks[object_name][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_balls(self, frames):
        """Per-frame ball candidates from the dedicated model, or None."""
        if self.ball_model is None or not frames:
            return None
        if not self.ball_image_size:
            height, width = frames[0].shape[:2]
            self.ball_image_size = choose_ball_image_size(width, height)
        candidates = []
        for i in range(0, len(frames), DETECT_BATCH_SIZE):
            results = self.ball_model.predict(
                frames[i:i + DETECT_BATCH_SIZE],
                conf=BALL_MODEL_CONFIDENCE,
                imgsz=self.ball_image_size,
                verbose=False,
            )
            for result in results:
                candidates.append(
                    [
                        ([float(v) for v in box], float(score))
                        for box, score in zip(
                            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
                        )
                    ]
                )
        return candidates

    def detect_frames(self, frames, conf=None, imgsz=None):
        detections = []
        if conf is None:
            # Without a ball model the ball comes from this model and needs the
            # low threshold; with one, only people are wanted.
            conf = DETECT_CONFIDENCE if self.ball_model is not None else BALL_CONFIDENCE

        for i in range(0, len(frames), DETECT_BATCH_SIZE):
            batch = frames[i:i + DETECT_BATCH_SIZE]
            size = imgsz or self._image_size_for(batch[0])
            # The model runs at the ball's low threshold; people are filtered back
            # up to DETECT_CONFIDENCE in get_object_tracks. verbose defaults to
            # True and printed one line per frame.
            detections_batch = self.model.predict(
                batch, conf=conf, imgsz=size, verbose=False
            )
            detections += detections_batch

        return detections

    def _role_for(self, track_id, observed_goalkeeper):
        votes = self._role_votes.setdefault(int(track_id), [0, 0])
        votes[1] += 1
        if observed_goalkeeper:
            votes[0] += 1
        if votes[0] >= GOALKEEPER_VOTE_SHARE * votes[1]:
            return "goalkeeper"
        return "player"

    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None,
        reset_offsets=None,
        skip_offsets=None,
    ):
        """Tracks for a batch of frames.

        ``reset_offsets``: frames that start a new shot - tracking restarts.
        ``skip_offsets``: frames inside a dissolve - not detected, not tracked,
        returned empty, because boxes on a blend of two shots belong to neither.
        """

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        skip = set(skip_offsets or ())
        resets = set(reset_offsets or ())
        active = [index for index in range(len(frames)) if index not in skip]
        active_frames = [frames[index] for index in active]
        detected = self.detect_frames(active_frames) if active_frames else []
        detected_balls = self.detect_balls(active_frames) if active_frames else None
        detections = [None] * len(frames)
        ball_lists = None if (self.ball_model is None) else [[] for _ in frames]
        for position, index in enumerate(active):
            detections[index] = detected[position]
            if ball_lists is not None and detected_balls is not None:
                ball_lists[index] = detected_balls[position]

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            if frame_num in resets:
                self.reset_tracking()
            if detection is None:
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})
                continue
            frame = frames[frame_num]
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            player_id = cls_names_inv.get(PLAYER_CLASS)
            goalkeeper_id = cls_names_inv.get(GOALKEEPER_CLASS)
            ball_id = cls_names_inv.get(BALL_CLASS)
            referee_ids = {
                class_id for name, class_id in cls_names_inv.items() if _is_referee_name(name)
            }

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Ball boxes are collected before the people filter and judged
            # separately; people below the normal threshold are dropped.
            ball_candidates = []
            keep = np.ones(len(detection_supervision), dtype=bool)
            for index, (bbox, class_id, confidence) in enumerate(
                zip(
                    detection_supervision.xyxy,
                    detection_supervision.class_id,
                    detection_supervision.confidence,
                )
            ):
                if ball_id is not None and class_id == ball_id:
                    if ball_lists is None:
                        ball_candidates.append(([float(v) for v in bbox], float(confidence)))
                    keep[index] = False
                    continue
                if confidence < DETECT_CONFIDENCE:
                    keep[index] = False
                    continue
                if not is_on_pitch(frame, bbox):
                    keep[index] = False
            people = detection_supervision[keep]
            if ball_lists is not None:
                ball_candidates = ball_lists[frame_num]

            # Track goalkeepers as players so one tracker sees every person, but
            # remember which boxes were goalkeepers so the role survives.
            goalkeeper_boxes = []
            if goalkeeper_id is not None and player_id is not None:
                for index, class_id in enumerate(people.class_id):
                    if class_id == goalkeeper_id:
                        goalkeeper_boxes.append(np.asarray(people.xyxy[index], dtype=float))
                        people.class_id[index] = player_id

            detection_with_tracks = self.tracker.update_with_detections(people)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            player_heights = []
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = int(frame_detection[4]) + self._id_offset
                self._max_track_id = max(self._max_track_id, track_id)

                if player_id is not None and cls_id == player_id:
                    observed_goalkeeper = any(
                        _boxes_overlap(bbox, gk_box) for gk_box in goalkeeper_boxes
                    )
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "role": self._role_for(track_id, observed_goalkeeper),
                    }
                    player_heights.append(bbox[3] - bbox[1])
                elif cls_id in referee_ids:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            median_height = float(np.median(player_heights)) if player_heights else 0.0
            ball_bbox = self.ball_selector.select(ball_candidates, median_height)
            if ball_bbox is not None:
                tracks["ball"][frame_num][1] = {"bbox": ball_bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            cv2.putText(frame, str(track_id), (x_center, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control=None):

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # players
            for track_id, player in player_dict.items():
                color = (255, 0, 255) if player.get("role") == "goalkeeper" else (0, 0, 255)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # referees (safe even if empty)
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))

            # ball
            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0,255,0))

            output_video_frames.append(frame)

        return output_video_frames


def _boxes_overlap(bbox, other, min_iou=0.5):
    x1 = max(bbox[0], other[0])
    y1 = max(bbox[1], other[1])
    x2 = min(bbox[2], other[2])
    y2 = min(bbox[3], other[3])
    if x2 <= x1 or y2 <= y1:
        return False
    intersection = (x2 - x1) * (y2 - y1)
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    other_area = (other[2] - other[0]) * (other[3] - other[1])
    union = area + other_area - intersection
    return union > 0 and intersection / union >= min_iou
