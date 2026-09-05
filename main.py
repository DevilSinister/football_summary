import argparse
import os

import cv2

from match_config import MatchConfig
from player_ball_assigner import PlayerBallAssigner
from scorer_identifier import JerseyReader, ScorerIdentifier, format_goal_event
from team_assigner import TeamAssigner


def build_parser():
    parser = argparse.ArgumentParser(description="Detect football events and identify goal scorers")
    parser.add_argument("video", help="input match video")
    parser.add_argument("--output", default="output_videos/output.avi")
    parser.add_argument("--tracking-model", default="models/best.pt")
    parser.add_argument("--event-model", default="models/final_model_80.pth")
    parser.add_argument("--team-a-name", required=True)
    parser.add_argument("--team-a-color", required=True, help="primary kit color as #RRGGBB")
    parser.add_argument("--team-b-name", required=True)
    parser.add_argument("--team-b-color", required=True, help="primary kit color as #RRGGBB")
    parser.add_argument("--ocr-debug-dir", default="output_videos/ocr_debug")
    parser.add_argument("--player-image-dir", default="output_videos/player_images")
    parser.add_argument("--player-image-url-prefix")
    parser.add_argument("--no-display", action="store_true", help="process without opening a GUI window")
    return parser


def _save_scorer_image(scorer_identifier, scorer, image_dir, url_prefix):
    if not scorer or not image_dir:
        return
    state = scorer_identifier.identity_store.get(int(scorer.get("track_id", -1)))
    if state is None or not state.recent_crops:
        return
    os.makedirs(image_dir, exist_ok=True)
    filename = f"track-{state.track_id}-frame-{state.recent_crops[0].frame_number}.jpg"
    path = os.path.join(image_dir, filename)
    if cv2.imwrite(path, state.recent_crops[0].image):
        scorer["player_image"] = (
            f"{url_prefix.rstrip('/')}/{filename}" if url_prefix else path
        )


def _assign_teams(team_assigner, frame, players):
    for track_id, player in players.items():
        try:
            assignment = team_assigner.get_player_assignment(frame, player["bbox"], int(track_id))
        except (ValueError, RuntimeError):
            assignment = {
                "team_id": 0,
                "team_name": "unknown_team",
                "confidence": 0.0,
            }
        player["team"] = assignment["team_id"]
        player["team_name"] = assignment["team_name"]
        player["team_confidence"] = assignment["confidence"]


def _record_ball_possession(scorer_identifier, ball_assigner, frame_number, tracks):
    players = tracks["players"][0]
    balls = tracks["ball"][0]
    if not balls:
        return
    ball = next(iter(balls.values()))
    track_id = ball_assigner.assign_ball_to_player(players, ball["bbox"])
    if track_id >= 0:
        players[track_id]["has_ball"] = True
        scorer_identifier.record_possession(frame_number, int(track_id))


def _draw_team_labels(frame, players):
    for track_id, player in players.items():
        x1, y1, x2, _ = map(int, player["bbox"])
        team_id = int(player.get("team", 0))
        color = (0, 200, 0) if team_id == 1 else (200, 80, 0) if team_id == 2 else (128, 128, 128)
        label = f"{player.get('team_name', 'unknown_team')} | ID {track_id}"
        cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x2, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, max(12, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def process_video(args):
    # Heavy model imports are delayed so configuration validation and --help work
    # without initializing Ultralytics/Torch or their user-level cache folders.
    from event_detector import EventDetector
    from event_logic import EventLogic
    from trackers import Tracker

    match_config = MatchConfig.from_user_input(
        args.team_a_name,
        args.team_a_color,
        args.team_b_name,
        args.team_b_color,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video file: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open output video: {args.output}")

    tracker = Tracker(args.tracking_model)
    event_detector = EventDetector(args.event_model)
    event_logic = EventLogic()
    team_assigner = TeamAssigner(match_config)
    ball_assigner = PlayerBallAssigner()
    scorer_identifier = ScorerIdentifier(
        jersey_reader=JerseyReader(debug_dir=args.ocr_debug_dir)
    )

    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            tracks = tracker.get_object_tracks([frame], read_from_stub=False)
            tracker.add_position_to_tracks(tracks)
            players = tracks["players"][0]
            _assign_teams(team_assigner, frame, players)
            scorer_identifier.observe_players(frame_number, frame, players)
            _record_ball_possession(
                scorer_identifier, ball_assigner, frame_number, tracks
            )

            detected_events = event_detector.predict(frame, frame_number)
            confirmed_events = event_logic.update(detected_events, tracks, frame_number)
            for event in confirmed_events:
                if event.get("event") == "goal":
                    event["scorer"] = scorer_identifier.identify_goal(frame_number)
                    _save_scorer_image(
                        scorer_identifier,
                        event["scorer"],
                        getattr(args, "player_image_dir", None),
                        getattr(args, "player_image_url_prefix", None),
                    )
                    print(format_goal_event(event))
                else:
                    print(f"{str(event.get('event', 'event')).upper()} at frame {frame_number}")

            annotated = tracker.draw_annotations([frame], tracks)[0]
            _draw_team_labels(annotated, players)
            writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Football Analysis System", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_number += 1
    finally:
        capture.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    event_logic.generate_summary(fps)
    return event_logic.confirmed_events


def main():
    process_video(build_parser().parse_args())


if __name__ == "__main__":
    main()
