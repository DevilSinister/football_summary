import cv2
import os

from trackers import Tracker
from event_detector import EventDetector
from event_logic import EventLogic
from team_assigner.team_assigner import TeamAssigner


def main():

    video_path = "input_videos/football.mp4"
    output_path = "output_videos/output.avi"

    os.makedirs("output_videos", exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Cannot open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise Exception("VideoWriter failed to open")

    # ===== INIT MODELS =====
    tracker = Tracker("models/best.pt")
    event_detector = EventDetector("models/final_model_80.pth")
    event_logic = EventLogic()
    team_assigner = TeamAssigner()

    print("=" * 60)
    print("Football Analysis System Started")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    print("=" * 60)

    frame_num = 0
    team_calibrated = False

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # ===== TRACKING =====
        tracks = tracker.get_object_tracks([frame], read_from_stub=False)
        tracker.add_position_to_tracks(tracks)

        # ===== TEAM CALIBRATION (ONLY ONCE) =====
        if not team_calibrated and len(tracks["players"][0]) >= 6:
            try:
                team_assigner.assign_team_color(frame, tracks["players"][0])
                team_calibrated = True

                print(f"✅ Team calibration done at frame {frame_num}")
                print(f"Team 1 color: {team_assigner.team_colors[1]}")
                print(f"Team 2 color: {team_assigner.team_colors[2]}")

            except Exception as e:
                print(f"❌ Calibration error: {e}")

        # ===== TEAM ASSIGNMENT =====
        if team_calibrated:
            try:
                for player_id, player_info in tracks["players"][0].items():
                    bbox = player_info["bbox"]

                    team_id = team_assigner.get_player_team(
                        frame,
                        bbox,
                        player_id
                    )

                    player_info["team"] = team_id

            except Exception as e:
                print(f"❌ Team assignment error: {e}")
                team_calibrated = False

        # ===== EVENT DETECTION =====
        detected_events = event_detector.predict(frame, frame_num)

        confirmed_events = event_logic.update(detected_events)

        # ===== PRINT EVENTS (console only) =====
        if confirmed_events:
            for event in confirmed_events:
                print(f"✅ {event['event'].upper()} at frame {event['frame']}")

        # ===== DRAW TRACKING =====
        frame = tracker.draw_annotations([frame], tracks)[0]

        # ===== DRAW TEAM INFO =====
        if team_calibrated:
            for player_id, player in tracks["players"][0].items():

                bbox = player["bbox"]
                team_id = player.get("team", 0)

                x1, y1, x2, y2 = map(int, bbox)

                # Team colors (visual only)
                if team_id == 1:
                    color = (0, 255, 0)
                elif team_id == 2:
                    color = (255, 0, 0)
                else:
                    color = (128, 128, 128)

                # Draw label box
                cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, -1)

                cv2.putText(
                    frame,
                    f"T{team_id}",
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

        # ===== DISPLAY =====
        cv2.imshow("Football Analysis System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # ===== SAVE =====
        out.write(frame)

        frame_num += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ===== FINAL SUMMARY =====
    event_logic.generate_summary(fps)


if __name__ == "__main__":
    main()