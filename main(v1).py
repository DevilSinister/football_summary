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

    tracker = Tracker("models/best.pt")
    event_detector = EventDetector("models/final_model.pth")
    event_logic = EventLogic()
    team_assigner = TeamAssigner()

    print("="*60)
    print("Football Analysis System Started")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    print("="*60)

    frame_num = 0
    team_calibrated = False

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        tracks = tracker.get_object_tracks([frame], read_from_stub=False)
        tracker.add_position_to_tracks(tracks)

        # --- calibrate team colors on first frame with players ---
        # store first good frame
        if not team_calibrated and len(tracks["players"][0]) >= 6:
            try:
                # use THIS frame as reference (like original repo)
                reference_frame = frame.copy()

                team_assigner.assign_team_color(
                    reference_frame,
                    tracks["players"][0]
                )

                team_calibrated = True

                print(f"✅ Team calibration locked at frame {frame_num}")
                print(f"Team 1 color: {team_assigner.team_colors.get(1)}")
                print(f"Team 2 color: {team_assigner.team_colors.get(2)}")

            except Exception as e:
                print(f"❌ Team calibration error: {e}")

        # --- assign teams to players (STABLE TRACK-BASED) ---
        if team_calibrated:
            try:
                for player_id, player_info in tracks["players"][0].items():

                    # If already assigned before → reuse
                    if "team" in player_info:
                        continue

                    player_bbox = player_info["bbox"]

                    team_id = team_assigner.get_player_team(
                        frame,
                        player_bbox,
                        player_id
                    )

                    player_info["team"] = team_id

            except Exception as e:
                print(f"❌ Team assignment error: {e}")
                team_calibrated = False

        detected_event = event_detector.predict(frame, frame_num)

        confirmed_events = event_logic.update(detected_event)
        
        # Log confirmed events to console only (not on video)
        if confirmed_events:
            for event in confirmed_events:
                print(f"✅ CONFIRMED: {event['event'].upper()} at frame {event['frame']}")

        # --- draw tracking and team information ---
        frame = tracker.draw_annotations([frame], tracks)[0]
        
        # --- display team colors and player info on frame ---
        if team_calibrated:
            player_dict = tracks["players"][0]
            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                team_id = player.get("team", 0)
                
                # Draw team color indicator
                team_color = (0, 255, 0) if team_id == 1 else (255, 0, 0)
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), team_color, -1)
                cv2.putText(
                    frame,
                    f"Team {team_id}",
                    (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )


        cv2.imshow("Football Analysis System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    event_logic.generate_summary(fps)


if __name__ == "__main__":
    main()