# import cv2
# import os
# from trackers import Tracker
# from event_detector import EventDetector
# from event_logic import EventLogic
#
#
# def main():
#     video_path = 'input_videos/test.mp4'
#     output_path = 'output_videos/output.avi'
#
#     os.makedirs("output_videos", exist_ok=True)
#
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         raise Exception("Error opening video")
#
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#
#     # ✅ SAFE codec (important)
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#     if not out.isOpened():
#         raise Exception("VideoWriter failed")
#
#     # models
#     tracker = Tracker('models/best.pt')
#     event_detector = EventDetector('models/final_model_80.pth')
#     event_logic = EventLogic()
#
#     frame_num = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # --- tracking ---
#         tracks = tracker.get_object_tracks([frame], read_from_stub=False)
#         tracker.add_position_to_tracks(tracks)
#
#         # --- event detection ---
#         detected_event = event_detector.predict(frame, frame_num)
#
#         # --- event logic ---
#         confirmed_events = event_logic.update(detected_event, tracks, frame_num)
#
#         # --- draw events ---
#         for e in confirmed_events:
#             cv2.putText(frame,
#                         e['event'].upper(),
#                         (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,
#                         (0, 0, 255),
#                         2)
#
#         # --- draw tracking ---
#         frame = tracker.draw_annotations([frame], tracks)[0]
#
#         out.write(frame)
#
#         frame_num += 1
#
#     cap.release()
#     out.release()
#
#     # --- summary ---
#     event_logic.generate_summary(fps)
#
#
# if __name__ == '__main__':
#     main()





import cv2
import os

from trackers import Tracker
from event_detector import EventDetector
from event_logic import EventLogic


def main():

    video_path = "input_videos/test.mp4"
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
    event_detector = EventDetector("models/final_model_80.pth")
    event_logic = EventLogic()

    frame_num = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        tracks = tracker.get_object_tracks([frame], read_from_stub=False)
        tracker.add_position_to_tracks(tracks)

        detected_event = event_detector.predict(frame, frame_num)

        confirmed_events = event_logic.update(detected_event, tracks, frame_num)

        frame = tracker.draw_annotations([frame], tracks)[0]

        for event in confirmed_events:
            cv2.putText(
                frame,
                event["event"].upper(),
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
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