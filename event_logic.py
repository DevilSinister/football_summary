from scorer_identifier import format_goal_event


class EventLogic:
    def __init__(self, cooldown_frames=60):
        self.confirmed_events = []
        self.cooldown_frames = cooldown_frames
        self.last_event_frame = -100

    def update(self, detected_events, tracks, frame_num):
        confirmed = []
        if detected_events is None:
            return confirmed
        if isinstance(detected_events, dict):
            detected_events = [detected_events]

        for event in detected_events:
            event_frame = int(event.get("frame", frame_num))
            if event_frame - self.last_event_frame < self.cooldown_frames:
                continue
            event_type = event.get("event")
            confidence = float(event.get("confidence", 0.0))
            if event_type == "goal":
                if confidence >= 0.5 and self._goal_condition(tracks):
                    self._confirm(event, event_frame, confirmed)
            elif event_type == "foul":
                if confidence >= 0.7 and self._foul_condition(tracks):
                    self._confirm(event, event_frame, confirmed)
            elif confidence >= 0.7:
                self._confirm(event, event_frame, confirmed)
        return confirmed

    def _confirm(self, event, frame_num, confirmed):
        event["frame"] = frame_num
        self.confirmed_events.append(event)
        confirmed.append(event)
        self.last_event_frame = frame_num

    @staticmethod
    def _goal_condition(tracks):
        ball_tracks = tracks.get("ball", [])
        return bool(ball_tracks and ball_tracks[-1])

    @staticmethod
    def _foul_condition(tracks):
        player_tracks = tracks.get("players", [])
        return bool(player_tracks and len(player_tracks[-1]) > 1)

    def generate_summary(self, fps):
        print("\n" + "=" * 60)
        print("MATCH SUMMARY")
        print("=" * 60)
        if not self.confirmed_events:
            print("No events detected during the match.")
        for event in self.confirmed_events:
            time_seconds = event["frame"] / max(float(fps), 1.0)
            timestamp = f"{int(time_seconds // 60):02d}:{int(time_seconds % 60):02d}"
            if event.get("event") == "goal":
                print(f"{timestamp} → {format_goal_event(event)}")
            else:
                print(f"{timestamp} → {str(event.get('event', 'event')).upper()}")
        print("=" * 60)
