class EventLogic:
    def __init__(self):
        self.confirmed_events = []
        self.cooldown_frames = 60
        self.last_event_frame = -100

    def update(self, detected_event, tracks, frame_num):
        confirmed = []

        if detected_event is None:
            return confirmed

        # prevent duplicates
        if frame_num - self.last_event_frame < self.cooldown_frames:
            return confirmed

        event_type = detected_event["event"]

        if event_type == "goal":
            if self._goal_condition(tracks):
                self._confirm(detected_event, frame_num, confirmed)

        elif event_type == "foul":
            if self._foul_condition(tracks):
                self._confirm(detected_event, frame_num, confirmed)

        return confirmed

    def _confirm(self, event, frame_num, confirmed_list):
        self.confirmed_events.append(event)
        confirmed_list.append(event)
        self.last_event_frame = frame_num

    def _goal_condition(self, tracks):
        try:
            # basic check: ball exists
            ball_tracks = tracks.get("ball", [])
            return len(ball_tracks) > 0
        except:
            return False

    def _foul_condition(self, tracks):
        try:
            players = tracks.get("players", [])
            return len(players) > 1
        except:
            return False

    def generate_summary(self, fps):
        print("\n===== MATCH SUMMARY =====")
        for e in self.confirmed_events:
            time_sec = e["frame"] / fps
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)

            print(f"{minutes:02d}:{seconds:02d} → {e['event'].upper()}")