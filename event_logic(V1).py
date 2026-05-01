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
        confidence = detected_event.get("confidence", 0.0)

        if event_type == "goal":
            # Lower confidence threshold for goals (0.5)
            if confidence >= 0.5:
                if self._goal_condition(tracks):
                    self._confirm(detected_event, frame_num, confirmed)
            
        elif event_type == "foul":
            # Higher confidence threshold for fouls (0.7)
            if confidence >= 0.7:
                if self._foul_condition(tracks):
                    self._confirm(detected_event, frame_num, confirmed)

        return confirmed

    def _confirm(self, event, frame_num, confirmed_list):
        self.confirmed_events.append(event)
        confirmed_list.append(event)
        self.last_event_frame = frame_num

    def _goal_condition(self, tracks):
        try:
            # check if ball is detected in current frame
            ball_tracks = tracks.get("ball", [])
            if len(ball_tracks) == 0:
                return False
            # ball_tracks is a list of frames, get the last frame
            current_frame_ball = ball_tracks[-1] if ball_tracks else {}
            return len(current_frame_ball) > 0
        except Exception as e:
            print(f"Goal condition error: {e}")
            return False

    def _foul_condition(self, tracks):
        try:
            # check if multiple players detected in current frame
            player_tracks = tracks.get("players", [])
            if len(player_tracks) == 0:
                return False
            # player_tracks is a list of frames, get the last frame
            current_frame_players = player_tracks[-1] if player_tracks else {}
            return len(current_frame_players) > 1
        except Exception as e:
            print(f"Foul condition error: {e}")
            return False

    def generate_summary(self, fps):
        print("\n" + "="*60)
        print("MATCH SUMMARY")
        print("="*60)
        
        if not self.confirmed_events:
            print("No events detected during the match.")
        else:
            for e in self.confirmed_events:
                time_sec = e["frame"] / fps
                minutes = int(time_sec // 60)
                seconds = int(time_sec % 60)
                confidence = e.get("confidence", "N/A")
                
                if isinstance(confidence, float):
                    print(f"{minutes:02d}:{seconds:02d} → {e['event'].upper()} (confidence: {confidence:.2f})")
                else:
                    print(f"{minutes:02d}:{seconds:02d} → {e['event'].upper()}")
        
        print("="*60)
