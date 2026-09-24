from scorer_identifier import format_goal_event
from track_events import summarise_passes


# How far back a shot must have been seen for the learned model's "goal" to be
# believed, and how close two opponents must stand for its "foul" to be.
GOAL_SHOT_LOOKBACK_SECONDS = 4.0
FOUL_CONTACT_HEIGHTS = 1.0
FOUL_BALL_HEIGHTS = 2.5


class EventLogic:
    """Gate for the learned goal/foul detector.

    Measured on the three reference clips, that model never raised goal
    probability above 0.2 and fired "foul" on the opening frames of two of them.
    Its proposals are therefore only accepted when the tracks agree: a goal
    needs a recent shot towards the goalkeeper, a foul needs two opponents in
    contact near the ball.
    """

    def __init__(self, cooldown_frames=60, fps=25.0):
        self.confirmed_events = []
        self.cooldown_frames = cooldown_frames
        self.fps = max(float(fps), 1.0)
        self.last_event_frame = -100

    def update(self, detected_events, tracks, frame_num, track_events=None):
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
                if (
                    confidence >= 0.5
                    and self._goal_condition(tracks)
                    and self._recent_shot(track_events, event_frame)
                ):
                    self._confirm(event, event_frame, confirmed)
            elif event_type == "foul":
                if confidence >= 0.7 and self._foul_condition(tracks, track_events):
                    self._confirm(event, event_frame, confirmed)
            elif confidence >= 0.7:
                self._confirm(event, event_frame, confirmed)
        return confirmed

    def _confirm(self, event, frame_num, confirmed):
        event["frame"] = frame_num
        event.setdefault("source", "model")
        self.confirmed_events.append(event)
        confirmed.append(event)
        self.last_event_frame = frame_num

    @staticmethod
    def _goal_condition(tracks):
        ball_tracks = tracks.get("ball", [])
        return bool(ball_tracks and ball_tracks[-1])

    def _recent_shot(self, track_events, frame_num):
        if track_events is None:
            # No track context available: fall back to the old behaviour.
            return True
        lookback = GOAL_SHOT_LOOKBACK_SECONDS * self.fps
        return any(
            event.get("event") in ("shot", "penalty")
            and 0 <= frame_num - int(event.get("frame", 0)) <= lookback
            for event in track_events.events
        )

    @staticmethod
    def _foul_condition(tracks, track_events=None):
        player_tracks = tracks.get("players", [])
        if not (player_tracks and len(player_tracks[-1]) > 1):
            return False
        if track_events is None or track_events.scale <= 0:
            return True

        scale = track_events.scale
        players = player_tracks[-1]
        ball_tracks = tracks.get("ball", [])
        ball = next(iter(ball_tracks[-1].values()), None) if ball_tracks and ball_tracks[-1] else None
        ball_point = None
        if ball is not None:
            bbox = ball["bbox"]
            ball_point = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        feet = []
        for player in players.values():
            bbox = player["bbox"]
            feet.append((int(player.get("team", 0) or 0), (bbox[0] + bbox[2]) / 2.0, bbox[3]))
        for index, (team_a, xa, ya) in enumerate(feet):
            for team_b, xb, yb in feet[index + 1:]:
                if team_a == 0 or team_b == 0 or team_a == team_b:
                    continue
                if ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5 > FOUL_CONTACT_HEIGHTS * scale:
                    continue
                if ball_point is None:
                    return True
                midpoint = ((xa + xb) / 2.0, (ya + yb) / 2.0)
                if ((midpoint[0] - ball_point[0]) ** 2 + (midpoint[1] - ball_point[1]) ** 2) ** 0.5 <= FOUL_BALL_HEIGHTS * scale:
                    return True
        return False

    def generate_summary(self, fps, events=None):
        events = list(events) if events is not None else list(self.confirmed_events)
        print("\n" + "=" * 60)
        print("MATCH SUMMARY")
        print("=" * 60)
        if not events:
            print("No events detected during the match.")
        for event in events:
            time_seconds = event["frame"] / max(float(fps), 1.0)
            timestamp = f"{int(time_seconds // 60):02d}:{int(time_seconds % 60):02d}"
            if event.get("event") == "goal":
                print(f"{timestamp} → {format_goal_event(event)}")
            elif event.get("description"):
                print(f"{timestamp} → {event['description']}")
            else:
                print(f"{timestamp} → {str(event.get('event', 'event')).upper()}")
        passes = summarise_passes(events)
        if passes:
            print("-" * 60)
            for team, counts in passes.items():
                print(f"{team}: {counts['passes']} passes ({counts['crosses']} crosses)")
        print("=" * 60)
