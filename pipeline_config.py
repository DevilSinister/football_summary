"""Configuration and small helpers for goal-focused video analysis."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterator


@dataclass(frozen=True)
class PipelineConfig:
    name: str = "full"
    short_clip_seconds: float = 60.0
    short_clip_fps: float = 1_000_000.0
    long_clip_fps: float = 1_000_000.0
    low_resolution_size: int = 640
    high_resolution_size: int = 960
    high_resolution_height: int = 721
    goal_only: bool = False
    # The old LSTM (final_model_80.pth) is kept for comparison only; its goal
    # score is noise on every measured clip. main.py --legacy-event-model.
    use_legacy_event_model: bool = False
    use_pitch_calibration: bool = True
    # The E2E-Spot action-spotting model (models/checkpoint_088.pt). It needs
    # every decoded frame offered to it but only scores spotting_sample_fps.
    # Commentary speech to text (faster-whisper) is the primary source of
    # player names: scorer, assister, fouler and the player on the ball.
    use_commentary: bool = True
    whisper_model: str = "small"
    # Event types a goal-only profile returns; the rest stay internal evidence.
    output_events: tuple = ("goal",)
    use_spotting: bool = True
    spotting_sample_fps: float = 2.0
    spotting_min_confidence: float = 0.1
    detect_fouls: bool = True
    detect_cards: bool = True
    detect_penalties: bool = True
    read_scoreboard: bool = True
    highlight_clips: bool = True
    # Who to name. Off = the event is kept but its players stay unnamed (and,
    # for passes, no shirt reading or identity work is spent on them).
    name_goal_scorer: bool = True
    name_assister: bool = True
    name_pass_players: bool = True
    name_foul_players: bool = True
    # A foul / card needs two kinds of evidence (spotting/match_events.py).
    foul_double_check: bool = False
    card_double_check: bool = False
    # Set pieces from the spotting model alone.
    detect_corners: bool = False
    detect_offsides: bool = False
    detect_free_kicks: bool = False
    detect_substitutions: bool = False
    # Both scores rising by one at once on the graphic = one goal each.
    score_jump_goals: bool = False
    # The features.yaml switches this profile was built from, for the job result.
    features: tuple = ()
    scoreboard_scan_seconds: float = 2.0
    scoreboard_burst_seconds: float = 0.5
    scoreboard_burst_duration: float = 8.0
    scoreboard_permissive_after_misses: int = 3
    identity_crops: int = 8
    jersey_ocr_calls: int = 120
    debug_ocr: bool = True

    @classmethod
    def short_clip(cls, debug_ocr: bool = False, features: Dict[str, bool] | None = None) -> "PipelineConfig":
        """The profile API (phone app) jobs use, switched by features.yaml.

        ``features`` defaults to the file as it is now, read per job, so an
        edit applies to the next analysis without restarting the server.
        """
        from feature_flags import load_features, visible_event_types

        flags = dict(features) if features is not None else load_features()
        on = lambda key: bool(flags.get(key, False))  # noqa: E731
        return cls(
            name="short_clip",
            short_clip_fps=12.0,
            long_clip_fps=10.0,
            goal_only=True,
            use_legacy_event_model=False,
            use_pitch_calibration=False,
            detect_fouls=on("detect_fouls"),
            # Event types the job returns and stores; the rest stay internal
            # evidence (passes find the assister, shots and saves the goals).
            output_events=tuple(visible_event_types(flags)),
            detect_cards=on("detect_cards"),
            detect_penalties=on("detect_penalties"),
            use_commentary=on("commentary_names"),
            use_spotting=on("action_spotting"),
            read_scoreboard=on("read_scoreboard"),
            highlight_clips=on("highlight_clips"),
            name_goal_scorer=on("goal_scorer_name"),
            name_assister=on("assister_name"),
            name_pass_players=on("detect_passes") and on("passer_name"),
            name_foul_players=on("fouler_name"),
            foul_double_check=on("foul_double_check"),
            card_double_check=on("card_double_check"),
            detect_corners=on("detect_corners"),
            detect_offsides=on("detect_offsides"),
            detect_free_kicks=on("detect_free_kicks"),
            detect_substitutions=on("detect_substitutions"),
            score_jump_goals=on("score_jump_goals"),
            features=tuple(sorted(flags.items())),
            identity_crops=4,
            # Shirt reading is the second source of names after commentary:
            # every goal and foul participant gets reads, not just the scorer.
            jersey_ocr_calls=64,
            debug_ocr=debug_ocr,
        )

    def analysis_fps(self, duration_seconds: float) -> float:
        return self.short_clip_fps if duration_seconds <= self.short_clip_seconds else self.long_clip_fps

    def detector_size(self, frame_height: int) -> int:
        return self.low_resolution_size if int(frame_height) <= 720 else self.high_resolution_size


def scheduled_frames(total_frames: int, source_fps: float, target_fps: float) -> Iterator[int]:
    """Yield stable source-frame indices without accumulating rounding drift."""
    if total_frames <= 0:
        return
    source_fps = max(float(source_fps), 1.0)
    target_fps = min(source_fps, max(float(target_fps), 0.1))
    step = source_fps / target_fps
    cursor = 0.0
    last = -1
    while True:
        frame = min(total_frames - 1, int(round(cursor)))
        if frame > last:
            yield frame
            last = frame
        cursor += step
        if cursor >= total_frames:
            break


@dataclass
class PipelineMetrics:
    phases: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add_time(self, phase: str, seconds: float) -> None:
        self.phases[phase] = self.phases.get(phase, 0.0) + float(seconds)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(amount)

    def timer(self, phase: str):
        metrics = self

        class _Timer:
            def __enter__(self):
                self.started = perf_counter()
                return self

            def __exit__(self, *_args):
                metrics.add_time(phase, perf_counter() - self.started)

        return _Timer()

    def summary(self) -> dict:
        return {
            "phases_seconds": {key: round(value, 3) for key, value in self.phases.items()},
            "counts": dict(self.counts),
        }
