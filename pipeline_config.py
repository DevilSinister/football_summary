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
    use_legacy_event_model: bool = True
    use_pitch_calibration: bool = True
    detect_fouls: bool = True
    detect_cards: bool = True
    scoreboard_scan_seconds: float = 2.0
    scoreboard_burst_seconds: float = 0.5
    scoreboard_burst_duration: float = 8.0
    scoreboard_permissive_after_misses: int = 3
    identity_crops: int = 8
    jersey_ocr_calls: int = 120
    debug_ocr: bool = True

    @classmethod
    def short_clip(cls, debug_ocr: bool = False) -> "PipelineConfig":
        return cls(
            name="short_clip",
            short_clip_fps=12.0,
            long_clip_fps=10.0,
            goal_only=True,
            use_legacy_event_model=False,
            use_pitch_calibration=False,
            detect_fouls=False,
            detect_cards=False,
            identity_crops=4,
            jersey_ocr_calls=32,
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
