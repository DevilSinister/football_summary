"""Process-wide heavy resources and serialized CV job execution."""

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Callable


class SerialJobQueue:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="footy-cv")

    def submit(self, function: Callable, *args, **kwargs) -> Future:
        return self._executor.submit(function, *args, **kwargs)


class ModelCache:
    def __init__(self) -> None:
        self._lock = Lock()
        self._models: dict[str, object] = {}
        self._ocr = None
        self._ocr_attempted = False

    def yolo(self, path: str | Path):
        key = str(Path(path).resolve())
        with self._lock:
            if key not in self._models:
                from ultralytics import YOLO

                self._models[key] = YOLO(key)
            return self._models[key]

    def spotting(self):
        """The action-spotting model, or None when its weights or timm are missing."""
        with self._lock:
            if "spotting" not in self._models:
                try:
                    from spotting import load_spotting_bundle

                    self._models["spotting"] = load_spotting_bundle()
                except Exception as error:  # optional model: a corrupt file must not fail jobs
                    print(f"Spotting model unavailable: {error}")
                    self._models["spotting"] = None
            return self._models["spotting"]

    def reid(self):
        """The player re-identification embedder, or None when its weights are missing."""
        with self._lock:
            if "reid" not in self._models:
                try:
                    from identity import ReidEmbedder

                    self._models["reid"] = ReidEmbedder.load()
                except Exception as error:  # optional model: a corrupt file must not fail jobs
                    print(f"Re-identification model unavailable: {error}")
                    self._models["reid"] = None
            return self._models["reid"]

    def whisper(self):
        """The commentary speech-to-text model, or None when unavailable."""
        with self._lock:
            if "whisper" not in self._models:
                try:
                    from commentary import load_whisper

                    self._models["whisper"] = load_whisper()
                except Exception as error:  # optional model: a missing download must not fail jobs
                    print(f"Commentary model unavailable: {error}")
                    self._models["whisper"] = None
            return self._models["whisper"]

    def paddle_ocr(self):
        with self._lock:
            if self._ocr is None and not self._ocr_attempted:
                self._ocr_attempted = True
                from scorer_identifier import JerseyReader

                reader = JerseyReader(debug_dir=None)
                if not reader._ensure_engine():
                    return None
                self._ocr = reader.engine
            return self._ocr


job_queue = SerialJobQueue()
model_cache = ModelCache()
