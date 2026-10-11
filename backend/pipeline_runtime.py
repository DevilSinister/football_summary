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
