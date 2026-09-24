"""Deletes uploaded source videos once they are no longer needed.

Highlights are never deleted. Clips, the reel, player images, OCR crops and
team samples stay on disk indefinitely, because old matches may need their
clips shown again (owner decision, 2026-09-24). An earlier version also
expired that media after a number of days; it was removed for that reason.

What is deleted is only the uploaded source video:

- as soon as its job completes or fails (``delete_upload``);
- or, for an upload whose job never finished - abandoned at team
  confirmation, or orphaned by a restart, which empties the in-memory job
  table - once it is older than ``FOOTY_UPLOAD_MAX_AGE_HOURS`` (24). 0 keeps
  uploads forever.

Only uuid-named files this service wrote are touched; anything else in the
upload folder is left alone. It imports nothing from the database layer so
tests can run it on temporary folders.
"""

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_UPLOAD_MAX_AGE_HOURS = 24.0

_UPLOAD_FILE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"\.(mp4|mov|avi|mkv)",
    re.IGNORECASE,
)


def upload_max_age_hours() -> float:
    try:
        return float(os.environ.get("FOOTY_UPLOAD_MAX_AGE_HOURS", DEFAULT_UPLOAD_MAX_AGE_HOURS))
    except ValueError:
        return DEFAULT_UPLOAD_MAX_AGE_HOURS


@dataclass
class SweepReport:
    deleted_uploads: list[str] = field(default_factory=list)
    freed_bytes: int = 0

    def describe(self) -> str:
        return f"{len(self.deleted_uploads)} upload(s) deleted, {self.freed_bytes / 1048576:.1f} MB freed"


def _inside(path: Path, folder: Path) -> bool:
    try:
        return folder.resolve() in path.resolve().parents
    except OSError:
        return False


def delete_upload(path, upload_dir) -> int:
    """Delete one uploaded source video; returns the bytes freed.

    Refuses anything outside ``upload_dir``: the path comes from the job
    record, and a CLI or test caller may point a job at a file it owns.
    """
    if not path:
        return 0
    target = Path(path)
    if not target.is_file() or not _inside(target, Path(upload_dir)):
        return 0
    size = target.stat().st_size
    target.unlink(missing_ok=True)
    return size


def sweep(
    upload_dir,
    active_uploads: Iterable[str] = (),
    now: Optional[float] = None,
    upload_max_age: Optional[float] = None,
) -> SweepReport:
    """Delete uploads older than the limit whose job is not still running."""
    report = SweepReport()
    now = time.time() if now is None else now
    upload_max_age = upload_max_age_hours() if upload_max_age is None else upload_max_age
    folder = Path(upload_dir)
    if upload_max_age <= 0 or not folder.is_dir():
        return report
    active = {str(Path(p).resolve()).lower() for p in active_uploads if p}
    for item in folder.iterdir():
        if not item.is_file() or not _UPLOAD_FILE.fullmatch(item.name):
            continue
        if str(item.resolve()).lower() in active:
            continue
        if now - item.stat().st_mtime < upload_max_age * 3600:
            continue
        report.freed_bytes += item.stat().st_size
        item.unlink(missing_ok=True)
        report.deleted_uploads.append(item.name)
    return report
