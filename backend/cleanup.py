"""Retention for uploaded videos and per-job media.

Nothing used to be deleted: after a handful of test jobs backend/uploads held
143 MB and static/clips 21 MB, and a full match at the old clip bitrate would
have added 3-4 GB.

- An uploaded source video is deleted as soon as its job completes or fails
  (``delete_upload``). An upload whose job never finishes - abandoned at team
  confirmation, or orphaned by a restart, which empties the in-memory job
  table - is deleted once older than ``FOOTY_UPLOAD_MAX_AGE_HOURS`` (24).
- A job's clips and reel, player images, OCR debug crops and team samples are
  deleted ``FOOTY_MEDIA_RETENTION_DAYS`` (7) after they were last written.
  0 keeps them forever.

Only names this service generates are touched: uuid-named uploads, uuid-named
job directories and ``<uuid>-team-N.jpg`` samples. Anything else in those
folders is left alone.

It imports nothing from the database layer so tests can run it on temporary
folders.
"""

import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_UPLOAD_MAX_AGE_HOURS = 24.0
DEFAULT_MEDIA_RETENTION_DAYS = 7.0

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_JOB_DIR = re.compile(rf"{_UUID}")
_UPLOAD_FILE = re.compile(rf"{_UUID}\.(mp4|mov|avi|mkv)", re.IGNORECASE)
_TEAM_SAMPLE = re.compile(rf"({_UUID})-team-\d+\.jpg", re.IGNORECASE)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def upload_max_age_hours() -> float:
    return _env_float("FOOTY_UPLOAD_MAX_AGE_HOURS", DEFAULT_UPLOAD_MAX_AGE_HOURS)


def media_retention_days() -> float:
    return _env_float("FOOTY_MEDIA_RETENTION_DAYS", DEFAULT_MEDIA_RETENTION_DAYS)


@dataclass
class SweepReport:
    deleted_uploads: list[str] = field(default_factory=list)
    expired_jobs: list[str] = field(default_factory=list)  # job ids whose clips were deleted
    freed_bytes: int = 0

    def describe(self) -> str:
        return (
            f"{len(self.deleted_uploads)} upload(s) and media of {len(self.expired_jobs)} job(s) "
            f"deleted, {self.freed_bytes / 1048576:.1f} MB freed"
        )


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _newest_mtime(path: Path) -> float:
    if path.is_file():
        return path.stat().st_mtime
    times = [item.stat().st_mtime for item in path.rglob("*")]
    return max(times, default=path.stat().st_mtime)


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
    job_media_dirs: Iterable,
    team_sample_dir=None,
    clip_dir=None,
    active_job_ids: Iterable[str] = (),
    active_uploads: Iterable[str] = (),
    now: Optional[float] = None,
    upload_max_age: Optional[float] = None,
    retention_days: Optional[float] = None,
) -> SweepReport:
    """Delete expired uploads and job media. Jobs still running are skipped.

    ``job_media_dirs`` hold one sub-folder per job id. ``clip_dir`` names the
    one among them whose expiry must also clear the stored clip links, and is
    what ``expired_jobs`` reports.
    """
    report = SweepReport()
    now = time.time() if now is None else now
    upload_max_age = upload_max_age_hours() if upload_max_age is None else upload_max_age
    retention_days = media_retention_days() if retention_days is None else retention_days
    active_ids = {str(job_id).lower() for job_id in active_job_ids}
    active_paths = {str(Path(p).resolve()).lower() for p in active_uploads if p}

    upload_folder = Path(upload_dir)
    if upload_max_age > 0 and upload_folder.is_dir():
        for item in upload_folder.iterdir():
            if not item.is_file() or not _UPLOAD_FILE.fullmatch(item.name):
                continue
            if str(item.resolve()).lower() in active_paths:
                continue
            if now - item.stat().st_mtime < upload_max_age * 3600:
                continue
            report.freed_bytes += item.stat().st_size
            item.unlink(missing_ok=True)
            report.deleted_uploads.append(item.name)

    if retention_days <= 0:
        return report
    limit = retention_days * 86400
    clip_folder = Path(clip_dir).resolve() if clip_dir else None
    expired = set()
    for folder in [Path(f) for f in job_media_dirs]:
        if not folder.is_dir():
            continue
        for item in folder.iterdir():
            if not item.is_dir() or not _JOB_DIR.fullmatch(item.name):
                continue
            if item.name.lower() in active_ids or now - _newest_mtime(item) < limit:
                continue
            report.freed_bytes += _size(item)
            shutil.rmtree(item, ignore_errors=True)
            if clip_folder is not None and folder.resolve() == clip_folder:
                expired.add(item.name)

    if team_sample_dir and Path(team_sample_dir).is_dir():
        for item in Path(team_sample_dir).iterdir():
            match = _TEAM_SAMPLE.fullmatch(item.name)
            if not item.is_file() or match is None or match.group(1).lower() in active_ids:
                continue
            if now - item.stat().st_mtime < limit:
                continue
            report.freed_bytes += item.stat().st_size
            item.unlink(missing_ok=True)

    report.expired_jobs = sorted(expired)
    return report
