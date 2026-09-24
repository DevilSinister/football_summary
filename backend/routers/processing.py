from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4
import re
import threading

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from highlights import REEL_FILE_NAME, clips_from_events
from scorer_identifier import format_goal_event
from team_assigner import (
    TeamAssigner,
    TeamSample,
    cluster_team_samples,
    crop_sharpness,
    kit_color_bgr,
    render_team_sample,
    safe_player_crop,
)

from .. import cleanup
from ..database import SessionLocal, get_db
from ..models import Event, Match, Occurrence, Play, Player, Team, User
from ..roster import find_team, resolve_team
from ..schemas import TeamConfirmationRequest, VideoUrlRequest


router = APIRouter(tags=["processing"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "static" / "outputs"
OCR_DIR = PROJECT_ROOT / "backend" / "static" / "ocr_debug"
TEAM_SAMPLE_DIR = PROJECT_ROOT / "backend" / "static" / "team_samples"
PLAYER_IMAGE_DIR = PROJECT_ROOT / "backend" / "static" / "player_images"
CLIP_DIR = PROJECT_ROOT / "backend" / "static" / "clips"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
for directory in (
    UPLOAD_DIR,
    OUTPUT_DIR,
    OCR_DIR,
    TEAM_SAMPLE_DIR,
    PLAYER_IMAGE_DIR,
    CLIP_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# Team calibration sampling. The old code hardcoded 12 frames over the first 45
# seconds and would accept a two-team model built from as few as two crops.
CALIBRATION_MIN_FRAMES = 24
CALIBRATION_MAX_FRAMES = 60
CALIBRATION_MIN_SAMPLES = 40
CALIBRATION_MIN_PER_CLUSTER = 12
CALIBRATION_MIN_SEPARATION = 12.0

# The band of the progress bar that the frame loop reports into.
PROGRESS_ANALYSIS_START = 15
PROGRESS_ANALYSIS_END = 90
PROGRESS_PERSIST = 94

# Clip URLs are built from a job uuid and a generated file name; both are matched
# strictly so a path cannot escape CLIP_DIR.
_SAFE_JOB_ID = re.compile(r"[0-9a-fA-F-]{8,64}")
_SAFE_CLIP_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\.mp4")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

_tracker = None
_tracker_lock = threading.Lock()


def _get_tracker():
    """One YOLO model per process.

    Team detection and the processing run used to load models/best.pt separately,
    so every job paid for two loads.
    """
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            from trackers import Tracker

            _tracker = Tracker(str(PROJECT_ROOT / "models" / "best.pt"))
        return _tracker


def _set_job(job_id: str, **changes) -> None:
    with _jobs_lock:
        _jobs[job_id].update(changes)


def _job_snapshot(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def sweep_media() -> None:
    """Delete stale uploaded source videos (backend.cleanup).

    Highlights are never deleted: clips, reels and player images under
    backend/static stay so old matches can still show them.
    """
    with _jobs_lock:
        active = [
            job for job in _jobs.values() if job.get("status") not in ("completed", "failed")
        ]
    try:
        report = cleanup.sweep(UPLOAD_DIR, active_uploads=[job.get("video_path") for job in active])
    except OSError as exc:
        print(f"Upload clean-up failed: {exc}")
        return
    if report.deleted_uploads:
        print(f"Upload clean-up: {report.describe()}")


def _release_job_files(job_id: str) -> None:
    """A finished job no longer needs its uploaded source video."""
    job = _job_snapshot(job_id)
    if job is not None:
        cleanup.delete_upload(job.get("video_path"), UPLOAD_DIR)
    sweep_media()


def _create_job(video_path: Path, user_id: int | None) -> str:
    job_id = str(uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "uploaded",
            "stage": "Video uploaded",
            "progress_percent": 10,
            "video_path": str(video_path),
            "user_id": user_id,
            "team_a_name": None,
            "team_a_color": None,
            "team_b_name": None,
            "team_b_color": None,
            "team_candidates": None,
            "result": None,
            "error": None,
        }
    return job_id


def _rgb_hex(rgb: list[int] | tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*[int(value) for value in rgb])


def _detect_team_candidates(job_id: str) -> None:
    job = _job_snapshot(job_id)
    if job is None:
        return
    try:
        _set_job(
            job_id,
            status="detecting_teams",
            stage="Finding representative players",
            progress_percent=11,
        )

        capture = cv2.VideoCapture(job["video_path"])
        if not capture.isOpened():
            raise RuntimeError("The uploaded video could not be opened.")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

            # Scan the whole video, not just the first 45 seconds, and take more
            # frames from longer clips. The old fixed 12 frames over the opening
            # 45s is how the two-team model could end up resting on a couple of
            # players.
            duration = frame_count / max(fps, 1.0)
            sample_frames = int(
                min(CALIBRATION_MAX_FRAMES, max(CALIBRATION_MIN_FRAMES, duration / 2.0))
            )
            last_frame = max(frame_count - 1, 0)
            positions = (
                np.linspace(0, last_frame, num=sample_frames, dtype=int)
                if last_frame
                else np.asarray([0])
            )

            frames = []
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if ok:
                    frames.append(frame)
        finally:
            capture.release()
        if not frames:
            raise RuntimeError("No readable frames were found in the uploaded video.")

        tracker = _get_tracker()
        detections = tracker.detect_outfield_players(frames)
        # Do not admit crops so small the kit colour is mostly noise, but scale the
        # floor to the source resolution so 720p and 1080p are not held to a 480p
        # threshold.
        min_crop_height = max(28, int(frame_height * 0.06))
        samples: list[TeamSample] = []
        for frame, boxes in zip(frames, detections):
            for bbox in boxes:
                crop = safe_player_crop(
                    frame, bbox, min_height=min_crop_height, min_width=12
                )
                if crop is None:
                    continue
                color = kit_color_bgr(frame, bbox)
                if color is None:
                    continue
                samples.append(
                    TeamSample(
                        color=color,
                        crop=crop,
                        frame=frame,
                        bbox=bbox,
                        sharpness=crop_sharpness(crop),
                    )
                )

        centers_rgb, representatives = cluster_team_samples(
            samples,
            min_samples=CALIBRATION_MIN_SAMPLES,
            min_per_cluster=CALIBRATION_MIN_PER_CLUSTER,
            min_separation=CALIBRATION_MIN_SEPARATION,
        )
        candidates = []
        for index, representative in enumerate(representatives, start=1):
            # A zoomed-out view with the player outlined, at a fixed readable size.
            # Writing the raw box produced 28x46 px files the app then blew up ~25x.
            image = render_team_sample(representative)
            image_path = TEAM_SAMPLE_DIR / f"{job_id}-team-{index}.jpg"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError("A representative player image could not be saved.")
            candidates.append(
                {
                    "cluster_id": index,
                    "label": f"Detected team {index}",
                    "sample_image": f"/static/team_samples/{image_path.name}",
                }
            )
        _set_job(
            job_id,
            status="awaiting_team_confirmation",
            stage="Name the detected teams",
            progress_percent=13,
            team_candidates=candidates,
            detected_team_colors=centers_rgb,
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            stage="Team detection failed",
            progress_percent=0,
            error=str(exc),
        )
        _release_job_files(job_id)


def _get_or_create_team(db: Session, name: str, color: str) -> Team:
    """The team row for a name the user typed at the confirmation step.

    Resolution lives in backend.roster so the seeder matches teams the same
    way. Without that, "Real Madrid" and "real madrid cf" become two rows and
    the seeded squad attaches to only one of them, so _roster_player below
    finds nothing and every scorer comes back as a bare number.
    """
    return resolve_team(db, name, color)


def _roster_player(db: Session, team: Team | None, jersey_number, create_slot: bool = True) -> Player | None:
    """Roster row for a (team, shirt number), creating an unnamed slot if needed.

    The roster is the authority on names, never OCR. A slot with no name is
    inserted so a real line-up can fill it in later, keyed on the
    (team_id, jersey_no) unique constraint.
    """
    if jersey_number is None or team is None:
        return None
    player = db.scalar(
        select(Player).where(
            Player.team_id == team.team_id,
            Player.jersey_no == int(jersey_number),
        )
    )
    if player is None and create_slot:
        player = Player(team_id=team.team_id, jersey_no=int(jersey_number), name=None)
        db.add(player)
        db.flush()
    return player


def _clean_team_name(raw_team_name) -> str | None:
    # "unknown_team" is an internal token. Store NULL instead so the app can
    # degrade gracefully rather than printing it as a team name.
    if raw_team_name and str(raw_team_name) != "unknown_team":
        return str(raw_team_name)
    return None


def _persist_match(job: dict[str, Any], events: list[dict], fps: float) -> int:
    from main import describe_event

    with SessionLocal() as db:
        user_id = job.get("user_id")
        if user_id is not None and db.get(User, user_id) is None:
            user_id = None

        team_a = _get_or_create_team(db, job["team_a_name"], job["team_a_color"])
        team_b = _get_or_create_team(db, job["team_b_name"], job["team_b_color"])
        match = Match(
            match_date=datetime.now(timezone.utc).replace(tzinfo=None),
            location="Uploaded video",
            user_id=user_id,
            job_id=job["job_id"],
            source_video=job["video_path"],
            status="completed",
        )
        db.add(match)
        db.flush()
        plays = {
            team_a.team_name: Play(team_id=team_a.team_id, match_id=match.match_id, score=0),
            team_b.team_name: Play(team_id=team_b.team_id, match_id=match.match_id, score=0),
        }
        db.add_all(plays.values())
        db.flush()

        teams_by_name = {team_a.team_name: team_a, team_b.team_name: team_b}
        for detected in events:
            event_type = str(detected.get("event", "event")).lower()
            scorer = detected.get("scorer") or {}
            # Every participant of the event: passer and receiver, shooter and
            # keeper, booked player and referee. The primary actor comes first
            # and is the one the old single-occurrence readers see.
            actors = [dict(actor) for actor in (detected.get("actors") or [])]
            if not actors and scorer:
                actors = [dict(scorer)]
            if actors and scorer and actors[0].get("track_id") != scorer.get("track_id"):
                actors.sort(key=lambda actor: 0 if actor.get("track_id") == scorer.get("track_id") else 1)

            frame_number = int(detected.get("frame", 0))
            time_seconds = frame_number / max(fps, 1.0)
            end_frame = int(detected.get("end_frame", frame_number))
            end_seconds = max(time_seconds, end_frame / max(fps, 1.0))

            # Resolve every actor against the roster before writing anything, so
            # the description can carry the roster names.
            resolved: list[tuple[dict, Player | None, Team | None, str | None]] = []
            for actor in actors:
                team_name = _clean_team_name(actor.get("team_name"))
                team = teams_by_name.get(team_name) if team_name else None
                player = _roster_player(db, team, actor.get("jersey_number"))
                if actor.get("unconfirmed"):
                    # Goal scorer evidence too thin to name (goal_scorer.fuse_scorer):
                    # keep the number, attach no roster player or name.
                    player = None
                actor["player_name"] = player.name if player is not None else None
                resolved.append((actor, player, team, team_name))

            event_team_name = _clean_team_name(detected.get("team_name")) or (
                resolved[0][3] if resolved else None
            )
            play = plays.get(event_team_name) if event_team_name else None

            if event_type == "goal":
                primary_name = resolved[0][0].get("player_name") if resolved else None
                description = format_goal_event(
                    {**detected, "scorer": {**scorer, "player_name": primary_name}}
                )
            elif actors:
                description = describe_event({**detected, "actors": [row[0] for row in resolved]})
            else:
                description = detected.get("description") or f"{event_type.upper()} detected"

            event = Event(
                match_id=match.match_id,
                event_type=event_type,
                event_name=event_type.replace("_", " ").title(),
                description=description[:500],
                # Passes are recorded without a clip; highlight events carry one.
                video_path=detected.get("clip_url") or detected.get("clip_path"),
                time_seconds=time_seconds,
                clip_start_seconds=detected.get("clip_start_seconds"),
                clip_end_seconds=detected.get("clip_end_seconds"),
            )
            db.add(event)
            db.flush()

            if not resolved:
                db.add(
                    Occurrence(
                        event_id=event.event_id,
                        play_id=play.play_id if play else None,
                        start_time=time_seconds,
                        end_time=end_seconds,
                        detected_team_name=event_team_name,
                    )
                )
            for actor, player, team, team_name in resolved:
                actor_play = plays.get(team_name) if team_name else play
                db.add(
                    Occurrence(
                        event_id=event.event_id,
                        play_id=actor_play.play_id if actor_play else None,
                        player_id=player.player_id if player else None,
                        start_time=time_seconds,
                        end_time=end_seconds,
                        detected_track_id=actor.get("track_id"),
                        detected_jersey_no=actor.get("jersey_number"),
                        detected_player_name=actor.get("player_name"),
                        detected_team_name=team_name,
                        team_confidence=actor.get("team_confidence"),
                        jersey_confidence=actor.get("jersey_confidence"),
                        name_confidence=actor.get("name_confidence"),
                    )
                )
            if event_type == "goal" and play is not None:
                play.score += 1

        db.commit()
        return match.match_id


def _team_codes(name: str) -> str:
    """Scoreboard codes for a typed team name, from teams.short_code and
    three-letter aliases. Empty when the team or the database is unavailable -
    the pipeline then falls back to data/rosters and to spelling."""
    try:
        with SessionLocal() as db:
            team = find_team(db, name)
            if team is None:
                return ""
            codes = {team.short_code.upper()} if team.short_code else set()
            codes.update(
                alias.alias.upper()
                for alias in team.aliases
                if len(alias.alias) == 3 and alias.alias.isalpha()
            )
            return ",".join(sorted(codes))
    except Exception:
        return ""


def _process_job(job_id: str) -> None:
    job = _job_snapshot(job_id)
    if job is None:
        return
    try:
        _set_job(job_id, status="processing", stage="Detecting match events", progress_percent=15)
        capture = cv2.VideoCapture(job["video_path"])
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        capture.release()

        from main import process_video

        def report_progress(frame_index: int, total_frames: int) -> None:
            # The frame loop used to run start to finish with no progress channel
            # at all, so the bar sat at 15% for the whole job.
            if total_frames > 0:
                fraction = min(1.0, max(0.0, frame_index / float(total_frames)))
            else:
                fraction = 0.0
            _set_job(
                job_id,
                stage=f"Analysing frame {frame_index} of {total_frames or '?'}",
                progress_percent=int(
                    PROGRESS_ANALYSIS_START
                    + fraction * (PROGRESS_ANALYSIS_END - PROGRESS_ANALYSIS_START)
                ),
            )

        clip_dir = CLIP_DIR / job_id
        arguments = SimpleNamespace(
            video=job["video_path"],
            output=str(OUTPUT_DIR / f"{job_id}.mp4"),
            tracking_model=str(PROJECT_ROOT / "models" / "best.pt"),
            event_model=str(PROJECT_ROOT / "models" / "final_model_80.pth"),
            team_a_name=job["team_a_name"],
            team_a_color=job["team_a_color"],
            team_b_name=job["team_b_name"],
            team_b_color=job["team_b_color"],
            team_a_code=_team_codes(job["team_a_name"]),
            team_b_code=_team_codes(job["team_b_name"]),
            ocr_debug_dir=str(OCR_DIR / job_id),
            player_image_dir=str(PLAYER_IMAGE_DIR / job_id),
            player_image_url_prefix=f"/static/player_images/{job_id}",
            clip_dir=str(clip_dir),
            clip_url_prefix=f"/api/processing/clips/{job_id}",
            # The full annotated render is off: it produced a 327 MB MJPG AVI that
            # Android cannot decode, and cost a copy and an encode every frame.
            write_annotated_video=False,
            no_display=True,
        )
        events = process_video(arguments, progress_cb=report_progress)
        _set_job(job_id, stage="Saving match summary", progress_percent=PROGRESS_PERSIST)
        match_id = _persist_match(job, events, fps)

        # Keyed by team id, not by the typed name. Matching on the name string
        # meant any event without a confident team silently left the sheet 0-0.
        team_names = {1: job["team_a_name"], 2: job["team_b_name"]}
        score_by_team_id = {1: 0, 2: 0}
        player_results = []
        for event in events:
            if str(event.get("event", "")).lower() != "goal":
                continue
            scorer = event.get("scorer") or {}
            team_id = int(scorer.get("team_id") or 0)
            if team_id in score_by_team_id:
                score_by_team_id[team_id] += 1
            player_results.append(
                {
                    "team_name": team_names.get(team_id),
                    "jersey_number": scorer.get("jersey_number"),
                    "player_name": scorer.get("player_name"),
                    "player_image": scorer.get("player_image"),
                    "time_seconds": int(event.get("frame", 0)) / max(fps, 1.0),
                    "clip_url": event.get("clip_url"),
                }
            )
        from track_events import summarise_passes

        event_counts: dict[str, int] = {}
        # Shots on target per team. Stored in the events table like every
        # other event; the app's summary page does not list them.
        shots_on_target: dict[str, int] = {}
        for event in events:
            key = str(event.get("event", "event")).lower()
            event_counts[key] = event_counts.get(key, 0) + 1
            if key == "shot" and (event.get("details") or {}).get("on_target", True):
                team = _clean_team_name(event.get("team_name")) or "unknown team"
                shots_on_target[team] = shots_on_target.get(team, 0) + 1
        result = {
            "db_response": {"success": True, "matchId": match_id},
            "match_id": match_id,
            "events": events,
            "event_counts": event_counts,
            "shots_on_target": shots_on_target,
            "pass_summary": summarise_passes(events),
            "score_sheet": [
                {"team_name": team_names[team_id], "score": score}
                for team_id, score in score_by_team_id.items()
            ],
            "players": player_results,
            # One entry per merged clip; a goal and the shot before it share one.
            "clips": clips_from_events(events),
            "reel": _reel(job_id),
        }
        _set_job(
            job_id,
            status="completed",
            stage="Analysis complete",
            progress_percent=100,
            result=result,
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            stage="Processing failed",
            progress_percent=0,
            error=str(exc),
        )
    finally:
        _release_job_files(job_id)


def _reel(job_id: str | None) -> dict | None:
    """The joined highlight reel of a job, if one was written and not yet expired."""
    if not job_id or not _SAFE_JOB_ID.fullmatch(job_id):
        return None
    path = CLIP_DIR / job_id / REEL_FILE_NAME
    if not path.is_file():
        return None
    return {
        "url": f"/api/processing/clips/{job_id}/{REEL_FILE_NAME}",
        "size_bytes": path.stat().st_size,
    }


@router.post("/api/processing/submit-file", status_code=202)
async def submit_file(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    user_id: int | None = Form(None),
):
    suffix = Path(video.filename or "match.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=415, detail="Unsupported video format.")
    temporary_path = UPLOAD_DIR / f"{uuid4()}{suffix}"
    size = 0
    try:
        with temporary_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 2 GB limit.")
                destination.write(chunk)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    job_id = _create_job(temporary_path, user_id)
    background_tasks.add_task(_detect_team_candidates, job_id)
    return {"job_id": job_id, "status": "uploaded"}


@router.post("/api/processing/submit-url", status_code=202)
def submit_url(payload: VideoUrlRequest, background_tasks: BackgroundTasks):
    target = UPLOAD_DIR / f"{uuid4()}.mp4"
    try:
        with httpx.stream("GET", str(payload.video_url), follow_redirects=True, timeout=90) as response:
            response.raise_for_status()
            size = 0
            with target.open("wb") as destination:
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Video exceeds the 2 GB limit.")
                    destination.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    job_id = _create_job(target, payload.user_id)
    background_tasks.add_task(_detect_team_candidates, job_id)
    return {"job_id": job_id, "status": "uploaded"}


@router.post("/api/processing/confirm-teams/{job_id}", status_code=202)
def confirm_teams(
    job_id: str,
    payload: TeamConfirmationRequest,
    background_tasks: BackgroundTasks,
):
    team_1_name = payload.team_1_name.strip()
    team_2_name = payload.team_2_name.strip()
    if team_1_name.casefold() == team_2_name.casefold():
        raise HTTPException(status_code=422, detail="Enter a different name for each team.")
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Processing job not found.")
        if job["status"] != "awaiting_team_confirmation":
            raise HTTPException(
                status_code=409,
                detail=f"Team names cannot be confirmed while the job is {job['status']}.",
            )
        colors = job.get("detected_team_colors")
        if not colors or len(colors) != 2:
            raise HTTPException(status_code=409, detail="Detected team colors are unavailable.")
        job.update(
            {
                "team_a_name": team_1_name,
                "team_a_color": _rgb_hex(colors[0]),
                "team_b_name": team_2_name,
                "team_b_color": _rgb_hex(colors[1]),
                "status": "queued",
                "stage": "Team names confirmed",
                "progress_percent": 14,
            }
        )
    background_tasks.add_task(_process_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@router.get("/api/processing/clips/{job_id}/{file_name}")
def job_clip(job_id: str, file_name: str):
    """Serve one highlight clip.

    ExoPlayer will not play a stream without byte-range support, so this goes
    through FileResponse rather than returning the bytes directly.
    """
    if not _SAFE_JOB_ID.fullmatch(job_id) or not _SAFE_CLIP_NAME.fullmatch(file_name):
        raise HTTPException(status_code=404, detail="Clip not found.")

    path = (CLIP_DIR / job_id / file_name).resolve()
    # Defence in depth: the patterns above already exclude separators and dots.
    if not path.is_file() or CLIP_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


@router.get("/api/processing/status/{job_id}")
def job_status(job_id: str):
    job = _job_snapshot(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Processing job not found.")
    return {
        key: job.get(key)
        for key in (
            "job_id",
            "status",
            "stage",
            "progress_percent",
            "team_candidates",
            "error",
        )
    }


@router.get("/api/processing/result/{job_id}")
def job_result(job_id: str):
    job = _job_snapshot(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Processing job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}.")
    return {"job_id": job_id, "status": "completed", "result": job["result"]}


@router.get("/api/matches")
def get_matches(user_id: int | None = None, db: Session = Depends(get_db)):
    statement = select(Match).order_by(Match.match_date.desc())
    if user_id is not None:
        statement = statement.where(Match.user_id == user_id)
    matches = db.scalars(statement).all()
    return [
        {
            "matchId": match.match_id,
            "matchDate": match.match_date,
            "location": match.location,
            "userId": match.user_id,
            "status": match.status,
        }
        for match in matches
    ]


def _detected(occurrence_by_event: dict, event: Event, field: str):
    occurrence = occurrence_by_event.get(event.event_id)
    return getattr(occurrence, field, None) if occurrence is not None else None


@router.get("/api/matches/{match_id}")
def get_match_summary(match_id: int, db: Session = Depends(get_db)):
    match = db.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")
    plays = db.scalars(select(Play).where(Play.match_id == match_id)).all()
    events = db.scalars(select(Event).where(Event.match_id == match_id)).all()
    event_ids = [event.event_id for event in events]
    occurrences = (
        db.scalars(select(Occurrence).where(Occurrence.event_id.in_(event_ids))).all()
        if event_ids
        else []
    )
    team_ids = [play.team_id for play in plays if play.team_id is not None]
    teams = db.scalars(select(Team).where(Team.team_id.in_(team_ids))).all() if team_ids else []
    player_ids = [row.player_id for row in occurrences if row.player_id is not None]
    players = db.scalars(select(Player).where(Player.player_id.in_(player_ids))).all() if player_ids else []
    occurrence_by_event = {row.event_id: row for row in occurrences}
    reel = _reel(match.job_id)
    return {
        "match": {
            "matchId": match.match_id,
            "matchDate": match.match_date,
            "location": match.location,
            "userId": match.user_id,
        },
        "reelUrl": reel["url"] if reel else None,
        "teams": [
            {
                "teamId": team.team_id,
                "teamName": team.team_name,
                "primaryColor": team.primary_color,
            }
            for team in teams
        ],
        "play": [
            {
                "playId": play.play_id,
                "teamId": play.team_id,
                "matchId": play.match_id,
                "score": play.score,
            }
            for play in plays
        ],
        "events": [
            {
                "eventId": event.event_id,
                "eventType": event.event_type,
                "eventName": event.event_name,
                "description": event.description,
                "clipFileLocation": event.video_path,
                "timeSec": event.time_seconds,
                # Clip bounds in the source video; timeSec - clipStartSec is
                # where the event happens inside the clip.
                "clipStartSec": event.clip_start_seconds,
                "clipEndSec": event.clip_end_seconds,
                # An event with no occurrence row used to raise KeyError here and
                # fail the whole summary with a 500.
                "teamName": _detected(occurrence_by_event, event, "detected_team_name"),
                "jerseyNumber": _detected(occurrence_by_event, event, "detected_jersey_no"),
                "playerName": _detected(occurrence_by_event, event, "detected_player_name"),
            }
            for event in events
        ],
        "occurBy": [
            {
                "occurId": row.occurrence_id,
                "eventId": row.event_id,
                "playId": row.play_id,
                "playerId": row.player_id,
                "startTime": row.start_time,
                "endTime": row.end_time,
                "detectedPlayerTrackId": row.detected_track_id,
                "detectedJerseyNo": row.detected_jersey_no,
                "detectedPlayerName": row.detected_player_name,
                "teamName": row.detected_team_name,
                "teamConfidence": row.team_confidence,
                "jerseyConfidence": row.jersey_confidence,
                "nameConfidence": row.name_confidence,
            }
            for row in occurrences
        ],
        "players": [
            {
                "playerId": player.player_id,
                "name": player.name,
                "jerseyNo": player.jersey_no,
                "teamId": player.team_id,
            }
            for player in players
        ],
    }
