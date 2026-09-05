from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4
import threading

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from scorer_identifier import format_goal_event
from team_assigner import TeamAssigner, cluster_team_samples, safe_player_crop

from ..database import SessionLocal, get_db
from ..models import Event, Match, Occurrence, Play, Player, Team, User
from ..schemas import TeamConfirmationRequest, VideoUrlRequest


router = APIRouter(tags=["processing"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "static" / "outputs"
OCR_DIR = PROJECT_ROOT / "backend" / "static" / "ocr_debug"
TEAM_SAMPLE_DIR = PROJECT_ROOT / "backend" / "static" / "team_samples"
PLAYER_IMAGE_DIR = PROJECT_ROOT / "backend" / "static" / "player_images"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
for directory in (UPLOAD_DIR, OUTPUT_DIR, OCR_DIR, TEAM_SAMPLE_DIR, PLAYER_IMAGE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **changes) -> None:
    with _jobs_lock:
        _jobs[job_id].update(changes)


def _job_snapshot(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


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
        _set_job(job_id, status="detecting_teams", stage="Finding representative players", progress_percent=8)
        from trackers import Tracker

        capture = cv2.VideoCapture(job["video_path"])
        if not capture.isOpened():
            raise RuntimeError("The uploaded video could not be opened.")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        scan_end = min(max(frame_count - 1, 0), int(fps * 45))
        positions = np.linspace(0, scan_end, num=12, dtype=int) if scan_end else np.asarray([0])
        frames = []
        try:
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if ok:
                    frames.append(frame)
        finally:
            capture.release()
        if not frames:
            raise RuntimeError("No readable frames were found in the uploaded video.")

        tracker = Tracker(str(PROJECT_ROOT / "models" / "best.pt"))
        tracks = tracker.get_object_tracks(frames, read_from_stub=False)
        color_reader = TeamAssigner()
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for frame, players in zip(frames, tracks["players"]):
            for player in players.values():
                crop = safe_player_crop(frame, player["bbox"])
                if crop is None:
                    continue
                try:
                    color = color_reader.get_player_color(frame, player["bbox"])
                except (ValueError, RuntimeError):
                    continue
                samples.append((np.asarray(color, dtype=np.float32), crop))

        centers_rgb, crops = cluster_team_samples(samples)
        candidates = []
        for index, (rgb, crop) in enumerate(zip(centers_rgb, crops), start=1):
            image_path = TEAM_SAMPLE_DIR / f"{job_id}-team-{index}.jpg"
            if not cv2.imwrite(str(image_path), crop):
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
            progress_percent=12,
            team_candidates=candidates,
            detected_team_colors=centers_rgb,
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            stage="Team detection failed",
            error=str(exc),
        )


def _get_or_create_team(db: Session, name: str, color: str) -> Team:
    team = db.scalar(select(Team).where(func.lower(Team.team_name) == name.lower()))
    if team is None:
        team = Team(team_name=name, primary_color=color)
        db.add(team)
        db.flush()
    elif color:
        team.primary_color = color
    return team


def _persist_match(job: dict[str, Any], events: list[dict], fps: float) -> int:
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
            team_name = str(scorer.get("team_name") or "unknown_team")
            frame_number = int(detected.get("frame", 0))
            time_seconds = frame_number / max(fps, 1.0)
            description = (
                format_goal_event(detected)
                if event_type == "goal"
                else f"{event_type.upper()} detected"
            )
            event = Event(
                match_id=match.match_id,
                event_type=event_type,
                event_name=event_type.title(),
                description=description,
                time_seconds=time_seconds,
            )
            db.add(event)
            db.flush()

            play = plays.get(team_name)
            team = teams_by_name.get(team_name)
            jersey_number = scorer.get("jersey_number")
            player = None
            if jersey_number is not None and team is not None:
                player = db.scalar(
                    select(Player).where(
                        Player.team_id == team.team_id,
                        Player.jersey_no == int(jersey_number),
                    )
                )
                if player is None:
                    player = Player(
                        team_id=team.team_id,
                        jersey_no=int(jersey_number),
                        name=scorer.get("player_name"),
                    )
                    db.add(player)
                    db.flush()
                elif not player.name and scorer.get("player_name"):
                    player.name = str(scorer["player_name"])

            db.add(
                Occurrence(
                    event_id=event.event_id,
                    play_id=play.play_id if play else None,
                    player_id=player.player_id if player else None,
                    start_time=time_seconds,
                    end_time=time_seconds,
                    detected_track_id=scorer.get("track_id"),
                    detected_jersey_no=jersey_number,
                    detected_player_name=scorer.get("player_name"),
                    detected_team_name=team_name,
                    team_confidence=scorer.get("team_confidence"),
                    jersey_confidence=scorer.get("jersey_confidence"),
                    name_confidence=scorer.get("name_confidence"),
                )
            )
            if event_type == "goal" and play is not None:
                play.score += 1

        db.commit()
        return match.match_id


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

        output_path = OUTPUT_DIR / f"{job_id}.avi"
        arguments = SimpleNamespace(
            video=job["video_path"],
            output=str(output_path),
            tracking_model=str(PROJECT_ROOT / "models" / "best.pt"),
            event_model=str(PROJECT_ROOT / "models" / "final_model_80.pth"),
            team_a_name=job["team_a_name"],
            team_a_color=job["team_a_color"],
            team_b_name=job["team_b_name"],
            team_b_color=job["team_b_color"],
            ocr_debug_dir=str(OCR_DIR / job_id),
            player_image_dir=str(PLAYER_IMAGE_DIR / job_id),
            player_image_url_prefix=f"/static/player_images/{job_id}",
            no_display=True,
        )
        events = process_video(arguments)
        _set_job(job_id, stage="Saving match summary", progress_percent=92)
        match_id = _persist_match(job, events, fps)
        score_by_team = {job["team_a_name"]: 0, job["team_b_name"]: 0}
        player_results = []
        for event in events:
            if str(event.get("event", "")).lower() != "goal":
                continue
            scorer = event.get("scorer") or {}
            team_name = scorer.get("team_name") or "Unknown team"
            if team_name in score_by_team:
                score_by_team[team_name] += 1
            player_results.append(
                {
                    "team_name": team_name,
                    "jersey_number": scorer.get("jersey_number"),
                    "player_name": scorer.get("player_name"),
                    "player_image": scorer.get("player_image"),
                    "time_seconds": int(event.get("frame", 0)) / max(fps, 1.0),
                }
            )
        result = {
            "db_response": {"success": True, "matchId": match_id},
            "match_id": match_id,
            "events": events,
            "score_sheet": [
                {"team_name": name, "score": score}
                for name, score in score_by_team.items()
            ],
            "players": player_results,
            "annotated_video": f"/static/outputs/{job_id}.avi",
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
            error=str(exc),
        )


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
    return {
        "match": {
            "matchId": match.match_id,
            "matchDate": match.match_date,
            "location": match.location,
            "userId": match.user_id,
        },
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
                "teamName": occurrence_by_event[event.event_id].detected_team_name,
                "jerseyNumber": occurrence_by_event[event.event_id].detected_jersey_no,
                "playerName": occurrence_by_event[event.event_id].detected_player_name,
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
