"""Write the planned highlight clips and, optionally, the match reel."""

import os
from typing import Iterable, List, Optional, Sequence

import cv2

from .encoder import concat_clips, cut_clip, ffmpeg_binary, open_mp4_writer, probe
from .segments import ClipPlan, plan_clips

REEL_FILE_NAME = "reel.mp4"


def _cut_with_opencv(video_path: str, plan: ClipPlan, destination: str, fps: float) -> bool:
    """Silent fallback for machines without ffmpeg."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return False
    written = 0
    try:
        size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        writer = open_mp4_writer(destination, fps, size)
        if writer is None:
            return False
        capture.set(cv2.CAP_PROP_POS_FRAMES, plan.start)
        try:
            for _ in range(plan.frame_count):
                success, frame = capture.read()
                if not success:
                    break
                writer.write(frame)
                written += 1
        finally:
            writer.release()
    finally:
        capture.release()
    if written == 0 and os.path.exists(destination):
        os.remove(destination)
    return written > 0


def export_highlights(
    video_path: str,
    events: List[dict],
    fps: float,
    clip_dir: Optional[str],
    url_prefix: Optional[str] = None,
    boundaries: Iterable = (),
    total_frames: int = 0,
    reel: bool = True,
) -> dict:
    """Cut one clip per merged passage of play and annotate its events.

    Every event inside a clip gets ``clip_path``, ``clip_url``, ``clip_index``,
    ``clip_start_seconds``, ``clip_end_seconds`` and ``clip_offset_seconds``
    (where the event happens inside the clip). A goal, the shot before it and
    a save share one clip.
    """
    summary = {"clips": [], "reel": None, "encoder": None}
    if not clip_dir or not events:
        return summary
    fps = max(float(fps), 1.0)
    if total_frames <= 0:
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
        capture.release()

    plans = plan_clips(events, fps, total_frames, boundaries)
    if not plans:
        return summary
    os.makedirs(clip_dir, exist_ok=True)

    info = probe(video_path) if ffmpeg_binary() else None
    summary["encoder"] = "ffmpeg" if info is not None else "opencv"
    if info is None:
        print("ffmpeg/ffprobe not found: writing silent clips with OpenCV and no reel")

    written_paths: List[str] = []
    for number, plan in enumerate(plans, start=1):
        file_name = f"{plan.primary_type}-{number:02d}.mp4"
        path = os.path.join(clip_dir, file_name)
        start_seconds = plan.start / fps
        duration_seconds = plan.frame_count / fps
        if info is not None:
            ok = cut_clip(video_path, path, start_seconds, duration_seconds, info)
        else:
            ok = _cut_with_opencv(video_path, plan, path, fps)
        if not ok:
            continue
        written_paths.append(path)
        url = f"{url_prefix.rstrip('/')}/{file_name}" if url_prefix else None
        end_seconds = round((plan.end + 1) / fps, 3)
        for index in plan.event_indices:
            event = events[index]
            event["clip_path"] = path
            event["clip_index"] = number
            event["clip_start_seconds"] = round(start_seconds, 3)
            event["clip_end_seconds"] = end_seconds
            event["clip_offset_seconds"] = round(
                max(0.0, int(event.get("frame", 0) or 0) / fps - start_seconds), 3
            )
            if url:
                event["clip_url"] = url
        summary["clips"].append(
            {
                "index": number,
                "file_name": file_name,
                "path": path,
                "url": url,
                "primary_event": plan.primary_type,
                "event_types": list(plan.event_types),
                "start_seconds": round(start_seconds, 3),
                "end_seconds": end_seconds,
                "has_audio": bool(info and info.has_audio),
                "size_bytes": os.path.getsize(path),
            }
        )

    # A reel of one clip would be a second copy of it.
    if reel and info is not None and len(written_paths) >= 2:
        reel_path = os.path.join(clip_dir, REEL_FILE_NAME)
        if concat_clips(written_paths, reel_path):
            summary["reel"] = {
                "file_name": REEL_FILE_NAME,
                "path": reel_path,
                "url": f"{url_prefix.rstrip('/')}/{REEL_FILE_NAME}" if url_prefix else None,
                "duration_seconds": round(
                    sum(c["end_seconds"] - c["start_seconds"] for c in summary["clips"]), 3
                ),
                "size_bytes": os.path.getsize(reel_path),
            }
    return summary


def clips_from_events(events: Sequence[dict]) -> List[dict]:
    """The distinct clips referenced by a list of annotated events, in order."""
    clips = {}
    for event in events:
        index = event.get("clip_index")
        if index is None:
            continue
        clip = clips.setdefault(
            index,
            {
                "index": index,
                "url": event.get("clip_url"),
                "start_seconds": event.get("clip_start_seconds"),
                "end_seconds": event.get("clip_end_seconds"),
                "event_types": [],
            },
        )
        clip["event_types"].append(str(event.get("event", "")).lower())
    return [clips[index] for index in sorted(clips)]
