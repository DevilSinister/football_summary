"""Cut clips and join the highlight reel with ffmpeg.

Clips are re-encoded, not stream-copied. A copy can only start on a keyframe,
and both reference sources place one every 1.2 s, so a copied clip would open
with up to a second of the previous camera shot - undoing the cut-aware start
chosen in segments.py - and its stored start time would be wrong by as much.
Re-encoding with x264 at CRF 23, capped at the source's own video bitrate and
at a per-resolution ceiling, keeps the frame-exact start, carries the audio
track, and keeps every clip at or below the source bitrate. The OpenCV writer
this replaces wrote no audio at 3.7x the source bitrate (a 10 s 1080p goal was
58 MB), with the MP4 index at the end of the file.

The reel is a stream copy of the finished clips: one source and one encoder
configuration, so they concatenate without a second encode.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2

CRF = 23
PRESET = "veryfast"
AUDIO_BITRATE = "128k"
# Video bitrate ceilings by source frame height, bits per second. Whichever is
# lower of this and the source's own bitrate caps the encode.
BITRATE_CEILINGS = ((1080, 8_000_000), (720, 5_000_000), (0, 2_500_000))


def ffmpeg_binary() -> Optional[str]:
    return os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")


def ffprobe_binary() -> Optional[str]:
    return os.environ.get("FFPROBE_BINARY") or shutil.which("ffprobe")


@dataclass(frozen=True)
class SourceInfo:
    width: int
    height: int
    has_audio: bool
    video_bitrate: Optional[int]


def probe(path: str) -> Optional[SourceInfo]:
    """Size, audio presence and video bitrate of a file, or None without ffprobe."""
    ffprobe = ffprobe_binary()
    if not ffprobe:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "stream=codec_type,width,height,bit_rate:format=bit_rate",
                "-of", "json", path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        payload = json.loads(completed.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return None
    bitrate = video.get("bit_rate") or (payload.get("format") or {}).get("bit_rate")
    try:
        bitrate = int(bitrate) if bitrate else None
    except ValueError:
        bitrate = None
    return SourceInfo(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        video_bitrate=bitrate,
    )


def target_bitrate(info: SourceInfo) -> int:
    ceiling = next(limit for height, limit in BITRATE_CEILINGS if info.height >= height)
    return min(ceiling, info.video_bitrate) if info.video_bitrate else ceiling


def _run(command: List[str], timeout: float) -> Optional[str]:
    """Run ffmpeg; None on success, else the tail of its error output."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    if completed.returncode != 0:
        return (completed.stderr or "").strip()[-600:] or f"exit code {completed.returncode}"
    return None


def cut_clip(
    source: str,
    destination: str,
    start_seconds: float,
    duration_seconds: float,
    info: SourceInfo,
) -> bool:
    """Frame-accurate H.264/AAC clip with its index at the front (faststart)."""
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        return False
    rate = target_bitrate(info)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        # -ss before -i seeks fast; with a re-encode it is still frame-accurate.
        "-ss", f"{max(0.0, start_seconds):.3f}", "-i", source,
        "-t", f"{duration_seconds:.3f}",
        "-map", "0:v:0",
        "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
        "-maxrate", str(rate), "-bufsize", str(rate * 2),
        "-pix_fmt", "yuv420p", "-profile:v", "high",
    ]
    if info.has_audio:
        command += ["-map", "0:a:0", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2"]
    else:
        command += ["-an"]
    command += ["-movflags", "+faststart", destination]
    error = _run(command, timeout=max(120.0, duration_seconds * 20.0))
    if error is None and os.path.isfile(destination) and os.path.getsize(destination) > 0:
        return True
    print(f"ffmpeg could not cut {os.path.basename(destination)}: {error}")
    if os.path.exists(destination):
        os.remove(destination)
    return False


def concat_clips(paths: Sequence[str], destination: str) -> bool:
    """Join clips cut by cut_clip from one source into one file, without re-encoding."""
    ffmpeg = ffmpeg_binary()
    if not ffmpeg or not paths:
        return False
    list_path = destination + ".txt"
    with open(list_path, "w", encoding="utf-8") as handle:
        for path in paths:
            # The concat demuxer quotes with single quotes; forward slashes
            # keep Windows paths free of escape sequences.
            safe = os.path.abspath(path).replace(os.sep, "/").replace("'", "'\\''")
            handle.write(f"file '{safe}'\n")
    try:
        error = _run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-f", "concat", "-safe", "0", "-i", list_path,
                "-c", "copy", "-movflags", "+faststart", destination,
            ],
            timeout=600.0,
        )
    finally:
        os.remove(list_path)
    if error is None and os.path.isfile(destination) and os.path.getsize(destination) > 0:
        return True
    print(f"ffmpeg could not build the highlight reel: {error}")
    if os.path.exists(destination):
        os.remove(destination)
    return False


def open_mp4_writer(path, fps, size):
    """Open an OpenCV H.264 MP4 writer, falling back to MPEG-4 Part 2.

    Used for the optional annotated render, and for clips only when ffmpeg is
    missing: it writes no audio.
    """
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None
