"""Commentary speech to timed words (faster-whisper, local, CPU).

The video's audio is extracted with ffmpeg to 16 kHz mono and transcribed
with word timestamps. A video without an audio track (the Man City fixture
clip) gives no words; so does one with crowd noise or music only (the Real
Madrid fixture clip transcribes as nothing), and the rest of the pipeline then
runs as before.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPER_DIR = os.path.join(ROOT, "models", "whisper")
DEFAULT_WHISPER_MODEL = "small"


@dataclass(frozen=True)
class Word:
    start: float   # seconds from the start of the video
    end: float
    text: str
    probability: float = 1.0


def extract_audio(video_path: str, wav_path: str) -> bool:
    """16 kHz mono wav of the video's audio; False when it has no audio."""
    from highlights.encoder import ffmpeg_binary

    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        return False
    result = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
        capture_output=True,
    )
    return result.returncode == 0 and os.path.isfile(wav_path) and os.path.getsize(wav_path) > 1000


def load_whisper(model_size: str = DEFAULT_WHISPER_MODEL):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8", download_root=WHISPER_DIR)


def transcribe_video(video_path: str, model, language: Optional[str] = None) -> Tuple[List[Word], Optional[str]]:
    """(words, detected language). Empty when there is no usable audio."""
    handle, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    try:
        if not extract_audio(video_path, wav_path):
            return [], None
        segments, info = model.transcribe(
            wav_path, word_timestamps=True, vad_filter=True, language=language, beam_size=5
        )
        words = [
            Word(float(w.start), float(w.end), w.word.strip(), float(getattr(w, "probability", 1.0)))
            for segment in segments
            for w in (segment.words or [])
            if w.word.strip()
        ]
        return words, getattr(info, "language", None)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
