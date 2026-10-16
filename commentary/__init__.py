"""Commentary as the primary source of player names (speech to text, matched to squads)."""

from .assign import Attachment, attach_to_tracks, foul_roles, goal_roles
from .mentions import Candidate, Mention, find_mentions, normalise, ruled_out
from .transcribe import DEFAULT_WHISPER_MODEL, Word, load_whisper, transcribe_video

__all__ = [
    "Attachment",
    "Candidate",
    "DEFAULT_WHISPER_MODEL",
    "Mention",
    "Word",
    "attach_to_tracks",
    "find_mentions",
    "foul_roles",
    "goal_roles",
    "load_whisper",
    "normalise",
    "ruled_out",
    "transcribe_video",
]
