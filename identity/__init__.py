"""Few-shot player identification: reference images, embeddings, per-track belief."""

from .embedder import DEFAULT_REID_WEIGHTS, MIN_CROP_HEIGHT, ReidEmbedder, augment, body_crop
from .gallery import GALLERY_DIR, MAX_IMAGES_PER_PLAYER, PlayerGallery, TeamPrototypes, team_key
from .posterior import NAME_MIN_PROBABILITY, IdentityTracker

__all__ = [
    "DEFAULT_REID_WEIGHTS",
    "GALLERY_DIR",
    "IdentityTracker",
    "MAX_IMAGES_PER_PLAYER",
    "MIN_CROP_HEIGHT",
    "NAME_MIN_PROBABILITY",
    "PlayerGallery",
    "ReidEmbedder",
    "TeamPrototypes",
    "augment",
    "body_crop",
    "team_key",
]
