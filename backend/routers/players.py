"""Reference images of a player, for naming him in footage (1-10 per player).

Images are stored in data/player_gallery (identity.PlayerGallery), outside the
public static folder, keyed by the player's team and shirt number. The player
must already exist in the roster (scripts/seed_roster.py) with a team and a
shirt number.
"""

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from identity import MAX_IMAGES_PER_PLAYER, PlayerGallery
from identity.enroll import crops_from_images

from ..database import get_db
from ..models import Player

router = APIRouter(prefix="/api/players", tags=["players"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class _EnrolmentDetector:
    """A YOLO instance of its own, used under a lock.

    Ultralytics predictors keep per-instance state, so the model shared with
    the video job (ModelCache.yolo) must not be called from a request thread
    while a job is running.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._model = None

    def crops(self, images):
        with self._lock:
            if self._model is None:
                from ultralytics import YOLO

                self._model = YOLO(str(PROJECT_ROOT / "models" / "best.pt"))
            return crops_from_images(images, self._model)


detector = _EnrolmentDetector()


def _player(db: Session, player_id: int) -> Player:
    player = db.get(Player, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found.")
    if player.team is None or player.jersey_no is None:
        raise HTTPException(status_code=422, detail="The player needs a team and a shirt number first.")
    return player


def _listing(player: Player) -> dict:
    gallery = PlayerGallery()
    images = gallery.list_images(player.team.team_name, player.jersey_no)
    return {
        "player_id": player.player_id,
        "team": player.team.team_name,
        "jersey_no": player.jersey_no,
        "name": player.name,
        "images": [Path(path).stem for path in images],
        "max_images": MAX_IMAGES_PER_PLAYER,
    }


@router.get("/{player_id}/reference-images")
def list_reference_images(player_id: int, db: Session = Depends(get_db)):
    return _listing(_player(db, player_id))


@router.post("/{player_id}/reference-images", status_code=201)
def add_reference_images(
    player_id: int,
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    # A plain def: FastAPI runs it in its threadpool, so player detection does
    # not block the event loop.
    player = _player(db, player_id)
    if not 1 <= len(images) <= MAX_IMAGES_PER_PLAYER:
        raise HTTPException(status_code=422, detail=f"Send between 1 and {MAX_IMAGES_PER_PLAYER} images.")
    decoded = []
    for upload in images:
        data = upload.file.read(MAX_IMAGE_BYTES + 1)
        upload.file.close()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=f"{upload.filename}: image exceeds 10 MB.")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=415, detail=f"{upload.filename}: not a readable image.")
        decoded.append(image)
    crops, rejected = detector.crops(decoded)
    if not crops:
        raise HTTPException(status_code=422, detail="No player could be found in the images.")
    try:
        PlayerGallery().enroll(player.team.team_name, player.jersey_no, player.name, crops)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {**_listing(player), "stored": len(crops), "rejected": [images[i].filename for i in rejected]}


@router.delete("/{player_id}/reference-images/{image_id}", status_code=204)
def delete_reference_image(player_id: int, image_id: str, db: Session = Depends(get_db)):
    player = _player(db, player_id)
    if not PlayerGallery().remove_image(player.team.team_name, player.jersey_no, image_id):
        raise HTTPException(status_code=404, detail="Image not found.")
