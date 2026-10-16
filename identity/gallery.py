"""Reference images of named players: 1 to 10 per player, no training.

Layout (outside the public backend/static folder)::

    data/player_gallery/<team key>/<shirt number>/meta.json   {"team_name", "jersey_no", "name"}
    data/player_gallery/<team key>/<shirt number>/<id>.jpg     full-body crops
    data/player_gallery/<team key>/embeddings.npz             cache, rebuilt when images change

The team key is ``backend.team_naming.normalise_team_name``, the same key the
rosters and the database use, so "Real Madrid CF" and "real madrid" share one
gallery. Files rather than a database table: the command line runs without SQL
Server, and the gallery needs no schema change.

How few images are enough:

* 1-2 images: each image plus five augmented views (mirror, tighter and looser
  framing, lighter and darker) are embedded and averaged into a prototype.
* 3+ images for every enrolled player of a team: a logistic regression on the
  augmented embeddings is fitted per team as well, which separates team-mates
  better than a nearest prototype. Fitting takes milliseconds; the network
  itself is never retrained (that would need roughly 50+ images per player).
"""

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .embedder import augment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALLERY_DIR = os.path.join(ROOT, "data", "player_gallery")
MAX_IMAGES_PER_PLAYER = 10
CLASSIFIER_MIN_IMAGES = 3


def team_key(team_name: Optional[str]) -> str:
    from backend.team_naming import normalise_team_name

    key = normalise_team_name(team_name or "")
    return key.replace(" ", "_") or "unknown"


@dataclass
class TeamPrototypes:
    team_name: str
    jerseys: List[int]
    names: List[Optional[str]]
    prototypes: np.ndarray                      # (members, 512), L2-normalised
    image_counts: List[int]
    classifier: Optional[object] = None         # sklearn LogisticRegression or None
    classes: List[int] = field(default_factory=list)

    def member_probabilities(self, embedding: np.ndarray, temperature: float) -> np.ndarray:
        """P(member | embedding, the player is one of the enrolled members)."""
        if self.classifier is not None:
            probs = self.classifier.predict_proba(embedding.reshape(1, -1))[0]
            order = [list(self.classes).index(j) for j in self.jerseys]
            return probs[order]
        logits = (self.prototypes @ embedding) / temperature
        weights = np.exp(logits - logits.max())
        return weights / weights.sum()

    def best_similarity(self, embedding: np.ndarray) -> float:
        return float((self.prototypes @ embedding).max()) if len(self.prototypes) else -1.0


class PlayerGallery:
    def __init__(self, root: str = GALLERY_DIR) -> None:
        self.root = root

    def player_dir(self, team_name: str, jersey_no: int) -> str:
        return os.path.join(self.root, team_key(team_name), str(int(jersey_no)))

    def list_images(self, team_name: str, jersey_no: int) -> List[str]:
        folder = self.player_dir(team_name, jersey_no)
        if not os.path.isdir(folder):
            return []
        return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".jpg"))

    def enroll(self, team_name: str, jersey_no: int, name: Optional[str], crops: Sequence[np.ndarray]) -> List[str]:
        """Save 1-10 full-body crops of one player. Returns the saved paths."""
        existing = self.list_images(team_name, jersey_no)
        if not crops:
            raise ValueError("No usable player image was given.")
        if len(existing) + len(crops) > MAX_IMAGES_PER_PLAYER:
            raise ValueError(
                f"A player can have at most {MAX_IMAGES_PER_PLAYER} reference images "
                f"({len(existing)} already stored)."
            )
        folder = self.player_dir(team_name, jersey_no)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"team_name": team_name, "jersey_no": int(jersey_no), "name": name}, handle)
        saved = []
        for crop in crops:
            path = os.path.join(folder, f"{uuid.uuid4().hex}.jpg")
            if not cv2.imwrite(path, crop):
                raise OSError(f"Could not write {path}")
            saved.append(path)
        return saved

    def remove_image(self, team_name: str, jersey_no: int, image_id: str) -> bool:
        if not image_id.isalnum():
            return False
        path = os.path.join(self.player_dir(team_name, jersey_no), f"{image_id}.jpg")
        if not os.path.isfile(path):
            return False
        os.remove(path)
        if not self.list_images(team_name, jersey_no):
            shutil.rmtree(self.player_dir(team_name, jersey_no), ignore_errors=True)
        return True

    def members(self, team_name: str) -> List[dict]:
        folder = os.path.join(self.root, team_key(team_name))
        if not os.path.isdir(folder):
            return []
        out = []
        for entry in sorted((e for e in os.listdir(folder) if e.isdigit()), key=int):
            images = self.list_images(team_name, int(entry))
            if not images:
                continue
            meta_path = os.path.join(folder, entry, "meta.json")
            meta = {}
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as handle:
                    meta = json.load(handle)
            out.append({"jersey_no": int(entry), "name": meta.get("name"), "images": images})
        return out

    def team_prototypes(self, team_name: str, embedder) -> Optional[TeamPrototypes]:
        members = self.members(team_name)
        if not members:
            return None
        embeddings: Dict[str, np.ndarray] = self._cached_embeddings(team_name, members, embedder)
        prototypes, jerseys, names, counts = [], [], [], []
        train_x, train_y = [], []
        for member in members:
            vectors = [embeddings[path] for path in member["images"] if path in embeddings]
            if not vectors:
                continue
            vectors = np.concatenate(vectors)
            prototype = vectors.mean(axis=0)
            prototypes.append(prototype / max(np.linalg.norm(prototype), 1e-6))
            jerseys.append(member["jersey_no"])
            names.append(member["name"])
            counts.append(len(member["images"]))
            train_x.append(vectors)
            train_y += [member["jersey_no"]] * len(vectors)
        if not prototypes:
            return None
        classifier, classes = None, []
        if len(jerseys) >= 2 and min(counts) >= CLASSIFIER_MIN_IMAGES:
            from sklearn.linear_model import LogisticRegression

            classifier = LogisticRegression(C=1.0, max_iter=500)
            classifier.fit(np.concatenate(train_x), np.asarray(train_y))
            classes = [int(c) for c in classifier.classes_]
        return TeamPrototypes(team_name, jerseys, names, np.stack(prototypes), counts, classifier, classes)

    def _cached_embeddings(self, team_name, members, embedder) -> Dict[str, np.ndarray]:
        """Augmented embeddings per image, cached by file name, size and mtime."""
        cache_path = os.path.join(self.root, team_key(team_name), "embeddings.npz")
        cache: Dict[str, np.ndarray] = {}
        if os.path.isfile(cache_path):
            with np.load(cache_path, allow_pickle=False) as stored:
                cache = {key: stored[key] for key in stored.files}
        result, changed = {}, False
        for member in members:
            for path in member["images"]:
                stat = os.stat(path)
                key = f"{embedder.name}|{os.path.basename(path)}|{stat.st_size}|{int(stat.st_mtime)}"
                if key not in cache:
                    image = cv2.imread(path)
                    if image is None:
                        continue
                    cache[key] = embedder.embed(augment(image))
                    changed = True
                result[path] = cache[key]
        if changed:
            np.savez(cache_path, **cache)
        return result
