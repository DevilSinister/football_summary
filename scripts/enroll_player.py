"""Enrol 1-10 reference images of a player for naming him in footage.

    .venv\\Scripts\\python.exe -m scripts.enroll_player --team "Real Madrid" --jersey 7 \\
        --name "Vinicius Junior" photo1.jpg photo2.jpg

Each photo is cut to its largest detected player (models/best.pt). The crops
are stored in data/player_gallery and used by every later analysis of a match
with that team. No database is needed; the name falls back to data/rosters.
"""

import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from identity import MAX_IMAGES_PER_PLAYER, PlayerGallery  # noqa: E402
from identity.enroll import crops_from_images  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+")
    parser.add_argument("--team", required=True)
    parser.add_argument("--jersey", type=int, required=True)
    parser.add_argument("--name")
    parser.add_argument("--detector", default=os.path.join(ROOT, "models", "best.pt"))
    parser.add_argument("--no-detect", action="store_true", help="the images are already cropped to the player")
    args = parser.parse_args()

    if len(args.images) > MAX_IMAGES_PER_PLAYER:
        parser.error(f"at most {MAX_IMAGES_PER_PLAYER} images per player")
    images = []
    for path in args.images:
        image = cv2.imread(path)
        if image is None:
            parser.error(f"cannot read {path}")
        images.append(image)
    detector = None
    if not args.no_detect:
        from ultralytics import YOLO

        detector = YOLO(args.detector)
    crops, rejected = crops_from_images(images, detector)
    for index in rejected:
        print(f"skipped {args.images[index]}: no player found")
    if not crops:
        return 1
    name = args.name
    if name is None:
        from scorer_identifier.goal_scorer import squad_for

        name = next((p["name"] for p in squad_for(args.team) if int(p["jersey_no"]) == args.jersey), None)
    saved = PlayerGallery().enroll(args.team, args.jersey, name, crops)
    print(f"{args.team} #{args.jersey} ({name or 'no name'}): {len(saved)} image(s) stored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
