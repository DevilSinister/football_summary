"""Class names for the spotting model's outputs.

The checkpoint does not store its label order. Output 0 is background; outputs
1-17 follow E2E-Spot's data/soccernetv2/class.txt, which its
parse_soccernet.py writes as ``sorted()`` SoccerNet-v2 class names - so
alphabetical, not the SoccerNet dictionary order. Checked against frames on
2026-10-14 (scripts/eval_spotting.py): a saved shot reads "Shots on target"
0.85, and an open-play goal peaks on "Shots on target" then "Goal". The
dictionary order had read the saved shot as "Direct free-kick" and a
set-piece scene as "Goal". A ``models/spotting_classes.txt`` with one name per
line overrides this list.
"""

import os
from typing import List, Optional

SOCCERNET_V2_CLASSES = sorted(
    [
        "Penalty",
        "Kick-off",
        "Goal",
        "Substitution",
        "Offside",
        "Shots on target",
        "Shots off target",
        "Clearance",
        "Ball out of play",
        "Throw-in",
        "Foul",
        "Indirect free-kick",
        "Direct free-kick",
        "Corner",
        "Yellow card",
        "Red card",
        "Yellow->red card",
    ]
)

# Pipeline event names for the classes the rest of the code consumes.
EVENT_NAMES = {
    "Goal": "goal",
    "Kick-off": "kick_off",
    "Shots on target": "shot_on_target",
    "Shots off target": "shot_off_target",
    "Penalty": "penalty",
}


def load_labels(path: Optional[str] = None, num_classes: int = 17) -> List[str]:
    """Names for outputs 1..num_classes (output 0 is background)."""
    labels = list(SOCCERNET_V2_CLASSES)
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
    if len(labels) != num_classes:
        raise ValueError(f"Spotting labels: expected {num_classes} names, found {len(labels)}")
    return labels


def event_name(label: str) -> str:
    return EVENT_NAMES.get(label, label.lower().replace(" ", "_").replace("-", "_").replace(">", ""))
