import os
from dataclasses import dataclass
from typing import List, Optional

from .fusion import fuse_spotted_events
from .labels import SOCCERNET_V2_CLASSES, load_labels
from .model import load_config, load_spotting_model
from .spotter import DEFAULT_SAMPLE_FPS, ActionSpotter

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_SPOTTING_WEIGHTS = os.path.join(MODELS_DIR, "checkpoint_088.pt")
DEFAULT_SPOTTING_CONFIG = os.path.join(MODELS_DIR, "config.json")
DEFAULT_SPOTTING_LABELS = os.path.join(MODELS_DIR, "spotting_classes.txt")
# Classes the pipeline consumes; the rest are not turned into events.
USED_CLASSES = ("Goal", "Shots on target", "Shots off target", "Kick-off")


@dataclass
class SpottingBundle:
    model: object
    labels: List[str]
    clip_len: int


def load_spotting_bundle(
    weights: str = DEFAULT_SPOTTING_WEIGHTS,
    config_path: str = DEFAULT_SPOTTING_CONFIG,
    labels_path: Optional[str] = DEFAULT_SPOTTING_LABELS,
) -> SpottingBundle:
    """Load the spotting model; raises if the weights or ``timm`` are missing."""
    if not os.path.isfile(weights):
        raise FileNotFoundError(weights)
    config = load_config(config_path)
    labels = load_labels(labels_path, int(config["num_classes"]))
    return SpottingBundle(load_spotting_model(weights, config), labels, int(config["clip_len"]))


__all__ = [
    "ActionSpotter",
    "DEFAULT_SAMPLE_FPS",
    "DEFAULT_SPOTTING_WEIGHTS",
    "SOCCERNET_V2_CLASSES",
    "SpottingBundle",
    "USED_CLASSES",
    "fuse_spotted_events",
    "load_config",
    "load_labels",
    "load_spotting_bundle",
    "load_spotting_model",
]
