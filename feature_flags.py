"""Feature switches from features.yaml, for the pipeline and the phone app.

features.yaml is written for people who do not read code: one block per
feature with ``enabled: yes`` / ``enabled: no``. This module is the only reader.

* The API job profile (``PipelineConfig.short_clip``) reads the file at the
  start of every job, so a change applies to the next analysis without a
  server restart.
* ``GET /api/features`` (backend/routers/features.py) serves the same switches
  to the Flutter app, which shows or hides each event type from them.

A missing file, a missing feature, or a value that is not yes/no falls back to
the default in FEATURES below and logs a warning; a bad edit never stops a job.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FEATURES_FILE = Path(__file__).resolve().parent / "features.yaml"


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    default: bool
    # Event types ("event" values in the pipeline, events.event_type in the
    # database) this switch shows in the app. Empty for switches that only
    # change how the analysis works.
    event_types: Tuple[str, ...] = ()


# Order is the order the app lists them in. Defaults are the behaviour before
# features.yaml existed, so deleting the file changes nothing.
FEATURES: Tuple[Feature, ...] = (
    Feature("detect_goals", "Goals", True, ("goal",)),
    Feature("goal_scorer_name", "Goal scorer name", True),
    Feature("assister_name", "Assist name", True),
    Feature("detect_shots", "Shots on target", False, ("shot",)),
    Feature("detect_saves", "Saves", False, ("save",)),
    Feature("detect_fouls", "Fouls", True, ("foul",)),
    Feature("detect_cards", "Yellow and red cards", False, ("yellow_card", "red_card")),
    Feature("detect_penalties", "Penalties", False, ("penalty",)),
    Feature("detect_crosses", "Crosses", False, ("cross",)),
    Feature("detect_passes", "Passes", False, ("pass",)),
    Feature("passer_name", "Passer and receiver names", False),
    Feature("highlight_clips", "Highlight clips", True),
    Feature("commentary_names", "Names from commentary", True),
    Feature("read_scoreboard", "Score graphic reading", True),
    Feature("action_spotting", "Action-spotting model", True),
)
FEATURES_BY_KEY: Dict[str, Feature] = {feature.key: feature for feature in FEATURES}

_TRUE = {"yes", "y", "true", "on", "1"}
_FALSE = {"no", "n", "false", "off", "0"}


def _as_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def _read_file(path: Path) -> Tuple[dict, List[str]]:
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}, [f"{path.name} not found; every feature uses its default"]
    except Exception as error:  # a typo in the file must not stop the analysis
        return {}, [f"{path.name} could not be read ({error}); every feature uses its default"]
    if not isinstance(raw, dict):
        return {}, [f"{path.name} is not a list of features; every feature uses its default"]
    return raw, []


def load_feature_details(path: Optional[Path] = None) -> Tuple[List[dict], List[str]]:
    """Every feature with its switch, rebuild note and description, plus warnings."""
    path = Path(path) if path else FEATURES_FILE
    raw, warnings = _read_file(path)
    details = []
    for feature in FEATURES:
        entry = raw.get(feature.key)
        value = entry.get("enabled") if isinstance(entry, dict) else entry
        enabled = _as_bool(value) if value is not None else None
        if enabled is None:
            if raw:
                warnings.append(
                    f"{feature.key}: 'enabled' is {value!r}, not yes/no; using the default "
                    f"({'yes' if feature.default else 'no'})"
                )
            enabled = feature.default
        info = entry if isinstance(entry, dict) else {}
        details.append(
            {
                "key": feature.key,
                "label": feature.label,
                "enabled": enabled,
                "app_rebuild_needed": bool(_as_bool(info.get("app_rebuild_needed")) or False),
                "what_it_does": str(info.get("what_it_does") or ""),
                "accuracy": str(info.get("accuracy") or ""),
                "event_types": list(feature.event_types),
            }
        )
    for key in raw:
        if key not in FEATURES_BY_KEY:
            warnings.append(f"{key}: not a known feature, ignored (check the spelling)")
    return details, warnings


def load_features(path: Optional[Path] = None, log: bool = True) -> Dict[str, bool]:
    """``{feature key: on/off}`` for every known feature."""
    details, warnings = load_feature_details(path)
    if log:
        for warning in warnings:
            print(f"features.yaml: {warning}")
    return {item["key"]: item["enabled"] for item in details}


def visible_event_types(flags: Dict[str, bool]) -> List[str]:
    """Event types the switches turn on, in FEATURES order."""
    types: List[str] = []
    for feature in FEATURES:
        if flags.get(feature.key, feature.default):
            types.extend(feature.event_types)
    return types
