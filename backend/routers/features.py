"""Feature switches (features.yaml) for the phone app."""

from fastapi import APIRouter

from feature_flags import load_feature_details, visible_event_types

router = APIRouter(prefix="/api/features", tags=["features"])


@router.get("")
def list_features():
    """The switches as they are in features.yaml right now.

    The app reads ``visible_event_types`` to decide which event types it lists
    on the match summary and the highlights page; ``features`` carries every
    switch with its label, description and rebuild note. The file is read on
    every call, so an edit shows up without restarting the server.
    """
    details, warnings = load_feature_details()
    flags = {item["key"]: item["enabled"] for item in details}
    return {
        "features": details,
        "flags": flags,
        "visible_event_types": visible_event_types(flags),
        "warnings": warnings,
    }
