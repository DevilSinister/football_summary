"""Score the pipeline against labelled clips (data/eval/clips.json).

Runs the same analysis an app job runs (PipelineConfig.short_clip) on every
labelled clip, keeps each result in data/eval/results/<clip>.json, and reports
per event type: how many real events were found (recall), how many reported
events were real (precision), how many found events had the right team, and
how many had the right player named.

    .venv\\Scripts\\python.exe -m scripts.eval_events                 # run clips without a saved result, then score
    .venv\\Scripts\\python.exe -m scripts.eval_events --rerun         # run every clip again
    .venv\\Scripts\\python.exe -m scripts.eval_events --clip levante_sevilla --rerun
    .venv\\Scripts\\python.exe -m scripts.eval_events --score-only    # score the saved results only
    .venv\\Scripts\\python.exe -m scripts.eval_events --all-on        # every event switch on, not features.yaml

Labels (data/eval/clips.json), one entry per clip:

    {"id": "levante_sevilla", "video": "~/Downloads/....mp4",
     "teams": [{"name": "Levante", "color": "#7A1F3D"}, {"name": "Sevilla", "color": "#F2F2F2"}],
     "complete": ["goal", "card"],        # types whose every occurrence is labelled
     "events": [{"type": "goal", "t": 16.0, "team": "Sevilla", "player": "Robbie Ure"}]}

Only types listed in ``complete`` are scored for a clip: a clip where nobody
checked every foul cannot say a reported foul is false. A label without a
player is scored for detection and team only.
"""

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = ROOT / "data" / "eval" / "clips.json"
RESULTS = ROOT / "data" / "eval" / "results"

# How far (seconds) a reported event may be from the labelled moment and still
# be the same event. Goals are reported at the shot or at the ball in the net,
# a few seconds apart; a card is shown well after its foul.
TOLERANCE = {"goal": 6.0, "foul": 5.0, "card": 12.0, "default": 5.0}
# Pipeline event types grouped into the labelled categories.
CATEGORY = {"yellow_card": "card", "red_card": "card"}
# The player each category names.
PLAYER_ROLE = {"goal": "scorer", "foul": "fouler", "card": "player", "shot": "shooter", "save": "goalkeeper"}


def _fold(text: Optional[str]) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return " ".join("".join(c if c.isalnum() else " " for c in text.lower()).split())


def same_player(predicted: Optional[str], label: Optional[str]) -> bool:
    """Same person: equal names, or the same surname with a matching first name
    (or with no first name on one side)."""
    p, l = _fold(predicted).split(), _fold(label).split()
    if not p or not l:
        return False
    if p == l:
        return True
    if p[-1] != l[-1]:
        return False
    return len(p) == 1 or len(l) == 1 or p[0] == l[0]


def _named(event: dict, category: str) -> Tuple[Optional[str], Optional[str]]:
    """(player name, team) the pipeline gives the event's main player."""
    role = PLAYER_ROLE.get(category)
    actor = next((a for a in event.get("actors") or [] if a.get("role") == role), None)
    if actor is None and category in ("goal", "card", "foul"):
        actor = event.get("scorer")
    actor = actor or {}
    return actor.get("player_name"), actor.get("team_name")


def _fps(video: str) -> float:
    import cv2

    capture = cv2.VideoCapture(video)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    capture.release()
    return fps


def video_path(clip: dict) -> str:
    raw = str(clip["video"])
    return str(ROOT / raw) if not os.path.isabs(os.path.expanduser(raw)) else os.path.expanduser(raw)


def run_clip(clip: dict, flags: Dict[str, bool]) -> dict:
    """The pipeline's result for one clip, reduced to what scoring needs."""
    from main import build_parser, process_video
    from pipeline_config import PipelineConfig

    video = video_path(clip)
    (team_a, team_b) = clip["teams"]
    work = tempfile.mkdtemp(prefix="footy_eval_")
    args = build_parser().parse_args([
        video, "--no-display",
        "--team-a-name", team_a["name"], "--team-a-color", team_a["color"],
        "--team-b-name", team_b["name"], "--team-b-color", team_b["color"],
        "--clip-dir", os.path.join(work, "clips"),
        "--player-image-dir", os.path.join(work, "players"),
    ])
    args.pipeline_config = PipelineConfig.short_clip(features=flags)
    args.return_pipeline_result = True
    args.ocr_debug_dir = None
    result = process_video(args)
    fps = _fps(video)
    events = []
    for event in result["events"]:
        events.append({
            "type": str(event.get("event", "")).lower(),
            "t": round(int(event.get("frame", 0)) / fps, 2),
            "team": event.get("team_name"),
            "source": event.get("source"),
            "confidence": event.get("confidence"),
            "description": event.get("description"),
            "scorer": {k: (event.get("scorer") or {}).get(k) for k in ("player_name", "jersey_number", "team_name")},
            "actors": [
                {k: a.get(k) for k in ("role", "player_name", "jersey_number", "team_name")}
                for a in event.get("actors") or []
            ],
        })
    return {"clip": clip["id"], "features": result.get("features", {}), "metrics": result.get("metrics", {}),
            "score_by_team_id": result.get("score_by_team_id"), "events": events}


def score_clip(clip: dict, result: dict) -> Dict[str, dict]:
    """Per category: found, reported, right team, right player, and a line per event."""
    by_category: Dict[str, dict] = {}
    predicted_all = result["events"]
    for category in clip.get("complete", []):
        tolerance = TOLERANCE.get(category, TOLERANCE["default"])
        labels = [e for e in clip.get("events", []) if e["type"] == category]
        predicted = [e for e in predicted_all if CATEGORY.get(e["type"], e["type"]) == category]
        unmatched = list(range(len(predicted)))
        stats = {"labels": len(labels), "predicted": len(predicted), "found": 0, "team_right": 0,
                 "with_player": 0, "player_right": 0, "player_named": 0, "details": []}
        for label in sorted(labels, key=lambda e: e["t"]):
            # Within the tolerance, a prediction of the labelled team first, then
            # the nearest: two goals 0.6 s apart must not swap their labels.
            close = [i for i in unmatched if abs(predicted[i]["t"] - label["t"]) <= tolerance]
            best = min(
                close,
                key=lambda i: (
                    bool(label.get("team")) and _fold(predicted[i].get("team")) != _fold(label.get("team")),
                    abs(predicted[i]["t"] - label["t"]),
                ),
                default=None,
            )
            if best is None:
                stats["details"].append(f"MISSED {category} at {label['t']}s ({label.get('player') or label.get('team')})")
                continue
            unmatched.remove(best)
            hit = predicted[best]
            stats["found"] += 1
            name, _ = _named(hit, category)
            team_ok = not label.get("team") or _fold(hit.get("team")) == _fold(label["team"])
            stats["team_right"] += int(team_ok)
            line = f"found {category} at {label['t']}s -> {hit['t']}s, team {'ok' if team_ok else 'WRONG ' + str(hit.get('team'))}"
            if label.get("player"):
                stats["with_player"] += 1
                stats["player_named"] += int(bool(name))
                ok = same_player(name, label["player"])
                stats["player_right"] += int(ok)
                line += f", player {'ok' if ok else 'WRONG'} ({name or 'unnamed'} vs {label['player']})"
            stats["details"].append(line)
        for i in unmatched:
            stats["details"].append(f"FALSE {category} at {predicted[i]['t']}s ({predicted[i].get('source')})")
        by_category[category] = stats
    return by_category


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100.0 * part / whole:.0f}%)" if whole else "n/a"


def totals(scores: Dict[str, Dict[str, dict]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for clip_scores in scores.values():
        for category, s in clip_scores.items():
            t = out.setdefault(category, {k: 0 for k in ("labels", "predicted", "found", "team_right",
                                                         "with_player", "player_right", "player_named")})
            for key in t:
                t[key] += s[key]
    return out


def report(clips: List[dict], scores: Dict[str, Dict[str, dict]]) -> str:
    lines = ["| Event | Found (recall) | Real (precision) | Right team | Right player | Named players right |",
             "|---|---|---|---|---|---|"]
    for category, t in sorted(totals(scores).items()):
        lines.append(
            f"| {category} | {_pct(t['found'], t['labels'])} | {_pct(t['found'], t['predicted'])} | "
            f"{_pct(t['team_right'], t['found'])} | {_pct(t['player_right'], t['with_player'])} | "
            f"{_pct(t['player_right'], t['player_named'])} |"
        )
    lines.append("")
    for clip in clips:
        for category, s in (scores.get(clip["id"]) or {}).items():
            for detail in s["details"]:
                lines.append(f"- {clip['id']}: {detail}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", action="append", help="only this clip id (repeatable)")
    parser.add_argument("--rerun", action="store_true", help="run the pipeline even if a result is saved")
    parser.add_argument("--score-only", action="store_true", help="score saved results, run nothing")
    parser.add_argument("--all-on", action="store_true", help="every event switch on (not features.yaml)")
    parser.add_argument("--switch", action="append", default=[], metavar="KEY=yes|no",
                        help="override one features.yaml switch for this run (repeatable)")
    parser.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args(argv)

    from feature_flags import FEATURES, load_features

    flags = {f.key: True for f in FEATURES} if args.all_on else load_features()
    for override in args.switch:
        key, _, value = override.partition("=")
        if key not in flags or value.lower() not in ("yes", "no"):
            parser.error(f"--switch {override!r}: expected a features.yaml key and yes/no")
        flags[key] = value.lower() == "yes"
    clips = json.loads(args.labels.read_text(encoding="utf-8"))["clips"]
    if args.clip:
        clips = [c for c in clips if c["id"] in args.clip]
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores = {}
    for clip in clips:
        path = RESULTS / f"{clip['id']}.json"
        if not args.score_only and (args.rerun or not path.is_file()):
            if not os.path.isfile(video_path(clip)):
                print(f"== {clip['id']}: video not found ({video_path(clip)}), skipped")
                continue
            print(f"== running {clip['id']}", flush=True)
            path.write_text(json.dumps(run_clip(clip, flags), indent=1, ensure_ascii=False), encoding="utf-8")
        if not path.is_file():
            print(f"== {clip['id']}: no saved result, skipped")
            continue
        scores[clip["id"]] = score_clip(clip, json.loads(path.read_text(encoding="utf-8")))
    print(report(clips, scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
