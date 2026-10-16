import argparse
import os

import cv2
from time import perf_counter

from highlights import CLIP_EVENT_TYPES, export_highlights, open_mp4_writer
from match_config import MatchConfig
from player_ball_assigner import PlayerBallAssigner
from scorer_identifier import (
    JerseyReader,
    ScorerIdentifier,
    celebration_candidates,
    format_goal_event,
    fuse_scorer,
    live_candidate,
    read_caption_candidates,
    squad_for,
)
from scorer_identifier.goal_scorer import CELEBRATION_SECONDS, ScorerCandidate
from team_assigner import TeamAssigner, kit_color_bgr, safe_player_crop
from track_events.goal_merge import CLEAR_NAME_CONFIDENCE
from track_events import (
    TrackEventDetector,
    drop_replayed_goals,
    summarise_passes,
    withdraw_saves_of_goals,
)
from scene import SceneTimeline
from scene.cuts import find_boundaries, frame_signature
from scoreboard import (
    ScoreboardReader,
    attribute_goals,
    combined_expected,
    expected_codes_for,
    map_codes_to_teams,
    merge_board_goals,
    paddle_ocr_fn,
    parse_code_option,
)
from pipeline_config import PipelineConfig, PipelineMetrics, scheduled_frames


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# Trained on 2026-09-23: a single-class ball detector. Pass "none" to run
# without it.
DEFAULT_BALL_MODEL = os.path.join(MODELS_DIR, "ball_detector_best.pt")
# The pitch-keypoint model (models/Football_field.pt) is opt-in: it was trained
# with mosaic=1.0 and, measured 2026-10-07, places goal-end keypoints in the
# stands, so a wrong goal line would add false goals. Pass
# --pitch-model models/Football_field.pt to use it.
DEFAULT_PITCH_MODEL = "none"
PITCH_STRIDE_FRAMES = 5

# Frames are read and detected in chunks so the batching inside
# Tracker.detect_frames actually engages - it used to be handed a one-frame list
# every iteration, which pinned the batch size at 1.
FRAME_BATCH_SIZE = 16

# Frames sampled across the video to learn the two kit colours before the run.
CALIBRATION_MIN_FRAMES = 24
CALIBRATION_MAX_FRAMES = 60

# Extra shirt-number OCR calls each goal may spend on its celebration close-ups,
# on top of JerseyReader's whole-job budget.
GOAL_SCORER_OCR_ALLOWANCE = 48
# The commentary's scorer vote, times its own confidence (caption is 3.0).
WEIGHT_COMMENTARY = 3.5
# Skip the broadcast caption OCR only when two sources already agree this well.
CAPTION_SKIP_CONFIDENCE = 0.85
# The celebration window also runs this long past the score change on the
# graphic, which often comes after the replay.
CELEBRATION_AFTER_SCOREBOARD_SECONDS = 10.0

# Which participant of a track-derived event is "the" player for the summary.
PRIMARY_ROLE = {
    "pass": "passer",
    "cross": "passer",
    "shot": "shooter",
    "save": "goalkeeper",
    "penalty": "taker",
    "foul": "fouler",
    "yellow_card": "player",
    "red_card": "player",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Detect football events and identify goal scorers")
    parser.add_argument("video", help="input match video")
    parser.add_argument("--output", default="output_videos/output.mp4")
    parser.add_argument("--tracking-model", default="models/best.pt")
    parser.add_argument(
        "--event-model",
        default="models/final_model_80.pth",
        help="weights for --legacy-event-model",
    )
    parser.add_argument(
        "--legacy-event-model",
        action="store_true",
        help=(
            "also run the old LSTM event model (off by default: it reads a strip of a 64x64 "
            "thumbnail, not the features it was trained on, and its goal score is noise)"
        ),
    )
    parser.add_argument("--team-a-name", required=True)
    parser.add_argument("--team-a-color", required=True, help="primary kit color as #RRGGBB")
    parser.add_argument("--team-b-name", required=True)
    parser.add_argument("--team-b-color", required=True, help="primary kit color as #RRGGBB")
    parser.add_argument("--ocr-debug-dir", default="output_videos/ocr_debug")
    parser.add_argument("--player-image-dir", default="output_videos/player_images")
    parser.add_argument("--player-image-url-prefix")
    parser.add_argument("--clip-dir", help="directory for per-event highlight clips")
    parser.add_argument("--clip-url-prefix", help="URL prefix served for --clip-dir")
    parser.add_argument(
        "--no-highlight-reel",
        action="store_true",
        help="write the per-play clips only, not the joined reel.mp4",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="detector input size; defaults to 1280 for 1080p sources and 960 below that",
    )
    parser.add_argument(
        "--ball-model",
        default=DEFAULT_BALL_MODEL,
        help='dedicated ball detector weights; "none" uses the ball class of --tracking-model',
    )
    parser.add_argument(
        "--pitch-model",
        default=DEFAULT_PITCH_MODEL,
        help='pitch keypoint (pose) weights, e.g. models/Football_field.pt; default "none" (off)',
    )
    parser.add_argument(
        "--pitch-stride",
        type=int,
        default=PITCH_STRIDE_FRAMES,
        help="run the pitch model every N frames and hold the homography between",
    )
    parser.add_argument(
        "--team-a-code",
        help="scoreboard code(s) for team A, e.g. POR or POR,FCP; default from data/rosters",
    )
    parser.add_argument(
        "--team-b-code",
        help="scoreboard code(s) for team B, e.g. MCI; default from data/rosters",
    )
    parser.add_argument(
        "--no-scene-cuts",
        action="store_true",
        help="do not detect camera cuts and crossfades (not recommended for broadcast clips)",
    )
    parser.add_argument(
        "--no-commentary",
        action="store_true",
        help="do not transcribe the commentary for player names",
    )
    parser.add_argument("--whisper-model", help='speech-to-text model size (default "small")')
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="do not name players from enrolled reference images (data/player_gallery)",
    )
    parser.add_argument(
        "--no-spotting",
        action="store_true",
        help="do not run the action-spotting model (models/checkpoint_088.pt)",
    )
    parser.add_argument(
        "--no-scoreboard",
        action="store_true",
        help="do not read the broadcast score graphic for goals",
    )
    parser.add_argument(
        "--scoreboard-every",
        type=float,
        default=1.0,
        help="seconds between reads of the score graphic once it is found",
    )
    parser.add_argument(
        "--no-scorer-captions",
        action="store_true",
        help="do not read the broadcast's scorer caption after a goal (saves a few OCR passes per goal)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="skip shirt-number OCR (faster; players are reported by track id only)",
    )
    parser.add_argument(
        "--write-annotated-video",
        action="store_true",
        help="also render the full annotated match video (slow, and large on disk)",
    )
    parser.add_argument("--no-display", action="store_true", help="process without opening a GUI window")
    return parser


def _optional_model(path):
    """A weights path that exists, or None ("none", empty, or missing file)."""
    if not path or str(path).strip().lower() == "none":
        return None
    if not os.path.exists(path):
        print(f"Model not found, running without it: {path}")
        return None
    return str(path)


def _drop_model_goals_near_pitch_goals(events, fps, window_seconds=5.0):
    """A goal the pitch geometry saw supersedes the learned model's proposal."""
    pitch_goals = [
        int(event.get("frame", 0))
        for event in events
        if event.get("event") == "goal" and event.get("source") == "pitch"
    ]
    if not pitch_goals:
        return events
    window = window_seconds * max(float(fps), 1.0)
    return [
        event
        for event in events
        if not (
            event.get("event") == "goal"
            and event.get("source") != "pitch"
            and any(abs(int(event.get("frame", 0)) - frame) <= window for frame in pitch_goals)
        )
    ]


def _drop_model_fouls_near_track_fouls(events, fps, window_seconds=5.0):
    """The learned model's foul adds nothing where the tracks already saw one."""
    track_fouls = [
        int(event.get("frame", 0))
        for event in events
        if event.get("event") == "foul" and event.get("source") == "tracks"
    ]
    window = window_seconds * max(float(fps), 1.0)
    return [
        event
        for event in events
        if not (
            event.get("event") == "foul"
            and event.get("source") != "tracks"
            and any(abs(int(event.get("frame", 0)) - frame) <= window for frame in track_fouls)
        )
    ]


def _save_scorer_image(scorer_identifier, scorer, image_dir, url_prefix):
    if not scorer or not image_dir:
        return
    state = scorer_identifier.identity_store.get(int(scorer.get("track_id", -1)))
    if state is None or not state.recent_crops:
        return
    os.makedirs(image_dir, exist_ok=True)
    filename = f"track-{state.track_id}-frame-{state.recent_crops[0].frame_number}.jpg"
    path = os.path.join(image_dir, filename)
    if os.path.exists(path) or cv2.imwrite(path, state.recent_crops[0].image):
        scorer["player_image"] = (
            f"{url_prefix.rstrip('/')}/{filename}" if url_prefix else path
        )


def _calibrate_teams(tracker, team_assigner, video_path, total_frames, fps, frame_height):
    """Learn the two kits from frames spread over the whole video.

    Mirrors the API's team-detection stage so the command line behaves the same
    way: the configured hex colours name the clusters, they do not have to match
    a crop pixel for pixel.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return False
    try:
        duration = total_frames / max(fps, 1.0)
        sample_count = int(min(CALIBRATION_MAX_FRAMES, max(CALIBRATION_MIN_FRAMES, duration / 2.0)))
        last_frame = max(total_frames - 1, 0)
        step = max(1, last_frame // max(sample_count - 1, 1))
        frames = []
        for position in range(0, last_frame + 1, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
            if len(frames) >= sample_count:
                break
    finally:
        capture.release()
    if not frames:
        return False

    min_crop_height = max(28, int(frame_height * 0.06))
    colors = []
    for frame, boxes in zip(frames, tracker.detect_outfield_players(frames)):
        for bbox in boxes:
            if safe_player_crop(frame, bbox, min_height=min_crop_height, min_width=12) is None:
                continue
            color = kit_color_bgr(frame, bbox)
            if color is not None:
                colors.append(color)
    return team_assigner.calibrate_from_samples(colors)


def _assign_teams(team_assigner, frame, players):
    for track_id, player in players.items():
        try:
            assignment = team_assigner.get_player_assignment(frame, player["bbox"], int(track_id))
        except (ValueError, RuntimeError):
            assignment = {
                "team_id": 0,
                "team_name": "unknown_team",
                "confidence": 0.0,
            }
        player["team"] = assignment["team_id"]
        player["team_name"] = assignment["team_name"]
        player["team_confidence"] = assignment["confidence"]


def _team_squads(args, team_names):
    """{team id: [{jersey_no, name, position}]}.

    The API passes the squads it read from the players table (``args.squads``,
    keyed by team id); the command line reads data/rosters, which
    scripts.import_squads keeps in step with the database.
    """
    given = getattr(args, "squads", None) or {}
    return {team_id: list(given.get(team_id) or squad_for(name)) for team_id, name in team_names.items()}


def _first_choice_keeper(squad):
    """The squad's number-one keeper: shirt 1 if a keeper wears it, else the lowest-numbered keeper."""
    keepers = [p for p in squad or [] if str(p.get("position") or "").upper() == "GK" and p.get("jersey_no") is not None]
    if not keepers:
        return None
    return next((k for k in keepers if int(k["jersey_no"]) == 1), min(keepers, key=lambda k: int(k["jersey_no"])))


ASSIST_LOOKBACK_SECONDS = 15.0


def _attach_pass_assisters(events, fps):
    """Add the assister to each goal: the last pass a team-mate made to the scorer.

    Uses the pass events the detector recorded (kept internally even when the
    output is goals only). Commentary naming the assister later overrides it.
    """
    fps = max(float(fps), 1.0)
    passes = [e for e in events if e.get("event") in ("pass", "cross")]
    for goal in (e for e in events if e.get("event") == "goal"):
        participants = goal.setdefault("participants", [])
        if any(p.get("role") == "assister" for p in participants):
            continue
        scorer = next((p for p in participants if p.get("role") == "scorer"), None)
        if scorer is None or scorer.get("track_id") is None:
            continue
        frame = int(goal.get("frame", 0))
        for p in sorted(passes, key=lambda e: -int(e.get("frame", 0))):
            if not frame - ASSIST_LOOKBACK_SECONDS * fps <= int(p.get("frame", 0)) <= frame:
                continue
            people = {q.get("role"): q for q in p.get("participants") or []}
            receiver, passer = people.get("receiver"), people.get("passer")
            if receiver is None or passer is None or receiver.get("track_id") != scorer.get("track_id"):
                continue
            if passer.get("track_id") == scorer.get("track_id"):
                continue
            if scorer.get("team_id") and passer.get("team_id") and passer["team_id"] != scorer["team_id"]:
                continue
            participants.append({**passer, "role": "assister"})
            goal.setdefault("details", {})["assist_from"] = "detected pass"
            break


def _role_record(role, found, team_names):
    return {
        "role": role,
        "team_id": found["team_id"],
        "team_name": team_names.get(found["team_id"]),
        "jersey_number": found["jersey_number"],
        "player_name": found["player_name"],
        "jersey_confidence": found["confidence"],
        "name_confidence": found["confidence"],
        "identity_source": "commentary",
        "said": found.get("said"),
        "said_at_seconds": found.get("said_at_seconds"),
    }


def _merge_role(event, role, found, team_names, identity_tracker=None):
    """Commentary is the primary source: it names the actor in this role.

    The tracked actor in the role (if any) keeps its track and photo when the
    commentary agrees with its shirt number; otherwise the commentary wins
    and the photo, which shows someone else, is dropped. The named player is
    also attached to that track as identity evidence.
    """
    actors = event.setdefault("actors", [])
    actor = next((a for a in actors if a.get("role") == role), None)
    record = _role_record(role, found, team_names)
    if actor is None:
        actors.append(record)
        return record
    if actor.get("track_id") is not None and identity_tracker is not None:
        identity_tracker.add_name_evidence(int(actor["track_id"]), found["team_id"], found["jersey_number"], 0.9)
    if actor.get("jersey_number") not in (None, found["jersey_number"]):
        actor.pop("player_image", None)
    actor.update({k: v for k, v in record.items() if k != "role"})
    return actor


def _team_from_scoreboard(goal):
    """True when the score graphic decided the goal's team: names do not overrule it."""
    return goal.get("source") == "scoreboard" and bool(goal.get("team_id"))


def _correct_goal_team(event, team_id, team_names, evidence):
    """Credit the goal to ``team_id``: a clear name said the kit colours were wrong."""
    old = event.get("team_name") or "unknown_team"
    event["team_id"], event["team_name"] = int(team_id), team_names.get(int(team_id))
    event.setdefault("details", {})["team_corrected"] = {"from": old, "by": evidence}
    for actor in event.get("actors") or []:
        if actor.get("role") in ("scorer", "assister"):
            actor["team_id"], actor["team_name"] = event["team_id"], event["team_name"]


def _commentary_goal_roles(events, words, mentions, fps):
    """{id(goal event): roles} - scorer, assister, ruled-out phrase from the commentary.

    Each goal's window stops at its neighbours, so a name said in the next
    attack is not credited to this goal.
    """
    from commentary import goal_roles

    fps = max(float(fps), 1.0)
    goals = sorted((e for e in events if str(e.get("event", "")).lower() == "goal"), key=lambda e: int(e.get("frame", 0)))
    times = [int(g.get("frame", 0)) / fps for g in goals]
    roles = {}
    for index, goal in enumerate(goals):
        roles[id(goal)] = goal_roles(
            times[index],
            int(goal.get("team_id") or 0),
            mentions,
            words,
            next_goal_seconds=times[index + 1] if index + 1 < len(times) else None,
            previous_goal_seconds=times[index - 1] if index else None,
            other_team=not _team_from_scoreboard(goal),
        )
    return roles


def _apply_commentary(events, words, mentions, fps, team_names, identity_tracker=None, goal_roles_by_event=None):
    """Assister of every goal, fouler and fouled of every foul, from the commentary.

    The scorer is decided earlier, in _improve_goal_scorers, where the
    commentary is the heaviest of the votes; here it only fills a scorer the
    fusion left unnamed. Measured on Levante 3-3 Sevilla (2026-10-16): the
    commentary alone named 4 of 6 scorers right and 1 wrong (the transcript
    turned "Ndukwe" into "Lufel"), while the on-screen caption named all six -
    so a commentary guess must not overrule a caption.
    """
    from commentary import foul_roles

    fps = max(float(fps), 1.0)
    goal_roles_by_event = goal_roles_by_event or _commentary_goal_roles(events, words, mentions, fps)
    for event in events:
        kind = str(event.get("event", "")).lower()
        seconds = int(event.get("frame", 0)) / fps
        details = event.setdefault("details", {})
        if kind == "goal":
            roles = goal_roles_by_event.get(id(event)) or {}
            if roles.get("ruled_out"):
                details["commentary"] = f"ruled out? commentator: '{roles['ruled_out']}'"
            scorer = event.get("scorer") or {}
            found = roles.get("scorer")
            other_team = found is not None and event.get("team_id") and found["team_id"] != event["team_id"]
            if other_team and (
                _team_from_scoreboard(event) or float(found["confidence"]) < CLEAR_NAME_CONFIDENCE
            ):
                found = None  # not clear enough to say the kit colours were wrong
            caption_led = "caption" in (scorer.get("scorer_sources") or [])
            if found is not None and not caption_led and (
                scorer.get("jersey_number") is None or scorer.get("unconfirmed")
            ):
                if not event.get("team_id"):
                    event["team_id"], event["team_name"] = found["team_id"], team_names.get(found["team_id"])
                elif other_team:
                    _correct_goal_team(event, found["team_id"], team_names, ["commentary"])
                merged = _merge_role(event, "scorer", found, team_names, identity_tracker)
                previous = {k: v for k, v in scorer.items() if k != "player_image"}
                event["scorer"] = {
                    **previous, **merged, "role": "scorer", "unconfirmed": False,
                    "scorer_confidence": found["confidence"], "scorer_sources": ["commentary"],
                }
                details["scorer_from"] = "commentary"
            elif found is not None and scorer.get("jersey_number") == found["jersey_number"]:
                # The fusion agrees with the commentary: tie the name to the track.
                if scorer.get("track_id") is not None and identity_tracker is not None:
                    identity_tracker.add_name_evidence(
                        int(scorer["track_id"]), found["team_id"], found["jersey_number"], 0.9
                    )
            assister = roles.get("assister")
            if assister is not None and int(assister["team_id"]) in (0, int(event.get("team_id") or 0)):
                # A team-mate of the scorer: not one of the side a caption showed conceded.
                _merge_role(event, "assister", assister, team_names, identity_tracker)
                details["assister_from"] = "commentary"
        elif kind == "foul":
            roles = foul_roles(seconds, mentions)
            for role in ("fouler", "fouled"):
                if roles[role] is not None:
                    merged = _merge_role(event, role, roles[role], team_names, identity_tracker)
                    if role == "fouler":
                        event["scorer"] = {**(event.get("scorer") or {}), **merged}
                    details[f"{role}_from"] = "commentary"


def _name_keepers(events, team_squads, team_names):
    """A save (or any keeper actor) with no read shirt: the team's number-one keeper.

    Shots are usually saved by the starting keeper, so the roster's keeper is a
    good default; a shirt read or a commentary mention overrides it.
    """
    for event in events:
        for actor in event.get("actors") or []:
            if actor.get("role") != "goalkeeper" or actor.get("jersey_number") is not None:
                continue
            keeper = _first_choice_keeper(team_squads.get(int(actor.get("team_id") or 0)))
            if keeper is None:
                continue
            actor.update({
                "jersey_number": int(keeper["jersey_no"]),
                "player_name": keeper.get("name"),
                "jersey_confidence": 0.6,
                "name_confidence": 0.6,
                "identity_source": "roster keeper",
                "team_name": actor.get("team_name") or team_names.get(int(actor.get("team_id") or 0)),
            })
            if (event.get("scorer") or {}).get("role") == "goalkeeper" or event.get("event") == "save":
                event["scorer"] = {**(event.get("scorer") or {}), **actor}


def _transcribe_commentary(args, config, team_squads):
    """(words, player mentions, language) from the video's commentary; empty without speech."""
    from commentary import find_mentions, load_whisper, transcribe_video

    model = getattr(args, "whisper_model_instance", None)
    try:
        if model is None:
            model = load_whisper(getattr(args, "whisper_model", None) or config.whisper_model)
        words, language = transcribe_video(args.video, model)
    except Exception as error:  # optional source: no model, no ffmpeg, bad audio
        print(f"Commentary unavailable, running without it: {error}")
        return [], [], None
    mentions = find_mentions(words, team_squads)
    named = sorted({c.name for m in mentions for c in m.candidates[:1]})
    print(
        f"Commentary: {len(words)} words ({language or 'no speech'}); "
        f"{len(mentions)} player mention(s)" + (f": {', '.join(named[:12])}" if named else "")
    )
    return words, mentions, language


def _build_identity_tracker(args, team_names, fps, squads=None):
    """Per-track name probabilities from commentary, shirt reads, reference images and the roster.

    None when neither team has a roster or any enrolled player. Appearance is
    used only for teams with reference images and when the re-ID weights load;
    otherwise commentary mentions, shirt-number reads and the roster drive it.
    """
    if getattr(args, "no_reid", False):
        return None
    from identity import DEFAULT_REID_WEIGHTS, GALLERY_DIR, IdentityTracker, PlayerGallery, ReidEmbedder

    gallery = PlayerGallery(getattr(args, "player_gallery_dir", None) or GALLERY_DIR)
    enrolled = {team_id: gallery.members(name) for team_id, name in team_names.items()}
    squads = squads or {team_id: squad_for(name) for team_id, name in team_names.items()}
    if not any(enrolled.values()) and not any(squads.values()):
        return None
    embedder = getattr(args, "reid_embedder", None)
    if embedder is None and any(enrolled.values()) and os.path.isfile(DEFAULT_REID_WEIGHTS):
        try:
            embedder = ReidEmbedder.load()
        except Exception as error:  # optional: a truncated or LFS-pointer file must not fail the run
            print(f"Re-identification weights unavailable: {error}")
    galleries = {
        team_id: gallery.team_prototypes(team_names[team_id], embedder) if embedder and enrolled[team_id] else None
        for team_id in team_names
    }
    print(
        "Player identity: "
        + ", ".join(f"{team_names[t]} {len(enrolled[t])} enrolled / {len(squads[t])} in roster" for t in team_names)
        + f"; appearance {'on' if any(galleries.values()) else 'off'}"
    )
    return IdentityTracker(squads, galleries, embedder, fps)


def _apply_identity(identity_tracker, actors):
    """Feed shirt-number reads in; name actors the OCR could not read."""
    track_ids = [int(a["track_id"]) for a in actors if a.get("track_id") is not None]
    for actor in actors:
        if actor.get("jersey_number") is not None and actor.get("track_id") is not None:
            identity_tracker.add_number_read(
                int(actor["track_id"]), int(actor["jersey_number"]), float(actor.get("jersey_confidence") or 0.5)
            )
    for actor in actors:
        if actor.get("jersey_number") is not None or actor.get("track_id") is None:
            continue
        track_id = int(actor["track_id"])
        named = identity_tracker.name_for(track_id, concurrent=[t for t in track_ids if t != track_id])
        if named is None:
            continue
        actor["jersey_number"] = named["jersey_number"]
        actor["player_name"] = named["player_name"]
        actor["jersey_confidence"] = named["probability"]
        actor["name_confidence"] = named["probability"]
        # The combined per-track belief: commentary, shirt reads, appearance.
        actor["identity_source"] = "identity"


def _record_ball_possession(scorer_identifier, players, frame_number):
    """Feed the scorer history from the possession the event detector settled."""
    for track_id, player in players.items():
        if player.get("has_ball"):
            scorer_identifier.record_possession(frame_number, int(track_id))
            return


def _draw_team_labels(frame, players):
    for track_id, player in players.items():
        x1, y1, x2, _ = map(int, player["bbox"])
        team_id = int(player.get("team", 0))
        color = (0, 200, 0) if team_id == 1 else (200, 80, 0) if team_id == 2 else (128, 128, 128)
        role = " GK" if player.get("role") == "goalkeeper" else ""
        label = f"{player.get('team_name', 'unknown_team')}{role} | ID {track_id}"
        cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x2, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, max(12, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def _frame_tracks(tracks, offset):
    """A single-frame view of a batched tracks dict.

    EventLogic inspects tracks[...][-1], so it must see only the frame being
    judged. The inner dicts are shared rather than copied, so mutations made
    downstream still land on the batch.
    """
    return {key: [value[offset]] for key, value in tracks.items()}


def _resolve_identities(scorer_identifier, event, image_dir, url_prefix, identity_tracker=None):
    """Attach team and shirt number to every participant of an event.

    ``scorer`` keeps its historical name: it is the primary actor (scorer,
    passer, shooter, goalkeeper for a save, booked player) and is what the
    persistence layer and the app already read. ``actors`` carries everyone.
    """
    participants = event.get("participants") or []
    if event.get("source") == "scoreboard" and not participants:
        # The scoreboard proved the goal, but no shot or holder was found in
        # the window. Report the team; do not guess a player.
        event["scorer"] = {
            "team_id": event.get("team_id", 0),
            "team_name": event.get("team_name"),
            "jersey_number": None,
            "player_name": None,
        }
        return
    if participants:
        actors = scorer_identifier.identify_participants(participants)
        if identity_tracker is not None:
            _apply_identity(identity_tracker, actors)
        event["actors"] = actors
        primary_role = PRIMARY_ROLE.get(str(event.get("event", "")).lower())
        primary = next((actor for actor in actors if actor.get("role") == primary_role), None)
        if primary is None:
            primary = actors[0]
        for actor in actors:
            _save_scorer_image(scorer_identifier, actor, image_dir, url_prefix)
        event["scorer"] = dict(primary)
        if event.get("source") == "scoreboard":
            # The scoreboard names the scoring team; a kit-colour reading
            # of the shooter does not overrule it.
            for record in [event["scorer"], *[a for a in actors if a.get("role") == "scorer"]]:
                record["team_id"] = event.get("team_id", 0)
                record["team_name"] = event.get("team_name")
        if event.get("team_id", 0) == 0 and primary.get("team_id", 0):
            event["team_id"] = primary["team_id"]
            event["team_name"] = primary["team_name"]
    elif str(event.get("event", "")).lower() != "goal":
        # A spotted shot has no tracked shooter: report it without a player
        # rather than guess (and spend shirt-number OCR on) the last holder.
        event["scorer"] = {
            "team_id": event.get("team_id", 0),
            "team_name": event.get("team_name"),
            "jersey_number": None,
            "player_name": None,
        }
    else:
        scorer = scorer_identifier.identify_goal(int(event.get("frame", 0)))
        team_id = int(event.get("team_id") or 0)
        if team_id and int(scorer.get("team_id") or 0) not in (0, team_id):
            # The last player in possession plays for the other side (a
            # defender, a keeper): the goal's team stands, no player is named.
            scorer = {
                "team_id": team_id,
                "team_name": event.get("team_name"),
                "jersey_number": None,
                "player_name": None,
            }
        event["scorer"] = scorer
        _save_scorer_image(scorer_identifier, scorer, image_dir, url_prefix)


def _roster_name(team_name, jersey_number):
    """Player name from data/rosters for display, or None.

    The API resolves names from the players table instead; this is only for
    the command-line summary, which runs without SQL Server.
    """
    if not team_name or jersey_number is None:
        return None
    return next(
        (p["name"] for p in squad_for(team_name) if int(p["jersey_no"]) == int(jersey_number)),
        None,
    )


def _back_caption_with_shared_name(candidates, team_id):
    """A spoken surname several players share ("Romero") supports the one the caption names.

    The commentary resolves a shared surname by the goal's team, which comes
    from kit colours; on Levante-Sevilla that turned "Romero" into Sevilla's
    Isaac while the caption showed Levante's Iván.
    """
    captioned = {
        (int(c.team_id or team_id), int(c.jersey_number)): c
        for c in candidates if c.source == "caption" and c.jersey_number is not None
    }
    for candidate in candidates:
        if candidate.source != "commentary":
            continue
        for team, number in candidate.detail.get("shared_with") or []:
            caption = captioned.get((int(team), int(number)))
            if caption is not None:
                candidate.detail["moved_from"] = [candidate.team_id or team_id, candidate.jersey_number]
                candidate.team_id, candidate.jersey_number = int(team), int(number)
                candidate.player_name = caption.player_name
                break


def _improve_goal_scorers(
    events,
    scorer_identifier,
    video_path,
    fps,
    frame_height,
    ocr_fn=None,
    extra_ocr_allowance=GOAL_SCORER_OCR_ALLOWANCE,
    team_squads=None,
    commentary_roles=None,
    team_names=None,
):
    """Re-decide every goal's scorer from all the evidence, not one shirt read.

    Sources and weights are in scorer_identifier/goal_scorer.py: the broadcast
    caption naming the scorer (``ocr_fn`` None skips it), the shirt numbers of
    the scoring team's players in the celebration close-ups, and the live read
    of the shooter. The result replaces ``scorer`` (and the scorer actor) with
    the fused shirt number, a confidence, and a name only when confirmed.

    Captions and the commentary are matched against both squads: the goal's
    team comes from kit colours, which are often wrong, and a clear name from
    the other squad corrects it (``details.team_corrected``). A team the score
    graphic decided is not corrected; its goal searches its own squad only.
    """
    fps = max(float(fps), 1.0)
    team_names = team_names or {}
    for event in events:
        if str(event.get("event", "")).lower() != "goal":
            continue
        details = event.setdefault("details", {})
        team_id = int(event.get("team_id", 0) or 0)
        squad = (team_squads or {}).get(team_id) or squad_for(event.get("team_name"))
        locked = _team_from_scoreboard(event)
        squads = {
            int(t): members for t, members in (team_squads or {}).items()
            if members and (not locked or int(t) == team_id)
        }
        scorer = dict(event.get("scorer") or {})
        moment = int(event.get("frame", 0) or 0)
        board_frame = int(details.get("scoreboard_frame") or moment)

        candidates = live_candidate(scorer, str(details.get("attribution") or ""))
        scorer_identifier.jersey_reader.max_ocr_calls += max(0, int(extra_ocr_allowance))
        celebration_end = int(
            max(moment + CELEBRATION_SECONDS * fps, board_frame + CELEBRATION_AFTER_SCOREBOARD_SECONDS * fps)
        )
        candidates += celebration_candidates(
            scorer_identifier.identity_store,
            scorer_identifier.identify_track,
            team_id,
            moment,
            celebration_end,
            frame_height,
        )
        spoken = ((commentary_roles or {}).get(id(event)) or {}).get("scorer")
        if spoken is not None:
            # The commentary is the heaviest single vote (caption 3.0).
            candidates.append(ScorerCandidate(
                int(spoken["jersey_number"]), WEIGHT_COMMENTARY * float(spoken["confidence"]),
                "commentary", spoken["player_name"],
                detail={"said": spoken.get("said"), "shared_with": spoken.get("shared_with") or []},
                team_id=int(spoken.get("team_id") or 0) or None,
            ))
        preliminary = fuse_scorer(candidates, squad, squads, team_id)
        # Captions are read unless the evidence is already decisive: a
        # commentary guess alone is not (it named 1 of 6 wrong on
        # Levante-Sevilla while the caption named all six right).
        decisive = (
            preliminary is not None
            and not preliminary.get("unconfirmed")
            and preliminary["scorer_confidence"] >= CAPTION_SKIP_CONFIDENCE
            and len(preliminary.get("scorer_sources") or []) >= 2
        )
        if ocr_fn is not None and not decisive:
            candidates += read_caption_candidates(
                video_path, moment, fps, ocr_fn, squad, squads=squads, team_id=team_id
            )
        _back_caption_with_shared_name(candidates, team_id)
        details["scorer_evidence"] = [
            {
                "source": c.source,
                "team_id": c.team_id or team_id,
                "jersey_number": c.jersey_number,
                "player_name": c.player_name,
                "weight": round(c.weight, 3),
            }
            for c in candidates
        ]
        fused = fuse_scorer(candidates, squad, squads, team_id)
        if fused is None:
            continue
        fused_team = int(fused.pop("team_id") or 0)
        if team_id and fused_team != team_id:
            name_sources = sorted(set(fused["scorer_sources"]) & {"caption", "commentary"})
            if fused["unconfirmed"] or not name_sources:
                # The other squad's player is not named clearly enough to
                # move the goal: decide among the goal team's evidence only.
                own = [c for c in candidates if not c.team_id or int(c.team_id) == team_id]
                fused = fuse_scorer(own, squad, squads, team_id)
                if fused is None:
                    continue
                fused.pop("team_id", None)
            else:
                _correct_goal_team(event, fused_team, team_names, name_sources)
                team_id = fused_team
        elif not team_id and fused_team and not fused["unconfirmed"]:
            # No team was read from the kits; the named scorer gives it.
            event["team_id"], event["team_name"] = fused_team, team_names.get(fused_team)
            team_id = fused_team

        changed = fused["jersey_number"] != scorer.get("jersey_number")
        for record in [scorer, *[a for a in event.get("actors") or [] if a.get("role") == "scorer"]]:
            record.update(fused)
            if details.get("team_corrected") or not record.get("team_id"):
                record["team_id"], record["team_name"] = team_id, event.get("team_name")
            record.setdefault("team_name", event.get("team_name"))
            if changed:
                # The photo is of the tracked shooter, who is not the fused scorer.
                record.pop("player_image", None)
                record["jersey_confidence"] = fused["scorer_confidence"]
        event["scorer"] = scorer
        actors = event.setdefault("actors", [])
        if not any(a.get("role") == "scorer" for a in actors):
            # A goal no player was tracked for (spotted, or from the graphic):
            # the API stores one row per actor, so the named scorer needs one,
            # first, ahead of any assister.
            actors.insert(0, {**scorer, "role": "scorer"})


def _actor_label(actor):
    if not actor:
        return "unknown player"
    number = actor.get("jersey_number")
    label = f"#{number}" if number is not None else f"track {actor.get('track_id')}"
    if actor.get("player_name"):
        label += f" {actor['player_name']}"
    return label


def describe_event(event):
    """One line for the console summary and the stored description."""
    event_type = str(event.get("event", "event")).lower()
    if event_type == "goal":
        scorer = dict(event.get("scorer") or {})
        if not scorer.get("player_name") and not scorer.get("unconfirmed"):
            scorer["player_name"] = _roster_name(scorer.get("team_name"), scorer.get("jersey_number"))
        text = format_goal_event({**event, "scorer": scorer})
        score_after = (event.get("details") or {}).get("score_after")
        return f"{text} ({score_after})" if score_after else text
    team = event.get("team_name") or (event.get("scorer") or {}).get("team_name") or "unknown_team"
    actors = {actor.get("role"): actor for actor in event.get("actors") or []}
    if event_type in ("pass", "cross"):
        return (
            f"{event_type.upper()} — {team} — {_actor_label(actors.get('passer'))} "
            f"→ {_actor_label(actors.get('receiver'))}"
        )
    if event_type == "shot":
        outcome = (event.get("details") or {}).get("outcome", "unknown")
        return f"SHOT — {team} — {_actor_label(actors.get('shooter'))} ({outcome})"
    if event_type == "save":
        shooter = actors.get("shooter") or {}
        return (
            f"SAVE — {team} — {_actor_label(actors.get('goalkeeper'))} "
            f"denied {shooter.get('team_name') or 'unknown_team'} {_actor_label(shooter)}"
        )
    if event_type == "penalty":
        return f"PENALTY — {team} — taken by {_actor_label(actors.get('taker'))}"
    if event_type in ("yellow_card", "red_card"):
        colour = event_type.split("_")[0].upper()
        return f"{colour} CARD — {team} — {_actor_label(actors.get('player'))}"
    if event_type == "foul" and actors.get("fouler"):
        text = f"FOUL — {team} — {_actor_label(actors.get('fouler'))}"
        if actors.get("fouled"):
            text += f" on {_actor_label(actors.get('fouled'))}"
        return text
    return f"{event_type.upper()} — {team}"


def final_score_by_team(events, scoreboard_reader=None, code_to_team=None):
    """Latest confirmed graphic score, otherwise deduplicated detected goals."""
    score = {1: 0, 2: 0}
    if (
        scoreboard_reader is not None
        and scoreboard_reader.active
        and scoreboard_reader.tracker.state
        and code_to_team
    ):
        mapped = {
            int(code_to_team.get(code, 0)): int(value)
            for code, value in scoreboard_reader.tracker.state.items()
            if int(code_to_team.get(code, 0)) in score
        }
        if len(mapped) == 2:
            score.update(mapped)
            return score
    for event in events:
        if str(event.get("event", "")).lower() != "goal":
            continue
        scorer = event.get("scorer") or {}
        team_id = int(event.get("team_id") or scorer.get("team_id") or 0)
        if team_id in score:
            score[team_id] += 1
    return score


def goal_output_events(events, kinds=("goal",)):
    return [event for event in events if str(event.get("event", "")).lower() in kinds]


def should_calibrate_teams(args) -> bool:
    return not bool(getattr(args, "confirmed_team_centers", False))


def process_video(args, progress_cb=None):
    # Heavy model imports are delayed so configuration validation and --help work
    # without initializing Ultralytics/Torch or their user-level cache folders.
    from trackers import Tracker

    config = getattr(args, "pipeline_config", None) or PipelineConfig()
    metrics = PipelineMetrics()

    try:
        import torch

        # Match the physical core count; oversubscribing hurts on a 4C/8T laptop.
        torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    except ImportError:
        pass

    match_config = MatchConfig.from_user_input(
        args.team_a_name,
        args.team_a_color,
        args.team_b_name,
        args.team_b_color,
    )

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video file: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / max(fps, 1.0)
    # The full profile asks for "every frame" with a huge rate; the tracker
    # sizes its lost-track buffer from this, so it must be a real rate.
    target_fps = min(fps, config.analysis_fps(duration))
    detector_size = config.detector_size(height)

    write_video = bool(getattr(args, "write_annotated_video", False))
    show_video = not args.no_display
    writer = None
    if write_video:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        writer = open_mp4_writer(args.output, fps, (width, height))
        if writer is None:
            capture.release()
            raise RuntimeError(f"Cannot open output video: {args.output}")

    ball_model = _optional_model(getattr(args, "ball_model", DEFAULT_BALL_MODEL))
    pitch_model = (
        _optional_model(getattr(args, "pitch_model", DEFAULT_PITCH_MODEL))
        if config.use_pitch_calibration
        else None
    )
    tracker = Tracker(
        args.tracking_model,
        frame_rate=target_fps,
        image_size=getattr(args, "image_size", None) or detector_size,
        ball_model_path=ball_model,
        ball_image_size=detector_size,
        model=getattr(args, "person_model", None),
        ball_model=getattr(args, "ball_model_instance", None),
    )
    pitch_calibrator = None
    if pitch_model:
        from pitch import PitchCalibrator

        pitch_calibrator = PitchCalibrator(
            pitch_model, stride=int(getattr(args, "pitch_stride", PITCH_STRIDE_FRAMES) or PITCH_STRIDE_FRAMES)
        )
    print(f"Ball source: {ball_model or 'ball class of ' + str(args.tracking_model)}")
    print(f"Pitch model: {pitch_model or 'none'}")
    event_detector = None
    event_logic = None
    if config.use_legacy_event_model or getattr(args, "legacy_event_model", False):
        from event_detector import EventDetector
        from event_logic import EventLogic

        event_detector = EventDetector(getattr(args, "event_model", None) or "models/final_model_80.pth")
        event_logic = EventLogic(fps=fps)
    team_assigner = TeamAssigner(match_config)
    ball_assigner = PlayerBallAssigner()
    track_events = TrackEventDetector(
        fps,
        (width, height),
        ball_assigner=ball_assigner,
        detect_cards=config.detect_cards and bool(getattr(args, "detect_cards", True)),
        detect_fouls=config.detect_fouls,
        # Passes are always followed (cheap, rule-based): the last team-mate
        # pass to the scorer is the goal's assist. Goal-only output hides them.
        detect_passes=True,
        detect_penalties=not config.goal_only,
    )
    if getattr(args, "no_ocr", False):
        jersey_reader = JerseyReader(debug_dir=None, max_ocr_calls=0)
    else:
        jersey_reader = JerseyReader(
            debug_dir=(getattr(args, "ocr_debug_dir", None) if config.debug_ocr else None),
            engine=getattr(args, "ocr_engine", None),
            max_crops=config.identity_crops,
            max_ocr_calls=config.jersey_ocr_calls,
        )
    from scorer_identifier.identity import PlayerIdentityStore, min_readable_box_height

    scorer_identifier = ScorerIdentifier(
        jersey_reader=jersey_reader,
        identity_store=PlayerIdentityStore(
            max_crops_per_player=config.identity_crops,
            crop_stride=1,
            min_box_height=min_readable_box_height(height),
        ),
        frame_height=height,
    )

    # Goals from the broadcast score graphic. Codes come from the caller (the
    # API reads them from the database), else from data/rosters.
    team_names = {1: match_config.team_a.name, 2: match_config.team_b.name}
    known_codes = {
        1: parse_code_option(getattr(args, "team_a_code", None)) or expected_codes_for(team_names[1]),
        2: parse_code_option(getattr(args, "team_b_code", None)) or expected_codes_for(team_names[2]),
    }
    scoreboard_reader = None
    if not getattr(args, "no_scoreboard", False):
        if jersey_reader._ensure_engine():
            scoreboard_reader = ScoreboardReader(
                paddle_ocr_fn(jersey_reader.engine),
                fps,
                (height, width),
                expected_codes=combined_expected(known_codes),
                read_every_seconds=config.scoreboard_scan_seconds,
                scan_every_seconds=config.scoreboard_scan_seconds,
                burst_every_seconds=config.scoreboard_burst_seconds,
                burst_duration_seconds=config.scoreboard_burst_duration,
                permissive_after_misses=config.scoreboard_permissive_after_misses,
            )
        else:
            print(f"Scoreboard reading unavailable: {jersey_reader._engine_error}")

    spotter = None
    if config.use_spotting and not getattr(args, "no_spotting", False):
        bundle = getattr(args, "spotting_bundle", None)
        if bundle is None:
            try:
                from spotting import load_spotting_bundle

                bundle = load_spotting_bundle()
            except Exception as error:  # optional: a truncated or LFS-pointer file must not fail the run
                print(f"Spotting model unavailable, running without it: {error}")
        if bundle is not None:
            from spotting import ActionSpotter

            spotter = ActionSpotter(
                bundle.model, bundle.labels, fps, sample_fps=config.spotting_sample_fps, clip_len=bundle.clip_len
            )
            print(f"Spotting model on, sampling {spotter.sample_fps:g} frames a second")

    team_squads = _team_squads(args, team_names)
    identity_tracker = _build_identity_tracker(args, team_names, fps, team_squads)
    commentary_words, commentary_mentions, commentary_language = [], [], None
    if config.use_commentary and not getattr(args, "no_commentary", False):
        with metrics.timer("commentary"):
            commentary_words, commentary_mentions, commentary_language = _transcribe_commentary(
                args, config, team_squads
            )
    possession_log = []

    if not should_calibrate_teams(args):
        print("Using the team kit centres confirmed during upload.")
    elif _calibrate_teams(tracker, team_assigner, args.video, total_frames, fps, height):
        print("Team kits calibrated from the video; configured colours name the two kits.")
    else:
        print("Team kit calibration found no clear second kit; matching configured colours directly.")

    # Scene signatures are collected during the same sequential decode used
    # for inference. A rolling overlap catches dissolves spanning two batches.
    scenes = SceneTimeline([])
    scene_signatures = []
    seen_boundaries = set()

    frame_number = 0
    frames_with_ball = 0
    pending_events = []
    stop = False
    scheduled = iter(scheduled_frames(total_frames, fps, target_fps))
    next_scheduled = next(scheduled, None)

    def update_scenes():
        nonlocal scenes
        if getattr(args, "no_scene_cuts", False) or len(scene_signatures) < 2:
            return
        local = find_boundaries([signature for _, signature in scene_signatures])
        start_frame = scene_signatures[0][0]
        boundaries = list(scenes.boundaries)
        for boundary in local:
            absolute = type(boundary)(
                start_frame + boundary.start,
                start_frame + boundary.end,
                boundary.kind,
                boundary.strength,
            )
            key = (absolute.start, absolute.end, absolute.kind)
            if key not in seen_boundaries:
                seen_boundaries.add(key)
                boundaries.append(absolute)
        scenes = SceneTimeline(sorted(boundaries, key=lambda boundary: boundary.start))

    handled_scene_start = -1

    def process_batch(batch, frame_numbers):
        nonlocal frames_with_ball, stop, handled_scene_start
        skip_offsets = {o for o, number in enumerate(frame_numbers) if scenes.in_transition(number)}
        # Only some frames are analysed, so the exact first frame of a shot is
        # usually not among them. Reset on the first analysed frame at or after
        # any scene start not yet acted on - otherwise roughly half the cuts at
        # 25 fps (and most at 60 fps) never clear the previous shot's ball trail
        # and goal frame, and the "cut away after a shot" goal rule never fires.
        reset_offsets = set()
        for offset, number in enumerate(frame_numbers):
            if offset in skip_offsets:
                continue
            start = scenes.latest_start_at_or_before(number)
            if start is not None and start > handled_scene_start:
                reset_offsets.add(offset)
                handled_scene_start = start
        tracks = tracker.get_object_tracks(
            batch,
            reset_offsets=reset_offsets,
            skip_offsets=skip_offsets,
        )
        tracker.add_position_to_tracks(tracks)

        for offset, (frame, original_frame) in enumerate(zip(batch, frame_numbers)):
            single = _frame_tracks(tracks, offset)
            players = single["players"][0]
            referees = single["referees"][0]
            balls = single["ball"][0]
            ball_bbox = next(iter(balls.values()))["bbox"] if balls else None
            if ball_bbox is not None:
                frames_with_ball += 1
            in_transition = offset in skip_offsets
            if offset in reset_offsets:
                pending_events.extend(track_events.reset_scene(original_frame))
                # Ids restart (offset) at a cut; old tracks' team votes are dead.
                team_assigner.forget_players(players.keys())
                if identity_tracker is not None:
                    identity_tracker.end_scene()
                if pitch_calibrator is not None:
                    pitch_calibrator.reset()
            calibration = (
                pitch_calibrator.update(original_frame, frame)
                if pitch_calibrator is not None and not in_transition
                else None
            )
            if scoreboard_reader is not None:
                with metrics.timer("scoreboard_ocr"):
                    scoreboard_reader.update(original_frame, frame)

            new_events = []
            if not in_transition:
                _assign_teams(team_assigner, frame, players)
                if identity_tracker is not None:
                    identity_started = perf_counter()
                    identity_tracker.observe(original_frame, frame, players, calibration)
                    metrics.add_time("player_identity", perf_counter() - identity_started)
                new_events = track_events.update(
                    original_frame, players, ball_bbox, referees, frame, pitch=calibration
                )
                pending_events.extend(new_events)
                # Shirt crops for the ball holder and for every player big
                # enough to read (close-ups, replays): the fouled player, the
                # assister and a scorer seen before he touched the ball need
                # reads too. The size gate keeps wide shots to a few players.
                readable = scorer_identifier.identity_store.min_box_height
                candidates = {
                    track_id: player
                    for track_id, player in players.items()
                    if player.get("has_ball") or float(player["bbox"][3]) - float(player["bbox"][1]) >= readable
                }
                scorer_identifier.observe_players(original_frame, frame, candidates)
                _record_ball_possession(scorer_identifier, players, original_frame)
                holder = next((t for t, p in players.items() if p.get("has_ball")), None)
                if holder is not None:
                    # Whole-match possession timeline: a spoken name goes to
                    # whoever had the ball when it was said.
                    possession_log.append(
                        (original_frame, int(holder), int(players[holder].get("team", 0) or 0))
                    )
                if scoreboard_reader is not None and any(
                    event.get("event") == "shot" for event in new_events
                ):
                    scoreboard_reader.trigger_burst(original_frame)

            if event_detector is not None and event_logic is not None:
                detected_events = event_detector.predict(frame, original_frame)
                pending_events.extend(
                    event_logic.update(detected_events, single, original_frame, track_events=track_events)
                )

            if write_video or show_video:
                annotated = tracker.draw_annotations([frame], single)[0]
                _draw_team_labels(annotated, players)
                if writer is not None:
                    writer.write(annotated)
                if show_video:
                    cv2.imshow("Football Analysis System", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop = True
                        return

    try:
        while not stop:
            batch = []
            batch_numbers = []
            while len(batch) < FRAME_BATCH_SIZE and next_scheduled is not None:
                success, frame = capture.read()
                if not success:
                    next_scheduled = None
                    break
                original_frame = frame_number
                if not getattr(args, "no_scene_cuts", False):
                    scene_started = perf_counter()
                    scene_signatures.append((original_frame, frame_signature(frame)))
                    metrics.add_time("scene_analysis", perf_counter() - scene_started)
                    if len(scene_signatures) > 96:
                        del scene_signatures[:-64]
                if spotter is not None and spotter.wants(original_frame):
                    spotting_started = perf_counter()
                    spotter.add_frame(original_frame, frame)
                    metrics.add_time("spotting", perf_counter() - spotting_started)
                if original_frame == next_scheduled:
                    batch.append(frame)
                    batch_numbers.append(original_frame)
                    next_scheduled = next(scheduled, None)
                frame_number += 1
            if not batch:
                break
            update_scenes()
            process_batch(batch, batch_numbers)

            if progress_cb is not None:
                progress_cb(frame_number, total_frames)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show_video:
            cv2.destroyAllWindows()

    pending_events.extend(track_events.finish(frame_number))
    if spotter is not None:
        from spotting import USED_CLASSES, fuse_spotted_events

        with metrics.timer("spotting"):
            spotter.finish()
        spotted = spotter.events(min_confidence=config.spotting_min_confidence, classes=USED_CLASSES)
        before = len(pending_events)
        pending_events = fuse_spotted_events(pending_events, spotted, fps)
        print(
            f"Spotting: {len(spotter.sample_frames)} samples, {spotter.inference_seconds:.1f}s inference; "
            + (", ".join(f"{e['event']} {e['confidence']:.2f}@{e['frame'] / fps:.1f}s" for e in spotted) or "nothing")
            + f"; {len(pending_events) - before} event(s) added"
        )
    commentary_attached = []
    if commentary_mentions:
        from commentary import attach_to_tracks

        commentary_attached = attach_to_tracks(commentary_mentions, possession_log, fps, identity_tracker)
        print(
            f"Commentary: {len(commentary_attached)} of {len(commentary_mentions)} spoken names attached "
            "to the player on the ball"
        )
    pending_events = _drop_model_fouls_near_track_fouls(pending_events, fps)
    pending_events = _drop_model_goals_near_pitch_goals(pending_events, fps)
    code_to_team = None
    if scoreboard_reader is not None and scoreboard_reader.active:
        # Score-graphic goals are added to the track, pitch and model goals,
        # not substituted for them: a proposal inside a score change's window
        # is that same goal and gives way; any other proposal is kept, because
        # the graphic misses changes it cannot see (hidden during replays,
        # other codes, a single unconfirmed read).
        codes = scoreboard_reader.tracker.codes
        code_to_team = map_codes_to_teams(codes, team_names, known_codes)
        board_goals, _evidence = attribute_goals(
            scoreboard_reader.tracker.goals,
            codes,
            code_to_team,
            team_names,
            pending_events,
            fps,
            scorer_identifier.possession_history.samples,
        )
        pending_events, duplicates = merge_board_goals(pending_events, board_goals, fps)
        other_goals = sum(
            1 for event in pending_events if event.get("event") == "goal" and event.get("source") != "scoreboard"
        )
        summary = scoreboard_reader.summary()
        print(
            f"Scoreboard: codes {summary['codes']} -> teams {code_to_team}; "
            f"{summary['confirmed_readings']} readings in {summary['ocr_calls']} OCR calls; "
            f"score timeline {summary['timeline']}; {len(board_goals)} goal(s); "
            f"{len(duplicates)} other proposal(s) were the same goal; "
            f"{other_goals} goal(s) the graphic did not show kept"
        )
        for frame, reason in summary["rejected"]:
            print(f"  scoreboard note at frame {frame}: {reason}")
        for goal in board_goals:
            if not goal["details"]["detected_goal"]:
                # The score changed but the video rules saw no goal: a gap in
                # detection worth knowing about, not just a scorer to fill in.
                print(
                    f"  score changed at frame {goal['details']['scoreboard_frame']} with no goal "
                    f"detected from the video; placed at frame {goal['frame']} "
                    f"({goal['details']['attribution']})"
                )
    else:
        if scoreboard_reader is not None:
            print(
                f"Scoreboard: no score graphic read ({scoreboard_reader.ocr_calls} OCR calls); "
                "goals come from the track and pitch rules"
            )
    # Spotted kick-offs only separate goals from replays (drop_replayed_goals,
    # after the scorers are named); they are not events a user sees.
    kick_off_frames = [int(e.get("frame", 0)) for e in pending_events if e.get("event") == "kick_off"]
    pending_events = [event for event in pending_events if event.get("event") != "kick_off"]
    # A shot that went in was not saved.
    pending_events = withdraw_saves_of_goals(pending_events)
    pending_events.sort(key=lambda event: int(event.get("frame", 0)))
    # Before passes are hidden: the last team-mate pass to the scorer.
    _attach_pass_assisters(pending_events, fps)

    # Shots and saves remain available above as attribution evidence. The web
    # profile exposes and persists only the configured event types.
    if config.goal_only:
        pending_events = goal_output_events(pending_events, config.output_events)

    print(f"Ball selected in {frames_with_ball} of {tracker.person_inference_frames} analysed frames")
    if pitch_calibrator is not None:
        stats = pitch_calibrator.summary()
        print(
            "Pitch calibration: "
            f"{stats['accepted_keyframes']} of {stats['keyframes']} keyframes accepted, "
            f"{stats['frames_calibrated']} of {stats['frames_seen']} frames calibrated; "
            f"rejections {stats['rejections']}"
        )

    # Read shirt numbers once per track, over the crops the whole run collected.
    # Highlight-worthy events are resolved first so they get the OCR budget.
    image_dir = getattr(args, "player_image_dir", None)
    url_prefix = getattr(args, "player_image_url_prefix", None)
    def resolution_order(event):
        kind = str(event.get("event", "")).lower()
        return 0 if kind == "goal" else 1 if kind in CLIP_EVENT_TYPES else 2

    ordered = sorted(pending_events, key=resolution_order)
    with metrics.timer("scorer_ocr"):
        for event in ordered:
            _resolve_identities(scorer_identifier, event, image_dir, url_prefix, identity_tracker)

    # Goals: fuse the caption, the celebration close-ups and the live read.
    caption_ocr = None
    if (
        not getattr(args, "no_ocr", False)
        and not getattr(args, "no_scorer_captions", False)
        and jersey_reader._ensure_engine()
    ):
        caption_ocr = paddle_ocr_fn(jersey_reader.engine)
    commentary_roles = (
        _commentary_goal_roles(pending_events, commentary_words, commentary_mentions, fps)
        if commentary_mentions or commentary_words
        else {}
    )
    with metrics.timer("scorer_ocr"):
        _improve_goal_scorers(
            pending_events,
            scorer_identifier,
            args.video,
            fps,
            height,
            caption_ocr,
            extra_ocr_allowance=(0 if config.goal_only else GOAL_SCORER_OCR_ALLOWANCE),
            team_squads=team_squads,
            commentary_roles=commentary_roles,
            team_names=team_names,
        )
    # The commentary names the assister, the fouler and the fouled player, and
    # a scorer the fusion left unnamed. Keepers default to the roster's number one.
    if commentary_roles or commentary_mentions:
        _apply_commentary(
            pending_events, commentary_words, commentary_mentions, fps, team_names, identity_tracker,
            goal_roles_by_event=commentary_roles,
        )
    # One goal is shown live and again in replays. Decided only now, once
    # captions and commentary have named the scorers: two goals close together
    # with different named scorers are two goals (highlight packages).
    goals_before = sum(1 for e in pending_events if e.get("event") == "goal")
    pending_events = drop_replayed_goals(pending_events, fps, kick_off_frames=kick_off_frames)
    replays = goals_before - sum(1 for e in pending_events if e.get("event") == "goal")
    if replays:
        print(f"Goals: {replays} replay(s) of an earlier goal dropped")
    _name_keepers(pending_events, team_squads, team_names)
    for event in pending_events:
        event["description"] = describe_event(event)

    # One clip per passage of play, cut at the camera shots, with the source
    # audio; the reel joins them.
    with metrics.timer("highlights"):
        highlights = export_highlights(
            args.video,
            pending_events,
            fps,
            getattr(args, "clip_dir", None),
            getattr(args, "clip_url_prefix", None),
            boundaries=scenes.boundaries,
            total_frames=total_frames,
            reel=not getattr(args, "no_highlight_reel", False),
        )
    if highlights["clips"]:
        reel = highlights["reel"]
        print(
            f"Highlights ({highlights['encoder']}): {len(highlights['clips'])} clip(s) for "
            f"{sum(len(c['event_types']) for c in highlights['clips'])} event(s)"
            + (f"; reel {reel['duration_seconds']} s" if reel else "")
        )

    for event in pending_events:
        print(f"[frame {event.get('frame', 0)}] {event['description']}")
    if event_logic is not None:
        event_logic.generate_summary(fps, events=pending_events)

    metrics.add_time("person_detection", tracker.person_inference_seconds)
    metrics.add_time("ball_detection", tracker.ball_inference_seconds)
    metrics.increment("decoded_frames", frame_number)
    metrics.increment("analysed_frames", tracker.person_inference_frames)
    metrics.increment("person_inference_frames", tracker.person_inference_frames)
    metrics.increment("ball_inference_frames", tracker.ball_inference_frames)
    if scoreboard_reader is not None:
        metrics.increment("scoreboard_ocr_calls", scoreboard_reader.ocr_calls)
    metrics.increment("scorer_ocr_calls", getattr(jersey_reader, "_ocr_calls", 0))
    score_by_team_id = final_score_by_team(pending_events, scoreboard_reader, code_to_team)
    result = {
        "events": pending_events,
        "score_by_team_id": score_by_team_id,
        "metrics": metrics.summary(),
        "player_tracks": identity_tracker.summary() if identity_tracker is not None else [],
        "commentary": {
            "language": commentary_language,
            "words": len(commentary_words),
            "mentions": [
                {"at_seconds": round(m.start, 2), "said": m.text,
                 "players": [{"team_id": c.team_id, "jersey_number": c.jersey_no, "name": c.name} for c in m.candidates]}
                for m in commentary_mentions
            ],
            "attached": [
                {"at_seconds": round(a.mention.start, 2), "track_id": a.track_id,
                 "team_id": a.candidate.team_id, "jersey_number": a.candidate.jersey_no, "name": a.candidate.name}
                for a in commentary_attached
            ],
        },
    }
    return result if getattr(args, "return_pipeline_result", False) else pending_events


def main():
    process_video(build_parser().parse_args())


if __name__ == "__main__":
    main()
