import argparse
import os

import cv2

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
from scorer_identifier.goal_scorer import CELEBRATION_SECONDS
from team_assigner import TeamAssigner, kit_color_bgr, safe_player_crop
from track_events import TrackEventDetector, summarise_passes
from scene import SceneTimeline, detect_scene_boundaries
from scoreboard import (
    ScoreboardReader,
    attribute_goals,
    combined_expected,
    expected_codes_for,
    map_codes_to_teams,
    paddle_ocr_fn,
    parse_code_option,
)


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# Trained on 2026-09-23: a single-class ball detector and a 32-keypoint pitch
# model. Both are optional; pass "none" to either option to run without it.
DEFAULT_BALL_MODEL = os.path.join(MODELS_DIR, "ball_detector_best.pt")
DEFAULT_PITCH_MODEL = os.path.join(MODELS_DIR, "football_field_best.pt")
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
    parser.add_argument("--event-model", default="models/final_model_80.pth")
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
        help='pitch keypoint (pose) weights for pitch coordinates; "none" disables them',
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


def _resolve_identities(scorer_identifier, event, image_dir, url_prefix):
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
    else:
        scorer = scorer_identifier.identify_goal(int(event.get("frame", 0)))
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


def _improve_goal_scorers(events, scorer_identifier, video_path, fps, frame_height, ocr_fn=None):
    """Re-decide every goal's scorer from all the evidence, not one shirt read.

    Sources and weights are in scorer_identifier/goal_scorer.py: the broadcast
    caption naming the scorer (``ocr_fn`` None skips it), the shirt numbers of
    the scoring team's players in the celebration close-ups, and the live read
    of the shooter. The result replaces ``scorer`` (and the scorer actor) with
    the fused shirt number, a confidence, and a name only when confirmed.
    """
    fps = max(float(fps), 1.0)
    for event in events:
        if str(event.get("event", "")).lower() != "goal":
            continue
        details = event.setdefault("details", {})
        team_id = int(event.get("team_id", 0) or 0)
        squad = squad_for(event.get("team_name"))
        scorer = dict(event.get("scorer") or {})
        moment = int(event.get("frame", 0) or 0)
        board_frame = int(details.get("scoreboard_frame") or moment)

        candidates = live_candidate(scorer, str(details.get("attribution") or ""))
        if ocr_fn is not None:
            candidates += read_caption_candidates(video_path, moment, fps, ocr_fn, squad)
        scorer_identifier.jersey_reader.max_ocr_calls += GOAL_SCORER_OCR_ALLOWANCE
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
        details["scorer_evidence"] = [
            {
                "source": c.source,
                "jersey_number": c.jersey_number,
                "player_name": c.player_name,
                "weight": round(c.weight, 3),
            }
            for c in candidates
        ]
        fused = fuse_scorer(candidates, squad)
        if fused is None:
            continue

        changed = fused["jersey_number"] != scorer.get("jersey_number")
        for record in [scorer, *[a for a in event.get("actors") or [] if a.get("role") == "scorer"]]:
            record.update(fused)
            record.setdefault("team_id", team_id)
            record.setdefault("team_name", event.get("team_name"))
            if changed:
                # The photo is of the tracked shooter, who is not the fused scorer.
                record.pop("player_image", None)
                record["jersey_confidence"] = fused["scorer_confidence"]
        event["scorer"] = scorer


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


def process_video(args, progress_cb=None):
    # Heavy model imports are delayed so configuration validation and --help work
    # without initializing Ultralytics/Torch or their user-level cache folders.
    from event_detector import EventDetector
    from event_logic import EventLogic
    from trackers import Tracker

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
    pitch_model = _optional_model(getattr(args, "pitch_model", DEFAULT_PITCH_MODEL))
    tracker = Tracker(
        args.tracking_model,
        frame_rate=fps,
        image_size=getattr(args, "image_size", None),
        ball_model_path=ball_model,
    )
    pitch_calibrator = None
    if pitch_model:
        from pitch import PitchCalibrator

        pitch_calibrator = PitchCalibrator(
            pitch_model, stride=int(getattr(args, "pitch_stride", PITCH_STRIDE_FRAMES) or PITCH_STRIDE_FRAMES)
        )
    print(f"Ball source: {ball_model or 'ball class of ' + str(args.tracking_model)}")
    print(f"Pitch model: {pitch_model or 'none'}")
    event_detector = EventDetector(args.event_model)
    event_logic = EventLogic(fps=fps)
    team_assigner = TeamAssigner(match_config)
    ball_assigner = PlayerBallAssigner()
    track_events = TrackEventDetector(fps, (width, height), ball_assigner=ball_assigner)
    if getattr(args, "no_ocr", False):
        jersey_reader = JerseyReader(debug_dir=None, max_ocr_calls=0)
    else:
        jersey_reader = JerseyReader(debug_dir=args.ocr_debug_dir)
    scorer_identifier = ScorerIdentifier(jersey_reader=jersey_reader, frame_height=height)

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
                read_every_seconds=float(getattr(args, "scoreboard_every", 1.0) or 1.0),
            )
        else:
            print(f"Scoreboard reading unavailable: {jersey_reader._engine_error}")

    if _calibrate_teams(tracker, team_assigner, args.video, total_frames, fps, height):
        print("Team kits calibrated from the video; configured colours name the two kits.")
    else:
        print("Team kit calibration found no clear second kit; matching configured colours directly.")

    # Camera cuts and crossfades, found in one cheap decode pass up front so
    # the frame loop knows where every shot starts and which frames are blends.
    scenes = None
    if not getattr(args, "no_scene_cuts", False):
        scenes = SceneTimeline(detect_scene_boundaries(args.video))
        print(f"Scene boundaries: {scenes.describe()}")

    frame_number = 0
    frames_with_ball = 0
    pending_events = []
    stop = False
    try:
        while not stop:
            batch = []
            while len(batch) < FRAME_BATCH_SIZE:
                success, frame = capture.read()
                if not success:
                    break
                batch.append(frame)
            if not batch:
                break

            skip_offsets = set()
            reset_offsets = set()
            if scenes is not None:
                skip_offsets = {o for o in range(len(batch)) if scenes.in_transition(frame_number + o)}
                reset_offsets = {o for o in range(len(batch)) if scenes.starts_scene(frame_number + o)}
            tracks = tracker.get_object_tracks(
                batch,
                read_from_stub=False,
                reset_offsets=reset_offsets,
                skip_offsets=skip_offsets,
            )
            tracker.add_position_to_tracks(tracks)

            for offset, frame in enumerate(batch):
                single = _frame_tracks(tracks, offset)
                players = single["players"][0]
                referees = single["referees"][0]
                balls = single["ball"][0]
                ball_bbox = next(iter(balls.values()))["bbox"] if balls else None
                if ball_bbox is not None:
                    frames_with_ball += 1
                in_transition = offset in skip_offsets
                if offset in reset_offsets:
                    # A new camera shot: nothing tracked in the old one carries over.
                    pending_events.extend(track_events.reset_scene(frame_number))
                    if pitch_calibrator is not None:
                        pitch_calibrator.reset()
                calibration = (
                    pitch_calibrator.update(frame_number, frame)
                    if pitch_calibrator is not None and not in_transition
                    else None
                )
                if scoreboard_reader is not None:
                    scoreboard_reader.update(frame_number, frame)

                if not in_transition:
                    # Inside a crossfade two shots are blended: no team votes,
                    # no shirt crops, no events.
                    _assign_teams(team_assigner, frame, players)
                    scorer_identifier.observe_players(frame_number, frame, players)

                    # Possession, passes, crosses, shots, saves, penalties and
                    # cards from the tracks themselves.
                    pending_events.extend(
                        track_events.update(
                            frame_number, players, ball_bbox, referees, frame, pitch=calibration
                        )
                    )
                    _record_ball_possession(scorer_identifier, players, frame_number)

                # The learned goal/foul model proposes; the tracks confirm.
                detected_events = event_detector.predict(frame, frame_number)
                pending_events.extend(
                    event_logic.update(
                        detected_events, single, frame_number, track_events=track_events
                    )
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
                            break

                frame_number += 1

            if progress_cb is not None:
                progress_cb(frame_number, total_frames)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show_video:
            cv2.destroyAllWindows()

    pending_events.extend(track_events.finish(frame_number))
    pending_events = _drop_model_fouls_near_track_fouls(pending_events, fps)
    if scoreboard_reader is not None and scoreboard_reader.active:
        # The score graphic is the authority on goals while it can be read:
        # every other goal proposal is either confirmed by a score change (and
        # becomes its moment and scorer) or dropped.
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
        dropped = [event for event in pending_events if event.get("event") == "goal"]
        pending_events = [event for event in pending_events if event.get("event") != "goal"] + board_goals
        summary = scoreboard_reader.summary()
        print(
            f"Scoreboard: codes {summary['codes']} -> teams {code_to_team}; "
            f"{summary['confirmed_readings']} readings in {summary['ocr_calls']} OCR calls; "
            f"score timeline {summary['timeline']}; {len(board_goals)} goal(s); "
            f"{len(dropped)} other goal proposal(s) replaced"
        )
        for frame, reason in summary["rejected"]:
            print(f"  scoreboard note at frame {frame}: {reason}")
    else:
        if scoreboard_reader is not None:
            print(
                f"Scoreboard: no score graphic read ({scoreboard_reader.ocr_calls} OCR calls); "
                "goals come from the track and pitch rules"
            )
        pending_events = _drop_model_goals_near_pitch_goals(pending_events, fps)
    pending_events.sort(key=lambda event: int(event.get("frame", 0)))

    print(f"Ball selected in {frames_with_ball} of {frame_number} frames")
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
    for event in ordered:
        _resolve_identities(scorer_identifier, event, image_dir, url_prefix)

    # Goals: fuse the caption, the celebration close-ups and the live read.
    caption_ocr = None
    if (
        not getattr(args, "no_ocr", False)
        and not getattr(args, "no_scorer_captions", False)
        and jersey_reader._ensure_engine()
    ):
        caption_ocr = paddle_ocr_fn(jersey_reader.engine)
    _improve_goal_scorers(pending_events, scorer_identifier, args.video, fps, height, caption_ocr)
    for event in pending_events:
        event["description"] = describe_event(event)

    # One clip per passage of play, cut at the camera shots, with the source
    # audio; the reel joins them.
    highlights = export_highlights(
        args.video,
        pending_events,
        fps,
        getattr(args, "clip_dir", None),
        getattr(args, "clip_url_prefix", None),
        boundaries=scenes.boundaries if scenes is not None else (),
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
    event_logic.generate_summary(fps, events=pending_events)
    return pending_events


def main():
    process_video(build_parser().parse_args())


if __name__ == "__main__":
    main()
