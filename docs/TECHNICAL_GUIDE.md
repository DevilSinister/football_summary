# Footy AI - Technical Guide

Developer reference for the Footy AI system: the FastAPI backend and video
analysis pipeline (`football_summary`, branch `jugaad2.0`) and the Flutter
Android app (`footyai/Footy_AI/footy_ai_app`, branch `sinister/experimental`).

It is written for developers and AI agents who need to change the system. A
separate plain-language guide (`docs/HOW_TO_CHANGE_THINGS.md`) covers the same
ground for non-technical readers.

How this document was produced, and how far to trust it:

- Every statement about behaviour comes from reading the code as it stood on
  2026-11-06, including the uncommitted working-tree changes of that day
  (`features.yaml`, `feature_flags.py`, `backend/routers/features.py`, the
  edits to `main.py`, `pipeline_config.py`, `backend/app.py`,
  `backend/routers/processing.py`, and in the app `lib/services/feature_service.dart`
  and the edits to `ai_match_summary.dart`, `match_highlights_list.dart`,
  `ai_predictions_screen.dart`).
- File references are relative to the repository they belong to. Backend paths
  (`main.py`, `backend/...`) are relative to `football_summary/`. App paths
  (`lib/...`, `test/...`, `android/...`) are relative to
  `footyai/Footy_AI/footy_ai_app/`. Line numbers are approximate and will drift;
  function names are the stable reference.
- "Measured" figures in section 10 come from the project's Obsidian vault
  (`Football Summary/80 - Delivery/Verification Evidence.md` and
  `70 - Bugs and Risks/Known Risks and Gaps.md`). They are history, not a
  re-run.
- Where something is unknown or was not verified, the text says so.
- No credentials, tokens, private URLs or real LAN addresses appear here. Use
  `<PC LAN IP>` for the address of the computer that runs the server.

---

## Contents

1. [Overview and architecture](#1-overview-and-architecture)
2. [Running everything locally](#2-running-everything-locally)
3. [Feature switches (`features.yaml`)](#3-feature-switches-featuresyaml)
4. [The analysis pipeline](#4-the-analysis-pipeline)
5. [REST API reference](#5-rest-api-reference)
6. [Database](#6-database)
7. [The Flutter app](#7-the-flutter-app)
8. [Recipes: how do I...](#8-recipes-how-do-i)
9. [Testing and verification](#9-testing-and-verification)
10. [Known limits and measured accuracy](#10-known-limits-and-measured-accuracy)
11. [Troubleshooting](#11-troubleshooting)
12. [Glossary](#12-glossary)
13. [Appendix: code findings and stale documentation](#13-appendix-code-findings-and-stale-documentation)

---

## 1. Overview and architecture

### 1.1 What the system does

A user records or downloads a football video, uploads it from the Android app,
confirms which team wears which kit by looking at player images, and receives a
match summary: the score, goals with the scorer (and assister) when they can be
named, other events that are switched on (fouls by default), and short
highlight clips with a joined reel.

### 1.2 Components and where they run

| Component | Where | Technology | Entry point |
| --- | --- | --- | --- |
| Phone app | Android phone or emulator | Flutter 3.44 / Dart SDK `^3.10.7`, packages `http`, `shared_preferences`, `image_picker`, `video_player`, `file_selector`, `google_fonts` (`pubspec.yaml`) | `lib/main.dart` |
| HTTP API | Windows PC, port 8000 | FastAPI + Uvicorn, Python 3.12 venv `.venv` | `backend/app.py` (`app`) |
| Job queue | Inside the API process | One-thread `ThreadPoolExecutor` (`backend/pipeline_runtime.py`, `SerialJobQueue`) | `job_queue.submit(...)` |
| Model cache | Inside the API process | Lazy singletons (`backend/pipeline_runtime.py`, `ModelCache`) | `model_cache.yolo()`, `.spotting()`, `.reid()`, `.whisper()`, `.paddle_ocr()` |
| Analysis pipeline | Inside the API process (job thread), or stand-alone from the command line | OpenCV, Ultralytics YOLO, supervision ByteTrack, PaddleOCR 3.x, faster-whisper, PyTorch (E2E-Spot, OSNet), scikit-learn | `main.py` (`process_video`) |
| Database | Same PC | SQL Server Express (default instance `.\SQLEXPRESS`, database `FootballDB`), via pyodbc + SQLAlchemy 2 | `backend/database.py` |
| Media files | Same PC, under `backend/static/` and `backend/uploads/` | Plain files served by FastAPI `StaticFiles` and `FileResponse` | `backend/app.py`, `backend/routers/processing.py` |
| Web page (set aside) | Served by the API at `/` | Static HTML/CSS/JS | `backend/static/index.html`, `app.css`, `app.js` |

The owner has set the web page aside; the Flutter app is the main client (see
also the user memory note "APK is the main client"). The web page still calls
the same endpoints (`submit-file`, `submit-url`, `status`, `result`,
`/api/teams`, `confirm-teams` - `backend/static/app.js`), so do not break it by
accident, but new user-facing work goes into the app.

### 1.3 Architecture diagram

```
 Android phone (Flutter app)                         Windows PC
 ---------------------------                         -----------------------------------------------
                                                     uvicorn backend.app:app  --host 0.0.0.0 --port 8000
  AppConfig.apiBaseUrl  ---- HTTP (LAN, :8000) --->  FastAPI app (backend/app.py)
  (build default or                                    |-- /api/users/*        routers/users.py      --+
   saved in Settings)                                  |-- /api/processing/*   routers/processing.py   |
                                                       |-- /api/matches[/id]   routers/processing.py   |
  upload  -> submit-file (multipart)                   |-- /api/teams          routers/teams.py        |
  poll    -> status/{job_id} every 2 s                 |-- /api/features       routers/features.py     |
  confirm -> confirm-teams/{job_id}                    |-- /api/players/*      routers/players.py      |
  summary -> /api/matches/{id}, /api/features          |-- /static/*  (StaticFiles: team samples,      |
  clips   -> /api/processing/clips/{job}/{file}        |              player images, clips dir, web)   |
                                                       |-- /health, /, /docs (Swagger)                 |
                                                       |                                               |
                                                       |  _jobs dict (in memory, lost on restart)      |
                                                       v                                               |
                                              SerialJobQueue (1 worker thread "footy-cv")              |
                                                 1) _detect_team_candidates(job_id)                    |
                                                 2) _process_job(job_id)                               |
                                                       |                                               |
                                                       v                                               |
                                              main.process_video(args)  <-- PipelineConfig.short_clip()|
                                                 tracking, teams, events, scoreboard OCR,  (reads      |
                                                 spotting, commentary, naming, clips       features.yaml)
                                                       |                                               |
                                    +------------------+-------------------+                           |
                                    v                                      v                           v
                     backend/static/clips/<job>/*.mp4,          SQL Server Express  FootballDB
                     reel.mp4, player_images/<job>/,            users, user_logs, teams, team_aliases,
                     team_samples/<job>-team-N.jpg              players, matches, plays, events,
                     backend/uploads/<uuid>.mp4 (deleted        occurrences
                     when the job ends)
```

### 1.4 Ports, hosts and LAN testing

- The API listens on port **8000**. `--host 0.0.0.0` is required for a phone to
  reach it; `127.0.0.1` only accepts connections from the PC itself.
- The app's build-time default is `http://10.0.2.2:8000`
  (`lib/config/app_config.dart`, `_buildApiBaseUrl`). `10.0.2.2` is the Android
  emulator's alias for the host PC; it does not work on a physical phone.
- On a physical phone, either build with
  `--dart-define=API_BASE_URL=http://<PC LAN IP>:8000` or type the address in
  the app's Settings tab (saved in SharedPreferences key `api_base_url`).
- Phone and PC must be on the same network, and Windows Firewall must allow
  inbound TCP 8000 for the Python process (section 2.3).
- All traffic is plain HTTP. `android/app/src/main/AndroidManifest.xml` sets
  `android:usesCleartextTraffic="true"`, which `video_player` needs for
  `http://` clip URLs.
- CORS is wide open (`allow_origins=["*"]`, `backend/app.py`). There is no
  authentication on any processing, match, team, feature or player endpoint;
  only `/api/users/login` checks a password, and nothing downstream checks the
  result. Treat the server as LAN-only.

### 1.5 Repositories and what to ignore

| Path | Branch | Notes |
| --- | --- | --- |
| `D:\Projects\football_summary` | `jugaad2.0` | Backend + pipeline. Main branch for PRs is `main`. |
| `D:\Projects\footyai` (git root) | `sinister/experimental` | Contains `Footy_AI/footy_ai_app` (the Flutter app). |
| `footyai/main.py`, `footyai/database.py`, `footyai/models.py`, `footyai/schemas.py` | untracked | Stubs of an older FastAPI backend (they import `routers.users`, `routers.processing` that are not there). Origin unknown, not used by anything. Ignore them. |
| `footyai/Footy_AI/{ai_match_summary, ai_processing_loading_screen, ...}`, `server.js`, `node_modules` | - | Sibling folders and a Node file next to the app. Not used by the app or backend as far as this review found. |
| `footyai/tools/jdk17/...` | - | The JDK 17 that the machine's Flutter is configured to use (`flutter config --jdk-dir`). |

---

## 2. Running everything locally

### 2.1 Prerequisites

| Need | Detail | Where it is used |
| --- | --- | --- |
| Python 3.12 venv at `.venv` | Packages from `requirements.txt` (numpy, opencv-python, pandas, scikit-learn, torch, timm>=1.0, faster-whisper>=1.2, ultralytics, supervision, paddlepaddle>=3.0, paddleocr>=3.0, fastapi, uvicorn[standard], sqlalchemy, pyodbc, python-multipart, bcrypt, email-validator) | everything |
| PyYAML and httpx | Listed in `requirements.txt` since 2026-11-07. `yaml` is imported by `feature_flags.py`, `httpx` by `backend/routers/processing.py` (installed in the current `.venv`: PyYAML 6.0.2, httpx 0.28.1). | feature switches; `submit-url` |
| SQL Server Express + "ODBC Driver 18 for SQL Server" | Windows authentication (`Trusted_Connection=yes`) | `backend/database.py` |
| ffmpeg and ffprobe on `PATH` | Optional but strongly advised. Without them clips are cut silently with OpenCV, no reel is written (`highlights/export.py`), and commentary has no audio to transcribe (`commentary/transcribe.py`, `extract_audio`). Override paths with env vars `FFMPEG_BINARY` / `FFPROBE_BINARY` (`highlights/encoder.py`). | clips, reel, commentary |
| Model files in `models/` | `best.pt` (players, keepers, referees, ball), `ball_detector_best.pt` (ball), `checkpoint_088.pt` + `config.json` (action spotting), `reid/osnet_x1_0_msmt17.pt` (re-identification), `whisper/` (faster-whisper "small" cache), `Football_field.pt` (pitch keypoints, off), `final_model_80.pth` (legacy LSTM, off). `.gitignore` excludes `models/*.pt` and `models/reid/`; only `best.pt` and `final_model_80.pth` are tracked, so a fresh clone must get the other weights separately. Missing optional models (spotting, re-ID, ball) are skipped with a console message rather than failing the job; faster-whisper would try to download "small" into `models/whisper/` on first use and, if that fails, commentary is skipped. | pipeline |
| Flutter SDK + JDK 17 | Flutter 3.44.9 on this machine; Gradle 8.14; Android target `JavaVersion.VERSION_17` (`android/app/build.gradle.kts`) | app build |

### 2.2 Start the server

From `football_summary/`:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The same command is stored for the Claude desktop preview in
`.claude/launch.json` (configuration name `footy-api`, `runtimeExecutable`
`.venv/Scripts/python.exe`, `runtimeArgs` `-m uvicorn backend.app:app --host 0.0.0.0 --port 8000`, port 8000).

What happens at import and startup (`backend/app.py`):

1. `backend/database.py` runs at import: builds the ODBC string from
   `FOOTY_SQL_SERVER` (default `.\SQLEXPRESS`), `FOOTY_SQL_DATABASE` (default
   `FootballDB`, must match `[A-Za-z0-9_]+`), `FOOTY_ODBC_DRIVER` (default
   `ODBC Driver 18 for SQL Server`), connects to `master` and runs
   `CREATE DATABASE` if the database is missing (`ensure_database_exists`). If
   SQL Server is down, the import fails and the server does not start.
2. `Base.metadata.create_all(bind=engine)` creates any missing table.
3. `ensure_schema(engine)` (`backend/schema.py`) applies the additive column
   changes (section 6.3).
4. Routers are mounted: `users`, `processing`, `players`, `teams`, `features`;
   `/static` is mounted on `backend/static/`.
5. Startup hooks: `delete_stale_uploads` (calls `processing.sweep_media()`,
   section 4.13) and `seed_demo_user` (creates the demo account if its email is
   missing; the password is hard-coded in `backend/app.py` - read it there, it
   is deliberately not repeated here).

Importing `backend.routers.processing` also creates the media folders
(`backend/uploads`, `backend/static/{outputs,ocr_debug,team_samples,player_images,clips}`).
Heavy models are **not** loaded at startup; `ModelCache` loads each one the
first time a job needs it, so the first job after a restart is slower.

Environment variables read by the backend:

| Variable | Default | Read in |
| --- | --- | --- |
| `FOOTY_SQL_SERVER` | `.\SQLEXPRESS` | `backend/database.py` |
| `FOOTY_SQL_DATABASE` | `FootballDB` | `backend/database.py` |
| `FOOTY_ODBC_DRIVER` | `ODBC Driver 18 for SQL Server` | `backend/database.py` |
| `FOOTY_UPLOAD_MAX_AGE_HOURS` | `24` (`0` keeps uploads forever) | `backend/cleanup.py` |
| `FFMPEG_BINARY`, `FFPROBE_BINARY` | `shutil.which("ffmpeg")` / `("ffprobe")` | `highlights/encoder.py` |
| `PADDLE_PDX_CACHE_HOME` | set by the code to `.cache/paddlex` if unset | `scorer_identifier/jersey_reader.py` (`_ensure_engine`) |

### 2.3 Windows Firewall

A phone on the LAN cannot reach port 8000 until Windows allows it. Either
accept the "Allow access" prompt Windows shows the first time `python.exe`
listens on `0.0.0.0`, or add an inbound rule from an elevated PowerShell
(this changes system security settings, so it is the owner's decision):

```powershell
New-NetFirewallRule -DisplayName "Footy AI API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

The `Private` profile assumes the Wi-Fi network is marked private. On a
`Public` network profile the rule above does not apply.

### 2.4 Health check and Swagger

| Check | Command / URL | Expected |
| --- | --- | --- |
| Server alive (on the PC) | `curl http://127.0.0.1:8000/health` | `{"status":"healthy"}` |
| Reachable from the LAN | open `http://<PC LAN IP>:8000/health` in the phone's browser | same JSON |
| Interactive API docs | `http://127.0.0.1:8000/docs` (Swagger UI); `/redoc`; `/openapi.json` | FastAPI defaults, title "Footy AI API", version 2.0.0 |
| Feature switches as the app sees them | `curl http://127.0.0.1:8000/api/features` | JSON with `features`, `flags`, `visible_event_types`, `warnings` |
| Team list as the app sees it | `curl http://127.0.0.1:8000/api/teams` | JSON array; empty if no squads were seeded |

`/health` does not touch the database. It proves only that the process is up.
`/api/teams` proves the database is reachable.

### 2.5 Point the phone at the server

Two ways, in order of precedence at run time:

1. **Settings tab** in the app (`lib/screens/settings_screen.dart`): type
   `<PC LAN IP>` or `http://<PC LAN IP>:8000`. `AppConfig.validateAndNormalizeApiBaseUrl`
   accepts `http`/`https` only, adds `http://` when missing, defaults the port
   to 8000, and rejects user info, a query, a fragment or any path other than
   `/`. Saved with `AppConfig.saveApiBaseUrl` into SharedPreferences key
   `api_base_url`; "Reset" removes it (`resetApiBaseUrl`).
2. **Build-time default**: `--dart-define=API_BASE_URL=http://<PC LAN IP>:8000`
   when building. Without it the default is `http://10.0.2.2:8000` (emulator).

`AppConfig.load()` runs before `runApp` (`lib/main.dart`) and replaces the
build default with the saved address if one exists. Two further build-time
overrides exist but are not used by the current build instructions:
`DB_API_BASE_URL` (auth and match reads, `AppConfig.dbApiBaseUrl`) and
`PROCESSING_API_BASE_URL` (uploads, jobs, teams, features, clips,
`AppConfig.processingApiBaseUrl`). Both fall back to the single
`apiBaseUrl`.

Note: `ApiClient` (`lib/services/api_client.dart`) caches `ApiConstants.baseUrl`
in a field at construction, but the services do not use `ApiClient`'s field;
they read `ApiConstants.baseUrl` / `AppConfig.processingApiBaseUrl` on every
call, so a Settings change applies immediately without restarting the app.

### 2.6 Running the pipeline from the command line

`main.py` can analyse a file without the API or SQL Server (it reads squads
from `data/rosters/*.json` instead of the database):

```powershell
.venv\Scripts\python.exe main.py "input_videos\test.mp4" --team-a-name "Portugal" --team-a-color "#C8102E" --team-b-name "Egypt" --team-b-color "#FFFFFF" --clip-dir output_videos\clips --no-display
```

The colours here are illustrative. Important differences from an API job:

| Aspect | API job | Command line (default) |
| --- | --- | --- |
| Profile | `PipelineConfig.short_clip()` (reads `features.yaml`) | `PipelineConfig()` - name `"full"`, every frame analysed, every event type returned (`goal_only=False`), `features.yaml` **not read** |
| Frame rate analysed | 12 fps for clips <= 60 s, 10 fps above (capped at the source rate) | every source frame |
| Kit calibration | skipped; the centres confirmed at upload are used (`confirmed_team_centers=True`) | `_calibrate_teams` samples 24-60 frames first |
| Squads | from the database (`_db_squad`) | from `data/rosters` (`squad_for`) |
| Scoreboard codes | from the database (`_team_codes`) | `--team-a-code/--team-b-code`, else `data/rosters` |
| Pitch model | off (`pitch_model=None`, `use_pitch_calibration=False`) | off unless `--pitch-model models/Football_field.pt` |
| OCR debug crops | off | written to `--ocr-debug-dir` (default `output_videos/ocr_debug`) because `debug_ocr=True` |
| GUI window | none | opens an OpenCV window unless `--no-display` |

Every CLI flag is defined in `main.py`, `build_parser()`. The full list:
`video`, `--output`, `--tracking-model`, `--event-model`,
`--legacy-event-model`, `--team-a-name`, `--team-a-color`, `--team-b-name`,
`--team-b-color` (the four team flags are required), `--ocr-debug-dir`,
`--player-image-dir`, `--player-image-url-prefix`, `--clip-dir`,
`--clip-url-prefix`, `--no-highlight-reel`, `--image-size`, `--ball-model`,
`--pitch-model`, `--pitch-stride`, `--team-a-code`, `--team-b-code`,
`--no-scene-cuts`, `--no-commentary`, `--whisper-model`, `--no-reid`,
`--no-spotting`, `--no-scoreboard`, `--scoreboard-every`,
`--no-scorer-captions`, `--no-ocr`, `--write-annotated-video`, `--no-display`.

`--scoreboard-every` is parsed but `process_video` does not read it; the
reader cadence comes from `PipelineConfig.scoreboard_scan_seconds`. The
`--image-size` help text says "defaults to 1280 for 1080p sources and 960
below that", but the code default is `PipelineConfig.detector_size`: 640 for
frame heights <= 720 and 960 above.

Do not run the pipeline casually on a shared machine: a 3-minute 1080p50 clip
took 30-40 minutes of CPU in the measured runs (section 10), and an API job
writes a match into SQL Server.

---

## 3. Feature switches (`features.yaml`)

New on 2026-11-06. One YAML file at the repository root turns analysis
features on or off for **API jobs** and tells the app which event types to
show. Files involved:

| File | Role |
| --- | --- |
| `features.yaml` | The switches, written for non-programmers. |
| `feature_flags.py` | The only reader. Registry `FEATURES`, parser, defaults, warnings. |
| `pipeline_config.py`, `PipelineConfig.short_clip()` | Turns the switches into a job profile. |
| `main.py` | Honours the profile (`apply_name_switches`, gated stages). |
| `backend/routers/features.py` | `GET /api/features` for the app. |
| `backend/routers/processing.py`, `_process_job` | Builds the profile per job; copies `features` into the job result. |
| `lib/services/feature_service.dart` | App-side `FeatureFlags` and `FeatureService.load()`. |
| `lib/screens/ai_match_summary.dart`, `lib/screens/match_highlights_list.dart` | Filter what is listed. |
| `tests/test_feature_flags.py`, `test/feature_flags_test.dart` | Tests. |

### 3.1 File format

One top-level key per feature; each is a mapping:

```yaml
detect_fouls:
  enabled: yes            # yes / no (the only key the code acts on)
  app_rebuild_needed: no  # information only; reported by /api/features
  what_it_does: Finds fouls from player contact and gives each a highlight clip.
  accuracy: Weak. The one checked foul was probably a false alarm and a real one was missed.
```

Parsing rules (`feature_flags.py`):

- `yaml.safe_load`. YAML 1.1 already turns bare `yes`/`no` into booleans.
  `_as_bool` also accepts the strings `yes y true on 1` and `no n false off 0`
  (case-insensitive) and the integers 0/1.
- A feature may also be written as a bare value (`detect_fouls: yes`); the
  code reads `entry.get("enabled")` only when the entry is a mapping.
- `app_rebuild_needed`, `what_it_does`, `accuracy` are passed through to
  `/api/features` and change no behaviour.
- `load_feature_details(path=None)` returns `(details, warnings)`. `details` is
  a list in `FEATURES` order of
  `{key, label, enabled, app_rebuild_needed, what_it_does, accuracy, event_types}`.
- `load_features(path=None, log=True)` returns `{key: bool}` and prints each
  warning as `features.yaml: <warning>` to the server console.
- `visible_event_types(flags)` returns the event types of every switched-on
  feature, in `FEATURES` order.

Fallback behaviour - a bad edit never stops a job:

| Problem | Result | Warning text (abridged) |
| --- | --- | --- |
| File missing | every feature uses its default | `features.yaml not found; every feature uses its default` |
| YAML syntax error, or PyYAML not installed | every default | `features.yaml could not be read (<error>); ...` |
| Top level is not a mapping | every default | `features.yaml is not a list of features; ...` |
| `enabled:` not yes/no, `enabled:` missing, or the whole feature block missing (file otherwise readable) | that feature's default | `<key>: 'enabled' is <value>, not yes/no; using the default (yes/no)` (value printed as `None` when missing) |
| Unknown top-level key (typo) | ignored | `<key>: not a known feature, ignored (check the spelling)` |

Note a subtlety in `load_feature_details`: a per-feature warning is only added
`if raw:` - when the file could not be read at all, only the file-level
warning is reported, not fifteen per-feature ones.

### 3.2 The registry (`feature_flags.FEATURES`)

`Feature(key, label, default, event_types)`. Order is the order the app would
list them. Defaults reproduce the behaviour before the file existed, so
deleting the file changes nothing.

| Key | Label | Default | Event types it shows | Current value in `features.yaml` |
| --- | --- | --- | --- | --- |
| `detect_goals` | Goals | yes | `goal` | yes |
| `goal_scorer_name` | Goal scorer name | yes | - | yes |
| `assister_name` | Assist name | yes | - | yes |
| `detect_shots` | Shots on target | no | `shot` | no |
| `detect_saves` | Saves | no | `save` | no |
| `detect_fouls` | Fouls | yes | `foul` | yes |
| `detect_cards` | Yellow and red cards | no | `yellow_card`, `red_card` | no |
| `detect_penalties` | Penalties | no | `penalty` | no |
| `detect_crosses` | Crosses | no | `cross` | no |
| `detect_passes` | Passes | no | `pass` | no |
| `passer_name` | Passer and receiver names | no | - | no |
| `highlight_clips` | Highlight clips | yes | - | yes |
| `commentary_names` | Names from commentary | yes | - | yes |
| `read_scoreboard` | Score graphic reading | yes | - | yes |
| `action_spotting` | Action-spotting model | yes | - | yes |

With the shipped file, `visible_event_types` is `["goal", "foul"]` (checked
with `python -c` on 2026-11-06; no warnings).

### 3.3 What each switch touches in code

`PipelineConfig.short_clip(debug_ocr=False, features=None)` builds the API
profile. `features=None` means "read `features.yaml` now" (`load_features()`).
The helper `on(key)` is `bool(flags.get(key, False))`; `load_features` always
returns every key, so the `False` default only matters when a caller passes
its own partial dict (tests do).

| Switch | `PipelineConfig` field(s) | What changes in the pipeline when it is **off** |
| --- | --- | --- |
| `detect_goals` | `output_events` (via `visible_event_types`) | Goals are still detected internally (the scoreboard, shot, spotting and merge logic still run) but are filtered out by `goal_output_events(...)` before naming, so they are not named, not clipped, not stored. The final score can still come from the score graphic (`final_score_by_team`), otherwise it is 0-0. |
| `goal_scorer_name` | `name_goal_scorer` | `_improve_goal_scorers` (caption OCR, celebration vote, commentary vote, fusion) is skipped; `apply_name_switches` then clears `jersey_number`, `player_name`, confidences and `identity_source` on the scorer and scorer actors. The per-goal live OCR inside `_resolve_identities` still runs. |
| `assister_name` | `name_assister` | `apply_name_switches` removes actors with role `assister` from goals. `_attach_pass_assisters` and the commentary still compute it first. |
| `detect_shots` | `output_events` | Shots are still detected (they drive goals, saves and score-change attribution) but are not returned/stored. |
| `detect_saves` | `output_events` | Same as shots. |
| `detect_fouls` | `detect_fouls`, `output_events` | `TrackEventDetector` is built with `detect_fouls=False`, so `FoulDetector` is not created at all (`track_events/detector.py`, `__init__`). |
| `detect_cards` | `detect_cards`, `output_events` | `CardDetector` not created. `_process_job` also passes `detect_cards=config.detect_cards` in `args`; `process_video` uses `config.detect_cards and args.detect_cards`. Cards also imply fouls when on (`_foul_for_card`). |
| `detect_penalties` | `detect_penalties`, `output_events` | `TrackEventDetector(detect_penalties=False)` skips `_update_penalty`. |
| `detect_crosses` | `output_events` | Crosses are detected with passes regardless (passes are always on internally), only filtered. |
| `detect_passes` | `output_events`, and gates `name_pass_players` | Passes are **always** tracked (`detect_passes=True` is hard-coded in `process_video` because the last team-mate pass to the scorer is the assist). Off = filtered from output. |
| `passer_name` | `name_pass_players = detect_passes and passer_name` | When off, `_resolve_identities` is skipped for pass/cross events (no shirt OCR spent on them) and `apply_name_switches` clears names on them. It only has an effect when `detect_passes` is on (otherwise passes are already filtered out before naming). |
| `highlight_clips` | `highlight_clips` | `export_highlights` is not called; events carry no `clip_url`/`clip_*` fields; `result.clips` is `[]`, `result.reel` is `None`. |
| `commentary_names` | `use_commentary` | `_transcribe_commentary` is skipped, so no words/mentions; `_commentary_goal_roles`, `attach_to_tracks` and `_apply_commentary` have nothing to work with. Saves 1-15 min per job according to the file. The Whisper model is still **loaded** once into the cache by `_process_job` (`whisper_model_instance=model_cache.whisper()`), just not used. |
| `read_scoreboard` | `read_scoreboard` | `ScoreboardReader` is not created; no score-graphic goals; final score comes from counted goals. |
| `action_spotting` | `use_spotting` | `ActionSpotter` not created; no spotted goals/shots/kick-offs. The spotting model is still loaded into the cache by `_process_job` (`spotting_bundle=model_cache.spotting()`). |

Fields `short_clip` always sets, regardless of switches: `name="short_clip"`,
`short_clip_fps=12.0`, `long_clip_fps=10.0`, `goal_only=True`,
`use_legacy_event_model=False`, `use_pitch_calibration=False`,
`identity_crops=4`, `jersey_ocr_calls=64`, `features=tuple(sorted(flags.items()))`.

`goal_only=True` is what makes `output_events` act as a filter (`main.py`,
`if config.goal_only: pending_events = goal_output_events(pending_events, config.output_events)`).
The name is historical: the profile now returns whatever `output_events`
lists, not only goals.

`process_video` copies `dict(config.features)` into its result as
`features`, and `_process_job` puts it into the job result
(`result["features"]`), so every stored job records the switches it ran with.
That copy lives only in the in-memory job result; it is **not** written to the
database.

### 3.4 Timing: when a change takes effect

| Consumer | When it reads the file | Consequence |
| --- | --- | --- |
| Analysis | Once per job, at the start of `_process_job` (`PipelineConfig.short_clip(debug_ocr=False)`) | The next confirmed job uses the new switches. A job already running is unaffected. No server restart. |
| `GET /api/features` | On every request (`load_feature_details()`) | Immediate. |
| App match summary | Every time `AIMatchSummary` loads (`_loadMatchEvents`, `Future.wait([fetchMatchSummary, FeatureService.load()])`) | Showing/hiding applies to old matches too. |
| App highlights page | Every `_loadContent()` (`initState`, Retry) | Same. |
| Command-line runs | Never (default `PipelineConfig()`) | CLI ignores `features.yaml`. To use it from a script call `PipelineConfig.short_clip()`. |

Turning a detector on later does not fill in old matches: an event type that
was off at analysis time was never stored.

### 3.5 What `app_rebuild_needed` means

It is a note for the person editing the file. `yes` would mean the installed
app has no screen or label for the feature yet, so a developer must change the
Flutter app and ship a new APK before the feature is visible. The code only
passes it through (`load_feature_details` -> `/api/features` ->
`features[].app_rebuild_needed`); the app ignores it. Every shipped feature has
`no`, because the app already filters by raw event type.

### 3.6 App behaviour

`lib/services/feature_service.dart`:

- `FeatureService.load()` does `GET {processingApiBaseUrl}/api/features` with a
  10 s timeout. Any non-200 status, exception or timeout returns
  `FeatureFlags.fallback`.
- `FeatureFlags.fromJson` reads `visible_event_types` (lower-cased into a set)
  and `flags` (only boolean values kept). It ignores `features` and
  `warnings`.
- `FeatureFlags.fallback.visibleEventTypes` = `{goal, foul}` - the server's
  default switches, so an unreachable `/api/features` shows the same types as
  a server running the shipped `features.yaml`.
- `shows(type)` = set membership after trim + lower-case. `isOn(key)` is
  defined but not used by any screen in `lib/` (checked with grep).

`ai_match_summary.dart` skips any occurrence row whose event type
`!features.shows(type)`. `match_highlights_list.dart` drops highlights whose
display type is `pass` or `save` (`_noClipTypes`, never listed there because
they have no clip), and drops categories that are switched off
(`_isSwitchedOn`; display type `Card` is on if `yellow_card` or `red_card` is
shown). Its filter chips (`_filters`) are "All", always "Goals", then one chip
per other category in `_filterLabels` that has at least one visible highlight.

### 3.7 Adding a new switch, end to end

Example: a hypothetical `detect_corners` that shows a new event type
`corner`. Adjust the steps you do not need (a switch that only changes
analysis has no event types and needs no app work).

1. **Registry** - `feature_flags.py`, add to `FEATURES` in the position the
   app should list it:
   `Feature("detect_corners", "Corners", False, ("corner",)),`
   The default must reproduce today's behaviour (usually `False` for a new
   detector).
2. **Config field** - `pipeline_config.py`: add `detect_corners: bool = False`
   to `PipelineConfig` (frozen dataclass; give it a default so the `"full"`
   profile and tests still construct), and in `short_clip` set
   `detect_corners=on("detect_corners")`. If the switch shows an event type,
   nothing else is needed for output: `output_events=tuple(visible_event_types(flags))`
   picks up `"corner"` automatically.
3. **Pipeline use** - `main.py`, `process_video`: pass the field to whatever
   produces the event (for a track rule, a new constructor flag on
   `TrackEventDetector` like `detect_penalties`). Events must use the string
   `"corner"` in `event` so `goal_output_events` lets them through.
   If the event deserves a clip, add it to `CLIP_EVENT_TYPES` and
   `EVENT_PRIORITY` in `highlights/segments.py` (section 8.2). If it has a
   primary participant, add it to `PRIMARY_ROLE` in `main.py` and a branch in
   `describe_event`.
4. **features.yaml** - add the block with `enabled`, `app_rebuild_needed`,
   `what_it_does`, `accuracy`. Keep two-space indentation.
5. **App** - an unknown event type is already handled generically:
   `MatchEvent._eventTitle` falls through to `"<Team> corner"`, and
   `ApiService._highlightType` title-cases it to `Corner`. For a filter chip on
   the highlights page add `'corner': 'Corners'` to `_filterLabels` in
   `match_highlights_list.dart`; for an icon/colour extend `_getEventColor` /
   `_getEventIcon` in `ai_match_summary.dart`. If any of that is needed,
   set `app_rebuild_needed: yes` until the new APK is installed.
6. **Tests** - `tests/test_feature_flags.py`:
   `test_the_shipped_file_parses_without_warnings_and_matches_todays_behaviour`
   will fail until the YAML block exists (a registry key missing from the
   file produces a warning) and, if the new switch is on by default and has an
   event type, until its expected `["goal", "foul"]` list is updated; add a
   profile test like
   `test_switches_reach_the_job_profile`; in the app add a case to
   `test/feature_flags_test.dart` if titles change.
7. Run both test suites (section 9) and, if possible, one job with the switch
   on.

---

## 4. The analysis pipeline

### 4.1 Job lifecycle

All job state lives in the module-level dict `_jobs` in
`backend/routers/processing.py`, guarded by `_jobs_lock`. It is **in memory
only**: a server restart forgets every job (finished matches stay in SQL
Server; their clips stay on disk).

```
 App                         API (request thread)                    footy-cv worker thread
 ---                         --------------------                    ----------------------
 POST submit-file  ------->  save upload to backend/uploads/<uuid>.<ext>
                             _create_job()  status=uploaded, 10%
                             status=queued "Queued for team detection"
                             job_queue.submit(_detect_team_candidates) ----> _detect_team_candidates(job_id)
 <------- 202 {job_id, status:"queued"}                                      status=detecting_teams, 11%
                                                                             sample 24-60 frames over the video
 GET status (every 2 s) -->  _job_snapshot()                                 Tracker.detect_outfield_players
                                                                             kit colours -> rank_team_samples
                                                                             write team_samples/<job>-team-N[-k].jpg
                                                                             status=awaiting_team_confirmation, 13%
 user picks two teams
 POST confirm-teams ------>  _confirmed_team_names() (DB lookup)
                             409 unless awaiting_team_confirmation
                             store names + detected colours (hex)
                             status=queued "Team names confirmed", 14%
                             job_queue.submit(_process_job)  -------------> _process_job(job_id)
 <------- 202                                                                status=processing, 15%
                                                                             PipelineConfig.short_clip()  (reads features.yaml)
 GET status (every 2 s) -->                                                  main.process_video(args, progress_cb)
                                                                               progress 15..90 from the frame loop
                                                                             stage "Saving match summary", 94%
                                                                             _persist_match() -> match_id
                                                                             build result dict
                                                                             status=completed, 100%
                                                                             finally: delete the upload, sweep_media()
 GET result -------------->  result (409 until completed)
 open /summary with matchId -> GET /api/matches/{id}
```

Because `SerialJobQueue` has exactly one worker, **jobs never overlap**: a
second upload's team detection waits until the first job's whole analysis has
finished (tested by `tests/test_pipeline_optimization.py`,
`test_serial_queue_never_overlaps_jobs`). This keeps one copy of each model in
memory and stops CPU contention, at the cost of queueing.

#### Job status values

| `status` | `stage` text set by the code | `progress_percent` | Set in |
| --- | --- | --- | --- |
| `uploaded` | "Video uploaded" | 10 | `_create_job` (immediately overwritten) |
| `queued` | "Queued for team detection" | 10 | `submit_file`, `submit_url` |
| `detecting_teams` | "Finding representative players" | 11 | `_detect_team_candidates` |
| `awaiting_team_confirmation` | "Name the detected teams" | 13 | `_detect_team_candidates` |
| `queued` | "Team names confirmed" | 14 | `confirm_teams` |
| `processing` | "Detecting match events", then "Analysing frame N of M" | 15, then 15-90 | `_process_job`, `report_progress` |
| `processing` | "Saving match summary" | 94 (`PROGRESS_PERSIST`) | `_process_job` |
| `completed` | "Analysis complete" | 100 | `_process_job` |
| `failed` | "Team detection failed" or "Processing failed" | 0 | either worker; `error` holds `str(exception)` |

`report_progress(frame_index, total_frames)` maps the fraction of decoded
frames linearly onto `PROGRESS_ANALYSIS_START`=15 ..
`PROGRESS_ANALYSIS_END`=90. It is called once per batch of 16 analysed
frames. Two consequences worth knowing:

- Commentary transcription runs **before** the frame loop, so the bar sits at
  15% "Detecting match events" for the whole transcription (1-15 minutes per
  `features.yaml`).
- Everything after the loop (spotting inference finish, scoreboard
  attribution, shirt OCR for every event, caption OCR, commentary roles,
  clip encoding) happens while the bar sits at about 90%, until 94% at
  persistence.

#### Team detection (`_detect_team_candidates`)

1. Opens the upload with OpenCV; reads FPS (default 25), frame count, height.
2. `sample_frames = int(min(CALIBRATION_MAX_FRAMES=60, max(CALIBRATION_MIN_FRAMES=24, duration/2)))`
   positions spread with `np.linspace` over the **whole** video; each is read
   by seeking.
3. `_get_tracker()` builds a `Tracker` on the cached `models/best.pt` with
   `image_size=640`; `detect_outfield_players(frames)` returns boxes.
4. For each box: `safe_player_crop(frame, bbox, min_height=max(28, int(h*0.06)), min_width=12)`
   and `kit_color_bgr(frame, bbox)` (masked torso median, `team_assigner/calibration.py`),
   giving a `TeamSample(color, crop, frame, bbox, sharpness)`.
5. `rank_team_samples(samples, min_samples=40, min_per_cluster=12, min_separation=12.0, per_team=6)`
   clusters the colours into two kits with KMeans (`n_clusters=2`,
   `n_init=10`, `random_state=7`) in weighted Lab space
   (`KIT_FEATURE_WEIGHTS = [0.5, 3.0, 1.0]`), and returns
   `(centers_rgb, ranked)` where `ranked[i]` is up to 6 samples of kit `i`,
   the first being the most presentable and the rest coming from different
   frames first. It raises `ValueError` ("Could not find enough clear player
   images...", "The two team kits could not be separated...", "One of the two
   kits was seen too few times...") which becomes a `failed` job with that
   message.
6. Each sample is rendered with `render_team_sample(choice)` (zoomed-out view
   with the player outlined) and written as JPEG quality 90 to
   `backend/static/team_samples/<job_id>-team-<n>.jpg` (first image) and
   `<job_id>-team-<n>-<k>.jpg` (alternatives).
7. The job gets `team_candidates` (see 5.3), `detected_team_colors`
   (= `centers_rgb`, RGB triplets) and `team_sampling_seconds`.

The kit colours confirmed here are reused by the analysis
(`confirmed_team_centers=True` -> `should_calibrate_teams` returns False), so
the run does not calibrate again. `confirm_teams` stores them as hex
(`_rgb_hex`) in `team_a_color`/`team_b_color`; kit 1 is always team A
(`team_id` 1), kit 2 team B (`team_id` 2).

#### What `_process_job` passes to `process_video`

`args` is a `SimpleNamespace` (not argparse), so `process_video` reads most
options with `getattr(args, name, default)`. Values set by the API:

| Attribute | Value |
| --- | --- |
| `video` | the upload path |
| `output` | `backend/static/outputs/<job_id>.mp4` (unused: `write_annotated_video=False`) |
| `tracking_model`, `person_model` | `models/best.pt` path and the cached YOLO instance |
| `ball_model_instance` | cached YOLO for `models/ball_detector_best.pt` if the file exists |
| `ocr_engine` | `model_cache.paddle_ocr()` (may be `None` if PaddleOCR is unavailable) |
| `pipeline_config` | `PipelineConfig.short_clip(debug_ocr=False)` |
| `spotting_bundle`, `reid_embedder`, `whisper_model_instance` | cached models (each may be `None`) |
| `squads` | `{1: _db_squad(team_a), 2: _db_squad(team_b)}` - `[{jersey_no, name, position}]` from `players` |
| `confirmed_team_centers` | `True` |
| `return_pipeline_result` | `True` (returns a dict, not a list) |
| `team_a_name/color`, `team_b_name/color` | from the confirmation |
| `team_a_code`, `team_b_code` | `_team_codes(name)`: `teams.short_code` plus 3-letter alphabetic aliases, comma-joined |
| `ocr_debug_dir` | `None` |
| `player_image_dir`, `player_image_url_prefix` | `backend/static/player_images/<job_id>`, `/static/player_images/<job_id>` |
| `clip_dir`, `clip_url_prefix` | `backend/static/clips/<job_id>`, `/api/processing/clips/<job_id>` |
| `pitch_model` | `None` |
| `detect_cards` | `config.detect_cards` |
| `no_ocr`, `no_scoreboard` | `ocr_engine is None` |
| `no_scorer_captions` | `False` |
| `write_annotated_video` | `False` (a full render was a 327 MB MJPG AVI Android could not decode) |
| `no_display` | `True` |

### 4.2 Frame sampling and fps for API jobs

`pipeline_config.py`:

- `PipelineConfig.analysis_fps(duration)` returns `short_clip_fps` (12.0) when
  `duration <= short_clip_seconds` (60.0), else `long_clip_fps` (10.0).
- `process_video` sets `target_fps = min(source_fps, config.analysis_fps(duration))`.
- `scheduled_frames(total_frames, source_fps, target_fps)` yields **source**
  frame indices with step `source_fps / target_fps`, computed from a running
  float cursor so rounding never drifts (a 25 fps source at 12 fps yields
  0, 2, 4, 6, 8, 10, 13, ...).
- The decode loop still reads **every** frame sequentially
  (`capture.read()`); every decoded frame feeds the scene-cut signature
  (`frame_signature`) and is offered to the spotter (`spotter.wants(frame)` ->
  it keeps only its own 2 fps sample, `spotting_sample_fps`). Only scheduled
  frames are batched (`FRAME_BATCH_SIZE` = 16) and sent to detection,
  tracking, team assignment, events and shirt crops.
- Event `frame` values are always source frame numbers. Seconds are
  `frame / source_fps` everywhere (`_persist_match` re-reads the source fps
  with OpenCV).
- `Tracker(frame_rate=target_fps, ...)` - ByteTrack's lost-track buffer is
  sized from the analysed rate, not the source rate.
- Detector input size: `PipelineConfig.detector_size(height)` = 640 for
  heights <= 720, 960 above (`low_resolution_size`, `high_resolution_size`);
  the ball model uses the same size.
- Thresholds in `track_events` are expressed in seconds and player heights,
  so they hold at any analysed rate (`tests/test_fps_invariance.py`).

The CLI's `"full"` profile asks for 1,000,000 fps, i.e. every source frame.

### 4.3 `process_video` stage by stage

`main.py`, `process_video(args, progress_cb=None)`. Stages in execution order:

| # | Stage | Code | Notes |
| --- | --- | --- | --- |
| 1 | Threads | `torch.set_num_threads(max(1, cpu_count // 2))` | "match the physical core count" |
| 2 | Match config | `MatchConfig.from_user_input(...)` (`match_config.py`) | validates `#RRGGBB` colours and names; raises `ValueError` otherwise |
| 3 | Open video | OpenCV | width, height, fps (default 25), total frames |
| 4 | Models | `_optional_model` for ball/pitch paths; `Tracker(...)`; `PitchCalibrator` only if a pitch model is given and `use_pitch_calibration` | API: pitch off |
| 5 | Legacy LSTM | `EventDetector` + `EventLogic` only if `use_legacy_event_model` or `--legacy-event-model` | off for API |
| 6 | Team assigner | `TeamAssigner(match_config)` (`team_assigner/team_assigner.py`) | per-track vote with `VOTE_DECAY` 0.97, rechecked every `SETTLED_RECHECK_EVERY` 5 |
| 7 | Event detector | `TrackEventDetector(fps, (w,h), ball_assigner, detect_cards, detect_fouls, detect_passes=True, detect_penalties)` | switches from section 3 |
| 8 | Shirt reader | `JerseyReader(debug_dir, engine, max_crops=identity_crops, max_ocr_calls=jersey_ocr_calls)`; `max_ocr_calls=0` with `--no-ocr` | API: 4 crops, 64 calls |
| 9 | Scorer identifier | `ScorerIdentifier(jersey_reader, PlayerIdentityStore(max_crops_per_player, crop_stride=1, min_box_height=min_readable_box_height(h)), frame_height)` | holds crops and possession history |
| 10 | Scoreboard codes | `known_codes = {1: parse_code_option(team_a_code) or expected_codes_for(name), 2: ...}` | DB codes first, then `data/rosters` |
| 11 | Scoreboard reader | `ScoreboardReader(paddle_ocr_fn(engine), fps, (h,w), expected_codes=combined_expected(known_codes), read_every_seconds=2.0, scan_every_seconds=2.0, burst_every_seconds=0.5, burst_duration_seconds=8.0, permissive_after_misses=3)` if `read_scoreboard` and an OCR engine loads | |
| 12 | Spotter | `ActionSpotter(bundle.model, bundle.labels, fps, sample_fps=2.0, clip_len=bundle.clip_len)` if `use_spotting` and the bundle loads | `spotting/spotter.py` |
| 13 | Squads | `_team_squads(args, team_names)`: `args.squads` else `squad_for(name)` | |
| 14 | Identity tracker | `_build_identity_tracker(args, team_names, fps, squads)` | `None` if neither team has a squad nor enrolled gallery images |
| 15 | Commentary | `_transcribe_commentary` if `use_commentary` and not `--no-commentary` | before the frame loop |
| 16 | Kit calibration | skipped for API jobs; CLI `_calibrate_teams` | |
| 17 | Frame loop | decode, scene signatures, spotting samples, batches of 16 -> `process_batch` | see 4.4 |
| 18 | Finish tracks | `track_events.finish(frame_number)` | flushes open shots/fouls |
| 19 | Spotting fusion | `spotter.finish()`, `spotter.events(min_confidence=0.1, classes=USED_CLASSES)`, `fuse_spotted_events(...)` | `spotting/fusion.py` |
| 20 | Commentary to tracks | `attach_to_tracks(mentions, possession_log, fps, identity_tracker)` | `commentary/assign.py` |
| 21 | Learned-model clean-up | `_drop_model_fouls_near_track_fouls`, `_drop_model_goals_near_pitch_goals` (5 s windows) | only matter with the legacy model / pitch |
| 22 | Scoreboard goals | `map_codes_to_teams`, `attribute_goals`, `merge_board_goals` | see 4.6 |
| 23 | Kick-offs | collected for the replay rule, then removed from the event list | |
| 24 | Saves of goals | `withdraw_saves_of_goals` | a shot whose outcome is `goal` was not saved |
| 25 | Sort + assists | sort by frame; `_attach_pass_assisters(events, fps)` | before passes are filtered |
| 26 | Output filter | `goal_output_events(events, config.output_events)` when `goal_only` | the switches in section 3 |
| 27 | Identities | `_resolve_identities` for each event, goals first, then `CLIP_EVENT_TYPES`, then the rest (OCR budget order); skipped for pass/cross when `name_pass_players` is off | see 4.7 |
| 28 | Caption OCR fn | `paddle_ocr_fn(engine)` unless `--no-ocr`/`--no-scorer-captions` | |
| 29 | Commentary goal roles | `_commentary_goal_roles(...)` | |
| 30 | Goal scorers | `_improve_goal_scorers(...)` if `name_goal_scorer` | caption, celebration, live, commentary fused |
| 31 | Commentary roles | `_apply_commentary(...)` | assisters, fouler/fouled, unnamed scorers |
| 32 | Replays | `drop_replayed_goals(events, fps, kick_off_frames)` | after naming, see 4.6 |
| 33 | Keepers | `_name_keepers(events, team_squads, team_names)` | roster number-one keeper for saves |
| 34 | Name switches | `apply_name_switches(events, config)` | |
| 35 | Descriptions | `event["description"] = describe_event(event)` | |
| 36 | Highlights | `export_highlights(...)` if `highlight_clips` | see 4.9 |
| 37 | Metrics | `PipelineMetrics` phases and counts | see 9.4 |
| 38 | Final score | `final_score_by_team(events, scoreboard_reader, code_to_team)` | graphic score if both codes map, else counted goals |
| 39 | Result | dict with `events`, `score_by_team_id`, `metrics`, `player_tracks`, `features`, `commentary` | returned when `return_pipeline_result` |

### 4.4 Inside the frame loop (`process_batch`)

For each batch of up to 16 scheduled frames:

1. `update_scenes()` runs `find_boundaries` over the rolling list of frame
   signatures (kept to the last 64-96) and merges new boundaries into a
   `SceneTimeline` (`scene/cuts.py`, twin-comparison on HSV histograms,
   `LOW_THRESHOLD` 0.08, `HIGH_THRESHOLD` 0.28).
2. Frames inside a dissolve are **skipped** (`scenes.in_transition`) - no
   detection, team votes or events, because blended frames give garbage kit
   colours.
3. The first analysed frame at or after each new scene start is a **reset**:
   `track_events.reset_scene(frame)` (may emit pending events),
   `team_assigner.forget_players(...)`, `identity_tracker.end_scene()`,
   `pitch_calibrator.reset()`; the tracker restarts ids (`reset_offsets`).
4. `tracker.get_object_tracks(batch, reset_offsets, skip_offsets)` runs YOLO on
   the batch (person model at `DETECT_CONFIDENCE` 0.25, ball at
   `BALL_MODEL_CONFIDENCE` 0.15 with the dedicated ball model) and ByteTrack;
   `add_position_to_tracks`.
5. Per frame: the scoreboard reader `update(frame_number, frame)` (it decides
   internally whether to OCR this frame); `_assign_teams` (kit colour ->
   `team`, `team_name`, `team_confidence` on each player);
   `identity_tracker.observe(...)`; `track_events.update(frame, players, ball_bbox, referees, frame_image, pitch)`
   which returns new events; `scorer_identifier.observe_players(...)` for the
   ball holder and every player tall enough to read
   (`min_readable_box_height`); possession recorded for the scorer history
   and the whole-match `possession_log` used by commentary; a recorded shot
   triggers `scoreboard_reader.trigger_burst(frame)` (read every 0.5 s for 8 s).
6. `progress_cb(frame_number, total_frames)` after each batch.

### 4.5 Track-derived events (`track_events/`)

`track_events/detector.py`, `TrackEventDetector`. All distances are in player
heights (running median of box heights) so thresholds hold at any
resolution. Each event is emitted by `_emit(...)` with `source: "tracks"`.

| Event | Rule (summary of the module docstring) | Key constants (`track_events/detector.py` unless noted) |
| --- | --- | --- |
| `pass` | possession moves to a team-mate and the ball really travels | `CONFIRM_FRAMES` 2, `MAX_FLIGHT_SECONDS` 2.5, `LOOSE_BALL_TIMEOUT_SECONDS` 3.0, `MIN_PASS_DISTANCE` 0.8 heights, `ID_SWITCH_MAX_GAP_SECONDS` 0.25 |
| `cross` | long, lateral, airborne pass landing near the goalkeeper | `CROSS_MIN_LATERAL` 3.5, `CROSS_MIN_SECONDS` 0.45, `CROSS_MIN_ARC` 0.4, `CROSS_MAX_KEEPER_DISTANCE` 4.5 |
| `shot` | ball leaves a player fast, heading at the keeper; always `details.on_target=True` | `SHOT_MIN_SPEED` 6.0 heights/s, `SHOT_MAX_ANGLE_DEGREES` 35, `SHOT_KEEPER_REACH` 2.5, `KEEPER_MEMORY_SECONDS` 2.0 |
| `save` | a shot the keeper gathers or that reverses at him | (shot constants) |
| `goal` (from a shot) | after a shot, the ball entered the goal mouth (found near the keeper by `track_events/goal_mouth.py`), never came out, nobody held it, then vanished or came to rest; judged once `GOAL_JUDGE_SECONDS` passed, at a cut, or at the end | `GOAL_JUDGE_SECONDS` 2.5, `GOAL_VANISH_SECONDS` 0.6, `GOAL_REST_SECONDS` 0.4, `GOAL_HOLD_SECONDS` 0.5, `MOUTH_SEARCH_HEIGHTS` 8.0 |
| `penalty` | long-still ball, taker beside it, keeper at spot distance, others away, then a kick at the keeper | `PENALTY_STILL_SECONDS` 1.5, `PENALTY_KEEPER_DISTANCE` (2.5, 8.0), `PENALTY_MIN_OTHERS` 4, `PENALTY_KICK_SPEED` 4.0 |
| `foul` | `track_events/fouls.py`: opponents in contact near the ball, then a player down or play stopped; every card also implies a foul | `CONTACT_HEIGHTS` 0.8, `CONTACT_MIN_IOU` 0.15, `BALL_NEAR_HEIGHTS` 2.5, `OUTCOME_SECONDS` 3.0, confidences 0.75/0.6/0.5 |
| `yellow_card`, `red_card` | `track_events/cards.py`: a card-coloured blob held up near a tracked referee | `YELLOW_HUE` (18,36), `CARD_MIN_BOX_HEIGHT` 48 px referee, `CARD_COOLDOWN_SECONDS` 6.0 |

Per-type cooldowns: `EVENT_COOLDOWN_SECONDS = {"shot": 1.0, "save": 1.0, "penalty": 10.0, "cross": 0.5, "goal": 10.0, "foul": 3.0}`.
With a pitch calibration (off for API jobs) extra metre-based rules apply
(`PITCH_*` constants, goal-line goals with `source: "pitch"`).

### 4.6 How goals are merged

Goals can come from five sources; the `source` field records which.

| `source` | Produced by | Strength order in `drop_replayed_goals` |
| --- | --- | --- |
| `tracks` | `TrackEventDetector._judge_goal` (ball into the mouth after a shot) | 0 (strongest) |
| `pitch` | pitch goal line (only with a pitch model) | 0 |
| `scoreboard` | `scoreboard.attribute_goals` from a score change | handled separately, always kept |
| `spotting` | `spotting.fuse_spotted_events` | 1 (`SOURCE_RANK`) |
| `model` | legacy LSTM via `EventLogic` | 2 |

Order of operations:

1. **Spotting fusion** (`spotting/fusion.py`, `fuse_spotted_events`). A spotted
   goal within `SAME_GOAL_SECONDS` 5 of a video goal only adds its confidence.
   Otherwise it becomes a goal alone at `GOAL_ALONE` >= 0.5, or at
   `GOAL_WITH_SHOT` >= 0.15 when a shot (tracked or spotted) lies
   `SHOT_BEFORE_GOAL_SECONDS` 5 before / `SHOT_AFTER_GOAL_SECONDS` 1 after it -
   that shot's shooter becomes the scorer and the shot's `details.outcome` is
   set to `goal`. A spotted shot >= `SHOT_MIN` 0.3 with no tracked shot within
   `SAME_SHOT_SECONDS` 2 is added as a shot with no participants. Spotted
   kick-offs >= `KICK_OFF_MIN` 0.3 are kept for the replay rule.
2. **Score graphic** (`scoreboard/attribution.py`, `attribute_goals`). The
   reader (`scoreboard/reader.py`) believes a new score only after several
   agreeing reads, and only a rise of one or two goals for one team counts. For
   each score change, the window runs from the last read of the old score
   (minus `GRAPHIC_LAG_SECONDS` 6, or `DEFAULT_LOOKBACK_SECONDS` 60 if the old
   score was never read) to the first read of the new one, padded by
   `WINDOW_PAD_SECONDS` 1. Inside it, in order: an already detected goal (the
   earliest); the latest shot by the scoring team; the latest shot by a player
   read weakly (< `OTHER_TEAM_MAX_CONFIDENCE` 0.5) as the other team; the
   latest shot by anyone (shooter not named if confidently the other team);
   the last scoring-team player in possession; nothing. The produced goal has
   `source: "scoreboard"`, `confidence` 0.95 (with a source event) or 0.85,
   and `details` = `score_before`, `score_after` (e.g. `"POR 0-1 MCI"`),
   `scoreboard_code`, `scoreboard_frame`, `window_start_frame`,
   `attribution` (human-readable reason), `detected_goal`, `detected_by`.
   `merge_board_goals` then removes other proposals inside a score change's
   window (they are the same goal) and keeps every other proposal.
3. **Saves of goals** - `withdraw_saves_of_goals` drops a save at the same
   frame as a shot whose outcome is `goal`.
4. **Naming** (4.7) happens next, deliberately before replay removal.
5. **Replays** (`track_events/goal_merge.py`, `drop_replayed_goals`). Non-board
   goals are visited strongest source first, then in time order. A goal within
   `GOAL_REPLAY_SECONDS` 20 s of a kept goal is a replay **unless** a spotted
   kick-off lies between them, or both goals have different clearly named
   scorers (`named_scorer`: a caption or commentary source, not unconfirmed,
   `scorer_confidence` >= `CLEAR_NAME_CONFIDENCE` 0.7). A goal within 20 s of
   a scoreboard goal (and not differently named) is that goal's replay. When a
   replay names the scorer and the original does not, `_adopt_scorer` moves the
   name over (except across teams for a scoreboard goal).
   `details.replays_dropped` counts removed replays.
6. **Final score** (`main.py`, `final_score_by_team`): if the reader is active,
   has a state and both codes map to teams 1 and 2, the graphic's latest score
   wins; otherwise goals are counted per `team_id`. `_persist_match` writes it
   into `plays.score`.

### 4.7 Naming players

Names flow from several sources into the per-event `actors` list. The shirt
itself is never read for a name (`JerseyReader` reads numbers only).

| Source | Code | What it gives | Weight / threshold |
| --- | --- | --- | --- |
| Shirt-number OCR | `scorer_identifier/jersey_reader.py` (`JerseyReader.read`), via `ScorerIdentifier.identify_track` / `identify_participants` | `jersey_number`, `jersey_confidence` per track, read once and cached; budget `max_ocr_calls` (API 64), `max_crops` (API 4); needs >= 2 supporting frames (`min_supporting_frames`), candidate conf >= 0.50, consensus >= 0.58 | PaddleOCR PP-OCRv5 mobile det/rec, `enable_mkldnn=False` (required on this stack) |
| Identity posterior | `identity/posterior.py`, `IdentityTracker` (built by `_build_identity_tracker`, applied by `_apply_identity`) | per-track probability over the squad from shirt reads, commentary name evidence, enrolled appearance (OSNet re-ID) and a goalkeeper prior; one name per on-screen player (Hungarian assignment); tracklets re-linked across cuts | names only at `NAME_MIN_PROBABILITY` 0.7; `identity_source: "identity"` |
| Broadcast caption | `scorer_identifier/goal_scorer.py`, `read_caption_candidates` | words in the lower third / beside the score bug, from the goal to `CAPTION_SECONDS` 45 later, every 1.5 s, max 30 reads, stop after 2 agreeing reads; matched to both squads with difflib >= `CAPTION_MIN_RATIO` 0.84 | `WEIGHT_CAPTION` 3.0; skipped when the preliminary fusion is already decisive (>= `CAPTION_SKIP_CONFIDENCE` 0.85 from >= 2 sources) |
| Celebration close-ups | `celebration_candidates` | shirt numbers of scoring-team players seen large for `CELEBRATION_SECONDS` 30 after the goal (or 10 s past the graphic change) | `WEIGHT_CELEBRATION` 1.5 |
| Live read | `live_candidate` | the shooter's / last holder's number | `WEIGHT_LIVE_SHOT` 1.0, `WEIGHT_LIVE_POSSESSION` 0.6 |
| Commentary | `commentary/` (`transcribe_video` -> `find_mentions` -> `goal_roles`, `foul_roles`, `attach_to_tracks`) | scorer, assister, fouler, fouled; name evidence on the track holding the ball when a name was said | scorer vote `WEIGHT_COMMENTARY` 3.5 x confidence; English cue words only |
| Roster keeper | `_name_keepers` | a `goalkeeper` actor without a number gets the squad's number-one keeper (shirt 1 if a GK, else lowest-numbered GK) | confidence 0.6, `identity_source: "roster keeper"` |
| Database roster | `_persist_match`, `_roster_player` | the **stored** name: `players.name` for (team, shirt number) | final authority at persistence |

**Goal scorer fusion** (`main.py`, `_improve_goal_scorers`;
`scorer_identifier/goal_scorer.py`, `fuse_scorer`). Candidates' weights are
summed per (team, shirt). A number the squad does not have is multiplied by
`NOT_IN_SQUAD_FACTOR` 0.3; otherwise by `POSITION_PRIOR` (FW 1.2, MF 1.0,
DF 0.7, GK 0.1). Confidence = `share * (0.5 + 0.5 * min(1, best / STRENGTH_FOR_FULL_CONFIDENCE=2.5))`.
A caption-backed winner at least `CAPTION_LEAD_FACTOR` 1.5x the runner-up is
raised to at least 0.7. Below `UNCONFIRMED_BELOW` 0.7 the result is
`unconfirmed`: the number is kept but no name is attached, and the
description reads `"#9 (unconfirmed) scored"`. Captions and commentary search
**both** squads; a confident name from the other squad corrects the goal's
team (`_correct_goal_team`, `details.team_corrected`), except when the score
graphic decided the team (`_team_from_scoreboard`). A goal with no scorer
actor gets one inserted first (`actors.insert(0, ...)`), so persistence stores
the scorer before any assister. `_back_caption_with_shared_name` moves a
shared spoken surname ("Romero") onto the player the caption names.

**Assister**: `_attach_pass_assisters` takes the last pass/cross within
`ASSIST_LOOKBACK_SECONDS` 15 before the goal whose receiver is the scorer's
track and whose passer is a team-mate (`details.assist_from = "detected pass"`).
`_apply_commentary` can then set the assister from commentary
(`details.assister_from = "commentary"`) if the named player is on the goal's
team.

**Commentary** details (`commentary/`):

- `transcribe.py`: ffmpeg extracts 16 kHz mono WAV; faster-whisper
  `WhisperModel(size, device="cpu", compute_type="int8", download_root=models/whisper)`
  with `word_timestamps=True, vad_filter=True, beam_size=5`. Default size
  `"small"` (`PipelineConfig.whisper_model`, `--whisper-model`).
- `mentions.py`: name keys per squad player (full name, surname with particle,
  unique first names of >= 5 letters); casefolded and accent-free; fuzzy
  >= 0.84 for long keys; surnames that are common English words count only with
  the first name; cue words (scoring, passing, foul, victim/agent markers)
  around each mention within `CUE_WINDOW` 4 words; "ruled out" / "stands"
  phrases.
- `assign.py`: a mention goes to the holder between 2.5 s before and 0.5 s
  after it (speech lag 0.7 s) if teams agree; goal windows 10 s before /
  12 s after; foul windows 3 s before / 15 s after; `ROLE_MIN_SCORE` 1.0.
- `_apply_commentary` never overrides a caption-led scorer
  (`"caption" in scorer_sources`); it fills a scorer only when the fusion left
  none or left it unconfirmed. Measured reason: on Levante 3-3 Sevilla the
  commentary named 4 of 6 scorers right and 1 wrong; captions named all six.

**What the database stores** (`_persist_match`): for every actor it looks up
`players` by (team from the actor's `team_name`, `jersey_number`), creating an
unnamed slot if absent (`_roster_player(..., create_slot=True)`), and **replaces**
`actor["player_name"]` with that row's name (or `None`). An `unconfirmed`
scorer gets no player row link and no name. So the name stored in
`occurrences.detected_player_name` is always the database roster's name for
that shirt, whatever source found the number. A seeded squad is therefore a
precondition for names in the app.

### 4.8 Event dictionary shape

Events are plain dicts. Fields by stage (all optional unless stated):

```jsonc
{
  "event": "goal",              // required: goal | shot | save | pass | cross | foul |
                                //   yellow_card | red_card | penalty | kick_off (internal)
  "frame": 106,                 // required: source frame number of the moment
  "end_frame": 106,             // last frame of the event (shots, fouls span time)
  "confidence": 0.95,           // 0..1, rule- or model-specific
  "team_id": 2,                 // 0 unknown, 1 = team A (kit 1), 2 = team B (kit 2)
  "team_name": "Man City",      // "unknown_team" when unknown (never shown; stored as NULL)
  "source": "scoreboard",       // tracks | pitch | scoreboard | spotting | model
  "participants": [             // from the detector: who took part, before identification
    {"role": "scorer", "track_id": 41, "team_id": 2, "team_name": "Man City", "team_confidence": 0.71}
  ],
  "details": {                  // free-form evidence, varies by source and stage, e.g.
    "score_before": "POR 0-0 MCI", "score_after": "POR 0-1 MCI",
    "attribution": "goal detected (tracks), confirmed by the score graphic",
    "on_target": true, "outcome": "goal",            // shots
    "scorer_evidence": [ {"source": "caption", "team_id": 2, "jersey_number": 9,
                          "player_name": "...", "weight": 3.6} ],
    "assist_from": "detected pass", "assister_from": "commentary",
    "team_corrected": {"from": "Porto", "by": ["caption"]},
    "commentary": "ruled out? commentator: '...'", "replays_dropped": 1
  },
  "actors": [                   // after _resolve_identities: every participant, identified
    {"role": "scorer", "track_id": 41, "team_id": 2, "team_name": "Man City",
     "team_confidence": 0.71, "jersey_number": 9, "jersey_confidence": 0.82,
     "player_name": "Erling Haaland", "name_confidence": 0.82,
     "identity_source": "identity", "player_image": "/static/player_images/<job>/track-41-frame-120.jpg",
     "scorer_confidence": 0.86, "scorer_sources": ["caption", "celebration"],
     "unconfirmed": false},
    {"role": "assister", "jersey_number": 17, "player_name": "...", "identity_source": "commentary"}
  ],
  "scorer": { /* copy of the primary actor (historical name: also used for non-goals) */ },
  "description": "GOAL — Man City — #9 Erling Haaland scored (POR 0-1 MCI)",
  "clip_path": "...\\backend\\static\\clips\\<job>\\goal-01.mp4",
  "clip_url": "/api/processing/clips/<job>/goal-01.mp4",
  "clip_index": 1,
  "clip_start_seconds": 0.0, "clip_end_seconds": 9.2, "clip_offset_seconds": 4.24
}
```

The JSON above is illustrative (the values and names are made up); field
names are exact. `ocr_error` may also appear on actors (from
`identify_track`). `said` / `said_at_seconds` appear on commentary-sourced
actors (`_role_record`).

### 4.9 Roles per event type

| Event | Roles in `participants`/`actors` | Primary role (`PRIMARY_ROLE` in `main.py`) |
| --- | --- | --- |
| `goal` | `scorer` and `goalkeeper` (track goals, `_judge_goal`); `scorer` only (scoreboard and spotting goals); `assister` added later by `_attach_pass_assisters` or the commentary | `scorer` (goal handling does not use `PRIMARY_ROLE`) |
| `shot` | `shooter`, `goalkeeper` | `shooter` |
| `save` | `goalkeeper`, `shooter` | `goalkeeper` |
| `pass`, `cross` | `passer`, `receiver` | `passer` |
| `foul` | `fouler`, `fouled` | `fouler` |
| `yellow_card`, `red_card` | `player` (booked), `referee` | `player` |
| `penalty` | `taker`, `goalkeeper` (when found) | `taker` |

`occurrences.role` stores these strings (`VARCHAR(20)`).

### 4.10 Descriptions

`main.py`, `describe_event` (used for the console and for `events.description`
of non-goals) and `scorer_identifier/identity.py`, `format_goal_event` (goals):

| Event | Format |
| --- | --- |
| goal | `GOAL — <team> — #<n> <name> scored`, `... — #<n> scored`, `... — #<n> (unconfirmed) scored`, `... — <name> scored`, `... — scorer number unavailable`; plus ` (<score_after>)` in `describe_event` |
| pass/cross | `PASS — <team> — #8 Name → #10 Name`; `PASS — <team>` when nobody is named |
| shot | `SHOT — <team> — <shooter> (<outcome>)` |
| save | `SAVE — <team> — <keeper> denied <team> <shooter>` |
| penalty | `PENALTY — <team> — taken by <taker>` |
| card | `YELLOW CARD — <team> — <player>` / `RED CARD ...` |
| foul | `FOUL — <team> — <fouler> on <fouled>` |

An actor label is `#<n>`, else `track <id>`, plus the name when known
(`_actor_label`). `_persist_match` regenerates the description from the
database names before storing (truncated to 500 characters); for goals it
calls `format_goal_event` directly, so the stored goal description has no
score suffix.

### 4.11 Highlights export

`highlights/segments.py` plans, `highlights/export.py` writes,
`highlights/encoder.py` encodes.

- **Which events get clips**: `CLIP_EVENT_TYPES = {"goal", "foul", "shot", "cross", "penalty", "yellow_card", "red_card"}`.
  Passes (hundreds per match) and saves (owner decision 2026-09-24: the shot's
  clip already shows it) get none. Only events that survived the output filter
  are planned, so with the shipped switches only goals and fouls get clips.
- **Window per event** (`event_window`): start = first frame of the latest
  camera shot that began between `LEAD_MAX_SECONDS` 6 and `LEAD_MIN_SECONDS` 2
  before the event, else `LEAD_DEFAULT_SECONDS` 5 before. End = for most
  events the end of the first shot closing between `TAIL_MIN_SECONDS` 1.5 and
  `TAIL_MAX_SECONDS` 4 after `end_frame`, else `TAIL_DEFAULT_SECONDS` 3; for a
  goal the end of the last shot closing between `GOAL_TAIL_MIN_SECONDS` 3 and
  `GOAL_TAIL_MAX_SECONDS` 5, else `GOAL_TAIL_DEFAULT_SECONDS` 4. Trimmed to
  `MAX_CLIP_SECONDS` 10 (owner rule), lead first (never below 2 s), then tail.
- **Placement** (`plan_clips`): events in `EVENT_PRIORITY` order
  (`goal, penalty, red_card, yellow_card, foul, shot, cross`). An event inside
  a placed clip joins it; else a clip within `MERGE_GAP_SECONDS` 1 grows to the
  union if that fits 10 s and hits no other clip; else a new clip cut back so
  nothing overlaps. A goal and the shot before it share one clip.
- **Files**: `<clip_dir>/<primary_type>-<NN>.mp4` (e.g. `goal-01.mp4`,
  `foul-02.mp4`); with ffmpeg the clips are re-encoded with x264
  (`CRF` 23, `PRESET` veryfast, AAC `AUDIO_BITRATE` 128k, bitrate capped at the
  source's and at `BITRATE_CEILINGS` 8/5/2.5 Mbit/s for 1080/720/lower); the
  reel `reel.mp4` is a stream-copy concat of the clips, written only with
  ffmpeg and only when there are >= 2 clips. Without ffmpeg: silent OpenCV
  clips, no reel.
- **Fields added to each clipped event**: `clip_path`, `clip_url` (if a URL
  prefix is given), `clip_index`, `clip_start_seconds`, `clip_end_seconds`,
  `clip_offset_seconds`.
- `clips_from_events(events)` (used for `result.clips`) groups events by
  `clip_index`.
- Clips are served by `GET /api/processing/clips/{job_id}/{file_name}` with
  byte-range support (ExoPlayer needs it). Clips, reels, player images and team
  samples are never deleted (`backend/cleanup.py`).

### 4.12 The job result (`GET /api/processing/result/{job_id}`)

Built in `_process_job`; kept in memory only.

| Key | Content |
| --- | --- |
| `db_response` | `{"success": true, "matchId": <int>}` (the app reads `matchId` from here) |
| `match_id` | same id |
| `events` | the final event dicts (4.8) |
| `event_counts` | `{event_type: count}` |
| `shots_on_target` | `{team_name: count}` for shots with `details.on_target` (empty unless shots are switched on) |
| `pass_summary` | `track_events.summarise_passes`: `{team_name: {"passes": n, "crosses": n}}` (empty unless passes/crosses are switched on) |
| `score_sheet` | `[{"team_name", "score"}]` for team ids 1 and 2 |
| `players` | one entry per goal: `team_name`, `jersey_number`, `player_name`, `player_image`, `name_source` (= `identity_source`), `assister` (`{jersey_number, player_name, name_source}` or null), `time_seconds`, `clip_url` |
| `clips` | `clips_from_events(events)`: `[{index, url, start_seconds, end_seconds, event_types}]` |
| `reel` | `{"url", "size_bytes"}` or null (`_reel`) |
| `pipeline_metrics` | `{"phases_seconds": {...}, "counts": {...}}` plus `team_sampling` and `persistence` phases |
| `features` | the switches this job ran with (`{key: bool}`) |

`process_video` also returns `player_tracks` (identity summary) and
`commentary` (`language`, `words`, `mentions`, `attached`), but `_process_job`
does not copy them into the job result.

### 4.13 Files, retention and clean-up

| Path | Written by | Deleted? |
| --- | --- | --- |
| `backend/uploads/<uuid>.<ext>` | `submit_file` / `submit_url` | yes: when the job completes or fails (`_release_job_files` -> `cleanup.delete_upload`), or by `sweep_media()` at startup and after each job once older than `FOOTY_UPLOAD_MAX_AGE_HOURS` (24 h) and not used by an active job. Only uuid-named video files are touched. |
| `backend/static/team_samples/<job>-team-N[-k].jpg` | team detection | never |
| `backend/static/player_images/<job>/track-<id>-frame-<n>.jpg` | `_save_scorer_image` | never |
| `backend/static/clips/<job>/*.mp4`, `reel.mp4` | `export_highlights` | never |
| `backend/static/outputs/` | old annotated renders (now off) | never (outside the retention rule) |
| `backend/static/ocr_debug/` | only with `debug_ocr` (CLI) | never |
| `data/player_gallery/<team key>/<shirt>/` | `/api/players/.../reference-images`, `scripts/enroll_player.py` | by `DELETE` endpoint only |
| `.cache/paddlex` | PaddleOCR model cache | n/a |

---

## 5. REST API reference

Base URL: `http://<PC LAN IP>:8000`. All bodies are JSON unless stated. Error
responses from `HTTPException` have the FastAPI shape `{"detail": "<message>"}`;
validation errors (422 from Pydantic) have `{"detail": [ {loc, msg, type, ...} ]}`.
No endpoint requires authentication.

### 5.1 Endpoint index

| Method | Path | Router / function | Called by (app) | Called by (web page) |
| --- | --- | --- | --- | --- |
| GET | `/` | `backend/app.py`, `root` | - | (serves it) |
| GET | `/health` | `backend/app.py`, `health` | - (only manual checks) | - |
| GET | `/static/{path}` | `StaticFiles` mount | images (`KitSampleViewer` via `_imageUrls`) | yes |
| GET | `/docs`, `/redoc`, `/openapi.json` | FastAPI built-in | - | - |
| POST | `/api/users/register` | `users.register` | `AuthService.register` (`signup_screen.dart`) | - |
| POST | `/api/users/login` | `users.login` | `AuthService.login` (`login_screen.dart`) | - |
| GET | `/api/users/{user_id}` | `users.get_user` | `AuthService.fetchUser` (defined; no screen calls it - grep) | - |
| POST | `/api/processing/submit-file` | `processing.submit_file` | `ProcessingService.submitVideoFile` (`ai_video_upload.dart`) | yes |
| POST | `/api/processing/submit-url` | `processing.submit_url` | `ProcessingService.submitVideoUrl` (defined; no screen calls it - grep) | yes |
| GET | `/api/processing/status/{job_id}` | `processing.job_status` | `ProcessingService.getStatus` (`ai_processing_screen.dart`) | yes |
| POST | `/api/processing/confirm-teams/{job_id}` | `processing.confirm_teams` | `ProcessingService.confirmTeams` | yes |
| GET | `/api/processing/result/{job_id}` | `processing.job_result` | `ProcessingService.getResult` | yes |
| GET | `/api/processing/clips/{job_id}/{file_name}` | `processing.job_clip` | `ClipVideoPlayer` (`clip_video_player.dart`) | yes |
| GET | `/api/matches` | `processing.get_matches` | `ApiService.fetchRecentMatches` (home, predictions, highlights) | - |
| GET | `/api/matches/{match_id}` | `processing.get_match_summary` | `ApiService.fetchMatchSummary` (`ai_match_summary.dart`, highlights) | - |
| GET | `/api/teams` | `teams.list_teams` | `ProcessingService.listTeams` | yes |
| GET | `/api/features` | `features.list_features` | `FeatureService.load` | - |
| GET | `/api/players/{player_id}/reference-images` | `players.list_reference_images` | - | - |
| POST | `/api/players/{player_id}/reference-images` | `players.add_reference_images` | - | - |
| DELETE | `/api/players/{player_id}/reference-images/{image_id}` | `players.delete_reference_image` | - | - |

### 5.2 Users (`backend/routers/users.py`)

Schemas in `backend/schemas.py`.

#### `POST /api/users/register` -> 201

Request (`RegisterRequest`, aliases accepted in brackets):

| Field | Type | Rules |
| --- | --- | --- |
| `username` (`Username`, `fullName`) | string | 2-100 chars, stripped |
| `email` (`Email`) | email | `EmailStr`; stored lower-cased |
| `password` (`Password`) | string | 6-128 chars; stored as a bcrypt hash |

Response (`AuthResponse`):

```json
{"success": true, "message": "Account created successfully.",
 "user": {"userId": 7, "username": "sam", "email": "sam@example.com"}}
```

Errors: 409 `"Email or username is already registered."` (case-insensitive
check on both); 422 validation. Writes a `users` row and a `user_logs` row
(`action="register"`, client IP).

#### `POST /api/users/login` -> 200

Request (`LoginRequest`): `email` (`Email`), `password` (`Password`).
Response: same `AuthResponse` shape with `"message": "Login successful."`.
Errors: 401 `"Invalid email or password."`. Writes `user_logs`
(`action="login"`). No token or session is issued; the app keeps the user id
locally (`SessionService.saveLogin`).

#### `GET /api/users/{user_id}` -> 200

Response (`UserResponse`): `{"userId": 7, "username": "sam", "email": "sam@example.com"}`.
Errors: 404 `"User not found."`.

### 5.3 Processing (`backend/routers/processing.py`)

#### `POST /api/processing/submit-file` -> 202

`multipart/form-data`:

| Field | Type | Notes |
| --- | --- | --- |
| `video` | file | extension must be `.mp4`, `.mov`, `.avi` or `.mkv` (case-insensitive), else 415 |
| `user_id` | int, optional | stored on the match if the user exists |

The upload is streamed to `backend/uploads/<uuid><ext>` in 1 MiB chunks; more
than 2 GiB (`MAX_UPLOAD_BYTES`) -> 413 and the partial file is deleted.

Response: `{"job_id": "<uuid>", "status": "queued"}`.

#### `POST /api/processing/submit-url` -> 202

JSON (`VideoUrlRequest`): `{"video_url": "<http(s) URL>", "user_id": 7}`
(`user_id` optional). The server downloads with `httpx.stream(..., follow_redirects=True, timeout=90)`
inside the request (the request blocks for the whole download) and saves it
as `<uuid>.mp4` whatever the source format. Errors: 413 over 2 GiB; a failed
download raises (HTTP status error or network error) and FastAPI returns 500.
Response as for `submit-file`.

#### `GET /api/processing/status/{job_id}` -> 200

```json
{
  "job_id": "1b4e...",
  "status": "awaiting_team_confirmation",
  "stage": "Name the detected teams",
  "progress_percent": 13,
  "team_candidates": [
    {"cluster_id": 1, "label": "Detected team 1",
     "sample_image": "/static/team_samples/1b4e...-team-1.jpg",
     "sample_images": ["/static/team_samples/1b4e...-team-1.jpg",
                       "/static/team_samples/1b4e...-team-1-1.jpg"]},
    {"cluster_id": 2, "label": "Detected team 2",
     "sample_image": "/static/team_samples/1b4e...-team-2.jpg",
     "sample_images": ["/static/team_samples/1b4e...-team-2.jpg"]}
  ],
  "error": null
}
```

Only these six keys are returned. `team_candidates` is `null` until team
detection finishes; `sample_images` (up to 6 per kit, first = `sample_image`)
lives inside each candidate, not at the top level. Status values and
percentages: section 4.1. Errors: 404 `"Processing job not found."` (also
after a server restart).

#### `POST /api/processing/confirm-teams/{job_id}` -> 202

JSON (`TeamConfirmationRequest`):

| Field | Type | Meaning |
| --- | --- | --- |
| `team_1_id` | int, optional | `teams.team_id` for kit 1 (`team_candidates[0]`) |
| `team_2_id` | int, optional | for kit 2 |
| `team_1_name` | string <= 100, optional | typed name (older clients); used only when the id is absent |
| `team_2_name` | string <= 100, optional | |

Resolution (`_confirmed_team_names`): an id must exist (`db.get(Team, id)`),
else 422 `"Choose a team from the list."`; the stored `team_name` is used. A
name without an id is mapped with `find_team` (literal, alias, normalised) to
the existing team's stored name, or kept as typed if nothing matches; an
empty name -> 422 `"Choose a team for each kit."`. The two names compared
case-insensitively must differ, else 422
`"Choose a different team for each kit."`.

Then: 404 if the job is unknown; 409
`"Team names cannot be confirmed while the job is <status>."` if it is not
`awaiting_team_confirmation`; 409 `"Detected team colors are unavailable."` if
the detection stored no two colours. On success the job is queued for
analysis. Response: `{"job_id": "<uuid>", "status": "queued"}`.

The app sends both ids and names (`ProcessingService.confirmTeams`); the id
wins. A typed name that matches nothing creates a **new team row** later at
persistence (`resolve_team`).

#### `GET /api/processing/result/{job_id}` -> 200

`{"job_id": "<uuid>", "status": "completed", "result": { ... }}` - the result
keys are listed in section 4.12. Errors: 404 unknown job; 409
`"Job is <status>."` until completed. The result is only in memory: it is lost
on restart even though the match stays in the database.

#### `GET /api/processing/clips/{job_id}/{file_name}` -> 200 `video/mp4`

`job_id` must match `[0-9a-fA-F-]{8,64}` and `file_name`
`[A-Za-z0-9_-]{1,64}\.mp4`; the resolved path must be inside
`backend/static/clips`. Anything else -> 404 `"Clip not found."`. Served by
`FileResponse` with `Accept-Ranges: bytes` and `Cache-Control: public, max-age=3600`.
Works for old matches as long as the files exist (they are never deleted).

#### `GET /api/matches` -> 200

Query: `user_id` (int, optional). Ordered by `match_date` descending.

```json
[{"matchId": 42, "matchDate": "2026-11-06T09:12:30.123000", "location": "Uploaded video",
  "userId": 1, "status": "completed"}]
```

No team names or scores in this list. The app always sends `user_id`; when
nobody is logged in `ApiService.fetchRecentMatches` sends `user_id=1`.

#### `GET /api/matches/{match_id}` -> 200

The match summary the app renders. Errors: 404 `"Match not found."`.

```json
{
  "match": {"matchId": 42, "matchDate": "2026-11-06T09:12:30.123000", "location": "Uploaded video", "userId": 1},
  "reelUrl": "/api/processing/clips/1b4e.../reel.mp4",
  "teams": [{"teamId": 5, "teamName": "Porto", "primaryColor": "#1D428A"},
            {"teamId": 9, "teamName": "Manchester City", "primaryColor": "#A7C8E8"}],
  "play": [{"playId": 83, "teamId": 5, "matchId": 42, "score": 0},
           {"playId": 84, "teamId": 9, "matchId": 42, "score": 1}],
  "events": [{
    "eventId": 311, "eventType": "goal", "eventName": "Goal",
    "description": "GOAL — Manchester City — #9 Erling Haaland scored",
    "clipFileLocation": "/api/processing/clips/1b4e.../goal-01.mp4",
    "timeSec": 4.24, "clipStartSec": 0.0, "clipEndSec": 9.2,
    "teamName": "Manchester City", "jerseyNumber": 9, "playerName": "Erling Haaland"
  }],
  "occurBy": [{
    "occurId": 901, "eventId": 311, "playId": 84, "playerId": 120,
    "startTime": 4.24, "endTime": 4.24, "detectedPlayerTrackId": 41,
    "detectedJerseyNo": 9, "detectedPlayerName": "Erling Haaland",
    "teamName": "Manchester City", "teamConfidence": 0.71,
    "jerseyConfidence": 0.86, "nameConfidence": 0.86, "role": "scorer"
  }],
  "players": [{"playerId": 120, "name": "Erling Haaland", "jerseyNo": 9, "teamId": 9}]
}
```

(Values illustrative; keys exact.) Field sources:

| Key | Source column |
| --- | --- |
| `reelUrl` | `_reel(match.job_id)` - only if `backend/static/clips/<job>/reel.mp4` exists |
| `teams[]` | `teams` rows of the match's `plays` (order: as returned by `IN (...)`, not guaranteed) |
| `play[]` | `plays` rows (`score` = final score) |
| `events[].clipFileLocation` | `events.video_path` (the clip URL) |
| `events[].timeSec` | `events.time_seconds` |
| `events[].clipStartSec`/`clipEndSec` | `events.clip_start_seconds`/`clip_end_seconds` |
| `events[].teamName`/`jerseyNumber`/`playerName` | from the event's **first** occurrence (ordered by `occurrence_id`), i.e. the primary actor: the scorer of a goal, the passer of a pass. (Before 2026-11-07 the last row was used; see section 13.) |
| `occurBy[]` | every `occurrences` row of the match's events, `role` included |
| `players[]` | `players` rows referenced by those occurrences |

`occurBy[].startTime`/`endTime` are seconds (`frame / fps`), not clock
strings. `teamName` on events and occurrences is `detected_team_name`
(NULL when the team was unknown).

### 5.4 Teams (`backend/routers/teams.py`)

#### `GET /api/teams` -> 200

Every team with **at least one player whose `name` is not NULL**, ordered by
`team_name`, with aliases eagerly loaded:

```json
[{"team_id": 3, "team_name": "Atletico Madrid", "short_code": "ATM",
  "aliases": ["atletico", "atletico madrid", "atm"]}]
```

(Example values illustrative.) `aliases` are the stored normalised spellings, sorted. The filter hides team
rows created from typed names that never got a squad (e.g. a misspelt
"Athletico Madrid"). About 116-120 rows; the app filters locally
(`matchTeams` in `lib/models/team_option.dart`). An empty list means no squad
has been seeded (section 6.6).

### 5.5 Features (`backend/routers/features.py`)

#### `GET /api/features` -> 200

```json
{
  "features": [
    {"key": "detect_goals", "label": "Goals", "enabled": true, "app_rebuild_needed": false,
     "what_it_does": "Finds goals from shots, ...", "accuracy": "About 7 of 9 ...",
     "event_types": ["goal"]}
  ],
  "flags": {"detect_goals": true, "goal_scorer_name": true, "detect_shots": false},
  "visible_event_types": ["goal", "foul"],
  "warnings": []
}
```

`features` has all 15 entries in `FEATURES` order; `flags` all 15 keys
(abridged above). The file is re-read on every call. The app reads only
`visible_event_types` and `flags`.

### 5.6 Players' reference images (`backend/routers/players.py`)

For naming players by appearance (re-identification). No app screen uses these
yet; the CLI equivalent is `scripts/enroll_player.py`.

| Endpoint | Request | Response | Errors |
| --- | --- | --- | --- |
| `GET /api/players/{player_id}/reference-images` | - | `{"player_id", "team", "jersey_no", "name", "images": [<image id>], "max_images": 10}` | 404 player not found; 422 `"The player needs a team and a shirt number first."` |
| `POST /api/players/{player_id}/reference-images` -> 201 | multipart `images` (1-10 files, each <= 10 MB) | the listing plus `"stored": <n>`, `"rejected": [<file names with no player found>]` | 422 count out of range / no player found / gallery error; 413 image > 10 MB; 415 unreadable image |
| `DELETE /api/players/{player_id}/reference-images/{image_id}` -> 204 | - | empty | 404 `"Image not found."` |

Each uploaded image is cut to its largest detected player by a private YOLO
instance under a lock (`_EnrolmentDetector`, not the job's shared model) and
stored by `identity.PlayerGallery` in `data/player_gallery/<normalised team>/<shirt>/`.

### 5.7 Other

- `GET /` returns `backend/static/index.html` (the set-aside web page).
- `GET /health` returns `{"status": "healthy"}`.
- The `background_tasks: BackgroundTasks` parameters on `submit_file`,
  `submit_url` and `confirm_teams` are unused; work goes through
  `job_queue.submit`.

---

## 6. Database

### 6.1 Connection

`backend/database.py` builds this ODBC string (values from the environment,
section 2.2):

```
DRIVER={ODBC Driver 18 for SQL Server};SERVER=.\SQLEXPRESS;DATABASE=FootballDB;
Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes;
```

SQLAlchemy URL: `mssql+pyodbc:///?odbc_connect=<quoted string>`;
`create_engine(..., pool_pre_ping=True, future=True)`;
`SessionLocal = sessionmaker(autocommit=False, autoflush=False, future=True)`;
`get_db()` is the FastAPI dependency. **The module connects at import time**
(`ensure_database_exists()`), so importing anything that imports
`backend.database` (`backend.app`, any router, `scripts/seed_roster.py`)
needs a live SQL Server. Modules written to avoid that:
`backend/schema.py`, `backend/team_naming.py`, `backend/cleanup.py`,
`backend/squad_import.py`, `scoreboard/teams.py`. Tests replace
`backend.database` with an in-memory SQLite stand-in in `sys.modules`
before importing routers (e.g. `tests/test_roster_seed.py`,
`tests/test_persist_match.py`, `tests/test_team_confirmation.py`).

### 6.2 Tables and columns

Declared in `backend/models.py` (SQLAlchemy 2 typed mappings). Types below
are the SQL Server types the mappings produce (`String(n)` -> `VARCHAR(n)`,
`Unicode(n)` -> `NVARCHAR(n)`, `Integer` -> `INT`, `Float` -> `FLOAT`,
`DateTime` -> `DATETIME`). Primary keys are `INT IDENTITY`.

#### `users` (`User`)

| Column | Type | Null | Key / constraint | Notes |
| --- | --- | --- | --- | --- |
| `user_id` | INT | no | PK, identity | |
| `username` | VARCHAR(100) | no | UNIQUE | |
| `email` | VARCHAR(150) | no | UNIQUE, indexed | stored lower-case by `register` |
| `password_hash` | VARCHAR(255) | no | | bcrypt |

Relationships: `matches` (1-N), `logs` (1-N). Written by `users.register` and
`app.seed_demo_user`; read by `users.login`, `users.get_user`,
`_persist_match` (checks the `user_id` exists, else stores NULL).

#### `user_logs` (`UserLog`)

| Column | Type | Null | Key | Notes |
| --- | --- | --- | --- | --- |
| `log_id` | INT | no | PK | |
| `user_id` | INT | no | FK `users.user_id` | |
| `action` | VARCHAR(50) | no | | `"register"` or `"login"` |
| `timestamp` | DATETIME | yes (default) | | `utc_now()` naive UTC |
| `ip_address` | VARCHAR(50) | yes | | `request.client.host` |

Written by register/login. Nothing reads it.

#### `teams` (`Team`)

| Column | Type | Null | Key | Notes |
| --- | --- | --- | --- | --- |
| `team_id` | INT | no | PK | |
| `team_name` | VARCHAR(100) | no | UNIQUE | as first typed or as in the roster file |
| `primary_color` | VARCHAR(7) | yes | | `#RRGGBB`, overwritten by every job that resolves the team (`resolve_team`) |
| `short_code` | VARCHAR(3) | yes | index `ix_teams_short_code` (not unique) | scoreboard code; added by `ensure_schema` |

Relationships: `players`, `plays`, `aliases` (cascade `all, delete-orphan`).
Written by `roster.resolve_team` (persistence), `roster.seed_roster`. Read by
`find_team` (confirmation, `_team_codes`, `_db_squad`), `teams.list_teams`,
`get_match_summary`.

#### `team_aliases` (`TeamAlias`)

| Column | Type | Null | Key | Notes |
| --- | --- | --- | --- | --- |
| `alias_id` | INT | no | PK | |
| `alias` | VARCHAR(100) | no | UNIQUE, indexed | always stored **normalised** (`normalise_team_name`) |
| `team_id` | INT | no | FK `teams.team_id` | |

Written by `roster.add_alias` (from `resolve_team` for a new typed team, and
from `seed_roster` for `team_name`, every `aliases` entry and the
`short_code`). An alias already bound to another team is never moved.

#### `players` (`Player`)

| Column | Type | Null | Key | Notes |
| --- | --- | --- | --- | --- |
| `player_id` | INT | no | PK | |
| `name` | NVARCHAR(100) | yes | | NULL = an "unnamed slot" the pipeline created for a shirt number it read with no roster entry |
| `jersey_no` | INT | yes | UNIQUE with `team_id` (`uq_player_team_jersey`) | |
| `team_id` | INT | yes | FK `teams.team_id` | |
| `position` | VARCHAR(2) | yes | | `GK`, `DF`, `MF`, `FW`; added by `ensure_schema`; drives the keeper default and the scorer position prior |

Written by `seed_roster` (insert, fill slot, rename, set position) and
`_roster_player` (insert empty slot). Read by `_roster_player`, `_db_squad`,
`teams.list_teams` (squad filter), `players` router, `get_match_summary`.

#### `matches` (`Match`)

| Column | Type | Null | Key | Notes |
| --- | --- | --- | --- | --- |
| `match_id` | INT | no | PK | |
| `match_date` | DATETIME | yes (default) | | time of persistence (naive UTC), not the match's real date |
| `location` | VARCHAR(200) | yes | | always `"Uploaded video"` |
| `user_id` | INT | yes | FK `users.user_id` | NULL if the uploader id was absent or unknown |
| `job_id` | VARCHAR(36) | yes | UNIQUE | the processing job uuid; links to `backend/static/clips/<job_id>/` |
| `source_video` | VARCHAR(500) | yes | | upload path (the file itself is deleted after the job) |
| `status` | VARCHAR(30) | yes (default) | | always `"completed"` |

Relationships: `plays`, `events` (both cascade `all, delete-orphan`).

#### `plays` (`Play`) - one row per team per match

| Column | Type | Null | Key | Notes |
| --- | --- | --- | --- | --- |
| `play_id` | INT | no | PK | |
| `team_id` | INT | yes | FK `teams.team_id` | |
| `match_id` | INT | no | FK `matches.match_id` | |
| `score` | INT | yes (default 0) | | final score: graphic score when available, else counted goals |

#### `events` (`Event`) - one row per detected event

| Column | Type | Null | Key | Notes |
| --- | --- | --- | --- | --- |
| `event_id` | INT | no | PK | |
| `match_id` | INT | no | FK `matches.match_id` | |
| `event_type` | VARCHAR(50) | no | | raw type: `goal`, `foul`, `pass`, ... (lower-case) |
| `event_name` | VARCHAR(150) | yes | | `event_type.replace("_"," ").title()` e.g. `Yellow Card` |
| `description` | VARCHAR(500) | yes | | 4.10, truncated to 500 |
| `video_path` | VARCHAR(500) | yes | | the clip URL (`clip_url`, else `clip_path`); NULL without a clip |
| `time_seconds` | FLOAT | yes | | `frame / fps` of the source |
| `clip_start_seconds` | FLOAT | yes | | added by `ensure_schema` |
| `clip_end_seconds` | FLOAT | yes | | added by `ensure_schema` |

Note `description` is VARCHAR, not NVARCHAR: player names with characters
outside the server's code page are stored correctly in
`occurrences.detected_player_name` and `players.name` but may lose accents or
become `?` inside `events.description`. Not verified against the live
database.

#### `occurrences` (`Occurrence`) - one row per participant of an event

| Column | Type | Null | Key | Notes |
| --- | --- | --- | --- | --- |
| `occurrence_id` | INT | no | PK | |
| `event_id` | INT | no | FK `events.event_id` | |
| `play_id` | INT | yes | FK `plays.play_id` | the participant's team's play row |
| `player_id` | INT | yes | FK `players.player_id` | roster row for (team, shirt); NULL when no number or unconfirmed |
| `start_time` | FLOAT | yes | | seconds (event moment) |
| `end_time` | FLOAT | yes | | seconds (`end_frame / fps`) |
| `detected_track_id` | INT | yes | | tracker id (meaningless across matches) |
| `detected_jersey_no` | INT | yes | | number read / fused / from commentary |
| `detected_player_name` | NVARCHAR(100) | yes | | the roster name at persistence time |
| `detected_team_name` | VARCHAR(100) | yes | | NULL when unknown (`_clean_team_name` drops `unknown_team`) |
| `team_confidence` | FLOAT | yes | | |
| `jersey_confidence` | FLOAT | yes | | |
| `name_confidence` | FLOAT | yes | | |
| `role` | VARCHAR(20) | yes | | section 4.9; NULL on rows written before 2026-10-15; added by `ensure_schema` |

Write order (`_persist_match`): actors in the order of `event["actors"]`,
with the actor whose `track_id` equals the scorer's sorted first. An event
with no actors still gets one occurrence row (team only).

### 6.3 ER diagram

```
 users 1 ──< user_logs
   │
   │ 0..1
   └──< matches 1 ──< plays >── 0..1 teams 1 ──< team_aliases
           │            │               │
           │            │ 0..1          └──< players (team_id, jersey_no unique)
           │            │                        │ 0..1
           └──< events 1 ──< occurrences >───────┘
                              │
                              └── play_id ──> plays (0..1)

 matches.job_id ── (file system) backend/static/clips/<job_id>/*.mp4, reel.mp4
```

### 6.4 Schema changes: `ensure_schema` (no Alembic)

There is no migration tool. `Base.metadata.create_all` creates missing
**tables** but never alters an existing one. `backend/schema.py`,
`ensure_schema(engine)` applies idempotent changes and is called from
`backend/app.py` at import and from `scripts/seed_roster.py`. It inspects the
live table first, so it is safe on every start. Current steps:

| Step | Change |
| --- | --- |
| `_add_team_short_code` | `ALTER TABLE teams ADD short_code VARCHAR(3) NULL`; `CREATE INDEX ix_teams_short_code ON teams (short_code)` |
| `_add_event_clip_times` | `ALTER TABLE events ADD clip_start_seconds FLOAT NULL`, same for `clip_end_seconds` |
| `_add_column(players, position, VARCHAR(2) NULL)` | squad positions |
| `_add_column(occurrences, role, VARCHAR(20) NULL)` | participant roles |
| `_widen_to_unicode(players, name, 100)` | `ALTER COLUMN name NVARCHAR(100) NULL` (SQL Server only) |
| `_widen_to_unicode(occurrences, detected_player_name, 100)` | same |

The NVARCHAR widening is the one non-additive change. It was added after the
2026-10-15 squad import showed 84 names losing letters outside Windows-1252
("Drăgușin" -> "Dragusin") under a VARCHAR column; widening is lossless.
`_widen_to_unicode` is a no-op on other dialects (tests use SQLite).

**How to add a column safely**

1. Add the mapped column to the model in `backend/models.py`, nullable
   (`Mapped[T | None] = mapped_column(..., nullable=True)`). New databases get
   it from `create_all`.
2. Add one line to `ensure_schema` in `backend/schema.py`:
   `+ _add_column(engine, "<table>", "<column>", "<SQL type> NULL")`.
   Use T-SQL types (`VARCHAR(n)`, `NVARCHAR(n)` for names, `FLOAT`, `INT`).
   Only nullable, additive changes belong here; anything that rewrites or drops
   data needs a reviewed one-off script.
3. Write the value where rows are created (usually `_persist_match`) and expose
   it where it is read (usually `get_match_summary`).
4. Add a test against SQLite like `tests/test_squad_import.py` (which covers the
   `position`/`role` columns) or `tests/test_schema_codes.py`.
5. Restart the server; the column appears on startup. Run it twice to prove it
   is idempotent.

### 6.5 Team names, aliases and normalisation

`backend/team_naming.py`, `normalise_team_name(raw)`:

1. NFKD + casefold, combining marks dropped ("München" -> "munchen").
2. Dots and apostrophes deleted (`C.F.` -> `cf`); other punctuation -> space;
   whitespace collapsed.
3. Club affixes stripped **only from the ends**, while more than one token
   remains: `fc afc cf cfc sc ac as ss rc cd ud sd club de del futbol football fussball calcio`.
   "united", "city", "real", "athletic" are deliberately kept.
4. `""` for empty input (callers treat it as "no match").

Examples from the docstring: "Real Madrid C.F." -> `real madrid`;
"FC Barcelona" -> `barcelona`; "Manchester United" -> `manchester united`.

`backend/roster.py`, `find_team(db, name)` tries, cheapest first:
case-insensitive literal `team_name`; then `team_aliases.alias == normalised`;
then a scan normalising every `team_name`. `resolve_team(db, name, color)`
returns the found team (updating `primary_color`) or creates one and stores
its normalised name as an alias.

**Known duplicate team rows.** The vault and the comment in
`backend/routers/teams.py` record that `FootballDB` holds both "Athletico
Madrid" (an early typed misspelling, no squad, no code) and "Atletico Madrid"
(seeded, code ATM). Normalisation cannot merge them (they differ by a letter,
not an affix). Mitigations in code: `/api/teams` hides teams without a named
squad, and the app confirms by id. A typed "Athletico Madrid" from an older
client would still match the duplicate row by literal name. Merging requires
repointing `plays` and `occurrences` and has not been done.

### 6.6 Rosters: files, seeding and import

**Roster JSON** - `data/rosters/<slug>.json`, one per team (116 team files plus
`_template.json` on 2026-11-06; files starting with `_` are never seeded;
only 10 of the 116 carry a `short_code` - the rest rely on 3-letter aliases or
spelling for scoreboard codes):

```json
{
  "team_name": "My Local Team",
  "short_code": "MLT",
  "aliases": ["my local team fc", "mlt"],
  "season": "2025-26",
  "source": "hand-entered",
  "needs_review": false,
  "players": [
    { "jersey_no": 1, "name": "Example Keeper", "position": "GK" },
    { "jersey_no": 9, "name": "Example Forward", "position": "FW" }
  ]
}
```

| Field | Used by | Rules |
| --- | --- | --- |
| `team_name` | `seed_roster` (required), `squad_for`, `expected_codes_for` | matched with `find_team`, so an existing spelling is reused |
| `short_code` | `seed_roster` (sets `teams.short_code` only when empty, also adds it as an alias), `scoreboard/teams.py` | exactly 3 letters, else ignored with a note |
| `aliases` | `seed_roster` -> `team_aliases`; `squad_for` and `expected_codes_for` (3-letter aliases count as codes) | any spelling; stored normalised |
| `season`, `source` | information only | |
| `needs_review` | `seed_roster` prints a note | |
| `players[].jersey_no`, `.name` | required per entry (else skipped with a note) | one player per number per team |
| `players[].position` | upper-cased; `GK/DF/MF/FW` | |

The command line pipeline reads these files directly
(`scorer_identifier/goal_scorer.py`, `squad_for`, cached per process); API
jobs read the database (`_db_squad`). Keep the two in step by seeding after
every file change.

**`scripts/seed_roster.py`** (needs SQL Server; runs `create_all` and
`ensure_schema` first):

```powershell
.venv\Scripts\python.exe -m scripts.seed_roster                     # every file
.venv\Scripts\python.exe -m scripts.seed_roster --team "Real Madrid" # one team (normalised match)
.venv\Scripts\python.exe -m scripts.seed_roster --dry-run           # report, roll back
.venv\Scripts\python.exe -m scripts.seed_roster --report            # print stored teams and squads
.venv\Scripts\python.exe -m scripts.seed_roster --dir <folder>      # another roster folder
```

Per player (`roster._seed_player`): new number -> insert; existing unnamed
slot -> fill the name; different name -> rename (with a note); position set
when it differs. Re-running is safe and reports "already up to date". It is
deliberately not run at startup: seeding on every boot would overwrite names
corrected by hand in the database. A rename changes the name shown for old
matches too, because occurrences point at the (team, shirt) row.

**`scripts/import_squads.py`** (spreadsheet -> roster files -> database):

```powershell
.venv\Scripts\python.exe -m scripts.import_squads "<path>\football_squads_2026_27.xlsx" --dry-run
.venv\Scripts\python.exe -m scripts.import_squads "<path>\football_squads_2026_27.xlsx"
.venv\Scripts\python.exe -m scripts.import_squads "<path>\squads.xlsx" --files-only --season 2026-27 --dir data\rosters
```

The workbook needs, per sheet, a header with `Club` or `Country` (or `Team`),
`Jersey #`, `Player`, and optionally `Position` (`backend/squad_import.py`,
`rosters_from_workbook`; standard library only). Positions map through
`POSITION_CODES` (goalkeeper/keeper -> GK, defender -> DF, midfielder -> MF,
forward/striker/attacker/winger -> FW). A shirt number used twice keeps the
first player and prints a note. An existing roster file keeps its
`short_code` and aliases (`merge_with_existing`); teams not in the workbook
are left alone. Without `--files-only` it then runs `seed_roster` (dry-run if
`--dry-run`).

**`scripts/enroll_player.py`** (no database): stores 1-10 reference images of a
player in `data/player_gallery`:

```powershell
.venv\Scripts\python.exe -m scripts.enroll_player --team "Real Madrid" --jersey 7 --name "<Player Name>" photo1.jpg photo2.jpg
```

Flags: `--team` and `--jersey` required, `--name` (defaults to the roster
name), `--detector` (default `models/best.pt`), `--no-detect` (images already
cropped).

### 6.7 Example queries (T-SQL, read-only)

```sql
-- Latest matches with both teams and the score
SELECT TOP 20 m.match_id, m.match_date, t.team_name, p.score
FROM matches m
JOIN plays p ON p.match_id = m.match_id
LEFT JOIN teams t ON t.team_id = p.team_id
ORDER BY m.match_date DESC, p.play_id;

-- Every event of one match with all participants and roles
SELECT e.event_id, e.event_type, e.time_seconds, o.role,
       o.detected_jersey_no, o.detected_player_name, o.detected_team_name
FROM events e
LEFT JOIN occurrences o ON o.event_id = e.event_id
WHERE e.match_id = 42
ORDER BY e.time_seconds, o.occurrence_id;

-- Goals with their scorer and assister
SELECT e.event_id, e.time_seconds,
       MAX(CASE WHEN o.role = 'scorer'   THEN o.detected_player_name END) AS scorer,
       MAX(CASE WHEN o.role = 'assister' THEN o.detected_player_name END) AS assister
FROM events e JOIN occurrences o ON o.event_id = e.event_id
WHERE e.event_type = 'goal'
GROUP BY e.event_id, e.time_seconds;

-- Teams the app's picker will offer (same rule as GET /api/teams)
SELECT t.team_id, t.team_name, t.short_code
FROM teams t
WHERE EXISTS (SELECT 1 FROM players p WHERE p.team_id = t.team_id AND p.name IS NOT NULL)
ORDER BY t.team_name;

-- Unnamed shirt slots created by the pipeline (candidates for seeding)
SELECT t.team_name, p.jersey_no FROM players p JOIN teams t ON t.team_id = p.team_id
WHERE p.name IS NULL ORDER BY t.team_name, p.jersey_no;

-- Teams that have no squad (typed-name rows, possible duplicates)
SELECT t.team_id, t.team_name FROM teams t
WHERE NOT EXISTS (SELECT 1 FROM players p WHERE p.team_id = t.team_id AND p.name IS NOT NULL);

-- Which spelling resolves to which team
SELECT a.alias, t.team_name FROM team_aliases a JOIN teams t ON t.team_id = a.team_id ORDER BY a.alias;
```

From PowerShell: `sqlcmd -S .\SQLEXPRESS -d FootballDB -E -Q "<query>"`
(`-E` = Windows authentication). Back up before any hand-written UPDATE or
DELETE; the app has no admin screen.

SQL Server caution: a `UNIQUE` constraint allows only one NULL. `matches.job_id`
is nullable and unique, so at most one match may have a NULL job id (the API
always sets it).

---

## 7. The Flutter app

Location: `footyai/Footy_AI/footy_ai_app/`. Package name `footy_ai_app`,
Android `applicationId` `com.footyai.footy_ai_app`, version `1.0.0+1`.
No state-management package: screens are `StatefulWidget`s that call static
service methods and `setState`.

### 7.1 Folder map

| Path | Contents |
| --- | --- |
| `lib/main.dart` | `main()` (loads `AppConfig`, runs `FootyAIApp`), the route table |
| `lib/config/app_config.dart` | API base URL selection and persistence |
| `lib/theme.dart` | `AppColors` (`primary`, `backgroundLight`, `textPrimary`, `textSecondary`, `white`, ...) |
| `lib/services/api_client.dart` | `ApiConstants` (`baseUrl` = `AppConfig.dbApiBaseUrl`, user and match paths), `ApiClient` singleton (holds a base URL; not used by the services) |
| `lib/services/api_service.dart` | match list, match summary, highlights built from a summary, prediction cards |
| `lib/services/auth_service.dart` | register, login, fetch user (`UserModel`, `AuthResult`) |
| `lib/services/processing_service.dart` | upload (with progress), job status/result, team list (cached), confirm teams |
| `lib/services/feature_service.dart` | `FeatureFlags`, `FeatureService.load()` (new 2026-11-06) |
| `lib/services/session_service.dart` | SharedPreferences: `is_logged_in`, `user_id`, `username`, `email` |
| `lib/services/prediction_service.dart` | thin wrapper over `ApiService.fetchPredictions` |
| `lib/models/team_option.dart` | `TeamOption` (from `/api/teams`), `teamSearchKey`, `matchTeams` (search ranking) |
| `lib/models/match_highlight.dart` | `MatchHighlight` (`time, title, player, type, team, description, clipPath, scoreHome/Away, teamHome/Away, ...`), `MatchStat`, `RelatedHighlight` |
| `lib/models/ai_prediction.dart` | `AIPrediction` |
| `lib/screens/*.dart` | screens (7.3) |
| `lib/widgets/kit_sample_viewer.dart` | player images of one kit with an "Another player" button |
| `lib/widgets/team_picker_sheet.dart` | `showTeamPicker(context, teams:, title:, excludeId:)` searchable bottom sheet |
| `lib/widgets/clip_video_player.dart` | tap-to-load highlight clip player (`ClipVideoPlayer`, `clipUrl`) |
| `lib/widgets/analysis_card.dart` | `AnalysisCard` display widget |
| `test/*.dart` | widget and unit tests (9.2) |
| `android/` | Gradle (Kotlin DSL), `AndroidManifest.xml` (`INTERNET`, `usesCleartextTraffic="true"`) |

### 7.2 Routes (`lib/main.dart`)

| Route | Screen | Arguments |
| --- | --- | --- |
| `/` (initial) | `SplashScreen` | - |
| `/login` | `LoginScreen` | - |
| `/signup` | `SignUpScreen` | - |
| `/home` | `MainNavigationScreen` (bottom navigation) | - |
| `/processing` | `AIProcessingScreen` | `{'jobId': String}` |
| `/summary` | `AIMatchSummary` | `{'matchId': int or String}` |
| `/upload` | `AIVideoUpload` | - |
| `/highlights` | `MatchHighlightsList` | - |
| `/predictions` | `AIPredictionsScreen` | - |
| `/settings` | `SettingsScreen` | - |
| `/expanded` (via `onGenerateRoute`) | `ExpandedHighlightView` | a `MatchHighlight` |

`MainNavigationScreen` tabs: Home (`HomeScreen`), Predictions
(`AIPredictionsScreen`), AI Analyst (`AIVideoUpload`), Settings
(`SettingsScreen`).

No widget in `lib/` navigates to `/highlights` (grep for `'/highlights'` and
`MatchHighlightsList` finds only `main.dart`). The highlights page exists and
is maintained, but how a user reaches it in the current build is unclear;
verify on a device.

### 7.3 Screens and the API calls they make

| Screen (file) | Purpose | Calls | Navigates to |
| --- | --- | --- | --- |
| `SplashScreen` (`splash_screen.dart`) | animated splash | `SessionService.isLoggedIn()` | `/home` or `/login` |
| `LoginScreen` (`login_screen.dart`) | email + password | `AuthService.login` -> `POST /api/users/login`; on success `SessionService.saveLogin` | `/home`; link to `/settings` and `/signup` |
| `SignUpScreen` (`signup_screen.dart`) | create account | `AuthService.register` -> `POST /api/users/register` | `/home` after 600 ms. **Does not** call `saveLogin`, so the new user id is not stored (7.8) |
| `HomeScreen` (`home_screen.dart`) | recent analyses, actions | `ApiService.fetchRecentMatches(userId)` -> `GET /api/matches?user_id=` | `/upload`, `/predictions`, `/summary` (tap a match) |
| `AIPredictionsScreen` (`ai_predictions_screen.dart`) | today's uncommitted version lists match history newest first, refreshed every 8 s (`Timer.periodic`) | `ApiService.fetchRecentMatches` | `/summary` |
| `AIVideoUpload` (`ai_video_upload.dart`) | pick a video from the gallery (`ImagePicker.pickVideo`) and upload with a progress bar | `ProcessingService.submitVideoFile` -> `POST /api/processing/submit-file` | `/processing` with `jobId` |
| `AIProcessingScreen` (`ai_processing_screen.dart`) | poll the job; team confirmation; progress | `getStatus` every 2 s; `listTeams` (prefetched as soon as polling starts); `confirmTeams`; `getResult` | `/summary` (replacing itself) with `matchId` from `result.db_response.matchId` |
| `AIMatchSummary` (`ai_match_summary.dart`) | score and Match Events with clips | `ApiService.fetchMatchSummary` + `FeatureService.load` in parallel | back |
| `MatchHighlightsList` (`match_highlights_list.dart`) | latest match's highlights with filter chips and clips | `fetchLatestMatchSummary`, `fetchHighlights` (each fetches the match list and the latest summary again), `FeatureService.load` | `/upload`, `/expanded` |
| `ExpandedHighlightView` (`expanded_highlight_view.dart`) | one highlight, larger player | none (uses the passed `MatchHighlight`) | back |
| `SettingsScreen` (`settings_screen.dart`) | server address | `AppConfig.validateAndNormalizeApiBaseUrl`, `saveApiBaseUrl`, `resetApiBaseUrl` | - |

#### Team confirmation in detail (`ai_processing_screen.dart`)

- Polling: `_startPolling` (2 s `Timer.periodic` + an immediate poll),
  `_pollInFlight` prevents overlapping requests, `_stopPolling` cancels **and
  nulls** the timer (a dead timer once froze the bar at 15%).
- Polling stops at `awaiting_team_confirmation` and `failed`. At
  `completed` it fetches the result and opens `/summary`.
- `progress_percent` is parsed leniently (`num.tryParse(toString())`).
- For each of the two `team_candidates` the screen shows a
  `KitSampleViewer(imageUrls: _imageUrls(candidate), kitNumber: n)` keyed by
  job id. `_imageUrls` prefers `sample_images`, falls back to `sample_image`,
  and prefixes relative paths with `AppConfig.processingApiBaseUrl`.
- `KitSampleViewer` (`lib/widgets/kit_sample_viewer.dart`) shows one image at a
  time; "Another player" steps to the next image and wraps; with a single
  image the button is hidden.
- Tapping the team field opens `showTeamPicker(context, teams:, title: 'Team wearing kit N', excludeId: <other kit's team id>)`
  (`lib/widgets/team_picker_sheet.dart`): a searchable list over
  `ProcessingService.listTeams()`. Search uses `matchTeams` (`team_option.dart`):
  name prefix first, then a word/code/alias prefix, then substring; accents
  folded (`_folded` table), spaces optional ("realmadrid"). Enter picks the
  first match.
- `_confirmTeams` requires two different teams and calls
  `ProcessingService.confirmTeams(jobId, team1, team2)` which posts
  `team_1_id`, `team_2_id`, `team_1_name`, `team_2_name`. Then polling restarts.
- `ProcessingService.listTeams()` caches the `Future` for the app's lifetime;
  a failed fetch is dropped from the cache so the next call retries. A team
  seeded after the app started is not visible until the app restarts.

#### Match summary in detail (`ai_match_summary.dart`)

- Builds `teamNameById` from `teams`, `playTeamNameById` from `play`,
  `eventById` from `events`, `receiverByEventId` from `occurBy` rows with
  `role == 'receiver'`.
- Iterates `occurBy`; skips types `!features.shows(type)`; keeps only the first
  row per `eventId`; merges `{...eventObj, ...item, 'teamName': <play team>, receiver fields}`
  into `MatchEvent.fromJson`.
- Titles (`MatchEvent._eventTitle`): goal `"<Team> — #9 Name scored"`
  (`"<Team> scored"` with no number); pass/cross
  `"<Team> pass: #8 Name → #10 Name"`, `"unknown"` for a missing side,
  `"<Team> pass"` when neither is named; foul `"<Team> committed a foul"`;
  others `"<Team> <type with spaces>"`. Team and names are title-cased.
- Scoreboard: `teams[0]`/`teams[1]` names and the matching `play.score`.
- Each tile expands to the description and, when `clipFileLocation` is set, a
  `ClipVideoPlayer`.
- See section 13 for a field-precedence issue in `MatchEvent.fromJson`
  (event-level `jerseyNumber`/`playerName` win over the occurrence's).

#### Highlights page in detail (`match_highlights_list.dart`, `api_service.dart`)

- `ApiService._buildHighlightsFromSummary` makes one `MatchHighlight` per event
  (first occurrence row), using the occurrence's `detectedPlayerName`
  (or the `players[]` name via `playerId`), `#<jersey>` or
  `"Scorer unavailable"`; type via `_highlightType` (`*_card` -> `Card`, else
  title case); team via `_cleanTeamName` (drops `unknown_team`).
- `_visibleHighlights` removes `pass` and `save` (`_noClipTypes`) and
  categories switched off; `_filters` = "All" + "Goals" (always) + present
  categories from `_filterLabels` (`goal, foul, card, shot, penalty, cross`).

#### Clip playback (`lib/widgets/clip_video_player.dart`)

- Loads only on tap (several simultaneous ExoPlayers ran out of decoders);
  20 s load timeout (`_loadTimeout`); controls fill the video
  (`StackFit.expand` in `AspectRatio`); "play" at the end restarts.
- `ClipVideoPlayer.clipUrl(raw)`: absolute `http(s)` URL as is; a path
  starting with `/` is prefixed with `AppConfig.processingApiBaseUrl`; an
  absolute file-system path is converted by finding the `clips` (or `events`)
  segment and rebuilding `/api/processing/clips/<jobId>/<file>`.
- The app does not use `reelUrl` or `clipStartSec`/`clipEndSec` yet.

### 7.4 Services and models in brief

| Service method | HTTP | Timeout | Returns on error |
| --- | --- | --- | --- |
| `AuthService.register/login` | POST JSON | 20 s | `AuthResult(success: false, message: <detail or "Unable to reach ...">)` |
| `AuthService.fetchUser` | GET | 20 s | `null` |
| `ProcessingService.submitVideoFile` | POST multipart (`video`, `user_id`) via `_ProgressMultipartRequest` | none | `null`, `lastError` set |
| `ProcessingService.submitVideoUrl` | POST JSON | 90 s | `null` |
| `ProcessingService.getStatus/getResult` | GET | 60 s | `null` |
| `ProcessingService.listTeams` | GET `/api/teams` | 30 s | throws (`Exception`) |
| `ProcessingService.confirmTeams` | POST JSON | 30 s | `false`, `lastError` set |
| `FeatureService.load` | GET `/api/features` | 10 s | `FeatureFlags.fallback` |
| `ApiService.fetchRecentMatches` | GET `/api/matches?user_id=` (default 1) | 30 s | `[]` |
| `ApiService.fetchMatchSummary` | GET `/api/matches/{id}` | 30 s | `null` |

`ProcessingService.lastError` is a static string shared by all calls; each
call clears it first.

### 7.5 How the base URL is chosen

See section 2.5. Summary: saved Settings value > `--dart-define=API_BASE_URL`
> `http://10.0.2.2:8000`. Auth and match reads use `dbApiBaseUrl`
(`DB_API_BASE_URL` override), everything processing-related uses
`processingApiBaseUrl` (`PROCESSING_API_BASE_URL` override); both default to
the same address.

### 7.6 Changing the app

**Add a screen**

1. Create `lib/screens/<name>_screen.dart` with a `StatefulWidget`.
2. Register a route in `routes:` in `lib/main.dart` (or in `onGenerateRoute`
   when it needs a typed argument, like `/expanded`).
3. Navigate with `Navigator.pushNamed(context, '/<route>', arguments: {...})`
   and read arguments in `didChangeDependencies` with
   `ModalRoute.of(context)?.settings.arguments` (pattern used by
   `AIProcessingScreen`, `AIMatchSummary`).
4. Put HTTP calls in a service under `lib/services/`, build URLs from
   `AppConfig.processingApiBaseUrl` or `ApiConstants.baseUrl`, add a timeout,
   and return a safe value on error (the pattern every service follows).

**Show a new event type** - nothing is required for listing on the match
summary: any raw `eventType` the server lists in `visible_event_types` is
shown with the generic title. Optional polish: a title branch in
`MatchEvent._eventTitle`, colour/icon in `_getEventColor`/`_getEventIcon`,
a display type in `ApiService._highlightType`, a chip in `_filterLabels`.

**Add a filter chip** - add `'<display type lower-case>': '<Label>'` to
`_filterLabels` in `match_highlights_list.dart`. The key must equal
`MatchHighlight.type.toLowerCase()` as produced by `_highlightType`.

**Show a new API field** - read it in the screen from the summary map
(`eventObj['<key>']` or the occurrence `item['<key>']`), or carry it through
`MatchEvent`/`MatchHighlight` by adding a field and reading it in `fromJson`
/ `_buildHighlightsFromSummary`. Recipe 8.6 shows the backend side.

### 7.7 Build, install, test, analyse

```powershell
cd footyai\Footy_AI\footy_ai_app
flutter pub get                                   # once, or after pubspec changes
flutter analyze --no-pub                          # 0 errors / 0 warnings / 48 infos recorded 2026-11-06
flutter test --no-pub                             # 23 tests on 2026-11-07
flutter build apk --release --dart-define=API_BASE_URL=http://<PC LAN IP>:8000
# output: build\app\outputs\flutter-apk\app-release.apk  (about 50 MB recorded)
adb install -r build\app\outputs\flutter-apk\app-release.apk   # with USB debugging on
```

The `--no-pub` flag skips the implicit `pub get` (useful offline). The APK can
also be copied to the phone and opened (allow installs from unknown sources).
The release build type is signed with the **debug** key
(`signingConfig = signingConfigs.getByName("debug")` in
`android/app/build.gradle.kts`), which is fine for side-loading but not for a
store release.

**JDK**: this machine's Flutter is configured with `flutter config --jdk-dir`
pointing at a JDK 17 inside `footyai/tools/jdk17/`. The Android project
targets Java 17 (`sourceCompatibility`/`targetCompatibility`/`jvmTarget`
`VERSION_17`) on Gradle 8.14. Android Studio's bundled JDK 25 fails with
`Unsupported class file major version 69`. Do not change the global
`--jdk-dir` from one project's task; other projects share it.

### 7.8 App behaviours to be aware of

- No logout: `SessionService.clear()` exists but nothing calls it.
- Sign-up does not save the session; after sign-up the user id is `null`, uploads
  are sent without `user_id` (match stored with `user_id` NULL), and match lists
  request `user_id=1` - so a new user's uploads do not appear in their lists.
  Logging in afterwards fixes the session for later uploads.
- When not logged in, all match lists show user 1's matches.
- The team list is cached for the app run (7.3).
- The highlights page always shows the **latest** match of the user, not a
  chosen one.

---

## 8. Recipes: how do I...

Each recipe lists the files to touch in order. Run the tests in section 9
afterwards.

### 8.1 Add a new event type end to end

Example: `corner`.

1. **Detect it.** Add a rule that returns event dicts with `"event": "corner"`,
   `frame`, `end_frame`, `confidence`, `team_id`, `team_name`, `source`,
   `participants` (each `{role, track_id, team_id, team_name, team_confidence}`)
   and `details`. For a track rule, add a method to `TrackEventDetector`
   (`track_events/detector.py`) and emit with `self._emit("corner", ...)`;
   add a cooldown to `EVENT_COOLDOWN_SECONDS` if repeats are likely. For a
   spotting-model class, map it in `spotting/labels.py` (`EVENT_NAMES`), add
   it to `USED_CLASSES` in `spotting/__init__.py`, and convert it in
   `spotting/fusion.py`.
2. **Switch it.** Add a `Feature("detect_corners", "Corners", False, ("corner",))`
   to `feature_flags.FEATURES`, a `detect_corners` field and `on(...)` in
   `PipelineConfig.short_clip`, pass it to the detector in `process_video`,
   and add the block to `features.yaml` (section 3.7).
3. **Name its players.** Add the primary role to `PRIMARY_ROLE` in `main.py`
   (e.g. `"corner": "taker"`); `_resolve_identities` then copies that actor to
   `scorer`. Event types listed in `CLIP_EVENT_TYPES` get their identities
   resolved before other non-goal events (OCR budget order).
4. **Describe it.** Add a branch to `describe_event` in `main.py`, e.g.
   `CORNER — <team> — taken by #7 Name`. Without it the description is
   `CORNER — <team>`.
5. **Clip it (optional).** Add `"corner"` to `CLIP_EVENT_TYPES` and to
   `EVENT_PRIORITY` in `highlights/segments.py`.
6. **Persist it.** Nothing to do: `_persist_match` stores any event type, with
   `event_name` `"Corner"` and one occurrence per actor with its `role`.
7. **Show it in the app.** Nothing is required (generic title). Optional:
   `_eventTitle`, icon/colour in `ai_match_summary.dart`; `_highlightType` in
   `api_service.dart`; `_filterLabels` in `match_highlights_list.dart`. If the
   app needs changes, set `app_rebuild_needed: yes` until a new APK is out.
8. **Test.** Synthetic-track tests in the style of `tests/test_track_events.py`
   or `tests/test_fouls.py`; persistence in `tests/test_persist_match.py`;
   switch wiring in `tests/test_feature_flags.py`; titles in
   `test/feature_flags_test.dart`.

### 8.2 Change which events get highlight clips

`highlights/segments.py`:

```python
CLIP_EVENT_TYPES = frozenset(
    {"goal", "foul", "shot", "cross", "penalty", "yellow_card", "red_card"}
)
EVENT_PRIORITY = ("goal", "penalty", "red_card", "yellow_card", "foul", "shot", "cross")
```

- To clip saves again: add `"save"` to both (the comment in the file says so).
- To stop clipping fouls: remove `"foul"` from both.
- Also remember the output filter: an event type that is switched off in
  `features.yaml` is removed before clips are planned, so it gets no clip
  whatever `CLIP_EVENT_TYPES` says.
- The app's highlights page never lists `pass` or `save`
  (`_noClipTypes` in `match_highlights_list.dart`); if saves get clips again,
  remove `'save'` there to list them.
- Tests: `tests/test_highlights.py`.

### 8.3 Change clip length

Constants at the top of `highlights/segments.py`:

| Constant | Default | Effect |
| --- | --- | --- |
| `MAX_CLIP_SECONDS` | 10.0 | hard cap per clip (owner rule) |
| `LEAD_MIN_SECONDS` / `LEAD_DEFAULT_SECONDS` / `LEAD_MAX_SECONDS` | 2 / 5 / 6 | time before the event |
| `TAIL_MIN_SECONDS` / `TAIL_DEFAULT_SECONDS` / `TAIL_MAX_SECONDS` | 1.5 / 3 / 4 | time after non-goal events |
| `GOAL_TAIL_MIN_SECONDS` / `GOAL_TAIL_DEFAULT_SECONDS` / `GOAL_TAIL_MAX_SECONDS` | 3 / 4 / 5 | time after a goal |
| `MERGE_GAP_SECONDS` | 1.0 | how close two windows must be to merge |

The MIN/MAX pairs are the search range for a camera cut to snap to; DEFAULT is
used when no cut is in range. Raising `MAX_CLIP_SECONDS` is an owner decision
(the 10 s rule replaced clips of up to 60 s). `tests/test_highlights.py`
asserts the cap and the merge behaviour; update it in the same change.

### 8.4 Change the OCR budget

| What | Where | API value | CLI value |
| --- | --- | --- | --- |
| Shirt-number OCR calls per job | `PipelineConfig.jersey_ocr_calls` (set in `short_clip`) | 64 | 120 |
| Crops kept per player for OCR | `PipelineConfig.identity_crops` | 4 | 8 |
| Extra OCR per goal for celebration close-ups | `GOAL_SCORER_OCR_ALLOWANCE` in `main.py`, passed as `extra_ocr_allowance` | 0 (because `goal_only`) | 48 |
| Caption reads per goal | `CAPTION_MAX_READS` 30, `CAPTION_EVERY_SECONDS` 1.5, `CAPTION_SECONDS` 45, `CAPTION_AGREEING_READS` 2 in `scorer_identifier/goal_scorer.py` | same | same |
| Skip captions when evidence is decisive | `CAPTION_SKIP_CONFIDENCE` 0.85 in `main.py` | same | same |
| Score graphic cadence | `scoreboard_scan_seconds` 2.0, `scoreboard_burst_seconds` 0.5, `scoreboard_burst_duration` 8.0, `scoreboard_permissive_after_misses` 3 in `PipelineConfig` | same | same |
| Score graphic strips | `TOP_STRIP` 0.18, `BOTTOM_STRIP` 0.22, `LOST_AFTER_SECONDS` 20 in `scoreboard/reader.py` | same | same |

Each PaddleOCR call costs roughly 2-4 s on this CPU for a full strip (module
docstring of `scoreboard/reader.py`), so the budget is the main lever on
`scorer_ocr` and `scoreboard_ocr` time (section 9.4). Fewer calls means more
unconfirmed scorers. The relevant tests are in `tests/test_pipeline_optimization.py`
(`HybridScoreboardTests`) and `tests/test_goal_scorer.py`.

### 8.5 Add a team or a squad

Option A - one hand-made team:

1. Copy `data/rosters/_template.json` to `data/rosters/<slug>.json` (any name
   not starting with `_`).
2. Fill `team_name`, `short_code` (three letters as on the score graphic, if
   known), `aliases` (other spellings and codes), `players`
   (`jersey_no`, `name`, `position` GK/DF/MF/FW).
3. `.venv\Scripts\python.exe -m scripts.seed_roster --team "<team_name>" --dry-run`, check the output, then run without `--dry-run`.
4. Check `GET /api/teams` lists it. Restart the **app** (its team list is
   cached per run).

Option B - a whole spreadsheet: `scripts.import_squads` (section 6.6), dry-run
first.

Option C - fix one name in the database directly (no file): `UPDATE players SET name = N'...' WHERE team_id = ... AND jersey_no = ...;`
Use the `N'...'` prefix for Unicode. The next seeding run will overwrite it
from the file if the file says otherwise, so fix the file too.

### 8.6 Change the server address

- One phone: Settings tab -> type `<PC LAN IP>` -> Save (section 2.5).
- Every new install: rebuild with
  `--dart-define=API_BASE_URL=http://<PC LAN IP>:8000`.
- Server side: the port is the `--port` argument of uvicorn (and
  `.claude/launch.json`). If you change it, include the port in the app's
  address (the app defaults to 8000 when no port is typed) and update the
  firewall rule.
- The PC's LAN IP can change with DHCP; reserve it in the router or re-enter it
  in Settings.

### 8.7 Add an API field and show it in the app

Example: show `clip_offset_seconds` (where the event happens inside its clip).

1. **Store it** (if it is not already stored): add a column (8.8), e.g.
   `events.clip_offset_seconds FLOAT NULL`, and set it in `_persist_match`
   (`clip_offset_seconds=detected.get("clip_offset_seconds")` in the `Event(...)`
   constructor). Fields that can be derived need no column: here
   `timeSec - clipStartSec` already gives the offset (the comment in
   `get_match_summary` says so).
2. **Expose it**: add the key to the dict built in `get_match_summary`
   (`backend/routers/processing.py`), camelCase like its neighbours, e.g.
   `"clipOffsetSec": event.clip_offset_seconds`. For job-only data add it to the
   `result` dict in `_process_job`.
3. **Read it in the app**: in `ai_match_summary.dart` it arrives in `eventObj`
   and, via the spread, in the map passed to `MatchEvent.fromJson`; add a field
   to `MatchEvent` and read `json['clipOffsetSec']`. For the highlights page,
   add a field to `MatchHighlight` and set it in
   `ApiService._buildHighlightsFromSummary`.
4. **Keep old servers working**: read new keys with null-safe fallbacks
   (`json['x'] as num?`), because an older server will not send them.
5. **Test**: extend `tests/test_persist_match.py` (it builds the summary from
   SQLite) and a Dart unit test like `test/feature_flags_test.dart`.

### 8.8 Add a database column

Section 6.4, "How to add a column safely". In short: nullable mapped column
in `backend/models.py` -> one `_add_column(...)` line in `ensure_schema`
(`backend/schema.py`) -> write it in `_persist_match` -> read it in
`get_match_summary` -> a SQLite test -> restart the server.

### 8.9 Turn a feature on or off

1. Edit `features.yaml`, change `enabled: no` to `enabled: yes` (or back) under
   the feature. Keep the two-space indentation.
2. Check: `curl http://127.0.0.1:8000/api/features` - `warnings` must be `[]`
   and `flags.<key>` must show the new value.
3. Analyse a new video. No restart of the server or the app is needed. The app
   picks up visibility changes the next time it opens a match.

Dependencies: `passer_name` needs `detect_passes`. Showing an event type that
was off when a match was analysed shows nothing for that match (it was not
stored).

### 8.10 Tune analysis speed

| Lever | Where | Trade-off |
| --- | --- | --- |
| Analysed frame rate | `short_clip_fps` (12), `long_clip_fps` (10), `short_clip_seconds` (60) in `short_clip` | lower = faster; fewer ball samples for shots/passes |
| Detector size | `low_resolution_size` 640, `high_resolution_size` 960 | smaller = faster, worse ball recall |
| Commentary | `commentary_names: no` | saves 1-15 min; loses names from speech |
| Whisper size | `PipelineConfig.whisper_model` ("small") | larger is slower |
| Spotting | `action_spotting: no` | loses spotted goals/shots and kick-offs (replay separation) |
| Scoreboard | `read_scoreboard: no` | loses graphic goals and final score |
| OCR budget | 8.4 | fewer names |

### 8.11 Change the replay window or goal fusion thresholds

- Replays: `GOAL_REPLAY_SECONDS` (20) in `track_events/goal_merge.py`. It was
  60 s and dropped real goals in highlight packages; see the module docstring
  before changing it. Tests: `tests/test_goal_merge.py`.
- Spotted goals: `GOAL_ALONE` (0.5), `GOAL_WITH_SHOT` (0.15) in
  `spotting/fusion.py`. Tests: `tests/test_spotting.py`.
- Scorer confidence: `UNCONFIRMED_BELOW` (0.7), `STRENGTH_FOR_FULL_CONFIDENCE`
  (2.5), weights `WEIGHT_*`, `POSITION_PRIOR` in
  `scorer_identifier/goal_scorer.py`. Tests: `tests/test_goal_scorer.py`,
  `tests/test_pipeline_optimization.py` (`BothSquadsScorerTests`,
  `CaptionLedScorerTests`).

---

## 9. Testing and verification

### 9.1 Backend tests

```powershell
cd football_summary
.venv\Scripts\python.exe -m unittest discover -s tests -t .
.venv\Scripts\python.exe -m unittest tests.test_feature_flags -v     # one module
```

Result on 2026-11-07: **320 tests, OK** in about 8-24 s. The tests need no SQL Server (SQLite stand-ins), no GPU and no
footage; the ffmpeg cut test in `test_highlights.py` uses ffmpeg when present.
There is no linter or formatter configured for the Python code.

| Test file | Tests | Covers |
| --- | --- | --- |
| `test_cleanup.py` | 5 | upload deletion and the age sweep (`backend/cleanup.py`); highlights never deleted |
| `test_commentary.py` | 20 | name matching to squads, goal/foul roles, attaching spoken names to tracks |
| `test_event_logic.py` | 3 | gate on the legacy learned goal/foul model's proposals |
| `test_feature_flags.py` | 10 | `features.yaml` parsing, defaults, warnings, profile wiring, name switches, pass description without names |
| `test_fouls.py` | 7 | foul rules on synthetic tracks (contact, player down, play stopped) |
| `test_fps_invariance.py` | 6 | the same play gives the same events at different source frame rates |
| `test_goal_from_shot.py` | 7 | a shot whose ball ends in the goal mouth is a goal, without pitch calibration |
| `test_goal_merge.py` | 19 | replay removal, named-scorer separation, kick-offs, saves of goals |
| `test_goal_scorer.py` | 26 | caption matching, celebration votes, live read, fusion and confidence |
| `test_highlights.py` | 16 | clip windows, merging, cut snapping, 10 s cap, ffmpeg cutter and reel |
| `test_identity.py` | 14 | gallery, per-track probabilities, scene cuts, one-name exclusivity |
| `test_match_config.py` | 8 | `MatchConfig` / `parse_hex_color` validation |
| `test_persist_match.py` | 4 | multi-participant persistence and the summary (scorer, not assister, as the event's player), in SQLite |
| `test_pipeline_optimization.py` | 25 | frame schedule, detector sizes, calibration skip, fixture JSON completeness, goal-only output, serial queue, scoreboard cadence, assists, keepers, commentary vs caption, both-squad scorers |
| `test_pitch.py` | 15 | pitch calibration gate and pitch-aware rules (synthetic) |
| `test_players_api.py` | 6 | reference-image endpoints, in SQLite |
| `test_roster_seed.py` | 20 | team-name normalisation, `find_team`, seeding idempotence, shipped roster JSON validity |
| `test_scene_cuts.py` | 6 | cut/dissolve detection and what is forgotten at a cut |
| `test_schema_codes.py` | 7 | `teams.short_code` migration and seeder |
| `test_scoreboard.py` | 35 | parser on real OCR token shapes (UEFA one-line, LaLiga stacked), reader confirmation, attribution windows |
| `test_scorer_identifier.py` | 13 | shirt-number OCR consensus and acceptance rules (deterministic, no models) |
| `test_spotting.py` | 11 | spotting sampling, windowing, peaks, conservative fusion |
| `test_squad_import.py` | 3 | spreadsheet -> roster payloads; `position`/`role` columns |
| `test_team_confirmation.py` | 9 | `rank_team_samples` alternatives, `/api/teams` filter, ids/names in confirmation |
| `test_track_events.py` | 16 | passes, crosses, shots, saves, penalties, cards on synthetic tracks |
| `test_tracker_helpers.py` | 9 | ball selection and tracker helpers |

### 9.2 App tests

```powershell
cd footyai\Footy_AI\footy_ai_app
flutter test --no-pub
flutter analyze --no-pub lib test
```

Result on 2026-11-07: **23 passed**; analyzer 0 errors, 0 warnings, 48 infos
(all pre-existing, in files this work did not touch).

| Test file | Tests | Covers |
| --- | --- | --- |
| `test/widget_test.dart` | 1 | app smoke test |
| `test/clip_video_player_test.dart` | 3 | no stream until tapped; plays in a list with the seek bar on the video; replays after the end (fake video platform) |
| `test/team_picker_test.dart` | 10 | `matchTeams` ranking, codes/aliases/accents/no spaces, other-kit exclusion, no-match message, picker returns the tapped team, `KitSampleViewer` stepping and single-image case |
| `test/feature_flags_test.dart` | 6 | `FeatureFlags.fromJson`, fallback, named and unnamed pass titles, goal title from the scorer row |
| `test/ai_predictions_screen_test.dart` | 3 | match-history sorting and scoreline helpers (uncommitted, 2026-11-06) |

### 9.3 Fixture clips

`input_videos/` holds the reference clips. `.gitignore` excludes
`input_videos/*.mp4`; only `test.mp4` is tracked (it was added before the
rule), the other two exist only on this machine. `tests/fixtures/short_clips.json`
annotates all three:

| File | Teams (codes) | Final score | Goals | Notes |
| --- | --- | --- | --- | --- |
| `football_1 (2).mp4` | RMA, ATM | 0-0 | 0 | 854x480, 25 fps, LaLiga graphic with crests; music/crowd audio |
| `test.mp4` | POR, EGY | 1-0 | 1 at ~12 s | 26 s, 30 fps; commentator says the goal was ruled out while the graphic shows 1-0 |
| `videoplayback (2).mp4` | POR, MCI | 0-1 | 1 at ~12 s (fixture) | 1920x1080, 25 fps, no audio track |

`tests/fixtures/goal_clips/README.md` describes where locally licensed goal
clips would go; none are committed. Other clips used in the vault's
measurements (FC Mobile game footage, Levante 3-3 Sevilla highlights) are not
in the repository.

`test_bundled_short_clip_annotations_are_complete` checks the fixture JSON
has 3 entries, two codes each and consistent goal counts. No automated test
runs the full pipeline on a clip; end-to-end runs are manual.

### 9.4 Timing a run

Every API job result carries `pipeline_metrics`
(`pipeline_config.PipelineMetrics.summary()`):

```json
{"phases_seconds": {"commentary": 35.1, "scene_analysis": 3.2, "spotting": 12.0,
                    "scoreboard_ocr": 45.3, "player_identity": 4.1, "scorer_ocr": 100.2,
                    "highlights": 6.0, "person_detection": 120.5, "ball_detection": 42.7,
                    "team_sampling": 9.8, "persistence": 0.4},
 "counts": {"decoded_frames": 780, "analysed_frames": 312, "person_inference_frames": 312,
            "ball_inference_frames": 312, "scoreboard_ocr_calls": 40, "scorer_ocr_calls": 61}}
```

(Values illustrative.) Phase names come from `metrics.timer(...)` /
`add_time(...)` calls in `main.py`; `team_sampling` and `persistence` are
added by `_process_job`. To time a job: submit through the app or with
`curl`, then `GET /api/processing/result/{job_id}` and read
`result.pipeline_metrics`. For the CLI, `process_video` returns the metrics
only with `return_pipeline_result=True`; the console prints the main counts
(ball frames, scoreboard summary, spotting inference seconds).

### 9.5 Manual end-to-end check (writes to the database)

1. Start the server (2.2); confirm `/health`, `/api/teams`, `/api/features`.
2. Upload `input_videos/test.mp4` from the app (or
   `curl -F "video=@input_videos/test.mp4" http://127.0.0.1:8000/api/processing/submit-file`).
3. Poll `GET /api/processing/status/{job_id}` until
   `awaiting_team_confirmation`; open the `sample_images` URLs.
4. Confirm with ids from `/api/teams`:
   `curl -X POST -H "Content-Type: application/json" -d "{\"team_1_id\": <id>, \"team_2_id\": <id>}" http://127.0.0.1:8000/api/processing/confirm-teams/<job_id>`.
5. Poll to `completed`; read `result`; open `GET /api/matches/{match_id}`.
6. Compare with `tests/fixtures/short_clips.json`.

This inserts a match (and possibly team rows and unnamed player slots) into
`FootballDB`. Get the owner's agreement first.

---

## 10. Known limits and measured accuracy

Sources: the vault notes `Football Summary/80 - Delivery/Verification Evidence.md`
(gates 2026-09-14 to 2026-11-06) and `Football Summary/70 - Bugs and Risks/Known Risks and Gaps.md`,
plus the `accuracy:` lines in `features.yaml`. Nothing in this section was
re-measured for this guide. Machine for all measurements: Intel i5-8365U, no
CUDA, PyTorch on CPU.

There is **no labelled evaluation set**. Every figure below comes from a
handful of clips (three short fixtures, one game-footage clip, one 3-minute
highlights package), mostly checked by eye. Treat them as anecdotes, not
rates.

### 10.1 Per feature

| Feature | What was measured | Result | Main caveats |
| --- | --- | --- | --- |
| Goals (`detect_goals`) | 5 clips with 9 known goals | about 7 of 9 found; no false goals seen. RMA-ATM 0-0: none (right). POR-EGY: goal at 10.0 s, confirmed by the graphic. POR-MCI: goal at 4.24 s. FC Mobile game clip (no graphic): found at frame 216 (truth ~194). Levante 3-3 Sevilla: 4 of 6. | The FC Mobile find rested on a spurious shot whose window happened to cover the real goal. Goals with no visible shot and no readable graphic are missed. Two goals < ~16 s apart may share a shot. |
| Goal's team | Levante after fixes; 50 fps copy of POR-EGY | 4 of 4 on Levante; wrong (Egypt) on the 50 fps test copy | Kit colours are unreliable on striped kits at 480p; captions/commentary can correct the team. |
| Scorer name (`goal_scorer_name`) | Levante after caption fixes | 4 of 4 right - on the match the rules were tuned on | POR-EGY "#7 Cristiano Ronaldo" from the caption is unverified; POR-MCI "#3 (unconfirmed)" is probably wrong; an earlier run named a Porto defender as Man City's scorer. |
| Assister (`assister_name`) | none | unmeasured | Levante gave "assist Fofana" on one goal; commentary alone gave plausible assisters on 4 goals. |
| Commentary (`commentary_names`) | Levante (English) | 27 player mentions recognised (416 words, 73 s Whisper small); commentary alone named 4 of 6 scorers right, 1 wrong ("Ndukwe" heard as "Lufel"), 1 blank | None of the three fixture clips names a player. Cue words are English only. Costs 1-15 min per video. |
| Score graphic (`read_scoreboard`) | UEFA 1080p clip; Levante LaLiga package | UEFA: correct 0-1 change. Levante: 3 of 6 score changes read after the LaLiga fixes | Graphics with crests and no codes are not read; a 0-1 -> 1-2 jump between package clips is rejected as implausible; the graphic is hidden during replays. 106 of 116 roster files had no `short_code` at the time. |
| Action spotting (`action_spotting`) | 3 fixture clips via `scripts/eval_spotting.py` | Shots on target 0.85 on a saved shot, 0.42 on a goal's shot; Goal output 0.22 on one goal and ~0 on another; never fired on the 0-0 clip | Used only with corroboration (>= 0.15 next to a shot) or alone at >= 0.5. Clips < 50 s are zero-padded to its window (out of distribution). |
| Fouls (`detect_fouls`) | 1080p POR-MCI | one foul reported, probably a false positive; a real tangle in the box missed | Rule-based; owner asked for a fix "next pass". |
| Cards, penalties | synthetic frames only | - | No clip with a card or penalty exists. The card rule needs a referee >= 48 px tall and cannot read a broadcast card graphic. |
| Shots, saves | contact sheets on the fixtures | unmeasured as rates; false shot/save on replays; a goal can be read as a save | Shots/saves stay internal evidence when switched off. |
| Passes, crosses | contact sheets | false passes on close-up replays (players ~400 px tall make ball speed look slow); old runs reported phantom passes from false balls | Crosses use the keeper as a stand-in for the box (no pitch calibration). |
| Passer names (`passer_name`) | none | unmeasured | A name lasts until the next camera cut (track ids restart). |
| Shirt numbers | fixture clips | none readable at 480p; #27 and #22 read at 1080p | Wider reading (2026-10-16) produced more chances of a wrong number. |
| Re-identification | 14 tracklets on the 1080p clip | top-1 among team-mates 32-40% (chance ~15-20%) | Pedestrian OSNet; only nudges the belief. |
| Team detection at upload | POR-EGY on 2026-11-06 | awaiting confirmation 11 s after upload; 6 images per kit | KMeans can split one team into two "kits"; "Another player" cannot fix that. |
| Highlight clips | ffmpeg integration tests and two CLI runs | correct cut-aware starts; clips <= 10 s; H.264+AAC, faststart | The app does not yet use the reel or seek to the event inside a merged clip. |

### 10.2 Speed

| Clip | Total | Biggest phases |
| --- | --- | --- |
| POR-EGY, 26 s (web profile, 2026-10-16) | 397 s | detection 163 s, scorer OCR 100 s, scoreboard OCR 45 s, commentary 35 s, spotting 12 s |
| Levante 3-3 Sevilla, 195 s 1080p50 (web profile) | 1811-2414 s across runs | detection ~1000-1400 s, commentary 107-897 s (machine load varied), scoreboard OCR 222 s |

Rule of thumb from these: roughly 10x real time on this laptop for 1080p
broadcast footage, more with commentary.

### 10.3 Main risks

| Risk | Detail |
| --- | --- |
| No authentication | Every processing, match, team, feature and player endpoint is open; CORS `*`. LAN only. |
| Jobs are in memory | A restart loses job status and results (matches stay in SQL Server). A job running at restart is lost; its upload is swept after 24 h. |
| One job at a time | `SerialJobQueue` has one worker; a long job blocks the next upload's team detection. |
| Teams missing from the database | The picker offers only teams with a named squad; a match against an unseeded team (e.g. Egypt, any amateur side) cannot be confirmed from the app. |
| Duplicate team rows | "Athletico Madrid" vs "Atletico Madrid" in `FootballDB` (6.5). |
| Slot renames rewrite history | Occurrences point at (team, shirt) player rows; re-seeding a different #17 relabels old matches. |
| Kit split | KMeans may split one team (lit/shaded, keepers, referee); no re-cluster option. |
| Cached failures | `ModelCache.spotting()/reid()/whisper()` cache a failed load as `None` until restart; `paddle_ocr()` tries once per process. |
| Highlight packages | One-each score rises between package clips are rejected; replay separation depends on names and kick-offs. |
| Non-English commentary | Whisper transcribes it, but cue words are English. |

---

## 11. Troubleshooting

| Symptom | Likely cause | Check / fix |
| --- | --- | --- |
| App: "Unable to reach http://... Check Server Settings" on login | wrong address, server not running, or firewall | open `http://<PC LAN IP>:8000/health` in the phone's browser. If it fails: server started with `--host 0.0.0.0`? Firewall rule for TCP 8000 (2.3)? Same Wi-Fi? Network profile Private? |
| Works in the emulator, not on the phone | the build default `10.0.2.2` only exists in the emulator | Settings tab -> `<PC LAN IP>`, or rebuild with `--dart-define=API_BASE_URL=...` |
| Server does not start: pyodbc `OperationalError` / login failed / driver not found | SQL Server Express stopped, wrong instance, or driver missing | `services.msc` -> SQL Server (SQLEXPRESS) running; `FOOTY_SQL_SERVER`, `FOOTY_ODBC_DRIVER` env vars; install "ODBC Driver 18 for SQL Server" |
| `ValueError: FOOTY_SQL_DATABASE may contain only letters, numbers, and underscores` | bad env var | fix `FOOTY_SQL_DATABASE` |
| Team picker list empty / "The team list could not be loaded" | no squads seeded, DB unreachable, or the app cached an old list | `curl /api/teams`; seed rosters (6.6); restart the app (list cached per run) |
| The team the user needs is not in the picker | it has no named players in `players` | add a roster file and seed (8.5) |
| Status 422 "Choose a team from the list." on confirm | team id not in `teams` (deleted, or another database) | re-open the picker after an app restart |
| Status 409 on confirm | job already confirmed, failed, or not finished detecting | poll status; a `failed` job needs a new upload |
| Job `failed` at team detection: "Could not find enough clear player images..." / "The two team kits could not be separated..." / "One of the two kits was seen too few times..." | too few large, clear players (< 40 samples, < 12 per kit) or similar kits | use longer or higher-resolution footage (720p+); thresholds `CALIBRATION_MIN_SAMPLES` etc. at the top of `processing.py` |
| Status 404 "Processing job not found." | server restarted (jobs are in memory) | upload again; finished matches are still in `/api/matches` |
| Bar stuck at 15% for minutes | commentary transcription runs before the frame loop | expected; `commentary_names: no` skips it |
| Bar stuck near 90% | post-loop work: OCR, captions, clips | expected; check the server console for "Highlights", "Scoreboard" lines |
| Jobs very slow | CPU-only, 1080p, commentary on, another job queued | section 8.10; check `result.pipeline_metrics`; jobs run one at a time |
| `features.yaml` edit has no effect | YAML typo, wrong indentation, unknown key, PyYAML missing | `curl /api/features` -> `warnings`; server console lines starting `features.yaml:`; remember only **new** jobs use it |
| An event type switched on shows nothing for an old match | it was not stored when analysed | analyse again |
| Clip shows "Could not play this clip from ..." | address changed since the clip URL was built, file missing, or cleartext blocked | clip URLs are relative and prefixed at play time; check `GET /api/processing/clips/<job>/<file>` in a browser; `usesCleartextTraffic` must stay true |
| Clips have no sound / no reel | ffmpeg/ffprobe not on PATH (OpenCV fallback) | install ffmpeg or set `FFMPEG_BINARY`/`FFPROBE_BINARY`; console says "ffmpeg/ffprobe not found" |
| No names, only numbers | squad not seeded for that team, or scorer unconfirmed | `scripts.seed_roster --report`; numbers below confidence 0.7 are shown as "#N (unconfirmed)" by design |
| "Scoreboard reading unavailable: PaddleOCR is unavailable: ..." | Paddle import/constructor failure | the reader logs the constructor error; `enable_mkldnn=False` is required on this stack; check `paddlepaddle`/`paddleocr` versions |
| "Commentary unavailable, running without it" | no ffmpeg, no audio track, or Whisper model missing | install ffmpeg; check `models/whisper/` |
| Gradle: `Unsupported class file major version 69` | JDK 25 (Android Studio's JBR) used | `flutter doctor -v` shows the JDK; use JDK 17 (`flutter config --jdk-dir`), do not change it globally from one project |
| Gradle: `invalid source release: 21` | a project targets Java 21 on the shared JDK 17 | keep `VERSION_17` in `android/app/build.gradle.kts` |
| New user's uploads not in their match list | sign-up does not save the session (7.8) | log in once after signing up |
| Highlights page shows another user's matches | not logged in -> `user_id=1` | log in |
| Upload rejected 415 / 413 | extension not mp4/mov/avi/mkv, or > 2 GiB | convert or trim the video |
| `backend/uploads` keeps growing | uploads of abandoned jobs are kept 24 h | `FOOTY_UPLOAD_MAX_AGE_HOURS`; sweep runs at startup and after each job |
| Disk filling under `backend/static/clips` | clips are never deleted (owner decision) | delete old `clips/<job_id>/` folders by hand if acceptable; the match then shows no clip |

---

## 12. Glossary

| Term | Meaning in this code base |
| --- | --- |
| Job | One upload's processing run, keyed by a uuid `job_id`; lives in `_jobs` (memory). |
| Match | The stored result of a completed job (`matches` row). |
| Play | One team's participation in a match (`plays` row) carrying its final `score`. Not "a passage of play". |
| Event | A detected happening (`goal`, `foul`, `pass`, ...); an `events` row. |
| Occurrence | One participant of one event (`occurrences` row) with `role`, shirt number and name. |
| Actor / participant | `participants` = who the detector saw take part; `actors` = the same people after identification (number, name, image). |
| Role | A participant's part: `scorer`, `assister`, `shooter`, `goalkeeper`, `passer`, `receiver`, `fouler`, `fouled`, `player`, `referee`, `taker`. |
| Primary actor | The actor copied into `event["scorer"]` (historical name) and stored first: `PRIMARY_ROLE` in `main.py`. |
| Track / track id | One player followed across frames by ByteTrack. Ids restart at every camera cut. |
| Tracklet | A track's fragment within one camera shot; the identity tracker re-links them. |
| Cut / dissolve / scene boundary | A camera change (hard cut) or crossfade (dissolve) found by `scene/cuts.py`; resets tracking, possession and team votes. |
| Analysed frame | A source frame chosen by `scheduled_frames` (10-12 fps for API jobs) and sent to detection. Decoded frames include every frame. |
| Player height | The unit for distances and speeds in `track_events` (running median box height). |
| Kit / cluster | One of the two team colour groups found by KMeans at upload; kit 1 = team A = `team_id` 1. |
| Kit sample | A rendered player image of one kit shown on the confirmation screen (`sample_images`). |
| Spotting | The E2E-Spot action-spotting model (`models/checkpoint_088.pt`) scoring frames for Goal, Shots on/off target, Kick-off. |
| Kick-off | A spotted restart; used only to separate a second goal from a replay. |
| Replay | A repeated showing of a goal, removed by `drop_replayed_goals`. |
| OCR | Optical character recognition with PaddleOCR: shirt numbers, score graphic, scorer captions. |
| Score graphic / score bug / scoreboard | The broadcaster's on-screen score box (`scoreboard/`). |
| Short code | Three-letter team code as on the score graphic (`teams.short_code`, e.g. RMA). |
| Caption | The broadcaster's lower-third text naming the scorer after a goal. |
| Celebration | The close-ups after a goal, used to vote for the scorer's shirt number. |
| Live read | The shirt number of the shooter/last holder at the goal moment. |
| Fusion | Weighted voting of candidate scorers in `fuse_scorer`. |
| Unconfirmed | A fused scorer below confidence 0.7: number kept, no name. |
| Roster / squad | The list of (shirt number, name, position) per team: `players` table in the database, `data/rosters/*.json` on disk. |
| Slot / unnamed slot | A `players` row with a number and NULL name, created when a number was read for a team with no roster entry. |
| Alias | A normalised spelling in `team_aliases` that resolves to one team. |
| Normalise | `normalise_team_name`: casefold, strip accents and punctuation, drop club affixes at the ends. |
| Seeding | Loading roster files into the database (`scripts/seed_roster.py`). |
| Gallery / enrolment | Reference images of a player in `data/player_gallery`, used for re-identification. |
| Re-ID | Appearance embedding (OSNet) compared with gallery images. |
| Identity posterior | `identity/posterior.py`: per-track probability over squad players. |
| Commentary mention | A squad player's name found in the Whisper transcript, with cue words. |
| Profile | A `PipelineConfig` instance: `"full"` (CLI default) or `"short_clip"` (API). |
| Feature switch | One block in `features.yaml`. |
| Visible event types | Event types the switches turn on (`visible_event_types`); used both to filter job output and by the app. |
| Clip / reel | A highlight video of at most 10 s per passage of play / all clips joined (`reel.mp4`). |
| Pitch calibration | Homography from pitch keypoints to metres; off in API jobs (model not trustworthy). |
| Legacy model | The LSTM `final_model_80.pth` (`event_detector.py`, `event_logic.py`); off; kept for comparison. |

---

## 13. Appendix: code findings and stale documentation

Found while reading the code for this guide. Items marked "probable" were
traced by reading, not observed on a device or a live database.

**Fixed on 2026-11-07** (after this appendix was first written):

- *Event-level player fields came from the last occurrence.* `get_match_summary`
  now orders occurrences by `occurrence_id` and keeps the **first** row per
  event (the primary actor) for `events[].teamName/jerseyNumber/playerName`;
  before, a goal with an assist reported the assister as the scorer (proved by
  `tests/test_persist_match.py::test_match_summary_reports_the_scorer_not_the_assister`,
  which fails on the old line with `10 != 7`). The app's `MatchEvent.fromJson`
  now also prefers the occurrence row's `detectedJerseyNo`/`detectedPlayerName`
  over the event-level keys, so older servers show the right player too.
- `requirements.txt` now lists `httpx` and `pyyaml`.
- `FeatureFlags.fallback` now equals the server defaults (`goal`, `foul`).
- The highlights page shows the "Goals" chip only when goals are switched on.

### 13.1 Probable bugs still open

1. *(fixed 2026-11-07, see above)*
2. **Sign-up does not save the session** (`signup_screen.dart` never calls
   `SessionService.saveLogin`); see 7.8 for the consequences.
3. **`/highlights` has no in-app entry point** (7.2). Unverified on a device.
4. *(fixed 2026-11-07: `requirements.txt` lists `pyyaml` and `httpx`.)* Note
   that a missing PyYAML still makes every feature fall back to its default
   with only a console warning.
5. **`events.description` is VARCHAR(500)** while names are NVARCHAR; names
   with characters outside the server code page may be mangled in the stored
   description (unverified on the live database).
6. **Summary team order is not defined**: `teams` in `GET /api/matches/{id}`
   comes from `WHERE team_id IN (...)` with no `ORDER BY`, and the app treats
   `teams[0]` as the left/"home" side. Scores are matched by `teamId`, so they
   stay correct; only the left/right placement may not follow kit 1/kit 2.
7. **`AIPredictionsScreen` (uncommitted) reads `teams`/`homeTeam` from
   `GET /api/matches` rows**, which carry neither, so `matchScoreline` falls
   back to `"Match #<id>"` for every row.
8. **Unused or misleading CLI options**: `--scoreboard-every` is parsed and
   never read; `--image-size` help states 1280/960 defaults, the code uses
   640/960 (`PipelineConfig.detector_size`).
9. **Models load even when switched off**: `_process_job` always fetches the
   spotting, re-ID and Whisper models from the cache (loading them on the first
   job), even when `action_spotting` or `commentary_names` is off. Only the
   work is skipped, not the load or memory.
10. **`goal_scorer_name: no` still spends live shirt OCR on goals**
    (`_resolve_identities` runs for every output event before the switch is
    applied).
11. *(fixed 2026-11-07: the fallback is now `goal`, `foul`.)*
12. *(fixed 2026-11-07: the Goals chip follows `detect_goals`.)*

### 13.2 Where older notes no longer match the code

| Note | Says | Code on 2026-11-06 |
| --- | --- | --- |
| Vault `Known Risks and Gaps`, "Open (2026-11-06, team picker)" | "The Flutter app still sends free-text names" | The app picks teams from `/api/teams` and posts `team_1_id`/`team_2_id` plus names (`ProcessingService.confirmTeams`, commit `6a98c24`). |
| `backend/schemas.py` comment on `TeamConfirmationRequest` | "The web dropdown sends the database team ids; older app builds send typed names only" | The current app build sends ids too. |
| Vault `Known Risks and Gaps` (2026-10-16) | replay window 60 s | `GOAL_REPLAY_SECONDS = 20.0`. |
| Vault `Known Risks and Gaps` (2026-09-25) | clips have no audio, 3.7x bitrate, duplicates | resolved 2026-09-24 (`highlights/encoder.py`, `segments.py`). |
| `main.py --image-size` help | 1280 / 960 | 640 / 960 |

### 13.3 Details of the brief that the code refines

- `sample_images` is not a top-level key of the job status; it is a key of each
  entry in `team_candidates` (next to `cluster_id`, `label`, `sample_image`).
- `PipelineConfig.short_clip` has the signature
  `short_clip(cls, debug_ocr=False, features=None)`.
- `detect_goals`, `detect_shots`, `detect_saves`, `detect_crosses` and
  `detect_passes` do not switch any detector off; they only change
  `output_events`. Passes are always tracked internally (assists), shots and
  saves always (goal evidence), goals always (score, merge).
- `GET /api/teams` filters on players with a non-NULL `name`, not on any
  player.
- The app's `lib/services/` also contains `api_client.dart` and
  `prediction_service.dart`.
