# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

Install the complete environment, including PaddleOCR for jersey reading, with:

```powershell
python -m pip install -r requirements.txt
```

## Scorer identification

The scorer pipeline uses the last tracked player in possession before a confirmed
goal. It then reads the best torso crops from several frames and accepts a jersey
number or shirt name only when repeated OCR results agree. The command-line entry
point still accepts explicit kit colors for advanced and automated use:

```powershell
python main.py input_videos/test.mp4 `
  --team-a-name "Team A" --team-a-color "#D71920" `
  --team-b-name "Team B" --team-b-color "#1D428A" `
  --no-display
```

If the shirt name cannot be read, the result is still reported as, for example,
`GOAL — Team A — #7 scored`. When the number also lacks repeated evidence, the
summary says that the scorer number is unavailable instead of guessing. Enhanced
OCR crops are retained under `output_videos/ocr_debug/` by default.

## Match events from the tracks

Besides the goal/foul model, `track_events/` derives events from the player and
ball tracks themselves and reports them with a confidence and
`"source": "tracks"`:

| Event | How it is judged |
| --- | --- |
| `pass` | possession moves between two team-mates and the ball travels between them |
| `cross` | a long, lateral, airborne pass that lands near the opposing goalkeeper |
| `shot` | the ball leaves a player fast, heading at the goalkeeper |
| `save` | a shot the goalkeeper gathers, or that reverses direction at the goalkeeper |
| `penalty` | a ball still for 1.5 s, one taker beside it, the keeper a spot-kick away, everyone else standing off, then a kick at the keeper |
| `yellow_card` / `red_card` | a card-sized yellow or red blob held above a tracked referee's head for a third of a second |

Every participant of an event (passer and receiver, shooter and keeper, booked
player and referee) is stored as its own `occurrences` row with the shirt number
OCR read for that track and the name looked up from the `players` roster on
`(team_id, jersey_no)`. Passes are recorded without a highlight clip. The
goalkeeper's team is deduced as the side opposite the shooter, because keeper
kits match neither strip. All of these are rule-based approximations without
pitch calibration; see the vault note *Track Event Detection* for measured
results and the resources that would make them more accurate.

Run the tests with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

## Goals from the broadcast scoreboard

On broadcast footage the score graphic is the most reliable goal signal, so
`main.py` reads it (`scoreboard/`) with the PaddleOCR engine already loaded for
shirt numbers. The graphic is found once by scanning the top and bottom of the
frame, then only its small region is read every `--scoreboard-every` seconds
(default 1). A new score counts after two consecutive reads agree; a rise of
one goal for one team is a goal, a fall withdraws the last goal (VAR), and
anything else is logged and ignored.

The goal's moment and scorer come from the window between the last read of
the old score and the first read of the new one: a pitch goal-line event if
there is one, else the latest shot by the scoring team, else the last player of
that team in possession. Goals the graphic never confirms are kept and marked
`"scoreboard": "not confirmed"`; see *Goal sources* below.

Codes are matched to the two named teams from, in order: `--team-a-code` /
`--team-b-code` (the API passes `teams.short_code` and three-letter aliases),
`data/rosters/*.json`, then spelling ("MCI" in "Man City"). Disable with
`--no-scoreboard`. Clips without a score graphic - most amateur footage - get
no scoreboard goals.

## Goal sources and the action-spotting model

Goals come from four sources, merged at the end of `process_video`:

1. **Tracks** (`track_events/detector.py`): a shot whose ball enters the goal
   frame and vanishes, rests, or is cut away from.
2. **Spotting** (`spotting/`, `models/checkpoint_088.pt` + `models/config.json`):
   an E2E-Spot network trained on SoccerNet-v2, run on 2 frames a second of
   the same decode. Its outputs are in sorted class order (E2E-Spot writes
   `class.txt` with `sorted()`). On the fixture clips it is good at shots and
   weak at goals, so a spotted goal counts alone only at >= 0.5, at >= 0.15
   next to a shot; spotted shots fill in where tracking saw none; spotted
   kick-offs separate a second goal from a replay. `--no-spotting` disables
   it; `python -m scripts.eval_spotting --rates 2` re-measures it.
3. **Scoreboard** (`scoreboard/`): a confirmed score change; the window reaches
   6 s before the last old-score read because the graphic lags the goal.
4. **Legacy LSTM** (`event_detector.py`): off; `--legacy-event-model` opts in.

All windows in the event detector are in seconds, not frames: the API analyses
10-12 frames a second of a 25-60 fps source, and frame counts tuned at 25 fps
were the reason goals were found on some videos and missed on others.

## Squads from a spreadsheet

```powershell
.venv\Scripts\python.exe -m scripts.import_squads "$env:USERPROFILE\Downloads\football_squads_2026_27.xlsx" --dry-run
.venv\Scripts\python.exe -m scripts.import_squads "$env:USERPROFILE\Downloads\football_squads_2026_27.xlsx"
```

Sheets with `League, Club, Jersey #, Player, Position` (or `Country` for
national teams) become `data/rosters/*.json` and are seeded into `teams` /
`players`, including `players.position` (GK/DF/MF/FW). A team already in
`data/rosters` keeps its scoreboard code and aliases; a shirt number listed
twice keeps the first player and is reported. Re-running is safe.

## Player names: commentary first, then shirt numbers

Names come from, in order of trust:

1. **Commentary** (`commentary/`, faster-whisper `small`, `models/whisper/`):
   the audio is transcribed with word times, squad names are matched
   (misspellings, particles such as "van Dijk", single names such as "Rodri";
   surnames that are ordinary words need the first name). A spoken name is
   attached to the player holding the ball when it was said (same team only).
   Around each goal the scorer is the named player with a scoring word, named
   last before the goal and repeated after it; the assister a team-mate named
   with a passing word ("cross from X") or just before the scorer. "Foul by X /
   on Y" names fouler and fouled. "Ruled out / flag up" is recorded on the goal.
   `--no-commentary` disables it; `--whisper-model` picks another size.
2. **Shirt numbers** read from every readable player (not only the ball
   holder), resolved to names through the squad.
3. **Detected passes**: the last team-mate pass to the scorer is the assist
   when the commentary names none.
4. **Saves**: the keeper is the team's number one from the squad (shirt 1 if a
   keeper wears it) unless a shirt read or the commentary says otherwise.

Each `occurrences` row stores its `role` (scorer, assister, fouler, ...). Web
results list goals and fouls; each goal carries its assister.

## Naming players from 1-10 reference images

No network is trained per player. A pretrained person re-ID network (OSNet,
`models/reid/osnet_x1_0_msmt17.pt`) turns photos into embeddings; each player
gets a prototype (1-2 images, each with five augmented views) and, once every
enrolled player of a team has 3+ images, a per-team logistic regression.
During analysis every tracked player keeps a probability over his team's squad
(`identity/posterior.py`): appearance every 0.5 s, shirt-number reads, the
goalkeeper role, one name per player on screen (Hungarian assignment), and
re-linking of tracklets across camera cuts. A name is attached when its
probability is >= 0.7. Foot positions are recorded per track in the result's
`player_tracks` (pitch metres too when a pitch calibration is accepted).

Measured honestly (2026-10-14, 1080p fixture clip, tracklets as labels): the
pedestrian-trained OSNet picks the right player among team-mates only 32-40%
of the time, so appearance nudges the belief and shirt numbers decide. A
football-trained re-ID model (SoccerNet-ReID) is the upgrade path.

```powershell
.venv\Scripts\python.exe -m scripts.enroll_player --team "Real Madrid" --jersey 7 photo1.jpg photo2.jpg
```

The API equivalent is `POST /api/players/{player_id}/reference-images`
(multipart `images`, 1-10 files), with `GET` to list and
`DELETE /api/players/{player_id}/reference-images/{image_id}`. Images are
stored in `data/player_gallery/` (git-ignored, not publicly served).
`--no-reid` disables naming from images.

## Team and player tables

| Table | Primary key | Foreign keys | Other keys |
| --- | --- | --- | --- |
| `teams` | `team_id` | - | `team_name` unique; `short_code` (3 letters, indexed, not unique) |
| `team_aliases` | `alias_id` | `team_id` -> `teams.team_id` | `alias` unique (normalised spelling) |
| `players` | `player_id` | `team_id` -> `teams.team_id` | unique (`team_id`, `jersey_no`) |

`teams.short_code` is added to an existing database by `backend/schema.py`,
which runs at API startup and in the seeder. Seed or update squads with:

```powershell
.venv\Scripts\python.exe -m scripts.seed_roster --dry-run
.venv\Scripts\python.exe -m scripts.seed_roster
.venv\Scripts\python.exe -m scripts.seed_roster --report
```

## Trained ball and pitch models

`main.py` picks up two optional weights from `models/` automatically; the API
job flow uses them too, because it calls the same `process_video`.

| File | What it does | Disable with |
| --- | --- | --- |
| `models/ball_detector_best.pt` | single-class ball detector, run at 1280 px on 1080p and 960 px below; replaces the ball class of `models/best.pt` | `--ball-model none` |
| `models/Football_field.pt` | 32-keypoint pitch model (pose); every `--pitch-stride` frames (default 5) it fits an image-to-pitch homography in metres. **Off by default** (trained with mosaic, wrong at the goal end); opt in with `--pitch-model models/Football_field.pt` | default |

A homography is used only when at least six keypoints agree within a metre
(`pitch/calibration.py`). When one holds, crosses need a ball from the wide
channel into the penalty area, penalties need a still ball on the spot, shots
aim at the goal mouth even without a visible keeper, and a ball held behind the
goal line between the posts is reported as a `goal` with `"source": "pitch"`.
Frames without an accepted homography use the player-height rules unchanged.
The run prints how many frames had a ball and how many keyframes the pitch
gate accepted.

The keypoint order is the Roboflow sports `SoccerPitchConfiguration`; a
dataset numbered differently needs only `pitch/layout.py` `VERTICES` changed.
Train both models with `training/train_pitch_and_ball_colab.ipynb`. The pitch
model must be trained with `mosaic=0.0`: a mosaic of four pitch quarters is not
a pitch, and weights trained with it produce keypoints the gate rejects.

## Web app and SQL Server API

The unified FastAPI backend stores users and processed match summaries in the
local `FootballDB` SQL Server Express database and runs video jobs through the
same CV pipeline. Start it with:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the user-facing workflow. It does not ask for
hex colors: after upload, the backend calculates two kit clusters from the video
and returns a representative player image for each. The user names those two
visual teams, then processing resumes and returns the named score sheet plus the
detected scorer image, shirt number, and shirt name when OCR evidence is strong.

Interactive API documentation remains available at `http://localhost:8000/docs`.
The processing API follows the same two-stage flow: submit a file or URL, poll
the job until `awaiting_team_confirmation`, then call
`POST /api/processing/confirm-teams/{job_id}` with `team_1_name` and
`team_2_name`.
