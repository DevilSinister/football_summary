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
that team in possession. While the scoreboard is readable it is the only
source of goals; when no graphic is found, goals fall back to the track and
pitch rules.

Codes are matched to the two named teams from, in order: `--team-a-code` /
`--team-b-code` (the API passes `teams.short_code` and three-letter aliases),
`data/rosters/*.json`, then spelling ("MCI" in "Man City"). Disable with
`--no-scoreboard`. Clips without a score graphic - most amateur footage - get
no scoreboard goals.

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
| `models/football_field_best.pt` | 32-keypoint pitch model (pose); every `--pitch-stride` frames (default 5) it fits an image-to-pitch homography in metres | `--pitch-model none` |

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
