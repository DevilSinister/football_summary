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
