# Team Identity Resolution Design

## Goal

Automatically differentiate football teams and assign scoreboard-derived team names to player kit clusters without manual fallback. The system must work across varied videos where team abbreviations are not fixed and kit colors may be easy, similar, or inconsistent.

The resolver should prefer `unknown_team` or a plain `team_id` over a forced wrong team name when confidence is weak.

## Context

The current pipeline detects event clips first, then post-processes only those short clips for player identity, team assignment, and jersey OCR. This is the right performance shape: expensive OCR and team-name logic should run on event clips, not the full match.

Current relevant components:

- `ScoreboardReader`: OCRs team abbreviations and tries to sample small scoreboard marker colors.
- `TeamAssigner`: extracts player torso colors and clusters players into two kit groups.
- `main.py`: post-processes event clips, identifies last-touch player, and OCRs jersey number only for goal scorer candidates.
- `test_main_clips.py`: fast clip-only test runner capped at 500 frames.

## Proposed Approach

Use a multi-signal fusion resolver. No single signal is reliable enough across videos, especially because scoreboard color markers are very small and kit similarity varies.

The resolver combines:

- Scoreboard OCR tokens, such as `RMA`, `MCI`, `LIV`, `ARS`, or any other 2-5 letter uppercase team abbreviation.
- Scoreboard marker colors, when visible.
- Player torso/kit cluster colors.
- Clip-level temporal consistency across sampled frames.

The output is a mapping from internal team cluster IDs to team names:

```python
{
    1: {"name": "LIV", "confidence": 0.78},
    2: {"name": "MCI", "confidence": 0.76},
}
```

If confidence is too low, the system should leave team names unset and continue with `team_1`, `team_2`, or `unknown_team`.

## Components

### ScoreboardReader

Responsibilities:

- Crop the top scoreboard region.
- OCR arbitrary uppercase team tokens.
- Vote across several frames to reduce one-frame OCR mistakes.
- Sample tiny colored markers near each team label when available.
- Return structured scoreboard identity data.

Proposed output:

```python
ScoreboardIdentity(
    left_name="LIV",
    right_name="MCI",
    left_marker_color=...,
    right_marker_color=...,
    ocr_confidence=0.82,
)
```

The reader must not hardcode specific teams like `RMA` or `AMD`.

### TeamAssigner

Responsibilities:

- Crop player torso regions.
- Filter grass, shadows, glare, and low-saturation background.
- Extract dominant kit-color features.
- Cluster player appearances into two team clusters.

`TeamAssigner` should not own scoreboard-specific logic beyond storing final resolved names by cluster.

### TeamIdentityResolver

New component.

Responsibilities:

- Accept scoreboard identity data and kit cluster data.
- Compare marker colors against kit cluster colors when marker colors are usable.
- Weight marker color as a weak signal because markers may be tiny or noisy.
- Use consistency across frames and samples.
- Produce team-name mapping with confidence scores.

The resolver should be deterministic and testable without running YOLO directly. It should accept plain data structures: names, colors, cluster centers, sample counts, and optional quality scores.

### Event Post-Processing

The event post-processing flow should remain clip-only:

1. Open the raw event clip.
2. Read scoreboard identity across the first sampled frames.
3. Track players and ball inside the clip.
4. Cluster player kits.
5. Resolve team-name mapping.
6. Identify last-touch player before the event moment.
7. Assign that player:
   - `team_id`
   - resolved `team_name`, if confidence is high
   - jersey number, only for goal scorer candidates
8. Rename the event clip with the best available metadata.

## Data Flow

For each event clip:

1. Sample up to 120 frames for scoreboard OCR and marker color collection.
2. Track players and ball for up to the configured clip processing window.
3. Collect player torso features from good crops.
4. Fit two kit clusters.
5. Resolve scoreboard names to kit clusters.
6. Compute last-touch player before the event frame.
7. For goals only, run jersey OCR on the last-touch player's crop.

## Confidence Rules

Use these confidence bands:

- `>= 0.70`: assign team names to clusters.
- `0.45 - 0.69`: keep mapping internally as uncertain, but do not use it in final event names.
- `< 0.45`: do not assign team names.

Final event output examples:

```text
GOAL at 42.5s by LIV, jersey #7
GOAL at 42.5s by team_1, jersey #7
GOAL at 42.5s by unknown_team, jersey #7
```

No manual correction prompt is allowed. Fully automatic behavior means the fallback is uncertainty, not user input.

## Error Handling

- If scoreboard OCR fails, continue with team IDs only.
- If OCR finds only one team token, ignore scoreboard names for that clip.
- If marker colors are missing or too noisy, lower their weight rather than failing.
- If too few player crops are available, delay team calibration.
- If kit clusters are visually too close, avoid assigning names.
- If jersey OCR fails, still report event time, team ID/name if known, and player track ID.

## Testing Strategy

Start with `test_main_clips.py` because it isolates the problem and limits processing to the first 500 frames of each event clip.

The test runner should print:

- OCR team names.
- Whether marker colors were found.
- Kit cluster colors.
- Proposed cluster-name mapping.
- Mapping confidence.
- Last-touch player, team ID, team name, and jersey number if available.

Optional debug outputs should include:

- Scoreboard crop.
- Marker crop candidates.
- Torso samples grouped by cluster.

Only after clip tests are acceptable should the resolver be wired into the full `main.py` post-processing path.

## Non-Goals

- No hardcoded team abbreviations.
- No manual mapping prompts.
- No full-match OCR pass.
- No training a new team identity model in this iteration.
- No guaranteed name assignment when the evidence is weak.
