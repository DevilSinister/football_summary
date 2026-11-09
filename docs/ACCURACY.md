# Footy AI — Measured Accuracy

Every number below comes from a run recorded in the project's verification
notes, checked against the actual video frames, the broadcast caption, or the
score graphic. Nothing here is an estimate.

**Read this first.**

- **The test set is tiny:** 5 clips with 9 known goals in total.
- **Some rules were tuned on the clip they are measured on** (Levante–Sevilla),
  so those results are optimistic.
- **A percentage on 1–9 cases is not a real-world accuracy.** It only says what
  happened on these clips. The *Confidence* column says how much to trust each
  row.
- **"Unmeasured"** means no clip with a known answer has ever been checked.

## Latest measured run (7 Nov 2026, `scripts/eval_events.py`)

The pipeline now scores itself against the labelled clips in
`data/eval/clips.json`. Run it after any change:

```
.venv\Scripts\python.exe -m scripts.eval_events --rerun        # about 45-60 min on this PC
.venv\Scripts\python.exe -m scripts.eval_events --score-only   # re-score saved results in seconds
```

| Event | Found (recall) | Reported that were real (precision) | Right team | Right player |
|---|---|---|---|---|
| Goals, all 5 clips | 6/9 (67%) | 6/6 (100%) | 6/6 (100%) | 4/4 (100%) |
| Goals, real football only (no video game) | 6/8 (75%) | 6/6 (100%) | 6/6 (100%) | 4/4 (100%) |
| Fouls | not measurable - no labelled clip has a foul | | | |
| Cards | 0 reported on the 2 clips known to have none (no false cards) | | | |

Missed goals: Levante-Sevilla 16 s (Ure, before the score graphic was first
read, nothing else saw it), 58 s (Correia, only the graphic's 0-1 to 1-2 jump
shows it), and the EA FC Mobile goal (the game clip now produces no events).
With `score_jump_goals: yes` the Correia goal is counted but placed at 77.5 s,
so it scores as one miss plus one false goal; that is why it is off.

Fouls: with `foul_double_check: yes`, no foul was reported on any of the 5
clips. The contact rule found none, and the spotting model's only "foul"
(Levante 113.5 s, 0.20) had no second kind of evidence. Whether that is right
cannot be said until a clip with labelled fouls exists.

**To reach and prove 80% on fouls and fouler names, the next step is footage:**
5-10 broadcast clips (with English commentary if possible) that contain fouls
and cards, labelled in `data/eval/clips.json` (time, team, player).

### Test clips

| Clip | Footage | Known events |
|---|---|---|
| Real Madrid v Atlético | broadcast, 480p, 15 s | 0 goals; 1 saved shot; 1 shot off the post (replay) |
| Portugal v Egypt (`test.mp4`) | broadcast, 26 s | 1 goal (graphic 1-0) |
| Porto v Man City | broadcast, 1080p, 26 s, no audio | 1 goal (graphic 0-1); 1 penalty-box tangle |
| EA FC Mobile clip | video game, 1280×592 | 1 goal, no score graphic |
| Levante 3-3 Sevilla | broadcast highlights, 1080p50, 3 min 15 s, English commentary | 6 goals with scorer captions |

---

## 1. Summary table

| What | Result | % | Confidence | Notes |
|---|---|---|---|---|
| **Goals found** (recall) | 7 of 9 | **78%** | Low | 4/6 Levante, 1/1 Portugal, 1/1 Man City, 1/1 FC Mobile. Real football only: 6 of 8 (75%). |
| **Reported goals that were real** (precision) | 7 of 7 | **100%** | Low | No false goal on any of the 5 clips, including the 0-0 one. |
| **Goal credited to the right team** | 7 of 7 found goals | **100%** | Low | One exception on a test copy re-encoded to 50 fps, where Portugal's goal was given to Egypt. |
| **Final score right** | 4 of 4 broadcast clips | **100%** | Medium | Comes from the score graphic: 0-0, 1-0, 0-1, 3-3. |
| **Score-graphic changes read** | 5 of 8 | **63%** | Low | Levante 3/6 (one goal came before the first reading; two arrived together between package clips). Man City 1/1. Portugal 1/1. |
| **Goal scorer named correctly** | 4 of 4 checkable goals | **100%** | **Very low** | All 4 are Levante goals, the clip the naming rules were tuned on. Before tuning, the same clip gave 2 of 4 (50%). The other found goals can't be checked: the scorer isn't visible, or it's a video game. |
| **Scorer from commentary alone** | 4 of 6 | **67%** | Low | Same Levante clip. 1 wrong (17%), 1 blank (17%). |
| **Player names heard in commentary** | 27 mentions matched; 2 players never matched | n/a | Low | "Ure" and "Ndukwe" were misheard. The total number of mentions spoken wasn't counted, so no percentage. |
| **Assister named** | — | **Unmeasured** | — | Levante gave plausible assisters on 4 goals, but none was checked. |
| **Fouls** | 0 of 1 checked foul real; 1 real foul missed | **0%** | Low | Porto v Man City: the one reported foul was probably a false alarm, and the real tangle in the box wasn't flagged. |
| **Yellow / red cards** | — | **Unmeasured** | — | No test clip contains a card; tested on synthetic frames only. After a fix, 0 false card readings on 46 checked frame/box pairs (28 before the fix). |
| **Penalties** | — | **Unmeasured** | — | No test clip contains a penalty; synthetic tests only. |
| **Shots** | 2 of 4 real shots found | **50%** | Low | Missed: a shot off the post seen only in a close-up replay (read as 2 passes), and the FC Mobile volley (found only through a false "ball" on a boot). |
| **Saves** | 1 of 1 found; 0 false at latest runs | **100%** | Very low | Older runs reported 3 false saves (2 Man City, 1 FC Mobile); all removed by later fixes. |
| **Passes** | 2 of 4 checked passes were false | **50% false** | Low | Real Madrid clip: 2 false passes on a close-up replay; the other 2 looked plausible but weren't confirmed. |
| **Passer / receiver names** | — | **Unmeasured** | — | A name lasts only until the next camera cut. |
| **Crosses** | — | **Unmeasured** | — | |

---

## 2. Building blocks (what the events are built from)

| What | Result | % | Notes |
|---|---|---|---|
| Ball found (480p) | 29 of 30 sampled frames | **97%** | And the chosen box was a real ball in 29 of 29 (100%). The old model picked the TV crest in 6. |
| Ball found (1080p) | 28 of 30 sampled frames | **93%** | Over whole clips: 355/385 frames (92%) at 480p, 460/644 (71%) at 1080p. |
| Camera cuts and dissolves found | 8 of 8 | **100%** | 0 false cuts on 2 clips. |
| Shirt numbers read, 480p | 0 tracks | **0%** | Shirts below about 14% of the frame height (roughly below 720p) can't be read. |
| Shirt numbers read, 1080p | 2 numbers (#27, #22) | n/a | Whether they were right wasn't checked. |
| Recognising a player by appearance | 31–40% top-1 | **31–40%** | Chance is about 15–20%. It only nudges the choice; shirt numbers decide. |
| Pitch calibration accepted, 1080p | 5 of 129 keyframes | **4%** | All 5 accepted fits were wrong. Switched off. |
| Pitch calibration accepted, 480p | 14 of 77 keyframes | **18%** | Halfway-line fits right; some goal-end fits wrong. Switched off. |
| Team split on striped kits, 480p | — | not a % | Several striped players were labelled as the white team; fixed at 1080p. |
| Action-spotting model, goals | 0 of 2 goals confident | **0%** | Goal score 0.22 on one goal, about 0 on the other. Used only together with other evidence. |
| Action-spotting model, shots on target | 2 of 2 shots scored ≥ 0.42 | **100%** | Scores 0.85 and 0.42. |
| Old LSTM goal model | max goal score 0.13–0.20 | — | Never reached its 0.5 threshold; switched off. |

---

## 3. How fast it is (same PC, CPU only)

| Clip | Analysis time | Times slower than the video |
|---|---|---|
| 26 s broadcast clip | about 400 s (6.5 min) | ×15 |
| 3 min 15 s 1080p50 highlights | 1,811–2,414 s (30–40 min) | ×9–12 |
| Team detection at upload (26 s clip) | 11 s | — |

---

## 4. What would make these numbers trustworthy

1. **A labelled test set:** 30+ goals from 10+ matches the rules were *not*
   tuned on, with the goal second, team, scorer and assister, and a few fouls
   and cards.
2. **A script that runs the pipeline on it** and fills this table
   automatically, overall and per resolution and source.
3. **Saved detections per clip,** so changing a rule re-scores in seconds
   instead of re-running 30 minutes of video.

*Source: the Verification Evidence and Known Risks notes, 14 Sep – 7 Nov 2026.
Update this file whenever a new measured run is recorded.*
