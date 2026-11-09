# Footy AI — How to Change Things (plain-language guide)

This guide is for anyone, including people who don't write code. It tells you
**which file to open** for each kind of change, and whether the change is
something you can do yourself or something a developer must do.

For the full technical detail behind each change, see
[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md).

---

## 1. The three parts of Footy AI

Think of Footy AI as three boxes:

| Part | What it is | Where it lives |
|---|---|---|
| **The phone app** | What you install on Android (the APK). Shows matches, goals, highlights. | Folder `D:\Projects\footyai\Footy_AI\footy_ai_app` |
| **The server** | The program on the PC that watches the video, finds goals and names players. The phone talks to it over Wi-Fi. | Folder `D:\Projects\football_summary` |
| **The database** | Where teams, players and analysed matches are stored. | SQL Server on the PC, database name `FootballDB` |

**Most important rule:**
- Changes to the **server** or the **database** take effect without touching the phone.
- Changes to the **phone app's code** need a **new APK** to be built and installed.

---

## 2. Turning features on and off (no developer needed)

All the feature switches are in **one file**:

> `D:\Projects\football_summary\features.yaml`

### How to change a switch

1. Open `features.yaml` in Notepad (right-click → Open with → Notepad).
2. Find the feature, for example:
   ```
   detect_passes:
     enabled: no
   ```
3. Change `no` to `yes` (or `yes` to `no`). Keep the spaces exactly as they are.
4. Save the file.
5. Analyse a **new** video in the app. That's it — no restart, no new APK.

### What happens when you switch something

- **yes** → the server does the analysis for it **and** the app shows it.
- **no** → the server skips it **and** the app hides it.
- The app checks the switches every time you open a match, so hiding or showing
  also applies to old matches. But a match analysed while a feature was `no`
  never collected that information, so switching it on later won't fill it in.

### The switches

"Rebuild needed" means a developer must build a new APK before you see the
change. Every switch below says **No**: the current app (built 7 Nov 2026 or
later) follows them by itself.

| Switch in `features.yaml` | What it does | On by default? | Rebuild needed? |
|---|---|---|---|
| `detect_goals` | Finds goals | Yes | No |
| `goal_scorer_name` | Names who scored | Yes | No |
| `assister_name` | Names who gave the assist | Yes | No |
| `score_jump_goals` | Counts a goal the score graphic shows only as a jump of one for each team. Its time and scorer are a guess | No | No |
| `detect_shots` | Lists shots on target, with a clip | No | No |
| `detect_saves` | Lists goalkeeper saves (no clip) | No | No |
| `detect_fouls` | Finds fouls, with a clip | Yes | No |
| `foul_double_check` | Reports a foul only when two kinds of evidence agree (contact, the action-spotting model, a free kick or card right after, the commentator). Fewer false fouls | Yes | No |
| `fouler_name` | Names who committed the foul and who was fouled | Yes | No |
| `detect_cards` | Finds yellow and red cards | No | No |
| `card_double_check` | Reports a card only when two kinds of evidence agree, or the model is very sure | Yes | No |
| `detect_penalties` | Finds penalty kicks | No | No |
| `detect_crosses` | Finds crosses into the box | No | No |
| `detect_passes` | Lists every pass (no clips) | No | No |
| `passer_name` | Names who passed and who received ("#8 Modric → #10 Mbappe"). Only works if `detect_passes` is also yes | No | No |
| `detect_corners` | Lists corners (from the action-spotting model, no player) | No | No |
| `detect_offsides` | Lists offsides (from the action-spotting model, no player) | No | No |
| `detect_free_kicks` | Lists free kicks (from the action-spotting model, no player) | No | No |
| `detect_substitutions` | Lists substitutions (from the action-spotting model, no player) | No | No |
| `highlight_clips` | Cuts video clips and a highlight reel | Yes | No |
| `commentary_names` | Listens to English commentary to name players. Off = faster | Yes | No |
| `read_scoreboard` | Reads the score graphic on TV footage | Yes | No |
| `action_spotting` | A trained model that helps find goals and shots | Yes | No |

Each block in `features.yaml` also has an `accuracy:` line saying how well that
feature has been tested. Many say "Unmeasured" — treat those results with care.
The full table with numbers and percentages is in [ACCURACY.md](ACCURACY.md).

### Where things appear in the app

| Event type | Match Events list (match summary) | Highlights page |
|---|---|---|
| Goals, fouls, cards, penalties, crosses, shots | Yes, when switched on | Yes, when switched on (they have clips) |
| Passes, saves, corners, offsides, free kicks, substitutions | Yes, when switched on | No — they have no video clips |

### If you make a typo

If a line in `features.yaml` is broken, the server ignores that feature's
setting, uses its default, and writes a warning in the server window. You can
also see the switches the server is using by opening this address in a browser
on the PC: `http://localhost:8000/api/features`.

---

## 3. "I want to…" — which file to change

**Who** column: **You** = anyone can do it by following the steps. **Dev** =
needs a developer. **APK** = the phone app must be rebuilt afterwards.

### Features and analysis

| I want to… | File or place | Who |
|---|---|---|
| Turn a feature on or off | `football_summary\features.yaml` | You |
| Make highlight clips longer or shorter | `football_summary\highlights\segments.py` (the numbers `LEAD_…` and `TAIL_…` seconds) | Dev |
| Choose which events get a video clip | `football_summary\highlights\segments.py` (`CLIP_EVENT_TYPES`) | Dev |
| Make goal or foul detection stricter or looser | `football_summary\track_events\detector.py` and `track_events\fouls.py` | Dev |
| Change how many player pictures you can flip through when naming teams | `football_summary\backend\routers\processing.py` (`TEAM_SAMPLE_IMAGES_PER_TEAM`, now 6) | Dev |
| Change how hard the server tries to read shirt numbers | `football_summary\pipeline_config.py` (`jersey_ocr_calls`) | Dev |

### Teams and players

| I want to… | File or place | Who |
|---|---|---|
| Add a missing team (e.g. Egypt) or fix a squad | Copy `football_summary\data\rosters\_template.json`, fill in the team and players, save it in the same folder, then run the "load squads" command in section 4 | You (careful) or Dev |
| Load a whole spreadsheet of squads | Run the "import spreadsheet" command in section 4 | You (careful) or Dev |
| Fix a misspelt team name | The database (table `teams`) — ask a developer | Dev |
| Let a nickname find a team (e.g. "Barca") | Add it to `"aliases"` in that team's file in `data\rosters\`, then run "load squads" | You (careful) |

A team only appears in the app's team list when it has at least one player with
a name in the database.

### The phone app

| I want to… | File | Who |
|---|---|---|
| Point the app at a different PC / new Wi-Fi address | In the app: login screen → **Server settings** → type the PC's address → **Save API address** | You (no rebuild) |
| Change the default server address built into the app | Build command in section 4 (`API_BASE_URL=…`) | Dev + APK |
| Change wording on the team-naming screen | `footy_ai_app\lib\screens\ai_processing_screen.dart` | Dev + APK |
| Change the match summary screen | `footy_ai_app\lib\screens\ai_match_summary.dart` | Dev + APK |
| Change the highlights page or its filter buttons | `footy_ai_app\lib\screens\match_highlights_list.dart` | Dev + APK |
| Change the "Another player" picture button | `footy_ai_app\lib\widgets\kit_sample_viewer.dart` | Dev + APK |
| Change the team search box | `footy_ai_app\lib\widgets\team_picker_sheet.dart` | Dev + APK |
| Change the app's main colours | `footy_ai_app\lib\theme.dart` | Dev + APK |
| Change the app's name on the phone | `footy_ai_app\android\app\src\main\AndroidManifest.xml` (`android:label`) | Dev + APK |

### Server and data

| I want to… | File or place | Who |
|---|---|---|
| Add a new piece of information the app can ask the server for | `football_summary\backend\routers\` (one file per area) | Dev (+ APK to show it) |
| Store a new kind of information in the database | `football_summary\backend\models.py` and `backend\schema.py` | Dev |
| Find analysed matches | Database `FootballDB`, tables `matches`, `events`, `occurrences` | Dev |
| Find video clips on the PC | `football_summary\backend\static\clips\` (one folder per analysis) | You |

---

## 4. Everyday commands

Open **PowerShell**, then copy and paste one line at a time.

**Start the server** (keep this window open while you use the app):
```powershell
cd D:\Projects\football_summary; .\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**Check the server is running** (in a browser on the PC): open `http://localhost:8000/health` — it should say `healthy`.

**Find the PC's Wi-Fi address** (to type into the app's Server settings):
```powershell
ipconfig
```
Look for "IPv4 Address" under "Wireless LAN adapter Wi-Fi", for example `192.168.1.20`.

**Load squads from the `data\rosters` files into the database:**
```powershell
cd D:\Projects\football_summary; .\.venv\Scripts\python.exe -m scripts.seed_roster
```
Add `--dry-run` at the end first to see what would change without changing anything.

**Import a squads spreadsheet:**
```powershell
cd D:\Projects\football_summary; .\.venv\Scripts\python.exe -m scripts.import_squads "C:\path\to\squads.xlsx" --dry-run
```
Remove `--dry-run` to really import.

**Build a new APK** (developer; takes about 5 minutes):
```powershell
cd D:\Projects\footyai\Footy_AI\footy_ai_app; flutter build apk --release --dart-define=API_BASE_URL=http://192.168.1.20:8000
```
Replace the address with the PC's address. The APK appears at
`footy_ai_app\build\app\outputs\flutter-apk\app-release.apk`. Copy it to the
phone and open it to install.

---

## 5. When do I need a new APK?

| Change | New APK? |
|---|---|
| Any switch in `features.yaml` | **No** |
| Adding teams or players | **No** |
| Changing analysis rules on the server | **No** |
| Changing the PC's address (use Server settings in the app) | **No** |
| Changing anything in the `footy_ai_app` folder | **Yes** |
| A switch whose `app_rebuild_needed` says `yes` | **Yes** |

---

## 6. Quick fixes

| Problem | Try this |
|---|---|
| App says it can't reach the server | Is the server window open? Are phone and PC on the same Wi-Fi? Is the address in Server settings right (see `ipconfig`)? |
| The team list in the app is empty | The server can't reach the database, or no teams are loaded — run "Load squads". |
| My team isn't in the list | Add it with a roster file (section 3, Teams and players). |
| I changed `features.yaml` but nothing changed | Did you save the file? Did you analyse a **new** video? Check `http://localhost:8000/api/features` on the PC. |
| Analysis is very slow | Normal on this PC: about 7 minutes for a 30-second clip, 30+ minutes for 3 minutes. Turning `commentary_names` off saves time. |
| Both team pictures show the same team | Tap **Another player** on the wrong picture until it shows the other kit. |
