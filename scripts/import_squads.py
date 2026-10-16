"""Import squads from a spreadsheet into data/rosters and the database.

    .venv\\Scripts\\python.exe -m scripts.import_squads "%USERPROFILE%\\Downloads\\football_squads_2026_27.xlsx" --dry-run
    .venv\\Scripts\\python.exe -m scripts.import_squads "%USERPROFILE%\\Downloads\\football_squads_2026_27.xlsx"

Each team in the workbook becomes (or replaces the players of) one
data/rosters/*.json file; a team already there keeps its scoreboard code and
aliases. Teams not in the workbook (e.g. Porto) are left untouched. Then
scripts.seed_roster loads every roster file into the teams/players tables:
new shirt numbers are inserted, changed names and positions updated.
``--files-only`` skips the database (the command line reads the files).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.roster import ROSTER_DIR  # noqa: E402
from backend.squad_import import merge_with_existing, read_xlsx, rosters_from_workbook  # noqa: E402
from backend.team_naming import normalise_team_name  # noqa: E402


def _existing_rosters(directory: Path) -> dict:
    by_key = {}
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name in [payload.get("team_name"), *(payload.get("aliases") or [])]:
            key = normalise_team_name(name or "")
            if key:
                by_key.setdefault(key, (path, payload))
    return by_key


def _slug(team_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalise_team_name(team_name)).strip("_") or "team"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workbook")
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--files-only", action="store_true", help="write data/rosters only, not the database")
    parser.add_argument("--dir", type=Path, default=ROSTER_DIR)
    args = parser.parse_args(argv)

    payloads, notes = rosters_from_workbook(
        read_xlsx(args.workbook), os.path.basename(args.workbook), args.season
    )
    existing = _existing_rosters(args.dir)
    written = updated = 0
    for payload in payloads:
        match = existing.get(normalise_team_name(payload["team_name"]))
        path = match[0] if match else args.dir / f"{_slug(payload['team_name'])}.json"
        merged = merge_with_existing(payload, match[1] if match else None)
        updated += bool(match)
        written += not match
        if not args.dry_run:
            path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    players = sum(len(p["players"]) for p in payloads)
    keepers = sum(1 for p in payloads for q in p["players"] if q["position"] == "GK")
    print(f"{len(payloads)} teams, {players} players ({keepers} goalkeepers) in the workbook")
    print(f"roster files {'to write' if args.dry_run else 'written'}: {written} new, {updated} replaced")
    for note in notes:
        print(f"  note: {note}")
    if args.files_only:
        return 0

    from scripts.seed_roster import main as seed_main

    if args.dry_run:
        print("\ndatabase: dry run of the seeder over the current roster files")
        return seed_main(["--dir", str(args.dir), "--dry-run"])
    return seed_main(["--dir", str(args.dir)])


if __name__ == "__main__":
    raise SystemExit(main())
