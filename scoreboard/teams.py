"""Which team a three-letter scoreboard code belongs to.

The user names the two teams at the confirmation step; the graphic shows
codes such as POR and MCI. Codes come, in order of trust, from:

1. the caller (the API passes ``teams.short_code`` and 3-letter aliases from
   the database),
2. ``data/rosters/*.json`` (``short_code`` and aliases), so the command line
   works without SQL Server,
3. a spelling test: the code's letters appear in order in the team name and
   the first letters agree - "MCI" in "Man City", "RMA" in "Real Madrid".

Nothing here imports backend.database, which connects to SQL Server on import.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

from backend.team_naming import normalise_team_name

ROSTER_DIR = Path(__file__).resolve().parent.parent / "data" / "rosters"


def _codes_in(payload: dict) -> Set[str]:
    codes = set()
    short = (payload.get("short_code") or "").strip().upper()
    if len(short) == 3 and short.isalpha():
        codes.add(short)
    for alias in payload.get("aliases") or []:
        alias = str(alias).strip()
        if len(alias) == 3 and alias.isalpha():
            codes.add(alias.upper())
    return codes


def expected_codes_for(team_name: str, roster_dir: Optional[Path] = None) -> Set[str]:
    """Scoreboard codes the roster files list for a typed team name."""
    key = normalise_team_name(team_name)
    if not key:
        return set()
    directory = roster_dir or ROSTER_DIR
    if not directory.is_dir():
        return set()
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        names = {normalise_team_name(payload.get("team_name"))}
        names.update(normalise_team_name(alias) for alias in payload.get("aliases") or [])
        if key in names:
            return _codes_in(payload)
    return set()


def code_spells_name(code: str, team_name: str) -> bool:
    """True when the code's letters occur in order in the name, first letters equal."""
    letters = "".join(ch for ch in normalise_team_name(team_name) if ch.isalpha())
    code = code.lower()
    if not letters or not code or letters[0] != code[0]:
        return False
    position = 0
    for ch in code:
        position = letters.find(ch, position)
        if position < 0:
            return False
        position += 1
    return True


def map_codes_to_teams(
    codes: Iterable[str],
    team_names: Dict[int, str],
    known_codes: Optional[Dict[int, Iterable[str]]] = None,
) -> Dict[str, int]:
    """{code: team_id} for as many observed codes as can be decided."""
    codes = [code.upper() for code in codes]
    known = {team_id: {c.upper() for c in (values or [])} for team_id, values in (known_codes or {}).items()}
    mapping: Dict[str, int] = {}

    for code in codes:
        owners = [team_id for team_id, values in known.items() if code in values]
        if len(owners) == 1:
            mapping[code] = owners[0]

    for code in codes:
        if code in mapping:
            continue
        owners = [
            team_id
            for team_id, name in team_names.items()
            if team_id not in mapping.values() and code_spells_name(code, name)
        ]
        if len(owners) == 1:
            mapping[code] = owners[0]

    unmapped_codes = [code for code in codes if code not in mapping]
    unmapped_teams = [team_id for team_id in team_names if team_id not in mapping.values()]
    if len(unmapped_codes) == 1 and len(unmapped_teams) == 1:
        mapping[unmapped_codes[0]] = unmapped_teams[0]
    return mapping


def parse_code_option(value: Optional[str]) -> Set[str]:
    """'POR' or 'POR,FCP' from the command line -> {'POR', 'FCP'}."""
    if not value:
        return set()
    return {part.strip().upper() for part in str(value).split(",") if part.strip()}


def combined_expected(known: Dict[int, Iterable[str]]) -> Optional[Tuple[str, ...]]:
    """All codes to accept, or None (accept any) unless both teams are known."""
    sets = [set(values or []) for values in known.values()]
    if len(sets) == 2 and all(sets):
        return tuple(sorted(sets[0] | sets[1]))
    return None
