"""Squads from a spreadsheet: .xlsx -> roster payloads (data/rosters format).

The workbook has one sheet per league with ``League, Club, Jersey #, Player,
Position`` and an international sheet with ``Country`` instead of league and
club. The reader uses only the standard library (an .xlsx file is a zip of
XML), so no spreadsheet package is needed.

Positions are mapped to the roster codes GK/DF/MF/FW. A shirt number used
twice in one squad keeps its first player; the other is reported, because the
database allows one player per (team, shirt number).
"""

import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
POSITION_CODES = {
    "goalkeeper": "GK", "keeper": "GK", "gk": "GK",
    "defender": "DF", "df": "DF",
    "midfielder": "MF", "mf": "MF",
    "forward": "FW", "striker": "FW", "attacker": "FW", "winger": "FW", "fw": "FW",
}
TEAM_COLUMNS = ("Club", "Country", "Team")


def _column(ref: str) -> int:
    index = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        index = index * 26 + ord(ch) - 64
    return index - 1


def read_xlsx(path: str) -> Dict[str, List[List[Optional[str]]]]:
    """Sheet name -> rows of cell text (None for empty cells)."""
    with zipfile.ZipFile(path) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{_NS['m']}}}t")))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.get("Id"): rel.get("Target") for rel in relations}
        sheets: Dict[str, List[List[Optional[str]]]] = {}
        for sheet in workbook.find("m:sheets", _NS):
            target = targets[sheet.get(f"{{{_NS['r']}}}id")].lstrip("/")
            target = target if target.startswith("xl/") else "xl/" + target
            rows = []
            for row in ET.fromstring(archive.read(target)).iter(f"{{{_NS['m']}}}row"):
                cells: Dict[int, str] = {}
                for cell in row.findall("m:c", _NS):
                    kind, value = cell.get("t"), cell.find("m:v", _NS)
                    if kind == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iter(f"{{{_NS['m']}}}t"))
                    elif value is None:
                        continue
                    elif kind == "s":
                        text = shared[int(value.text)]
                    else:
                        text = value.text
                    cells[_column(cell.get("r"))] = text
                if cells:
                    rows.append([cells.get(i) for i in range(max(cells) + 1)])
            sheets[sheet.get("name")] = rows
    return sheets


def _jersey(value) -> Optional[int]:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return int(text) if text.isdigit() else None


def rosters_from_workbook(
    sheets: Dict[str, Sequence[Sequence]], source: str, season: str
) -> Tuple[List[dict], List[str]]:
    """(roster payloads, notes about rows that were skipped)."""
    payloads: Dict[str, dict] = {}
    notes: List[str] = []
    for sheet_name, rows in sheets.items():
        if not rows:
            continue
        header = [str(h or "").strip() for h in rows[0]]
        team_col = next((header.index(c) for c in TEAM_COLUMNS if c in header), None)
        if team_col is None or "Player" not in header or "Jersey #" not in header:
            notes.append(f"sheet {sheet_name!r}: no Club/Country, Jersey # and Player columns; skipped")
            continue
        player_col, jersey_col = header.index("Player"), header.index("Jersey #")
        position_col = header.index("Position") if "Position" in header else None

        def cell(row, index):
            return row[index] if index is not None and index < len(row) else None

        for line, row in enumerate(rows[1:], start=2):
            team, name = str(cell(row, team_col) or "").strip(), str(cell(row, player_col) or "").strip()
            jersey = _jersey(cell(row, jersey_col))
            if not team or not name:
                continue
            if jersey is None:
                notes.append(f"{sheet_name} row {line}: {name} ({team}) has no shirt number; skipped")
                continue
            payload = payloads.setdefault(team, {
                "team_name": team,
                "aliases": [],
                "season": season,
                "source": f"{source}, sheet {sheet_name}",
                "needs_review": False,
                "players": [],
            })
            taken = next((p for p in payload["players"] if p["jersey_no"] == jersey), None)
            if taken is not None:
                notes.append(f"{team} #{jersey}: {name} skipped, number already used by {taken['name']}")
                continue
            position = POSITION_CODES.get(str(cell(row, position_col) or "").strip().lower())
            payload["players"].append({"jersey_no": jersey, "name": name, "position": position})
    for payload in payloads.values():
        payload["players"].sort(key=lambda p: p["jersey_no"])
    return list(payloads.values()), notes


def merge_with_existing(payload: dict, existing: Optional[dict]) -> dict:
    """The new squad, keeping the hand-kept scoreboard code and aliases."""
    if not existing:
        return payload
    merged = dict(payload)
    merged["short_code"] = existing.get("short_code") or payload.get("short_code")
    aliases = list(existing.get("aliases") or [])
    aliases += [a for a in payload.get("aliases") or [] if a not in aliases]
    if existing.get("team_name") and existing["team_name"] != payload["team_name"]:
        aliases.append(existing["team_name"])
    merged["aliases"] = aliases
    return merged
