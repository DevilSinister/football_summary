"""Squad spreadsheet -> roster payloads, and the additive player/occurrence columns."""

import os
import tempfile
import unittest
import zipfile
from xml.sax.saxutils import escape

from sqlalchemy import create_engine, inspect, text

from backend.schema import ensure_schema
from backend.squad_import import merge_with_existing, read_xlsx, rosters_from_workbook


def _sheet_xml(rows):
    lines = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(
            f'<c r="{chr(65 + c)}{r}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>'
            for c, v in enumerate(row) if v is not None
        )
        lines.append(f'<row r="{r}">{cells}</row>')
    return ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(lines)}</sheetData></worksheet>')


def write_xlsx(path, sheets):
    with zipfile.ZipFile(path, "w") as z:
        entries = "".join(
            f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(sheets, 1)
        )
        z.writestr("xl/workbook.xml", (
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{entries}</sheets></workbook>"))
        rels = "".join(
            f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
            for i in range(1, len(sheets) + 1))
        z.writestr("xl/_rels/workbook.xml.rels",
                   f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>')
        for i, rows in enumerate(sheets.values(), 1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(rows))


class SquadImportTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".xlsx")
        os.close(handle)
        write_xlsx(self.path, {
            "La Liga": [
                ["League", "Club", "Jersey #", "Player", "Position"],
                ["La Liga", "Atlético Madrid", "13", "Jan Oblak", "Goalkeeper"],
                ["La Liga", "Atlético Madrid", "7", "Antoine Griezmann", "Forward"],
                ["La Liga", "Atlético Madrid", "7", "Someone Else", "Forward"],
                ["La Liga", "Atlético Madrid", "", "No Number", "Defender"],
            ],
            "International Teams": [
                ["Country", "Jersey #", "Player", "Position"],
                ["Portugal", "7", "Cristiano Ronaldo", "Forward"],
            ],
        })

    def tearDown(self):
        os.remove(self.path)

    def test_sheets_become_rosters_with_position_codes(self):
        payloads, notes = rosters_from_workbook(read_xlsx(self.path), "squads.xlsx", "2026-27")
        by_team = {p["team_name"]: p for p in payloads}
        self.assertEqual(set(by_team), {"Atlético Madrid", "Portugal"})
        self.assertEqual(
            by_team["Atlético Madrid"]["players"],
            [{"jersey_no": 7, "name": "Antoine Griezmann", "position": "FW"},
             {"jersey_no": 13, "name": "Jan Oblak", "position": "GK"}],
        )
        self.assertEqual(by_team["Portugal"]["players"][0]["name"], "Cristiano Ronaldo")
        self.assertTrue(any("Someone Else" in n for n in notes))   # duplicate number reported
        self.assertTrue(any("No Number" in n for n in notes))      # missing number reported

    def test_an_existing_file_keeps_its_code_and_aliases(self):
        merged = merge_with_existing(
            {"team_name": "Atlético Madrid", "aliases": [], "players": []},
            {"team_name": "Atletico Madrid", "short_code": "ATM", "aliases": ["atleti"]},
        )
        self.assertEqual(merged["short_code"], "ATM")
        self.assertEqual(merged["aliases"], ["atleti", "Atletico Madrid"])


class SchemaColumnTests(unittest.TestCase):
    def test_position_and_role_are_added_once(self):
        engine = create_engine("sqlite://", future=True)
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE players (player_id INTEGER PRIMARY KEY, name VARCHAR(100))"))
            connection.execute(text("CREATE TABLE occurrences (occurrence_id INTEGER PRIMARY KEY)"))
        applied = ensure_schema(engine)
        self.assertIn("players.position", applied)
        self.assertIn("occurrences.role", applied)
        self.assertIn("position", {c["name"] for c in inspect(engine).get_columns("players")})
        self.assertEqual([a for a in ensure_schema(engine) if a in ("players.position", "occurrences.role")], [])


if __name__ == "__main__":
    unittest.main()
