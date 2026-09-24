"""Load data/rosters/*.json into the teams and players tables.

    python -m scripts.seed_roster                  # seed every roster file
    python -m scripts.seed_roster --team "Real Madrid"
    python -m scripts.seed_roster --dry-run        # show what would change
    python -m scripts.seed_roster --report         # print the stored roster

Run by hand, not on application startup: seeding every boot would overwrite
names corrected directly in the database. Re-running is safe - a second run
reports zero writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.models import Team  # noqa: E402
from backend.roster import SeedStats, load_roster_files, seed_roster  # noqa: E402
from backend.schema import ensure_schema  # noqa: E402


def _report() -> int:
    """Print every team and its squad, so wrong numbers are easy to spot."""
    with SessionLocal() as db:
        teams = db.scalars(select(Team).order_by(Team.team_name)).all()
        if not teams:
            print("no teams in the database yet - run the seeder without --report")
            return 0

        for team in teams:
            aliases = sorted(alias.alias for alias in team.aliases)
            code = team.short_code or "---"
            print(f"\n{team.team_name}  [{code}]  (team_id={team.team_id})")
            print(f"  aliases: {', '.join(aliases) if aliases else '(none)'}")

            squad = sorted(
                team.players, key=lambda player: (player.jersey_no is None, player.jersey_no or 0)
            )
            if not squad:
                print("  squad:   (empty)")
                continue
            for player in squad:
                number = "--" if player.jersey_no is None else f"{player.jersey_no:>2}"
                # An unnamed row is a number the pipeline read off a shirt with
                # no roster entry to match it - exactly what seeding fixes.
                print(f"  #{number}  {player.name or '(unnamed slot)'}")
    return 0


def _print_summary(stats: SeedStats, dry_run: bool) -> None:
    verb = "would change" if dry_run else "changed"
    print(f"\nteams created {stats.teams_created}, matched {stats.teams_matched}")
    print(f"aliases added {stats.aliases_added}")
    print(f"short codes {'to set' if dry_run else 'set'}: {stats.codes_set}")
    print(
        f"players {verb}: {stats.players_inserted} inserted, "
        f"{stats.players_named} unnamed slots filled, "
        f"{stats.players_renamed} renamed, {stats.players_unchanged} unchanged"
    )
    for note in stats.notes:
        print(f"  note: {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", help="seed only the roster whose team_name matches")
    parser.add_argument("--dry-run", action="store_true", help="report changes, write nothing")
    parser.add_argument("--report", action="store_true", help="print the stored roster and exit")
    parser.add_argument("--dir", type=Path, default=None, help="roster directory to read")
    args = parser.parse_args(argv)

    # Creates teams, players and team_aliases if they are absent. It does not
    # add columns to a table that already exists - there is no migration tool
    # in this project, so a column change needs a hand-written ALTER.
    Base.metadata.create_all(bind=engine)
    # ...so the one additive column change is applied here instead.
    for change in ensure_schema(engine):
        print(f"schema: added {change}")

    if args.report:
        return _report()

    try:
        rosters = load_roster_files(args.dir, args.team)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not rosters:
        target = f" matching {args.team!r}" if args.team else ""
        print(f"no roster files found{target}", file=sys.stderr)
        return 1

    totals = SeedStats()
    with SessionLocal() as db:
        for path, payload in rosters:
            stats = seed_roster(db, payload, dry_run=args.dry_run)
            totals.merge(stats)
            print(
                f"{payload['team_name']:<22} {len(payload.get('players') or []):>3} in file  "
                f"({path.name})"
            )
        if args.dry_run:
            db.rollback()
        else:
            db.commit()

    _print_summary(totals, args.dry_run)
    if args.dry_run:
        print("\ndry run - nothing was written")
    elif not totals.wrote_anything:
        print("\nalready up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
