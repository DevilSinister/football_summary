"""Canonical form for a team name typed by a user.

Team names reach the database as free text from two TextFields in the app, so
the same club arrives as "Real Madrid", "real madrid cf" and "Real Madrid C.F."
across three jobs. Left alone those become three rows in `teams`, and the
roster lookup on (team_id, jersey_no) then misses on two of them and the
scorer comes back nameless.

This module owns the one function that collapses those spellings. It does not
import backend.database, which connects to SQL Server at import time, so the
seeder and the tests can use it without a live server.
"""

import re
import unicodedata

# Stripped from the ends of a name only, never from the middle. "Real Madrid CF"
# and "Real Madrid" are the same club, but "Atletico" is not "Atletico Madrid",
# so a token is only dropped when it is an affix. Deliberately excluded:
# "united", "city", "real", "athletic" - each carries meaning in the names that
# use it, and dropping them would merge distinct clubs.
_CLUB_AFFIXES = frozenset(
    {
        "fc",
        "afc",
        "cf",
        "cfc",
        "sc",
        "ac",
        "as",
        "ss",
        "rc",
        "cd",
        "ud",
        "sd",
        "club",
        "de",
        "del",
        "futbol",
        "football",
        "fussball",
        "calcio",
    }
)

# Dots and apostrophes are deleted rather than spaced, so "C.F." collapses to
# the single token "cf" and is recognised as an affix. Every other separator
# becomes a space, so "Saint-Germain" stays two words.
_JOINING_MARKS = re.compile(r"[.'’]", flags=re.UNICODE)
_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalise_team_name(raw: str | None) -> str:
    """Collapse a typed team name to the key used for roster matching.

    Casefolds, strips accents, drops punctuation, then removes club-form
    affixes from either end. Returns "" for anything empty or unusable, which
    callers must treat as "no match" rather than as a key.

    >>> normalise_team_name("Real Madrid C.F.")
    'real madrid'
    >>> normalise_team_name("FC Barcelona")
    'barcelona'
    >>> normalise_team_name("  liverpool  fc ")
    'liverpool'
    >>> normalise_team_name("Manchester United")
    'manchester united'
    """
    if not raw:
        return ""

    # NFKD splits an accented character into base + combining mark, so dropping
    # the marks turns "München" into "munchen" and matches a user who cannot
    # type an umlaut.
    folded = unicodedata.normalize("NFKD", str(raw)).casefold()
    folded = "".join(char for char in folded if not unicodedata.combining(char))

    folded = _JOINING_MARKS.sub("", folded)
    cleaned = _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", folded)).strip()
    if not cleaned:
        return ""

    tokens = cleaned.split(" ")
    while len(tokens) > 1 and tokens[0] in _CLUB_AFFIXES:
        tokens.pop(0)
    while len(tokens) > 1 and tokens[-1] in _CLUB_AFFIXES:
        tokens.pop()

    return " ".join(tokens)
