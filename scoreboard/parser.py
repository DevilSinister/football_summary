"""Turn OCR tokens from a broadcast score graphic into a score reading.

Two layouts occur in the reference clips and both are handled:

* one line - UEFA: ``POR 0 1 MCI``, which PaddleOCR returns as a single token
  such as "POR O 0 MCI" (letter O for zero) or "POR0 1 MCI";
* stacked - LaLiga: ``RMA 0`` above ``ATM 0``, returned as separate tokens
  that often arrive partial ("RIAA" for RMA) or merged ("00" spanning both
  score cells).

The clock ("52:58"), captions ("0-1 MIN.33") and sponsor text are rejected by
shape. When the two team codes are known in advance they are matched with one
edit of tolerance, which removes nearly every false reading.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]

# Three-letter words that appear on score graphics but are never team codes.
STOPWORDS = frozenset(
    {"MIN", "VAR", "FIN", "END", "AET", "PEN", "UCL", "UEL", "EXT", "HT", "FT", "TV", "LIVE", "GOL", "TIM"}
)

# Misreads that turn a digit into a letter. Applied only where a digit is expected.
_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                           "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"})
_DIGITISH = "0123456789OoDQIl|iSsBZz"

_CLOCK = re.compile(r"^\+?\d{1,3}[:.']\d{2}$|^\d{1,3}'$|^\+\d{1,2}'?$")
_LEADING_CLOCK = re.compile(r"^\+?\d{1,3}[:.]\d{2}\s*")
_ONE_LINE = re.compile(
    r"^([A-Z]{3})\s*([" + re.escape(_DIGITISH) + r"]{1,2})\s*[-–:|]?\s*("
    + r"[" + re.escape(_DIGITISH) + r"]{1,2})\s*([A-Z]{3})$"
)
MAX_SCORE = 20


@dataclass(frozen=True)
class Token:
    text: str
    confidence: float
    box: Box

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def width(self) -> float:
        return max(1.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(1.0, self.box[3] - self.box[1])


@dataclass(frozen=True)
class ScoreReading:
    codes: Tuple[str, str]          # screen order: left or top first
    scores: Tuple[int, int]
    code_boxes: Tuple[Box, Box]
    region: Box                     # union of every token the reading used


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def as_code(text: str, expected: Optional[Iterable[str]] = None) -> Optional[str]:
    """The team code a token spells, or None."""
    raw = text.strip()
    letters = re.sub(r"[^A-Za-z]", "", raw).upper()
    if expected:
        if not (2 <= len(letters) <= 4) or sum(ch.isdigit() for ch in raw) > 1:
            return None
        best, distance = None, 99
        for code in expected:
            d = _edit_distance(letters, code.upper())
            if d < distance:
                best, distance = code.upper(), d
        return best if distance <= 1 else None
    if len(letters) == 3 and len(raw) == 3 and raw.isalpha() and letters not in STOPWORDS:
        return letters
    return None


def as_score_text(text: str) -> Optional[str]:
    """Digits a token spells ("0", "1", "00", "12"), or None for anything else."""
    raw = text.strip().replace(" ", "")
    if not raw or _CLOCK.match(raw):
        return None
    if not any(ch.isdigit() for ch in raw) and len(raw) > 1:
        # A lone "O" is a zero in a score cell; "OO" or "SO" is more likely text.
        return None
    if any(ch not in _DIGITISH for ch in raw):
        return None
    digits = raw.translate(_TO_DIGIT)
    if not digits.isdigit() or len(digits) > 2:
        return None
    return digits


def _union(boxes: Sequence[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _screen_order(first: Tuple[str, Box], second: Tuple[str, Box]):
    (code_a, box_a), (code_b, box_b) = first, second
    height = max(box_a[3] - box_a[1], box_b[3] - box_b[1], 1.0)
    same_row = abs((box_a[1] + box_a[3]) - (box_b[1] + box_b[3])) / 2.0 < height
    key_a = (box_a[0] + box_a[2]) / 2.0 if same_row else (box_a[1] + box_a[3]) / 2.0
    key_b = (box_b[0] + box_b[2]) / 2.0 if same_row else (box_b[1] + box_b[3]) / 2.0
    return (first, second) if key_a <= key_b else (second, first)


def find_codes(tokens: Sequence[Token], expected: Optional[Iterable[str]] = None) -> Optional[List[Tuple[str, Box]]]:
    """The two team codes and their boxes, when exactly two distinct codes are read.

    Lets the reader find the graphic before it has managed to read a score:
    at 480p the code letters and the digits are rarely both legible in the
    same full-strip read, but they are once the small region is enlarged.
    """
    expected = [code.upper() for code in expected] if expected else None
    found: dict = {}
    for token in tokens:
        code = as_code(token.text, expected)
        if code is not None and (code not in found or token.confidence > found[code].confidence):
            found[code] = token
    if len(found) != 2:
        return None
    ordered = _screen_order(*((code, token.box) for code, token in found.items()))
    return list(ordered)


def parse_one_line(token: Token, expected: Optional[Iterable[str]] = None) -> Optional[ScoreReading]:
    text = _LEADING_CLOCK.sub("", token.text.strip().upper())
    match = _ONE_LINE.match(text)
    if not match:
        return None
    left, s1, s2, right = match.groups()
    code_left, code_right = as_code(left, expected), as_code(right, expected)
    if not code_left or not code_right or code_left == code_right:
        return None
    score_1, score_2 = int(s1.translate(_TO_DIGIT)), int(s2.translate(_TO_DIGIT))
    if score_1 > MAX_SCORE or score_2 > MAX_SCORE:
        return None
    x1, y1, x2, y2 = token.box
    third = (x2 - x1) / 3.0
    return ScoreReading(
        (code_left, code_right),
        (score_1, score_2),
        ((x1, y1, x1 + third, y2), (x2 - third, y1, x2, y2)),
        token.box,
    )


def parse_tokens(
    tokens: Sequence[Token],
    expected: Optional[Iterable[str]] = None,
    layout: Optional[Sequence[Tuple[str, Box]]] = None,
) -> Optional[ScoreReading]:
    """Read a score from the tokens of one graphic.

    ``layout`` is the pair of (code, box) remembered from an earlier reading
    of the same graphic; it lets a read succeed when the OCR catches the
    digits but not the small code letters, which happens often at 480p.
    """
    expected = [code.upper() for code in expected] if expected else None
    for token in tokens:
        reading = parse_one_line(token, expected)
        if reading is not None:
            return reading

    codes: dict = {}
    used = set()
    for index, token in enumerate(tokens):
        code = as_code(token.text, expected)
        if code is None:
            continue
        if code not in codes or token.confidence > codes[code][1].confidence:
            codes[code] = (index, token)
    for index, _ in codes.values():
        used.add(index)
    code_boxes = {code: token.box for code, (_, token) in codes.items()}
    if layout:
        for code, box in layout:
            code_boxes.setdefault(code, box)
    if len(code_boxes) != 2:
        return None

    (code_a, box_a), (code_b, box_b) = _screen_order(*code_boxes.items())
    centres = {
        code_a: ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0),
        code_b: ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0),
    }
    code_height = max(box_a[3] - box_a[1], box_b[3] - box_b[1], 1.0)
    reach = 6.0 * code_height
    stacked = abs(centres[code_a][1] - centres[code_b][1]) >= code_height

    candidates: List[Tuple[float, str, int, Token]] = []  # (distance, code, value, token)
    for index, token in enumerate(tokens):
        if index in used:
            continue
        digits = as_score_text(token.text)
        if digits is None:
            continue
        if len(digits) == 2:
            # One token covering both score cells: split it along the layout.
            spans_both = (
                stacked and token.box[1] <= centres[code_a][1] and token.box[3] >= centres[code_b][1]
            ) or (
                not stacked and min(centres[code_a][0], centres[code_b][0]) < token.cx
                < max(centres[code_a][0], centres[code_b][0])
            )
            if spans_both:
                for code, value in ((code_a, int(digits[0])), (code_b, int(digits[1]))):
                    candidates.append((0.0, code, value, token))
                continue
        value = int(digits)
        if value > MAX_SCORE:
            continue
        for code, (cx, cy) in centres.items():
            distance = ((token.cx - cx) ** 2 + (token.cy - cy) ** 2) ** 0.5
            if distance <= reach:
                candidates.append((distance, code, value, token))

    candidates.sort(key=lambda item: item[0])
    assigned: dict = {}
    taken_tokens: set = set()
    for distance, code, value, token in candidates:
        if code in assigned:
            continue
        token_key = (id(token), code) if distance == 0.0 else id(token)
        if token_key in taken_tokens:
            continue
        assigned[code] = (value, token)
        taken_tokens.add(token_key)
    if code_a not in assigned or code_b not in assigned:
        return None

    boxes = [box_a, box_b, assigned[code_a][1].box, assigned[code_b][1].box]
    return ScoreReading(
        (code_a, code_b),
        (assigned[code_a][0], assigned[code_b][0]),
        (box_a, box_b),
        _union(boxes),
    )
