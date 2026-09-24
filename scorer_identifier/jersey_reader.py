from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackedCrop:
    frame_number: int
    image: np.ndarray = field(compare=False, repr=False)
    quality: float = 1.0
    rear_facing_score: float = 0.5


@dataclass(frozen=True)
class OcrCandidate:
    text: str
    confidence: float
    frame_number: int
    crop_quality: float = 1.0


@dataclass
class IdentityReadResult:
    jersey_number: Optional[int] = None
    jersey_confidence: float = 0.0
    # Reserved for the roster lookup - the shirt itself is never read for a name.
    player_name: Optional[str] = None
    name_confidence: float = 0.0
    candidates: List[OcrCandidate] = field(default_factory=list)
    error: Optional[str] = None


class JerseyReader:
    """Read shirt numbers from several crops and combine their evidence.

    Numbers only. Reading a player's name off the shirt was removed: at the
    resolutions this pipeline sees a name is a handful of pixels tall, and the
    old free-text path happily voted sponsor and advertising-board text
    ("FLY EMIRATES" matched the name pattern) into the player field. A name now
    comes from a roster lookup on (team, shirt number) instead.
    """

    _NUMBER_TOKEN = re.compile(r"^\d{1,2}$")
    # Only applied to tokens that already contain a digit, so a lone "O" or "S"
    # cannot become a shirt number.
    _CONFUSIONS = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "I": "1",
            "i": "1",
            "l": "1",
            "|": "1",
            "S": "5",
            "s": "5",
            "B": "8",
            "Z": "2",
            "z": "2",
            "G": "6",
        }
    )

    def __init__(
        self,
        debug_dir: Optional[str] = None,
        engine: Any = None,
        max_crops: int = 8,
        min_supporting_frames: int = 2,
        min_candidate_confidence: float = 0.50,
        min_consensus_confidence: float = 0.58,
        max_ocr_calls: int = 120,
    ) -> None:
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.engine = engine
        self.max_crops = max_crops
        self.min_supporting_frames = min_supporting_frames
        self.min_candidate_confidence = min_candidate_confidence
        self.min_consensus_confidence = min_consensus_confidence
        self.max_ocr_calls = max_ocr_calls
        self._ocr_calls = 0
        self._engine_error: Optional[str] = None

    def _ensure_engine(self) -> bool:
        if self.engine is not None:
            return True
        if self._engine_error is not None:
            return False
        try:
            # Kept out of backend/static, which is served publicly.
            cache_dir = (Path(".cache") / "paddlex").resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
            # This project also uses PyTorch. On Windows, loading Torch first avoids
            # a native DLL collision when PaddleOCR imports ModelScope internally.
            try:
                import torch  # noqa: F401
            except ImportError:
                pass
            from paddleocr import PaddleOCR

            # enable_mkldnn=False is required, not an optimisation: with oneDNN on,
            # paddlepaddle 3.x raises
            #   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
            # at predict() time on this stack, so every read returned nothing. It
            # also cuts construction from ~25s to ~4s. use_textline_orientation adds
            # a third model pass per text line, which buys nothing for digits.
            attempts = (
                dict(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                ),
                dict(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                ),
                # PaddleOCR 2.x compatibility for existing university environments.
                dict(use_angle_cls=True, lang="en", show_log=False, enable_mkldnn=False),
                dict(use_angle_cls=True, lang="en", show_log=False),
            )
            last_error: Optional[Exception] = None
            for kwargs in attempts:
                try:
                    self.engine = PaddleOCR(**kwargs)
                    return True
                except (TypeError, ValueError) as exc:
                    last_error = exc
            raise RuntimeError(f"no supported constructor signature ({last_error})")
        except Exception as exc:  # Paddle is intentionally a lazy, optional import.
            self._engine_error = f"PaddleOCR is unavailable: {exc}"
            return False

    @staticmethod
    def _enhance_crop(crop: np.ndarray) -> List[np.ndarray]:
        if crop.size == 0:
            return []
        height, width = crop.shape[:2]
        # Digits need real height to be recognised: PP-OCR wants roughly 32px of
        # glyph, and the number occupies well under half of the band.
        scale = max(2.0, min(6.0, 192.0 / max(height, 1)))
        enlarged = cv2.resize(
            crop,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(clahe, (0, 0), 1.2)
        sharpened = cv2.addWeighted(clahe, 1.8, blurred, -0.8, 0)
        return [enlarged, cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)]

    def _run_ocr(self, image: np.ndarray) -> Any:
        if hasattr(self.engine, "predict"):
            try:
                return self.engine.predict(input=image)
            except TypeError:
                return self.engine.predict(image)
        if hasattr(self.engine, "ocr"):
            try:
                return self.engine.ocr(image, cls=True)
            except TypeError:
                return self.engine.ocr(image)
        raise TypeError("OCR engine must provide ocr() or predict()")

    @classmethod
    def _extract_text_confidence(cls, result: Any) -> Iterable[Tuple[str, float]]:
        """Handle both PaddleOCR 2.x nested results and 3.x result dictionaries."""
        if result is None:
            return
        if hasattr(result, "json"):
            result = result.json() if callable(result.json) else result.json
        if isinstance(result, dict):
            payload = result.get("res", result)
            texts = payload.get("rec_texts") or payload.get("texts")
            scores = payload.get("rec_scores") or payload.get("scores")
            if texts is not None and scores is not None:
                for text, confidence in zip(texts, scores):
                    yield str(text), float(confidence)
                return
            for value in payload.values():
                yield from cls._extract_text_confidence(value)
            return
        if isinstance(result, (list, tuple)):
            if (
                len(result) == 2
                and isinstance(result[0], str)
                and isinstance(result[1], (int, float, np.floating))
            ):
                yield result[0], float(result[1])
                return
            if (
                len(result) >= 2
                and isinstance(result[1], (list, tuple))
                and len(result[1]) == 2
                and isinstance(result[1][0], str)
            ):
                yield result[1][0], float(result[1][1])
                return
            for value in result:
                yield from cls._extract_text_confidence(value)

    @classmethod
    def _candidate_values(cls, candidate: OcrCandidate) -> List[int]:
        """Shirt numbers this OCR reading supports.

        The whole token has to read as a one- or two-digit number. The old code
        ran a number regex over the raw string before stripping letters, so a
        shirt read as "R0NALD0" cast two votes for jersey #0.
        """
        numbers: List[int] = []
        for token in re.split(r"[\s/\\|,._-]+", candidate.text.strip()):
            if not token or not any(character.isdigit() for character in token):
                continue
            normalized = token.translate(cls._CONFUSIONS)
            if cls._NUMBER_TOKEN.fullmatch(normalized):
                numbers.append(int(normalized))
        return numbers

    def _choose_consensus(
        self,
        observations: Dict[Any, List[OcrCandidate]],
    ) -> Tuple[Optional[Any], float]:
        ranked = []
        for value, items in observations.items():
            supporting_frames = {item.frame_number for item in items}
            total_weight = sum(item.confidence * item.crop_quality for item in items)
            quality_weight = sum(item.crop_quality for item in items)
            average_confidence = total_weight / max(quality_weight, 1e-9)
            ranked.append((total_weight, len(supporting_frames), average_confidence, value))
        ranked.sort(reverse=True, key=lambda row: (row[0], row[1], row[2]))
        if not ranked:
            return None, 0.0

        winner_score, support, confidence, winner = ranked[0]
        if support < self.min_supporting_frames or confidence < self.min_consensus_confidence:
            return None, confidence
        if len(ranked) > 1 and winner_score < ranked[1][0] * 1.15:
            return None, confidence
        return winner, round(float(confidence), 3)

    def fuse_candidates(self, candidates: Sequence[OcrCandidate]) -> IdentityReadResult:
        number_observations: Dict[int, List[OcrCandidate]] = {}
        accepted = []
        for candidate in candidates:
            if candidate.confidence < self.min_candidate_confidence:
                continue
            accepted.append(candidate)
            for number in self._candidate_values(candidate):
                number_observations.setdefault(number, []).append(candidate)

        jersey_number, jersey_confidence = self._choose_consensus(number_observations)
        return IdentityReadResult(
            jersey_number=jersey_number,
            jersey_confidence=jersey_confidence,
            candidates=accepted,
        )

    def read(self, crops: Sequence[TrackedCrop], track_id: int) -> IdentityReadResult:
        if not self._ensure_engine():
            return IdentityReadResult(error=self._engine_error)

        selected = sorted(
            crops,
            key=lambda crop: (crop.quality + 0.25 * crop.rear_facing_score, crop.frame_number),
            reverse=True,
        )[: self.max_crops]
        candidates: List[OcrCandidate] = []
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        for crop_index, crop in enumerate(selected):
            variants = self._enhance_crop(crop.image)
            for variant_index, image in enumerate(variants):
                # A whole-job budget, so a match with many goals cannot turn into
                # thousands of CPU OCR passes.
                if self._ocr_calls >= self.max_ocr_calls:
                    return self.fuse_candidates(candidates)
                if self.debug_dir:
                    path = self.debug_dir / (
                        f"track_{track_id}_frame_{crop.frame_number}_"
                        f"crop_{crop_index}_variant_{variant_index}.jpg"
                    )
                    cv2.imwrite(str(path), image)
                try:
                    self._ocr_calls += 1
                    result = self._run_ocr(image)
                    for text, confidence in self._extract_text_confidence(result):
                        candidates.append(
                            OcrCandidate(
                                text=text,
                                confidence=max(0.0, min(1.0, confidence)),
                                frame_number=crop.frame_number,
                                crop_quality=crop.quality,
                            )
                        )
                except Exception:
                    continue

        return self.fuse_candidates(candidates)
