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
    player_name: Optional[str] = None
    name_confidence: float = 0.0
    candidates: List[OcrCandidate] = field(default_factory=list)
    error: Optional[str] = None


class JerseyReader:
    """Read shirt numbers/names from several crops and combine their evidence."""

    _NAME_PATTERN = re.compile(r"^[A-Z][A-Z '\-]{2,23}$")
    _NUMBER_PATTERN = re.compile(r"(?<!\d)(\d{1,2})(?!\d)")
    _IGNORED_WORDS = {
        "GOAL", "TEAM", "PLAYER", "SCORE", "SPORT", "LIVE", "MATCH", "HOME", "AWAY"
    }

    def __init__(
        self,
        debug_dir: Optional[str] = None,
        engine: Any = None,
        max_crops: int = 8,
        min_supporting_frames: int = 2,
        min_candidate_confidence: float = 0.50,
        min_consensus_confidence: float = 0.58,
    ) -> None:
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.engine = engine
        self.max_crops = max_crops
        self.min_supporting_frames = min_supporting_frames
        self.min_candidate_confidence = min_candidate_confidence
        self.min_consensus_confidence = min_consensus_confidence
        self._engine_error: Optional[str] = None

    def _ensure_engine(self) -> bool:
        if self.engine is not None:
            return True
        if self._engine_error is not None:
            return False
        try:
            cache_dir = (
                self.debug_dir.parent / "paddle_cache"
                if self.debug_dir
                else Path(".cache") / "paddlex"
            ).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
            # This project also uses PyTorch. On Windows, loading Torch first avoids
            # a native DLL collision when PaddleOCR imports ModelScope internally.
            try:
                import torch  # noqa: F401
            except ImportError:
                pass
            from paddleocr import PaddleOCR

            try:
                self.engine = PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                )
            except (TypeError, ValueError):
                # PaddleOCR 2.x compatibility for existing university environments.
                self.engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            return True
        except Exception as exc:  # Paddle is intentionally a lazy, optional import.
            self._engine_error = f"PaddleOCR is unavailable: {exc}"
            return False

    @staticmethod
    def _enhance_crop(crop: np.ndarray) -> List[np.ndarray]:
        if crop.size == 0:
            return []
        height, width = crop.shape[:2]
        scale = max(2.0, min(4.0, 180.0 / max(height, 1)))
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
    def _candidate_values(cls, candidate: OcrCandidate) -> Tuple[List[int], List[str]]:
        normalized = re.sub(r"\s+", " ", candidate.text.upper()).strip()
        numbers = [int(value) for value in cls._NUMBER_PATTERN.findall(normalized)]
        name_text = re.sub(r"[^A-Z '\-]", " ", normalized)
        name_text = re.sub(r"\s+", " ", name_text).strip()
        names = []
        if (
            cls._NAME_PATTERN.fullmatch(name_text)
            and name_text not in cls._IGNORED_WORDS
            and not all(word in cls._IGNORED_WORDS for word in name_text.split())
        ):
            names.append(name_text)
        return numbers, names

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
        name_observations: Dict[str, List[OcrCandidate]] = {}
        accepted = []
        for candidate in candidates:
            if candidate.confidence < self.min_candidate_confidence:
                continue
            accepted.append(candidate)
            numbers, names = self._candidate_values(candidate)
            for number in numbers:
                number_observations.setdefault(number, []).append(candidate)
            for name in names:
                name_observations.setdefault(name, []).append(candidate)

        jersey_number, jersey_confidence = self._choose_consensus(number_observations)
        player_name, name_confidence = self._choose_consensus(name_observations)
        return IdentityReadResult(
            jersey_number=jersey_number,
            jersey_confidence=jersey_confidence,
            player_name=player_name,
            name_confidence=name_confidence,
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
                if self.debug_dir:
                    path = self.debug_dir / (
                        f"track_{track_id}_frame_{crop.frame_number}_"
                        f"crop_{crop_index}_variant_{variant_index}.jpg"
                    )
                    cv2.imwrite(str(path), image)
                try:
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
