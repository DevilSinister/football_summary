# Labelled goal-clip fixtures

Place locally licensed short clips in this directory when evaluating the full
models. Each clip should cover one of these labels: clear back number,
blurred/occluded number, front-facing player, similar kits, or conflicting OCR
frames. Real broadcast clips are intentionally not committed to the repository.

The deterministic unit tests in `tests/test_scorer_identifier.py` exercise these
acceptance rules without downloading footage or loading YOLO/PaddleOCR models.
