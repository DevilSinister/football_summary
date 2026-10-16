"""Measure the spotting model (models/checkpoint_088.pt) on the fixture clips.

The checkpoint stores neither its label order nor the frame rate its training
frames were sampled at. This script runs it on every clip in
tests/fixtures/short_clips.json at several sampling rates and prints, per clip:
the strongest classes overall, the strongest classes within 3 s of each known
goal, and the "Goal" output's peak and where it is. If the Goal output does not
peak at the known goals but another output does, the assumed SoccerNet-v2 order
is wrong; write the corrected order to models/spotting_classes.txt.

    .venv\\Scripts\\python.exe -m scripts.eval_spotting --rates 2 12.5 25
"""

import argparse
import json
import os
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spotting import ActionSpotter, load_config, load_labels, load_spotting_model  # noqa: E402


def spot_clip(model, labels, clip_len, path, sample_fps):
    capture = cv2.VideoCapture(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    spotter = ActionSpotter(model, labels, fps, sample_fps=sample_fps, clip_len=clip_len)
    frame_number = 0
    started = perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        spotter.add_frame(frame_number, frame)
        frame_number += 1
    capture.release()
    spotter.finish()
    return spotter, fps, frame_number, perf_counter() - started


def report(spotter, fps, labels, goals_seconds, top=5):
    scores = spotter.scores()
    seconds = np.asarray(spotter.sample_frames, dtype=float) / fps
    actions = scores[:, 1:]
    best = actions.max(axis=0)
    order = np.argsort(best)[::-1][:top]
    print("    strongest overall: " + ", ".join(
        f"{labels[i]} {best[i]:.2f}@{seconds[int(np.argmax(actions[:, i]))]:.1f}s" for i in order
    ))
    goal = labels.index("Goal") if "Goal" in labels else None
    if goal is not None:
        peak = int(np.argmax(actions[:, goal]))
        print(f"    'Goal' output peak {actions[peak, goal]:.3f} at {seconds[peak]:.1f}s")
    for second in goals_seconds:
        near = np.abs(seconds - second) <= 3.0
        if not near.any():
            continue
        local = actions[near].max(axis=0)
        local_order = np.argsort(local)[::-1][:top]
        print(f"    near the goal at ~{second:.0f}s: " + ", ".join(
            f"{labels[i]} {local[i]:.2f}" for i in local_order
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default=os.path.join(ROOT, "models", "checkpoint_088.pt"))
    parser.add_argument("--config", default=os.path.join(ROOT, "models", "config.json"))
    parser.add_argument("--labels", default=os.path.join(ROOT, "models", "spotting_classes.txt"))
    parser.add_argument("--fixtures", default=os.path.join(ROOT, "tests", "fixtures", "short_clips.json"))
    parser.add_argument("--videos", default=os.path.join(ROOT, "input_videos"))
    parser.add_argument("--rates", type=float, nargs="+", default=[12.5])
    args = parser.parse_args()

    config = load_config(args.config)
    labels = load_labels(args.labels, int(config["num_classes"]))
    model = load_spotting_model(args.weights, config)
    with open(args.fixtures, encoding="utf-8") as handle:
        clips = json.load(handle)

    for clip in clips:
        path = os.path.join(args.videos, clip["file"])
        if not os.path.isfile(path):
            print(f"{clip['file']}: missing, skipped")
            continue
        goals = [g["approximate_clip_second"] for g in clip.get("goals", [])]
        for rate in args.rates:
            spotter, fps, frames, elapsed = spot_clip(model, labels, int(config["clip_len"]), path, rate)
            print(
                f"{clip['file']} ({frames} frames at {fps:.2f} fps, goals at {goals or 'none'}) "
                f"sampled at {rate} fps: {len(spotter.sample_frames)} samples, "
                f"{elapsed:.1f}s total, {spotter.inference_seconds:.1f}s inference"
            )
            report(spotter, fps, labels, goals)


if __name__ == "__main__":
    main()
