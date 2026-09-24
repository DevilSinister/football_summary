"""Highlight clips: merging, camera-cut snapping, and the ffmpeg cutter."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from highlights import clips_from_events, event_window, export_highlights, plan_clips
from highlights.encoder import SourceInfo, ffmpeg_binary, ffprobe_binary, target_bitrate
from scene.cuts import SceneBoundary

FPS = 25.0
ROOT = Path(__file__).resolve().parents[1]
# 854x480, 25 fps, 385 frames, AAC audio; hard cuts at frames 200 and 293.
AUDIO_SOURCE = ROOT / "input_videos" / "football_1 (2).mp4"


def cut(frame):
    return SceneBoundary(frame, frame, "cut", 0.5)


def dissolve(first, last):
    return SceneBoundary(first, last, "dissolve", 0.45)


class PlanClipsTests(unittest.TestCase):
    def test_shot_and_goal_of_one_moment_become_one_clip(self):
        events = [
            {"event": "shot", "frame": 250},
            {"event": "save", "frame": 255},
            {"event": "goal", "frame": 300},
            {"event": "pass", "frame": 240},
        ]
        clips = plan_clips(events, FPS, total_frames=2000)
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].primary_type, "goal")
        # Saves and passes are recorded but never clipped.
        self.assertEqual(sorted(clips[0].event_indices), [0, 2])

    def test_a_save_on_its_own_gets_no_clip(self):
        self.assertEqual(plan_clips([{"event": "save", "frame": 500}], FPS, 5000), [])

    def test_start_snaps_to_the_cut_that_opens_the_play(self):
        start, _ = event_window({"event": "shot", "frame": 500}, FPS, 5000, starts=[400])
        self.assertEqual(start, 400)

    def test_a_cut_too_close_to_the_event_is_not_used(self):
        # One second before the shot: the play began in the previous camera shot.
        start, _ = event_window({"event": "shot", "frame": 500}, FPS, 5000, starts=[475])
        self.assertEqual(start, 500 - 125)

    def test_clip_opens_after_a_dissolve_not_inside_it(self):
        clips = plan_clips([{"event": "shot", "frame": 500}], FPS, 5000, [dissolve(390, 398)])
        self.assertEqual(clips[0].start, 399)

    def test_shot_clip_ends_when_the_camera_cuts_away(self):
        clips = plan_clips([{"event": "shot", "frame": 500}], FPS, 5000, [cut(560)])
        self.assertEqual(clips[0].end, 559)

    def test_goal_clip_runs_on_through_the_replay(self):
        # Live goal at 20 s, celebration from 24 s, replay 30-38 s, play resumes at 38 s.
        clips = plan_clips([{"event": "goal", "frame": 500}], FPS, 5000, [cut(600), cut(750), cut(950)])
        self.assertEqual(clips[0].end, 949)

    def test_without_cuts_the_default_window_is_used(self):
        clips = plan_clips([{"event": "goal", "frame": 500}], FPS, 5000)
        self.assertEqual((clips[0].start, clips[0].end), (375, 800))

    def test_distant_events_stay_separate(self):
        clips = plan_clips([{"event": "shot", "frame": 500}, {"event": "foul", "frame": 2000}], FPS, 5000)
        self.assertEqual([c.event_indices for c in clips], [[0], [1]])

    def test_window_is_clamped_to_the_video(self):
        clips = plan_clips([{"event": "goal", "frame": 20}], FPS, total_frames=100)
        self.assertEqual((clips[0].start, clips[0].end), (0, 99))

    def test_an_over_long_chain_is_split_without_writing_footage_twice(self):
        events = [{"event": "shot", "frame": frame} for frame in range(250, 3000, 150)]
        clips = plan_clips(events, FPS, 10000)
        self.assertGreater(len(clips), 1)
        for earlier, later in zip(clips, clips[1:]):
            self.assertGreater(later.start, earlier.end)
        for clip in clips:
            self.assertLessEqual(clip.frame_count, 60 * FPS)
        self.assertEqual(sorted(i for c in clips for i in c.event_indices), list(range(len(events))))

    def test_clips_from_events_lists_each_clip_once(self):
        events = [
            {"event": "shot", "clip_index": 1, "clip_url": "/c/goal-01.mp4", "clip_start_seconds": 0.0, "clip_end_seconds": 8.0},
            {"event": "goal", "clip_index": 1, "clip_url": "/c/goal-01.mp4", "clip_start_seconds": 0.0, "clip_end_seconds": 8.0},
            {"event": "pass"},
            {"event": "foul", "clip_index": 2, "clip_url": "/c/foul-02.mp4", "clip_start_seconds": 11.7, "clip_end_seconds": 15.4},
        ]
        self.assertEqual(
            [(c["url"], c["event_types"]) for c in clips_from_events(events)],
            [("/c/goal-01.mp4", ["shot", "goal"]), ("/c/foul-02.mp4", ["foul"])],
        )


class BitrateTests(unittest.TestCase):
    def test_capped_by_the_source_and_by_resolution(self):
        self.assertEqual(target_bitrate(SourceInfo(1920, 1080, False, 12_685_584)), 8_000_000)
        self.assertEqual(target_bitrate(SourceInfo(848, 478, True, 1_350_512)), 1_350_512)
        self.assertEqual(target_bitrate(SourceInfo(1280, 720, True, None)), 5_000_000)


def _probe(path):
    output = subprocess.run(
        [ffprobe_binary(), "-v", "error", "-show_entries",
         "stream=codec_type,codec_name:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(output)


def _top_level_atoms(path):
    atoms = []
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if len(header) < 8:
                return atoms
            size, kind = int.from_bytes(header[:4], "big"), header[4:].decode("latin-1")
            atoms.append(kind)
            if size == 1:
                size = int.from_bytes(handle.read(8), "big")
                handle.seek(size - 16, 1)
            elif size == 0:
                return atoms
            else:
                handle.seek(size - 8, 1)


def _frame(path, index):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    assert ok, f"could not read frame {index} of {path}"
    return cv2.resize(frame, (160, 90)).astype(np.float32)


@unittest.skipUnless(
    ffmpeg_binary() and ffprobe_binary() and AUDIO_SOURCE.is_file(),
    "needs ffmpeg/ffprobe on PATH and the 480p reference clip",
)
class FfmpegExportTests(unittest.TestCase):
    def test_clips_keep_audio_start_on_the_cut_and_join_into_a_reel(self):
        events = [{"event": "shot", "frame": 30}, {"event": "foul", "frame": 360}]
        with tempfile.TemporaryDirectory() as folder:
            summary = export_highlights(
                str(AUDIO_SOURCE), events, FPS, folder, "/api/processing/clips/job",
                boundaries=[cut(200), cut(293)], total_frames=385,
            )
            self.assertEqual(summary["encoder"], "ffmpeg")
            self.assertEqual([c["file_name"] for c in summary["clips"]], ["shot-01.mp4", "foul-02.mp4"])
            # Clip 1 runs to the end of its camera shot; clip 2 opens on the cut at 293.
            self.assertEqual(
                [(c["start_seconds"], c["end_seconds"]) for c in summary["clips"]],
                [(0.0, 8.0), (11.72, 15.4)],
            )
            self.assertEqual(events[0]["clip_url"], "/api/processing/clips/job/shot-01.mp4")
            self.assertEqual(events[1]["clip_offset_seconds"], round(360 / FPS - 11.72, 3))

            for clip in summary["clips"]:
                path = Path(folder) / clip["file_name"]
                info = _probe(path)
                codecs = {s["codec_type"]: s["codec_name"] for s in info["streams"]}
                self.assertEqual(codecs, {"video": "h264", "audio": "aac"})
                duration = float(info["format"]["duration"])
                self.assertAlmostEqual(duration, clip["end_seconds"] - clip["start_seconds"], delta=0.15)
                # Index first, so playback starts without a second request.
                atoms = _top_level_atoms(path)
                self.assertLess(atoms.index("moov"), atoms.index("mdat"))
                # 480p is capped at 2.5 Mbit/s of video plus 128 kbit/s of audio.
                self.assertLess(path.stat().st_size * 8 / duration, 3_000_000)

            # The first frame of clip 2 is source frame 293, not the shot before the cut.
            first = _frame(Path(folder) / "foul-02.mp4", 0)
            after_cut = np.abs(first - _frame(AUDIO_SOURCE, 293)).mean()
            before_cut = np.abs(first - _frame(AUDIO_SOURCE, 292)).mean()
            self.assertLess(after_cut, 10.0)
            self.assertLess(after_cut, before_cut / 3)

            reel = summary["reel"]
            self.assertIsNotNone(reel)
            reel_info = _probe(Path(folder) / "reel.mp4")
            self.assertAlmostEqual(float(reel_info["format"]["duration"]), reel["duration_seconds"], delta=0.3)
            self.assertEqual({s["codec_type"] for s in reel_info["streams"]}, {"video", "audio"})
            self.assertFalse((Path(folder) / "reel.mp4.txt").exists())


if __name__ == "__main__":
    unittest.main()
