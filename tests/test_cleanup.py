"""Retention of uploads and per-job media (backend.cleanup)."""

import os
import tempfile
import time
import unittest
from pathlib import Path

from backend import cleanup

JOB_OLD = "aaaaaaaa-1111-2222-3333-444444444444"
JOB_NEW = "bbbbbbbb-1111-2222-3333-444444444444"
JOB_RUNNING = "cccccccc-1111-2222-3333-444444444444"
DAY = 86400


def _write(path, days_old, size=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    stamp = time.time() - days_old * DAY
    os.utime(path, (stamp, stamp))
    return path


class SweepTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.uploads = root / "uploads"
        self.clips = root / "clips"
        self.images = root / "player_images"
        self.samples = root / "team_samples"
        for folder in (self.uploads, self.clips, self.images, self.samples):
            folder.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def sweep(self, **kwargs):
        options = {"upload_max_age": 24.0, "retention_days": 7.0}
        options.update(kwargs)
        return cleanup.sweep(
            self.uploads, [self.clips, self.images], team_sample_dir=self.samples,
            clip_dir=self.clips, **options,
        )

    def test_uploads_older_than_the_limit_go_unless_their_job_is_running(self):
        old = _write(self.uploads / f"{JOB_OLD}.mp4", days_old=2)
        recent = _write(self.uploads / f"{JOB_NEW}.mp4", days_old=0.1)
        running = _write(self.uploads / f"{JOB_RUNNING}.MOV", days_old=3)
        foreign = _write(self.uploads / "keep-me.mp4", days_old=30)

        report = self.sweep(active_uploads=[str(running)])

        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())
        self.assertTrue(running.exists())
        self.assertTrue(foreign.exists())  # not a name this service generates
        self.assertEqual(report.deleted_uploads, [old.name])

    def test_job_media_expires_after_the_retention_period(self):
        old_clip = _write(self.clips / JOB_OLD / "goal-01.mp4", days_old=8)
        old_image = _write(self.images / JOB_OLD / "track-3.jpg", days_old=8)
        old_sample = _write(self.samples / f"{JOB_OLD}-team-1.jpg", days_old=8)
        new_clip = _write(self.clips / JOB_NEW / "shot-01.mp4", days_old=1)
        running_clip = _write(self.clips / JOB_RUNNING / "foul-01.mp4", days_old=9)
        stray = _write(self.clips / "not-a-job" / "x.mp4", days_old=90)

        report = self.sweep(active_job_ids=[JOB_RUNNING])

        self.assertFalse(old_clip.parent.exists())
        self.assertFalse(old_image.parent.exists())
        self.assertFalse(old_sample.exists())
        self.assertTrue(new_clip.exists())
        self.assertTrue(running_clip.exists())
        self.assertTrue(stray.exists())
        self.assertEqual(report.expired_jobs, [JOB_OLD])  # only clip folders clear links
        self.assertEqual(report.freed_bytes, 3000)

    def test_a_recent_file_keeps_its_whole_job_folder(self):
        _write(self.clips / JOB_OLD / "goal-01.mp4", days_old=10)
        _write(self.clips / JOB_OLD / "reel.mp4", days_old=1)
        self.assertEqual(self.sweep().expired_jobs, [])

    def test_zero_retention_keeps_media_forever(self):
        clip = _write(self.clips / JOB_OLD / "goal-01.mp4", days_old=400)
        self.sweep(retention_days=0)
        self.assertTrue(clip.exists())

    def test_retention_comes_from_the_environment(self):
        os.environ["FOOTY_MEDIA_RETENTION_DAYS"] = "30"
        try:
            clip = _write(self.clips / JOB_OLD / "goal-01.mp4", days_old=8)
            cleanup.sweep(self.uploads, [self.clips], clip_dir=self.clips)
            self.assertTrue(clip.exists())
        finally:
            del os.environ["FOOTY_MEDIA_RETENTION_DAYS"]


class DeleteUploadTests(unittest.TestCase):
    def test_deletes_inside_the_upload_folder_only(self):
        with tempfile.TemporaryDirectory() as folder:
            uploads = Path(folder) / "uploads"
            inside = _write(uploads / f"{JOB_OLD}.mp4", days_old=0, size=500)
            outside = _write(Path(folder) / "input.mp4", days_old=0)
            self.assertEqual(cleanup.delete_upload(str(outside), uploads), 0)
            self.assertTrue(outside.exists())
            self.assertEqual(cleanup.delete_upload(str(inside), uploads), 500)
            self.assertFalse(inside.exists())
            self.assertEqual(cleanup.delete_upload(str(inside), uploads), 0)  # already gone
            self.assertEqual(cleanup.delete_upload(None, uploads), 0)


if __name__ == "__main__":
    unittest.main()
