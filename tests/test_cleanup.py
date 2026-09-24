"""Upload clean-up (backend.cleanup). Highlights are never deleted."""

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
        self.clips = root / "static" / "clips"
        self.uploads.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_uploads_older_than_the_limit_go_unless_their_job_is_running(self):
        old = _write(self.uploads / f"{JOB_OLD}.mp4", days_old=2)
        recent = _write(self.uploads / f"{JOB_NEW}.mp4", days_old=0.1)
        running = _write(self.uploads / f"{JOB_RUNNING}.MOV", days_old=3)
        foreign = _write(self.uploads / "keep-me.mp4", days_old=30)

        report = cleanup.sweep(self.uploads, active_uploads=[str(running)], upload_max_age=24.0)

        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())
        self.assertTrue(running.exists())
        self.assertTrue(foreign.exists())  # not a name this service generates
        self.assertEqual(report.deleted_uploads, [old.name])
        self.assertEqual(report.freed_bytes, 1000)

    def test_highlights_are_never_deleted_however_old(self):
        clip = _write(self.clips / JOB_OLD / "goal-01.mp4", days_old=400)
        reel = _write(self.clips / JOB_OLD / "reel.mp4", days_old=400)
        cleanup.sweep(self.uploads, upload_max_age=24.0)
        self.assertTrue(clip.exists())
        self.assertTrue(reel.exists())

    def test_zero_keeps_uploads_forever(self):
        old = _write(self.uploads / f"{JOB_OLD}.mp4", days_old=400)
        cleanup.sweep(self.uploads, upload_max_age=0)
        self.assertTrue(old.exists())

    def test_limit_comes_from_the_environment(self):
        os.environ["FOOTY_UPLOAD_MAX_AGE_HOURS"] = "72"
        try:
            upload = _write(self.uploads / f"{JOB_OLD}.mp4", days_old=2)
            cleanup.sweep(self.uploads)
            self.assertTrue(upload.exists())
        finally:
            del os.environ["FOOTY_UPLOAD_MAX_AGE_HOURS"]


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
