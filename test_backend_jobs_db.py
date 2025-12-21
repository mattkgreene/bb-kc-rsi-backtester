#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from backend.db.jobs import (
    append_event,
    claim_next_job,
    enqueue_job,
    get_events,
    get_job,
    init_jobs_db,
    mark_canceled,
    mark_failed,
    mark_succeeded,
    update_progress,
)


class JobsDbTests(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parent
        data_dir = repo_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(dir=data_dir)
        self.db_path = Path(self._tmpdir.name) / "jobs.db"
        init_jobs_db(self.db_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_enqueue_claim_progress_succeed(self):
        job_id = enqueue_job(self.db_path, job_type="optimize", payload={"hello": "world"}, priority=5)
        job = get_job(self.db_path, job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, "queued")

        claimed = claim_next_job(self.db_path, worker_id="w1")
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.id, job_id)
        self.assertEqual(claimed.status, "running")
        self.assertEqual(claimed.worker_id, "w1")

        update_progress(self.db_path, job_id=job_id, current=3, total=10, message="working")
        job2 = get_job(self.db_path, job_id)
        self.assertEqual(job2.progress_current, 3)
        self.assertEqual(job2.progress_total, 10)
        self.assertEqual(job2.progress_message, "working")

        append_event(self.db_path, job_id=job_id, level="info", message="hello")
        events = get_events(self.db_path, job_id=job_id, limit=10)
        self.assertTrue(any(e["message"] == "hello" for e in events))

        mark_succeeded(self.db_path, job_id=job_id, result={"ok": True})
        job3 = get_job(self.db_path, job_id)
        self.assertEqual(job3.status, "succeeded")
        self.assertEqual(job3.result, {"ok": True})

    def test_mark_failed(self):
        job_id = enqueue_job(self.db_path, job_type="discover", payload={}, priority=0)
        claim_next_job(self.db_path, worker_id="w1")
        mark_failed(self.db_path, job_id=job_id, error="boom")
        job = get_job(self.db_path, job_id)
        self.assertEqual(job.status, "failed")
        self.assertTrue(job.error)

    def test_cancel_prevents_success_override(self):
        job_id = enqueue_job(self.db_path, job_type="optimize", payload={}, priority=0)
        claim_next_job(self.db_path, worker_id="w1")
        mark_canceled(self.db_path, job_id=job_id)

        mark_succeeded(self.db_path, job_id=job_id, result={"ok": True})
        job = get_job(self.db_path, job_id)
        self.assertEqual(job.status, "canceled")


if __name__ == "__main__":
    unittest.main()
