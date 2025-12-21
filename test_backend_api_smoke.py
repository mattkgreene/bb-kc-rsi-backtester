#!/usr/bin/env python3
import os
import tempfile
import unittest
from pathlib import Path


try:
    from fastapi.testclient import TestClient  # type: ignore
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore


@unittest.skipIf(TestClient is None, "fastapi testclient not installed")
class BackendApiSmokeTests(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parent
        data_dir = repo_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(dir=data_dir)
        tmp_root = Path(self._tmpdir.name)

        os.environ["BACKEND_DB_PATH"] = str(tmp_root / "backend.db")
        os.environ["DISCOVERY_DB_PATH"] = str(tmp_root / "discovery.db")
        os.environ["MARKET_DB_PATH"] = str(tmp_root / "market.db")

        import importlib

        mod = importlib.import_module("backend.api.main")
        self.mod = importlib.reload(mod)
        self.client = TestClient(self.mod.app)

    def tearDown(self):
        self._tmpdir.cleanup()
        for k in ("BACKEND_DB_PATH", "DISCOVERY_DB_PATH", "MARKET_DB_PATH"):
            os.environ.pop(k, None)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("status"), "ok")

    def test_enqueue_and_get_job(self):
        resp = self.client.post("/v1/jobs", json={"job_type": "leaderboard_refresh", "payload": {}, "priority": 0})
        self.assertEqual(resp.status_code, 200)
        job = resp.json().get("job") or {}
        job_id = int(job.get("id"))
        self.assertGreater(job_id, 0)

        resp2 = self.client.get(f"/v1/jobs/{job_id}")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.json().get("id"), job_id)

    def test_backtest_requires_params(self):
        resp = self.client.post("/v1/backtest", json={"params": {}})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
