#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from backend.db.leaderboard import get_latest_snapshot, save_snapshot


class LeaderboardDbTests(unittest.TestCase):
    def setUp(self):
        repo_root = Path(__file__).resolve().parent
        data_dir = repo_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(dir=data_dir)
        self.db_path = Path(self._tmpdir.name) / "backend.db"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_save_and_load_latest_snapshot(self):
        snapshot_id = save_snapshot(
            self.db_path,
            sort_by="total_return",
            min_trades=10,
            top_n=25,
            payload={"stats": {"total_winners": 1}, "strategies": []},
        )
        self.assertIsInstance(snapshot_id, int)
        latest = get_latest_snapshot(self.db_path)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.id, snapshot_id)
        self.assertEqual(latest.sort_by, "total_return")
        self.assertEqual(latest.top_n, 25)
        self.assertEqual(latest.payload.get("stats", {}).get("total_winners"), 1)


if __name__ == "__main__":
    unittest.main()

