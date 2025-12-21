#!/usr/bin/env python3
import os
import unittest

from backend.config import get_settings


class BackendConfigTests(unittest.TestCase):
    def setUp(self):
        self._old_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_database_url_overrides_sqlite_paths(self):
        os.environ["DATABASE_URL"] = "postgresql://example/db"
        os.environ.pop("BACKEND_DB_PATH", None)
        os.environ.pop("DISCOVERY_DB_PATH", None)
        os.environ.pop("MARKET_DB_PATH", None)

        settings = get_settings()
        self.assertEqual(settings.database_url, "postgresql://example/db")
        self.assertEqual(settings.backend_db_path, "postgresql://example/db")
        self.assertEqual(settings.discovery_db_path, "postgresql://example/db")
        self.assertEqual(settings.market_db_path, "postgresql://example/db")

    def test_per_db_url_overrides_database_url(self):
        os.environ["DATABASE_URL"] = "postgresql://example/db"
        os.environ["JOBS_DATABASE_URL"] = "postgresql://example/jobs"
        os.environ["DISCOVERY_DATABASE_URL"] = "postgresql://example/discovery"
        os.environ["MARKET_DATABASE_URL"] = "postgresql://example/market"

        settings = get_settings()
        self.assertEqual(settings.backend_db_path, "postgresql://example/jobs")
        self.assertEqual(settings.discovery_db_path, "postgresql://example/discovery")
        self.assertEqual(settings.market_db_path, "postgresql://example/market")


if __name__ == "__main__":
    unittest.main()

