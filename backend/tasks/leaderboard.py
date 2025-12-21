from __future__ import annotations

from typing import Any, Callable, Optional

from backend.config import Settings
from backend.features.discovery.service import LeaderboardService


def run_job(
    payload: dict[str, Any],
    *,
    settings: Settings,
    report_progress: Callable[[int, int, Optional[str]], None],
    log: Callable[[str, str], None],
) -> dict[str, Any]:
    service = LeaderboardService(
        backend_db_path=str(settings.backend_db_path),
        discovery_db_path=str(settings.discovery_db_path),
    )
    return service.run(payload, report_progress=report_progress, log=log)
