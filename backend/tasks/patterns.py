from __future__ import annotations

from typing import Any, Callable, Optional

from backend.config import Settings
from backend.features.discovery.service import PatternsService


def run_job(
    payload: dict[str, Any],
    *,
    settings: Settings,
    report_progress: Callable[[int, int, Optional[str]], None],
    log: Callable[[str, str], None],
) -> dict[str, Any]:
    service = PatternsService(discovery_db_path=str(settings.discovery_db_path))
    return service.run(payload, report_progress=report_progress, log=log)
