from __future__ import annotations

from typing import Any, Callable, Optional

from backend.config import Settings
from backend.features.market.service import MarketDataService
from backend.features.optimization.service import OptimizationService


def run_job(
    payload: dict[str, Any],
    *,
    settings: Settings,
    report_progress: Callable[[int, int, Optional[str]], None],
    log: Callable[[str, str], None],
) -> dict[str, Any]:
    service = OptimizationService(
        market=MarketDataService(settings.market_db_path),
        use_spark=settings.use_spark,
        spark_master=settings.spark_master,
    )
    return service.run(payload, report_progress=report_progress, log=log)
