from __future__ import annotations

from typing import Any, Callable, Optional

from backend.config import Settings

from discovery.database import DiscoveryDatabase
from discovery.rules import find_winning_patterns


def run_job(
    payload: dict[str, Any],
    *,
    settings: Settings,
    report_progress: Callable[[int, int, Optional[str]], None],
    log: Callable[[str, str], None],
) -> dict[str, Any]:
    min_confidence = float(payload.get("min_confidence") or 0.3)
    min_occurrence_pct = float(payload.get("min_occurrence_pct") or 10.0)

    report_progress(0, 1, "Analyzing winners for patterns")
    db = DiscoveryDatabase(str(settings.discovery_db_path))
    db.initialize()

    rules = find_winning_patterns(db, min_confidence=min_confidence, min_occurrence_pct=min_occurrence_pct)
    report_progress(1, 1, "Pattern analysis complete")
    log("info", f"Discovered {len(rules)} rules")
    return {
        "discovered": len(rules),
        "min_confidence": min_confidence,
        "min_occurrence_pct": min_occurrence_pct,
        "discovery_db_path": str(settings.discovery_db_path),
    }

