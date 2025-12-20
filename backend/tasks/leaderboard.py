from __future__ import annotations

from typing import Any, Callable, Optional

from backend.config import Settings
from backend.db.leaderboard import save_snapshot

from discovery.database import DiscoveryDatabase
from discovery.leaderboard import Leaderboard


def run_job(
    payload: dict[str, Any],
    *,
    settings: Settings,
    report_progress: Callable[[int, int, Optional[str]], None],
    log: Callable[[str, str], None],
) -> dict[str, Any]:
    sort_by = str(payload.get("sort_by") or "total_return")
    top_n = int(payload.get("top_n") or 25)
    min_trades = int(payload.get("min_trades") or 10)

    report_progress(0, 1, "Loading discovery DB")
    db = DiscoveryDatabase(str(settings.discovery_db_path))
    db.initialize()

    lb = Leaderboard(db)
    stats = lb.get_stats()
    strategies = lb.get_top(n=top_n, sort_by=sort_by, min_trades=min_trades)

    snapshot_payload = {
        "stats": {
            "total_winners": stats.total_winners,
            "avg_return": stats.avg_return,
            "avg_drawdown": stats.avg_drawdown,
            "avg_profit_factor": stats.avg_profit_factor,
            "avg_win_rate": stats.avg_win_rate,
            "best_return": stats.best_return,
            "worst_drawdown": stats.worst_drawdown,
        },
        "strategies": [
            {
                "rank": s.rank,
                "params_hash": s.params_hash,
                "discovered_at": s.discovered_at,
                "metrics": {
                    "total_return": s.total_return,
                    "max_drawdown": s.max_drawdown,
                    "profit_factor": s.profit_factor,
                    "win_rate": s.win_rate,
                    "sharpe_ratio": s.sharpe_ratio,
                    "num_trades": s.num_trades,
                },
                "params": s.params,
            }
            for s in strategies
        ],
    }

    snapshot_id = save_snapshot(
        settings.backend_db_path,
        sort_by=sort_by,
        min_trades=min_trades,
        top_n=top_n,
        payload=snapshot_payload,
    )

    report_progress(1, 1, "Leaderboard snapshot saved")
    log("info", f"Saved leaderboard snapshot id={snapshot_id}")
    return {"snapshot_id": snapshot_id, **snapshot_payload}

