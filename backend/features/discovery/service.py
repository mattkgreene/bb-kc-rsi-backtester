from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

from backend.db.leaderboard import save_snapshot
from backend.processing.spark import get_spark_session
from ..market.service import MarketDataService
from .database import DiscoveryDatabase, WinCriteria
from .engine import DiscoveryConfig, run_discovery, run_discovery_parallel
from .leaderboard import Leaderboard
from .rules import find_winning_patterns


@dataclass(frozen=True)
class DiscoveryService:
    market: MarketDataService
    discovery_db_path: str | None
    use_spark: bool = False
    spark_master: Optional[str] = None

    def run(
        self,
        payload: dict[str, Any],
        *,
        report_progress: Callable[[int, int, Optional[str]], None],
        log: Callable[[str, str], None],
    ) -> dict[str, Any]:
        exchange = payload["exchange"]
        symbol = payload["symbol"]
        timeframe = payload["timeframe"]
        start_ts = pd.Timestamp(payload["start_ts"])
        end_ts = pd.Timestamp(payload["end_ts"])

        report_progress(0, 1, "Loading OHLCV data")
        df = self.market.fetch_range(
            exchange,
            symbol,
            timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if df is None or df.empty:
            raise ValueError("No OHLCV data available for requested range")

        if self.use_spark:
            spark = get_spark_session(app_name="bbkc-discovery", master=self.spark_master)
            if spark is None:
                log("warn", "USE_SPARK enabled but SparkSession unavailable; falling back to multiprocessing.")
            else:
                log("info", "SparkSession available (discovery still runs via multiprocessing in this version).")

        base_params = dict(payload.get("base_params") or {})
        base_params.update(
            {
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_ts": start_ts.isoformat(),
                "end_ts": end_ts.isoformat(),
            }
        )

        win_criteria_dict = payload.get("win_criteria") or {}
        win_criteria = WinCriteria(
            min_total_return=float(win_criteria_dict.get("min_total_return", 0.0)),
            max_drawdown=float(win_criteria_dict.get("max_drawdown", 20.0)),
            min_trades=int(win_criteria_dict.get("min_trades", 10)),
            min_profit_factor=float(win_criteria_dict.get("min_profit_factor", 1.0)),
            min_win_rate=float(win_criteria_dict.get("min_win_rate", 0.0)),
        )

        cfg = DiscoveryConfig(
            param_grid=dict(payload.get("param_grid") or {}),
            win_criteria=win_criteria,
            skip_tested=bool(payload.get("skip_tested", True)),
            batch_size=int(payload.get("batch_size", 50)),
            max_combinations=(int(payload["max_combinations"]) if payload.get("max_combinations") is not None else None),
        )

        discovery_db = DiscoveryDatabase(str(self.discovery_db_path))
        discovery_db.initialize()

        use_parallel = bool(payload.get("use_parallel", True))
        n_workers = payload.get("n_workers")

        def _progress(current: int, total: int, message: str) -> None:
            step = max(1, total // 100)
            if current == total or current % step == 0:
                report_progress(int(current), int(total), message)

        log("info", f"Starting discovery for {exchange} {symbol} {timeframe}")
        if use_parallel:
            result = run_discovery_parallel(
                df=df,
                base_params=base_params,
                db=discovery_db,
                config=cfg,
                n_workers=(int(n_workers) if n_workers is not None else None),
                progress_callback=_progress,
            )
        else:
            result = run_discovery(
                df=df,
                base_params=base_params,
                db=discovery_db,
                config=cfg,
                progress_callback=_progress,
            )

        report_progress(1, 1, "Discovery complete")
        return {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
            "summary": {
                "total_tested": result.total_tested,
                "new_tested": result.new_tested,
                "winners_found": result.winners_found,
                "best_return": result.best_return,
                "best_params": result.best_params,
                "duration_seconds": result.duration_seconds,
            },
            "discovery_db_path": str(self.discovery_db_path),
        }


@dataclass(frozen=True)
class LeaderboardService:
    backend_db_path: str | None
    discovery_db_path: str | None

    def run(
        self,
        payload: dict[str, Any],
        *,
        report_progress: Callable[[int, int, Optional[str]], None],
        log: Callable[[str, str], None],
    ) -> dict[str, Any]:
        sort_by = str(payload.get("sort_by") or "total_return")
        top_n = int(payload.get("top_n") or 25)
        min_trades = int(payload.get("min_trades") or 10)

        report_progress(0, 1, "Loading discovery DB")
        db = DiscoveryDatabase(str(self.discovery_db_path))
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
            self.backend_db_path,
            sort_by=sort_by,
            min_trades=min_trades,
            top_n=top_n,
            payload=snapshot_payload,
        )

        report_progress(1, 1, "Leaderboard snapshot saved")
        log("info", f"Saved leaderboard snapshot id={snapshot_id}")
        return {"snapshot_id": snapshot_id, **snapshot_payload}


@dataclass(frozen=True)
class PatternsService:
    discovery_db_path: str | None

    def run(
        self,
        payload: dict[str, Any],
        *,
        report_progress: Callable[[int, int, Optional[str]], None],
        log: Callable[[str, str], None],
    ) -> dict[str, Any]:
        min_confidence = float(payload.get("min_confidence") or 0.3)
        min_occurrence_pct = float(payload.get("min_occurrence_pct") or 10.0)

        report_progress(0, 1, "Analyzing winners for patterns")
        db = DiscoveryDatabase(str(self.discovery_db_path))
        db.initialize()

        rules = find_winning_patterns(db, min_confidence=min_confidence, min_occurrence_pct=min_occurrence_pct)
        report_progress(1, 1, "Pattern analysis complete")
        log("info", f"Discovered {len(rules)} rules")
        return {
            "discovered": len(rules),
            "min_confidence": min_confidence,
            "min_occurrence_pct": min_occurrence_pct,
            "discovery_db_path": str(self.discovery_db_path),
        }
