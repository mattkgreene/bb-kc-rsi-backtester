from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ..market.service import MarketDataService
from .engine import run_backtest


def _serialize_df(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None:
        return None
    if df.empty:
        return df.to_json(orient="split", date_format="iso")
    return df.to_json(orient="split", date_format="iso")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class BacktestService:
    market: MarketDataService

    def run(self, params: dict[str, Any]) -> dict[str, Any]:
        if not params:
            raise ValueError("Missing backtest parameters")

        exchange = params.get("exchange")
        symbol = params.get("symbol")
        timeframe = params.get("timeframe")
        start_ts = params.get("start_ts")
        end_ts = params.get("end_ts")
        if not all([exchange, symbol, timeframe, start_ts, end_ts]):
            raise ValueError("Missing required backtest fields")

        df = self.market.fetch_range(
            exchange,
            symbol,
            timeframe,
            start_ts=pd.Timestamp(start_ts),
            end_ts=pd.Timestamp(end_ts),
        )
        if df is None or df.empty:
            raise ValueError("No OHLCV data available for requested range")

        bt_params = {
            "bb_len": _safe_int(params.get("bb_len", 20), 20),
            "bb_std": _safe_float(params.get("bb_std", 2.0), 2.0),
            "bb_basis_type": params.get("bb_basis_type", "sma"),
            "kc_ema_len": _safe_int(params.get("kc_ema_len", 20), 20),
            "kc_atr_len": _safe_int(params.get("kc_atr_len", 14), 14),
            "kc_mult": _safe_float(params.get("kc_mult", 2.0), 2.0),
            "kc_mid_type": params.get("kc_mid_type", "ema"),
            "rsi_len_30m": _safe_int(params.get("rsi_len_30m", 14), 14),
            "rsi_ma_len": _safe_int(params.get("rsi_ma_len", 10), 10),
            "rsi_smoothing_type": params.get("rsi_smoothing_type", "ema"),
            "rsi_ma_type": params.get("rsi_ma_type", "sma"),
            "rsi_min": _safe_float(params.get("rsi_min", 70), 70),
            "rsi_ma_min": _safe_float(params.get("rsi_ma_min", 70), 70),
            "rsi_max": _optional_float(params.get("rsi_max")),
            "rsi_ma_max": _optional_float(params.get("rsi_ma_max")),
            "use_rsi_relation": bool(params.get("use_rsi_relation", True)),
            "rsi_relation": params.get("rsi_relation", ">="),
            "entry_band_mode": params.get("entry_band_mode", "Either"),
            "trade_direction": params.get("trade_direction", "Short"),
            "exit_channel": params.get("exit_channel", "BB"),
            "exit_level": params.get("exit_level", "mid"),
            "cash": _safe_float(params.get("cash", 10_000.0), 10_000.0),
            "commission": _safe_float(params.get("commission", 0.0005), 0.0005),
            "trade_mode": params.get("trade_mode", "Margin / Futures"),
            "use_stop": bool(params.get("use_stop", True)),
            "stop_mode": params.get("stop_mode", "Fixed %"),
            "stop_pct": _safe_float(params.get("stop_pct", 2.0), 2.0),
            "stop_atr_mult": _safe_float(params.get("stop_atr_mult", 2.0), 2.0),
            "use_trailing": bool(params.get("use_trailing", False)),
            "trail_pct": _safe_float(params.get("trail_pct", 1.0), 1.0),
            "max_bars_in_trade": _safe_int(params.get("max_bars_in_trade", 100), 100),
            "daily_loss_limit": _safe_float(params.get("daily_loss_limit", 3.0), 3.0),
            "risk_per_trade_pct": _safe_float(params.get("risk_per_trade_pct", 1.0), 1.0),
            "max_leverage": _optional_float(params.get("max_leverage")),
            "maintenance_margin_pct": _optional_float(params.get("maintenance_margin_pct")),
            "max_margin_utilization": _optional_float(params.get("max_margin_utilization")),
        }

        stats, ds, trades, equity_curve = run_backtest(df, timeframe=timeframe, **bt_params)

        return {
            "df": _serialize_df(df),
            "ds": _serialize_df(ds),
            "trades": _serialize_df(trades),
            "stats": stats.to_dict(),
            "equity_curve": (equity_curve.tolist() if equity_curve is not None else []),
            "params": dict(params),
        }
