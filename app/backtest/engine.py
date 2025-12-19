"""
Backtesting engine for the BB+KC+RSI short strategy.

This module contains the core backtesting logic including:
- Dataset preparation with indicators
- Strategy execution loop (entry/exit logic)
- Trade finalization and timestamp conversion
- Performance statistics calculation

The strategy is short-only, entering when price touches upper bands
with high RSI, and exiting on mean reversion to mid/lower bands or
via stop loss/trailing stop/time stop.

Trade Modes:
- Simple (1x spot-style): No leverage, position = equity / price
- Margin / Futures: Leverage-based sizing with liquidation simulation
"""

from __future__ import annotations

from typing import Literal, Dict, Tuple, List, Optional, TypedDict, Union
import math
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from core.indicators import add_bb_kc_rsi
from core.utils import cmp, vectorized_cmp


# =============================================================================
# Type Definitions
# =============================================================================

class TradeRecord(TypedDict, total=False):
    """Type definition for a single trade record."""
    EntryBar: int
    ExitBar: int
    EntryPrice: float
    ExitPrice: float
    Side: Literal["Short", "Long"]
    ExitReason: Literal["signal_exit", "stop_loss", "time_stop", "liquidation", "close_at_end"]
    Size: float
    NotionalEntry: float
    RealizedPnL: float
    Commission: float
    EquityBefore: float
    EquityAfter: float
    R_multiple: float
    EffectiveLeverage: float
    MarginUtilAtEntry: float
    LiqPrice: Optional[float]
    RiskDollars: float
    EntryCommission: float
    BestPrice: float
    StopPrice: Optional[float]
    BarsInTrade: int


# Type alias for equity curve
EquityCurve = NDArray[np.float64]

# Type alias for indicator parameters
IndicatorParams = Dict[str, Union[int, float, str]]


# =============================================================================
# Time/Index Helpers
# =============================================================================

_TF_TO_PD = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
    "1d": "1D", "3d": "3D", "1w": "7D"
}


def _tf_delta(timeframe: str) -> pd.Timedelta:
    """
    Convert a timeframe string to a pandas Timedelta.
    
    Args:
        timeframe: Timeframe string (e.g., '30m', '1h', '4h', '1d').
    
    Returns:
        Timedelta representing the duration of one candle.
    """
    return pd.to_timedelta(_TF_TO_PD.get(timeframe.lower(), "1H"))


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame index is timezone-aware UTC.
    
    Args:
        df: DataFrame with DatetimeIndex.
    
    Returns:
        DataFrame with UTC-localized index.
    """
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _bar_time(
    ds: pd.DataFrame,
    bar: int,
    timeframe: str,
    convention: Literal["open", "close"]
) -> pd.Timestamp:
    """
    Get the timestamp for a bar based on open or close convention.
    
    Args:
        ds: Dataset DataFrame with DatetimeIndex.
        bar: Bar index (0-based).
        timeframe: Candle timeframe for offset calculation.
        convention: 'open' returns bar start time, 'close' returns bar end time.
    
    Returns:
        UTC timestamp for the specified bar and convention.
    """
    if convention == "close":
        return (ds.index[bar] + _tf_delta(timeframe) - pd.Timedelta(milliseconds=1)).tz_convert("UTC")
    return ds.index[bar].tz_convert("UTC")


# =============================================================================
# Dataset Preparation
# =============================================================================

def build_dataset(
    df: pd.DataFrame,
    *,
    bb_len: int = 20,
    bb_std: float = 2.0,
    bb_basis_type: str = "sma",
    kc_ema_len: int = 20,
    kc_atr_len: int = 14,
    kc_mult: float = 2.0,
    kc_mid_type: str = "ema",
    rsi_len_30m: int = 14,
    rsi_ma_len: int = 10,
    rsi_smoothing_type: str = "ema",
    rsi_ma_type: str = "sma",
) -> pd.DataFrame:
    """
    Build a complete dataset with all indicators for backtesting.
    
    Takes raw OHLCV data and adds Bollinger Bands, Keltner Channels,
    and RSI indicators required for the strategy.
    
    Args:
        df: Raw OHLCV DataFrame with ['Open', 'High', 'Low', 'Close', 'Volume'].
        bb_len: Bollinger Bands period.
        bb_std: Bollinger Bands standard deviation multiplier.
        bb_basis_type: BB middle line type ('sma' or 'ema').
        kc_ema_len: Keltner Channel middle line period.
        kc_atr_len: Keltner Channel ATR period.
        kc_mult: Keltner Channel ATR multiplier.
        kc_mid_type: KC middle line type ('ema' or 'sma').
        rsi_len_30m: RSI period on 30-minute resampled data.
        rsi_ma_len: RSI moving average period.
        rsi_smoothing_type: RSI smoothing method.
        rsi_ma_type: RSI MA smoothing method.
    
    Returns:
        DataFrame with OHLCV + indicator columns, UTC-indexed.
    """
    ds = df.copy()
    ds = add_bb_kc_rsi(
        ds,
        bb_len=bb_len, bb_std=bb_std, bb_basis_type=bb_basis_type,
        kc_ema_len=kc_ema_len, kc_atr_len=kc_atr_len, kc_mult=kc_mult, kc_mid_type=kc_mid_type,
        rsi_len_30m=rsi_len_30m, rsi_ma_len=rsi_ma_len,
        rsi_smoothing_type=rsi_smoothing_type, rsi_ma_type=rsi_ma_type,
    )
    return _ensure_utc_index(ds)




# =============================================================================
# Strategy Loop
# =============================================================================

def run_strategy_loop(
    ds: pd.DataFrame,
    *,
    # Entry conditions
    rsi_min: float = 70.0,
    rsi_ma_min: float = 70.0,
    use_rsi_relation: bool = True,
    rsi_relation: Literal["<", "<=", ">", ">="] = ">=",
    entry_band_mode: Literal["KC", "BB", "Both", "Either"] = "Either",

    # Exit conditions (signal-based)
    exit_channel: Literal["BB", "KC"] = "BB",
    exit_level: Literal["mid", "lower"] = "mid",

    # Risk management and account settings
    cash: float = 10_000.0,
    commission: float = 0.0005,
    trade_mode: str = "Simple (1x spot-style)",
    use_stop: bool = True,
    stop_mode: Literal["Fixed %", "ATR"] = "Fixed %",
    stop_pct: float = 2.0,
    stop_atr_mult: float = 2.0,
    use_trailing: bool = False,
    trail_pct: float = 1.0,
    max_bars_in_trade: int = 100,
    daily_loss_limit: float = 0.0,
    risk_per_trade_pct: float = 1.0,
    max_leverage: Optional[float] = None,
    maintenance_margin_pct: Optional[float] = None,
    max_margin_utilization: Optional[float] = None,
) -> Tuple[pd.DataFrame, EquityCurve]:
    """
    Execute the short-only trading strategy on prepared dataset.
    
    This is the core simulation loop that:
    1. Checks entry conditions (band touch + RSI thresholds)
    2. Manages open positions (stops, trailing, time limits)
    3. Executes exits (signal, stop, liquidation)
    4. Tracks equity and P&L
    
    Args:
        ds: Dataset with OHLCV + indicators (from build_dataset).
        
        Entry Conditions:
            rsi_min: Minimum RSI value for entry.
            rsi_ma_min: Minimum RSI MA value for entry.
            use_rsi_relation: Whether to check RSI vs RSI MA.
            rsi_relation: Comparison operator for RSI vs RSI MA.
            entry_band_mode: Which bands must be touched ('KC', 'BB', 'Both', 'Either').
        
        Exit Conditions:
            exit_channel: Channel for signal exit ('BB' or 'KC').
            exit_level: Level for signal exit ('mid' or 'lower').
        
        Risk Management:
            cash: Starting capital.
            commission: Trading fee as fraction (e.g., 0.001 = 0.1%).
            trade_mode: 'Simple (1x spot-style)' or 'Margin / Futures'.
            use_stop: Enable stop loss.
            stop_mode: 'Fixed %' or 'ATR' based stop.
            stop_pct: Fixed stop loss percentage.
            stop_atr_mult: ATR multiplier for stop distance.
            use_trailing: Enable trailing stop.
            trail_pct: Trailing stop percentage.
            max_bars_in_trade: Time-based exit limit.
            daily_loss_limit: Max daily loss % before stopping new trades.
            risk_per_trade_pct: % of equity to risk per trade (Margin mode).
        
        Margin/Futures:
            max_leverage: Maximum leverage allowed.
            maintenance_margin_pct: Liquidation threshold %.
            max_margin_utilization: Max margin usage %.
    
    Returns:
        Tuple of (trades_df, equity_curve):
            - trades_df: DataFrame with trade records
            - equity_curve: numpy array of equity at each bar
    
    Trade Record Fields:
        - EntryBar, ExitBar: Bar indices
        - EntryPrice, ExitPrice: Trade prices
        - Side: Always 'Short' for this strategy
        - ExitReason: 'signal_exit', 'stop_loss', 'time_stop', 'liquidation', 'close_at_end'
        - Size: Position size in units
        - NotionalEntry: Position value at entry
        - RealizedPnL: Profit/loss after commission
        - EquityBefore, EquityAfter: Account equity
        - R_multiple: PnL / Risk (if stop-based sizing)
        - EffectiveLeverage, MarginUtilAtEntry, LiqPrice: Margin mode fields
    """
    # Extract numpy arrays for faster access
    close = ds["Close"].values
    high = ds["High"].values
    low = ds["Low"].values

    bb_up = ds["bb_up"].values
    bb_mid = ds["bb_mid"].values
    bb_low = ds["bb_low"].values

    kc_up = ds["kc_up"].values
    kc_mid = ds["kc_mid"].values
    kc_low = ds["kc_low"].values

    rsi = ds["rsi30"].values
    rma = ds["rsi30_ma"].values

    atr = ds["kc_atr"].values if ("kc_atr" in ds.columns and stop_mode == "ATR") else None

    n = len(ds)
    idx = ds.index

    trades: List[dict] = []

    # Account state
    equity = float(cash)
    cur_day = None
    daily_loss_pct_acc = 0.0
    
    # Track equity curve for drawdown calculations
    equity_curve = np.full(n, cash, dtype=float)

    # Active position (None when flat)
    pos = None

    # =========================================================================
    # Vectorized Pre-computation of Entry Signals (Performance Optimization)
    # =========================================================================
    # Pre-compute all entry conditions as boolean arrays to avoid repeated
    # per-bar calculations. This provides significant speedup for large datasets.
    
    # Band touch conditions (vectorized)
    # Use HIGH for "touch" to match typical intrabar band-touch semantics.
    touch_kc_vec = (high >= kc_up) & ~np.isnan(kc_up)
    touch_bb_vec = (high >= bb_up) & ~np.isnan(bb_up)
    
    if entry_band_mode == "KC":
        band_ok_vec = touch_kc_vec
    elif entry_band_mode == "BB":
        band_ok_vec = touch_bb_vec
    elif entry_band_mode == "Both":
        band_ok_vec = touch_kc_vec & touch_bb_vec
    else:  # "Either"
        band_ok_vec = touch_kc_vec | touch_bb_vec
    
    # RSI conditions (vectorized)
    rsi_ok_vec = ~np.isnan(rsi) & (rsi >= rsi_min)
    rma_ok_vec = ~np.isnan(rma) & (rma >= rsi_ma_min)
    
    # RSI relation (vectorized)
    if use_rsi_relation:
        rsi_relation_ok_vec = vectorized_cmp(rsi, rma, rsi_relation) & ~np.isnan(rsi) & ~np.isnan(rma)
    else:
        rsi_relation_ok_vec = np.ones(n, dtype=bool)
    
    # Combined entry signal: all conditions must be true
    entry_signal_vec = band_ok_vec & rsi_ok_vec & rma_ok_vec & rsi_relation_ok_vec & (close > 0)

    # --- Helper Functions ---
    
    def band_ok(i: int) -> bool:
        """Check if price touches required bands at bar i (uses pre-computed)."""
        return entry_signal_vec[i]

    def rsi_relation_ok(i: int) -> bool:
        """Check RSI vs RSI MA relation at bar i (uses pre-computed)."""
        return rsi_relation_ok_vec[i]

    def compute_stop_price_short(entry_px: float, i: int) -> Optional[float]:
        """Calculate stop loss price for a short position."""
        if not use_stop:
            return None
        if stop_mode == "Fixed %":
            # For shorts, stop is above entry
            return entry_px * (1.0 + stop_pct / 100.0)
        if stop_mode == "ATR":
            if atr is None or math.isnan(atr[i]) or stop_atr_mult is None:
                return None
            return entry_px + stop_atr_mult * atr[i]
        return None

    def update_trailing_stop_short(pos_dict: dict, i: int) -> None:
        """Update trailing stop based on price movement."""
        if not use_trailing:
            return
        cur_px = close[i]
        best_px = pos_dict.get("BestPrice", pos_dict["EntryPrice"])
        # For shorts, "best" is lowest price (most profit)
        if cur_px < best_px:
            best_px = cur_px
            pos_dict["BestPrice"] = best_px
            # Trail stop above the best price
            pos_dict["StopPrice"] = best_px * (1.0 + trail_pct / 100.0)

    # --- Main Strategy Loop ---
    
    for i in range(n):
        bar_ts = idx[i]
        day = bar_ts.date()

        # Reset daily loss accumulator on new day
        if cur_day is None or day != cur_day:
            cur_day = day
            daily_loss_pct_acc = 0.0
        
        # Update equity curve (reflects current equity at this bar)
        equity_curve[i] = equity

        # =====================================================================
        # ENTRY LOGIC (Short only)
        # =====================================================================
        if pos is None and equity > 0:
            # Check daily loss limit - skip new trades if limit hit
            if (
                daily_loss_limit is not None
                and daily_loss_limit > 0
                and daily_loss_pct_acc <= -daily_loss_limit
            ):
                pass  # Skip entry, daily limit reached
            else:
                # Check all entry conditions (uses pre-computed vectorized signal)
                if entry_signal_vec[i]:
                    entry_px = float(close[i])
                    if entry_px <= 0:
                        continue

                    equity_before = equity

                    # ==========================================================
                    # Position Sizing based on Trade Mode
                    # ==========================================================
                    
                    if "Simple" in trade_mode:
                        # SPOT MODE: No leverage, full equity as notional
                        notional = equity_before
                        size = notional / entry_px
                        effective_leverage = 1.0
                        margin_util = float("nan")
                        risk_dollars = float("nan")
                        stop_price = compute_stop_price_short(entry_px, i)

                    else:
                        # MARGIN / FUTURES MODE: Risk-based sizing with leverage
                        if use_stop:
                            # Calculate stop price first for risk-based sizing
                            stop_price = compute_stop_price_short(entry_px, i)
                            if stop_price is not None and not math.isnan(stop_price):
                                risk_per_unit = abs(stop_price - entry_px)
                            else:
                                # Fallback if stop can't be computed
                                risk_per_unit = entry_px * 0.01
                                stop_price = None

                            if risk_per_unit <= 0:
                                continue

                            # Size position so stop-out risks only X% of equity
                            if risk_per_trade_pct > 0 and equity_before > 0:
                                risk_dollars = equity_before * (risk_per_trade_pct / 100.0)
                                size = risk_dollars / risk_per_unit
                            else:
                                size = 1.0
                                risk_dollars = 0.0

                            notional = size * entry_px

                            # Cap notional at max leverage
                            if max_leverage is not None and max_leverage > 0:
                                notional_cap = equity_before * max_leverage
                                if notional > notional_cap and notional > 0:
                                    scale = notional_cap / notional
                                    size *= scale
                                    notional = size * entry_px
                        else:
                            # No stop: size from max_leverage directly
                            effective_max_lev = max_leverage if (max_leverage is not None and max_leverage > 0) else 1.0
                            notional = equity_before * effective_max_lev
                            size = notional / entry_px
                            risk_dollars = float("nan")
                            stop_price = None

                        # Calculate effective leverage and margin utilization
                        effective_leverage = notional / equity_before if equity_before > 0 else float("nan")
                        if max_leverage is not None and max_leverage > 0 and equity_before > 0:
                            margin_util = notional / (equity_before * max_leverage)
                        else:
                            margin_util = float("nan")

                        # Check margin utilization cap (skip trade if exceeded)
                        if (
                            trade_mode == "Margin / Futures"
                            and max_margin_utilization is not None
                            and max_margin_utilization > 0
                            and max_leverage is not None
                            and max_leverage > 0
                            and equity_before > 0
                        ):
                            required_margin = notional / max_leverage
                            margin_util_pct = (required_margin / equity_before) * 100.0
                            if margin_util_pct > max_margin_utilization:
                                continue  # Skip this trade

                    # Calculate commission cost on entry
                    entry_commission = notional * commission

                    # ==========================================================
                    # Open Position
                    # ==========================================================
                    pos = {
                        "EntryBar": i,
                        "EntryPrice": entry_px,
                        "Side": "Short",
                        "BarsInTrade": 0,
                        "Size": size,
                        "NotionalEntry": notional,
                        "RiskDollars": risk_dollars,
                        "EquityBefore": equity_before,
                        "BestPrice": entry_px,
                        "EffectiveLeverage": effective_leverage,
                        "MarginUtilAtEntry": margin_util,
                        "LiqPrice": None,
                        "EntryCommission": entry_commission,
                    }
                    pos["StopPrice"] = stop_price

                    # Calculate liquidation price for Margin/Futures mode
                    # Liquidation occurs when equity falls to maintenance margin
                    if (
                        trade_mode == "Margin / Futures"
                        and maintenance_margin_pct is not None
                        and maintenance_margin_pct > 0
                        and size > 0
                    ):
                        m = float(maintenance_margin_pct) / 100.0
                        EQ0 = equity_before
                        N = notional
                        # For short: Equity(P) = EQ0 + (Entry - P) * size
                        # Liquidation when Equity(P) <= N * m
                        # Solving: P = (EQ0 + N - N*m) / size
                        liq_price = (EQ0 + N - N * m) / size
                        pos["LiqPrice"] = liq_price

                    continue  # Move to next bar

        # =====================================================================
        # EXIT / STOP / LIQUIDATION MANAGEMENT
        # =====================================================================
        if pos is not None:
            pos["BarsInTrade"] += 1
            exit_reason = None
            exit_px = float(close[i])

            # 1) Update trailing stop
            if use_trailing:
                update_trailing_stop_short(pos, i)

            # 2) Check hard stop loss
            sp = pos.get("StopPrice")
            if use_stop and sp is not None and not math.isnan(sp):
                if high[i] >= sp:  # For shorts, price rising hits stop
                    exit_reason = "stop_loss"
                    exit_px = sp  # Exit at stop price

            # 3) Check time-based stop
            if exit_reason is None and max_bars_in_trade is not None:
                if pos["BarsInTrade"] >= max_bars_in_trade:
                    exit_reason = "time_stop"
                    exit_px = float(close[i])

            # 4) Check signal-based exit (price reaches mid/lower band)
            if exit_reason is None:
                if exit_channel == "BB":
                    thresh_arr = bb_mid if exit_level == "mid" else bb_low
                else:
                    thresh_arr = kc_mid if exit_level == "mid" else kc_low
                thresh = thresh_arr[i]
                if not math.isnan(thresh) and close[i] <= thresh:
                    exit_reason = "signal_exit"
                    exit_px = float(close[i])

            # 5) Check liquidation (Margin/Futures only)
            if (
                exit_reason is None
                and trade_mode == "Margin / Futures"
                and maintenance_margin_pct is not None
                and maintenance_margin_pct > 0
            ):
                liq_price = pos.get("LiqPrice")
                if liq_price is not None and not math.isnan(liq_price):
                    # For shorts, liquidation when price spikes UP
                    if high[i] >= liq_price:
                        exit_reason = "liquidation"
                        exit_px = liq_price

            # =================================================================
            # Execute Exit if any condition triggered
            # =================================================================
            if exit_reason is not None:
                entry = pos["EntryPrice"]
                side = pos["Side"]
                size = pos["Size"]
                notional = pos["NotionalEntry"]
                risk_dollars = pos.get("RiskDollars", float("nan"))
                equity_before = pos.get("EquityBefore", equity)
                entry_commission = pos.get("EntryCommission", 0.0)

                # Calculate P&L for short position
                if side == "Short":
                    pnl_per_unit = entry - exit_px
                else:
                    pnl_per_unit = exit_px - entry

                # Calculate exit commission
                exit_notional = size * exit_px
                exit_commission = exit_notional * commission
                total_commission = entry_commission + exit_commission

                # Final P&L after commissions
                pnl = pnl_per_unit * size - total_commission
                equity_after = equity_before + pnl

                # Calculate R-multiple (profit relative to risk)
                if not math.isnan(risk_dollars) and risk_dollars > 0:
                    R_mult = pnl / risk_dollars
                else:
                    R_mult = float("nan")

                # Update daily loss accumulator
                ret_pct_equity = (pnl / equity_before * 100.0) if equity_before != 0 else 0.0
                daily_loss_pct_acc += ret_pct_equity

                # Record trade
                trades.append({
                    "EntryBar": pos["EntryBar"],
                    "ExitBar": i,
                    "EntryPrice": entry,
                    "ExitPrice": exit_px,
                    "Side": side,
                    "ExitReason": exit_reason,
                    "Size": size,
                    "NotionalEntry": notional,
                    "RealizedPnL": pnl,
                    "Commission": total_commission,
                    "EquityAfter": equity_after,
                    "R_multiple": R_mult,
                    "EquityBefore": equity_before,
                    "EffectiveLeverage": pos.get("EffectiveLeverage", float("nan")),
                    "MarginUtilAtEntry": pos.get("MarginUtilAtEntry", float("nan")),
                    "LiqPrice": pos.get("LiqPrice"),
                })

                equity = equity_after
                pos = None

    # =========================================================================
    # Force close any open position at end of data
    # =========================================================================
    if pos is not None:
        entry = pos["EntryPrice"]
        side = pos["Side"]
        size = pos["Size"]
        notional = pos["NotionalEntry"]
        risk_dollars = pos.get("RiskDollars", float("nan"))
        equity_before = pos.get("EquityBefore", equity)
        entry_commission = pos.get("EntryCommission", 0.0)
        exit_px = float(close[-1])

        if side == "Short":
            pnl_per_unit = entry - exit_px
        else:
            pnl_per_unit = exit_px - entry

        exit_notional = size * exit_px
        exit_commission = exit_notional * commission
        total_commission = entry_commission + exit_commission

        pnl = pnl_per_unit * size - total_commission
        equity_after = equity_before + pnl

        if not math.isnan(risk_dollars) and risk_dollars > 0:
            R_mult = pnl / risk_dollars
        else:
            R_mult = float("nan")

        ret_pct_equity = (pnl / equity_before * 100.0) if equity_before != 0 else 0.0
        daily_loss_pct_acc += ret_pct_equity

        trades.append({
            "EntryBar": pos["EntryBar"],
            "ExitBar": n - 1,
            "EntryPrice": entry,
            "ExitPrice": exit_px,
            "Side": side,
            "ExitReason": "close_at_end",
            "Size": size,
            "NotionalEntry": notional,
            "RealizedPnL": pnl,
            "Commission": total_commission,
            "EquityAfter": equity_after,
            "R_multiple": R_mult,
            "EquityBefore": equity_before,
            "EffectiveLeverage": pos.get("EffectiveLeverage", float("nan")),
            "MarginUtilAtEntry": pos.get("MarginUtilAtEntry", float("nan")),
            "LiqPrice": pos.get("LiqPrice"),
        })

    return pd.DataFrame(trades), equity_curve


# =============================================================================
# Trade Finalization
# =============================================================================

def finalize_trades(
    ds: pd.DataFrame,
    raw_trades: pd.DataFrame,
    *,
    timeframe: str,
    time_convention: Literal["open", "close"] = "close",
) -> pd.DataFrame:
    """
    Add timestamps and return calculations to raw trade records.
    
    Converts bar indices to proper timestamps and calculates per-trade
    return percentages.
    
    Args:
        ds: Dataset DataFrame with DatetimeIndex.
        raw_trades: Trade DataFrame from run_strategy_loop.
        timeframe: Candle timeframe for timestamp calculation.
        time_convention: 'open' or 'close' for bar timestamps.
    
    Returns:
        DataFrame with additional columns:
            - EntryTimeUTC, ExitTimeUTC: Timestamps
            - ReturnPct: Return as decimal (0.05 = 5%)
            - EntryTimeConv: Time convention used
    """
    ds = _ensure_utc_index(ds)

    if raw_trades is None or raw_trades.empty:
        return pd.DataFrame(columns=[
            "EntryBar", "ExitBar", "EntryTimeUTC", "ExitTimeUTC",
            "EntryPrice", "ExitPrice", "Side", "ExitReason", "EntryTimeConv", "ReturnPct"
        ])

    t = raw_trades.copy()
    if "EntryBar" not in t.columns or "ExitBar" not in t.columns:
        raise ValueError("finalize_trades requires EntryBar and ExitBar in raw_trades")

    t["EntryBar"] = t["EntryBar"].astype(int)
    t["ExitBar"] = t["ExitBar"].astype(int)

    # Filter valid bar indices
    n = len(ds)
    mask = (t["EntryBar"].between(0, n - 1)) & (t["ExitBar"].between(0, n - 1))
    t = t[mask].reset_index(drop=True)
    if t.empty:
        return pd.DataFrame(columns=[
            "EntryBar", "ExitBar", "EntryTimeUTC", "ExitTimeUTC",
            "EntryPrice", "ExitPrice", "Side", "ExitReason", "EntryTimeConv", "ReturnPct"
        ])

    # Convert bar indices to timestamps
    t["EntryTimeUTC"] = [_bar_time(ds, ei, timeframe, time_convention) for ei in t["EntryBar"]]
    t["ExitTimeUTC"] = [_bar_time(ds, xi, timeframe, time_convention) for xi in t["ExitBar"]]

    # Ensure price columns exist
    if "EntryPrice" not in t.columns:
        t["EntryPrice"] = [float(ds["Close"].iloc[ei]) for ei in t["EntryBar"]]
    if "ExitPrice" not in t.columns:
        t["ExitPrice"] = [float(ds["Close"].iloc[xi]) for xi in t["ExitBar"]]
    if "Side" not in t.columns:
        t["Side"] = "Short"

    # Calculate per-unit return (price move percentage)
    t["ReturnPct"] = (t["EntryPrice"] - t["ExitPrice"]) / t["EntryPrice"]
    t["EntryTimeConv"] = time_convention

    if "ExitReason" not in t.columns:
        t["ExitReason"] = "signal_exit"

    # Order columns
    base_cols = [
        "EntryBar", "ExitBar", "EntryTimeUTC", "ExitTimeUTC",
        "EntryPrice", "ExitPrice", "Side", "ExitReason", "EntryTimeConv", "ReturnPct"
    ]
    extra_cols = [c for c in t.columns if c not in base_cols]

    return t[base_cols + extra_cols]


# =============================================================================
# Statistics Calculation
# =============================================================================

def compute_stats(
    trades: pd.DataFrame,
    ds: pd.DataFrame,
    params: Dict[str, Union[str, float, int]],
    equity_curve: Optional[EquityCurve] = None
) -> pd.Series:
    """
    Calculate comprehensive performance statistics for backtest results.
    
    Args:
        trades: Finalized trades DataFrame.
        ds: Dataset DataFrame.
        params: Dictionary with 'timeframe' and 'initial_cash' keys.
        equity_curve: Optional equity curve array for drawdown calculations.
    
    Returns:
        Series containing performance metrics:
            - trades: Number of trades
            - win_rate: Percentage of winning trades
            - avg_return_pct: Mean return per trade
            - median_return_pct: Median return per trade
            - best_return_pct: Maximum trade return
            - worst_return_pct: Minimum trade return
            - profit_factor: Gross profit / Gross loss
            - avg_duration: Average trade duration
            - total_equity_return_pct: Overall portfolio return
            - max_drawdown_pct: Maximum peak-to-trough decline
            - sharpe_ratio: Risk-adjusted return (annualized)
            - sortino_ratio: Downside-adjusted return (annualized)
            - calmar_ratio: Return / Max Drawdown
            - max_consecutive_wins: Longest winning streak
            - max_consecutive_losses: Longest losing streak
    """
    initial_cash = float(params.get("initial_cash", 10_000.0))
    
    if trades is None or trades.empty:
        return pd.Series({
            "trades": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "best_return_pct": 0.0,
            "worst_return_pct": 0.0,
            "profit_factor": 0.0,
            "avg_duration": pd.Timedelta(0),
            "total_equity_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        })

    # Prefer net PnL-based metrics when available (includes commissions and sizing).
    has_pnl = "RealizedPnL" in trades.columns
    has_eq_before = "EquityBefore" in trades.columns

    # "ReturnPct" is the per-unit price move and does NOT include fees.
    ret = trades["ReturnPct"].astype(float)

    if has_pnl:
        pnl = pd.to_numeric(trades["RealizedPnL"], errors="coerce").astype(float)
        wins = pnl > 0

        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(pnl[pnl < 0].sum())  # negative
        pos_sum = gross_profit
        neg_sum = gross_loss
    else:
        wins = ret > 0
        pos_sum = float(ret[ret > 0].sum())
        neg_sum = float(ret[ret < 0].sum())
    td = _tf_delta(params.get("timeframe", "1h"))
    avg_bars = (trades["ExitBar"] - trades["EntryBar"]).mean() if len(trades) else 0

    # Total equity return
    if "EquityAfter" in trades.columns and len(trades):
        final_equity = float(trades["EquityAfter"].iloc[-1])
        total_equity_return_pct = (final_equity / initial_cash - 1.0) * 100.0
    else:
        total_equity_return_pct = 0.0

    # Calculate drawdown metrics from equity curve
    max_drawdown_pct = 0.0
    if equity_curve is not None and len(equity_curve) > 0:
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max * 100.0
        max_drawdown_pct = float(np.nanmax(drawdown))

    # Calculate Sharpe and Sortino ratios (annualized)
    sharpe_ratio = 0.0
    sortino_ratio = 0.0
    if len(ret) > 1:
        # Annualization factor based on timeframe
        tf_hours = td.total_seconds() / 3600
        trades_per_year = 365 * 24 / tf_hours  # Approximate bars per year
        
        # Use equity-relative returns if possible; otherwise fall back to per-unit price move returns.
        if has_pnl and has_eq_before:
            eq_before = pd.to_numeric(trades["EquityBefore"], errors="coerce").astype(float)
            rets_for_risk = (pnl / eq_before.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        else:
            rets_for_risk = ret.replace([np.inf, -np.inf], np.nan).dropna()

        ret_mean = rets_for_risk.mean() if len(rets_for_risk) else 0.0
        ret_std = rets_for_risk.std() if len(rets_for_risk) else 0.0
        
        if ret_std > 0:
            sharpe_ratio = (ret_mean / ret_std) * math.sqrt(trades_per_year)
        
        # Sortino uses only downside deviation
        downside_returns = rets_for_risk[rets_for_risk < 0] if len(rets_for_risk) else rets_for_risk
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino_ratio = (ret_mean / downside_std) * math.sqrt(trades_per_year)

    # Calmar ratio
    calmar_ratio = 0.0
    if max_drawdown_pct > 0:
        calmar_ratio = total_equity_return_pct / max_drawdown_pct

    # Consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    if len(wins) > 0:
        current_streak = 0
        is_winning = None
        for w in wins:
            if is_winning is None:
                is_winning = w
                current_streak = 1
            elif w == is_winning:
                current_streak += 1
            else:
                if is_winning:
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                is_winning = w
                current_streak = 1
        # Final streak
        if is_winning:
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        else:
            max_consecutive_losses = max(max_consecutive_losses, current_streak)

    stats = {
        "trades": len(trades),
        "win_rate": float(wins.mean()) * 100.0,
        # Return stats: prefer equity-relative net returns when possible.
        "avg_return_pct": float(
            (pnl / pd.to_numeric(trades["EquityBefore"], errors="coerce").replace(0, np.nan)).mean() * 100.0
        ) if (has_pnl and has_eq_before) else float(ret.mean()) * 100.0,
        "median_return_pct": float(
            (pnl / pd.to_numeric(trades["EquityBefore"], errors="coerce").replace(0, np.nan)).median() * 100.0
        ) if (has_pnl and has_eq_before) else float(ret.median()) * 100.0,
        "best_return_pct": float(
            (pnl / pd.to_numeric(trades["EquityBefore"], errors="coerce").replace(0, np.nan)).max() * 100.0
        ) if (has_pnl and has_eq_before) else float(ret.max()) * 100.0,
        "worst_return_pct": float(
            (pnl / pd.to_numeric(trades["EquityBefore"], errors="coerce").replace(0, np.nan)).min() * 100.0
        ) if (has_pnl and has_eq_before) else float(ret.min()) * 100.0,
        "profit_factor": math.inf if neg_sum == 0 else (pos_sum / abs(neg_sum)),
        "avg_duration": avg_bars * td,
        "total_equity_return_pct": total_equity_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
    }

    return pd.Series(stats)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    timeframe: str,
    bb_len: int,
    bb_std: float,
    bb_basis_type: str,
    kc_ema_len: int,
    kc_atr_len: int,
    kc_mult: float,
    kc_mid_type: str,
    rsi_len_30m: int,
    rsi_ma_len: int,
    rsi_smoothing_type: str,
    rsi_ma_type: str,
    rsi_min: float,
    rsi_ma_min: float,
    use_rsi_relation: bool,
    rsi_relation: str,
    entry_band_mode: str,
    exit_channel: str,
    exit_level: str,
    cash: float = 10_000.0,
    commission: float = 0.0005,
    trade_mode: str = "Simple (1x spot-style)",
    use_stop: bool = True,
    stop_mode: str = "Fixed %",
    stop_pct: float = 2.0,
    stop_atr_mult: float = 2.0,
    use_trailing: bool = False,
    trail_pct: float = 1.0,
    max_bars_in_trade: int = 100,
    daily_loss_limit: float = 3.0,
    risk_per_trade_pct: float = 1.0,
    max_leverage: Optional[float] = None,
    maintenance_margin_pct: Optional[float] = None,
    max_margin_utilization: Optional[float] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, EquityCurve]:
    """
    Run a complete backtest with the BB+KC+RSI short strategy.
    
    This is the main entry point that orchestrates:
    1. Dataset preparation with indicators
    2. Strategy execution
    3. Trade finalization
    4. Statistics calculation
    
    Args:
        df: Raw OHLCV DataFrame.
        timeframe: Candle timeframe.
        bb_*: Bollinger Bands parameters.
        kc_*: Keltner Channel parameters.
        rsi_*: RSI parameters.
        entry_band_mode: Band touch requirement for entry.
        exit_channel, exit_level: Signal exit configuration.
        cash: Starting capital.
        commission: Trading fee fraction.
        trade_mode: 'Simple' or 'Margin / Futures'.
        use_stop, stop_mode, stop_pct, stop_atr_mult: Stop loss settings.
        use_trailing, trail_pct: Trailing stop settings.
        max_bars_in_trade: Time-based exit.
        daily_loss_limit: Daily loss % limit.
        risk_per_trade_pct: Risk sizing (Margin mode).
        max_leverage, maintenance_margin_pct, max_margin_utilization: Margin settings.
    
    Returns:
        Tuple of (stats, dataset, trades, equity_curve):
            - stats: Performance metrics Series
            - dataset: DataFrame with OHLCV + indicators
            - trades: Finalized trades DataFrame
            - equity_curve: Numpy array of equity at each bar
    """
    df = _ensure_utc_index(df)

    # Build dataset with indicators
    ds = build_dataset(
        df,
        bb_len=bb_len, bb_std=bb_std, bb_basis_type=bb_basis_type,
        kc_ema_len=kc_ema_len, kc_atr_len=kc_atr_len, kc_mult=kc_mult, kc_mid_type=kc_mid_type,
        rsi_len_30m=rsi_len_30m, rsi_ma_len=rsi_ma_len,
        rsi_smoothing_type=rsi_smoothing_type, rsi_ma_type=rsi_ma_type,
    )

    # Execute strategy
    trades_raw, equity_curve = run_strategy_loop(
        ds,
        rsi_min=rsi_min, rsi_ma_min=rsi_ma_min,
        use_rsi_relation=use_rsi_relation, rsi_relation=rsi_relation,
        entry_band_mode=entry_band_mode,
        exit_channel=exit_channel, exit_level=exit_level,
        cash=cash,
        commission=commission,
        trade_mode=trade_mode,
        use_stop=use_stop,
        stop_mode=stop_mode,
        stop_pct=stop_pct,
        stop_atr_mult=stop_atr_mult,
        use_trailing=use_trailing,
        trail_pct=trail_pct,
        max_bars_in_trade=max_bars_in_trade,
        daily_loss_limit=daily_loss_limit,
        risk_per_trade_pct=risk_per_trade_pct,
        max_leverage=max_leverage,
        maintenance_margin_pct=maintenance_margin_pct,
        max_margin_utilization=max_margin_utilization,
    )

    # Finalize trades with timestamps
    trades = finalize_trades(ds, trades_raw, timeframe=timeframe, time_convention="close")

    # Calculate statistics
    stats = compute_stats(
        trades, ds,
        params={"timeframe": timeframe, "initial_cash": cash},
        equity_curve=equity_curve
    )
    
    return stats, ds, trades, equity_curve
