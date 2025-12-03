from __future__ import annotations

from typing import Literal, Dict
import math
import pandas as pd
import numpy as np

from core.indicators import add_bb_kc_rsi


# Time/index helpers
_TF_TO_PD = {
    "1m": "1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min",
    "1h":"1H","2h":"2H","4h":"4H","6h":"6H","12h":"12H",
    "1d":"1D","3d":"3D","1w":"7D"
}

def _tf_delta(timeframe: str) -> pd.Timedelta:
    return pd.to_timedelta(_TF_TO_PD.get(timeframe.lower(), "1H"))

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def _bar_time(ds: pd.DataFrame, bar: int, timeframe: str, convention: Literal["open","close"]) -> pd.Timestamp:
    if convention == "close":
        return (ds.index[bar] + _tf_delta(timeframe) - pd.Timedelta(milliseconds=1)).tz_convert("UTC")
    return ds.index[bar].tz_convert("UTC")





def build_dataset(
    df: pd.DataFrame,
    *,
    bb_len=20, bb_std=2.0, bb_basis_type="sma",
    kc_ema_len=20, kc_atr_len=14, kc_mult=2.0, kc_mid_type="ema",
    rsi_len_30m=14, rsi_ma_len=10, rsi_smoothing_type="ema", rsi_ma_type="sma",
) -> pd.DataFrame:
    ds = df.copy()
    ds = add_bb_kc_rsi(
        ds,
        bb_len=bb_len, bb_std=bb_std, bb_basis_type=bb_basis_type,
        kc_ema_len=kc_ema_len, kc_atr_len=kc_atr_len, kc_mult=kc_mult, kc_mid_type=kc_mid_type,
        rsi_len_30m=rsi_len_30m, rsi_ma_len=rsi_ma_len,
        rsi_smoothing_type=rsi_smoothing_type, rsi_ma_type=rsi_ma_type,
    )
    return _ensure_utc_index(ds)


def _cmp(a: float, b: float, op: str) -> bool:
    if op == "<":  return a <  b
    if op == "<=": return a <= b
    if op == ">":  return a >  b
    if op == ">=": return a >= b
    return True


# Strategy loop (short-only)
def run_strategy_loop(
    ds: pd.DataFrame,
    *,
    # ENTRY
    rsi_min: float = 70.0,
    rsi_ma_min: float = 70.0,
    use_rsi_relation: bool = True,
    rsi_relation: Literal["<","<=",">",">="] = ">=",
    entry_band_mode: Literal["KC","BB","Both","Either"] = "Either",

    # EXIT (signal)
    exit_channel: Literal["BB","KC"] = "BB",
    exit_level: Literal["mid","lower"] = "mid",

    # RISK / STOPS / ACCOUNT
    cash: float = 10_000.0,
    commission: float = 0.0005,         # currently not used, but keep for later
    trade_mode: str = "Simple (1x spot-style)",
    use_stop: bool = True,
    stop_mode: Literal["Fixed %","ATR"] = "Fixed %",
    stop_pct: float = 2.0,
    stop_atr_mult: float = 2.0,
    use_trailing: bool = False,
    trail_pct: float = 1.0,
    max_bars_in_trade: int = 100,
    daily_loss_limit: float = 0.0,    
    risk_per_trade_pct: float = 1.0,    # % of equity to risk per trade (futures only)
    max_leverage: float | None = None,
    maintenance_margin_pct=None,        # used for liquidation (Margin / Futures)
    max_margin_utilization=None,        # cap margin usage in Margin/Futures mode
) -> pd.DataFrame:

    close = ds["Close"].values
    high  = ds["High"].values
    low   = ds["Low"].values

    bb_up  = ds["bb_up"].values
    bb_mid = ds["bb_mid"].values
    bb_low = ds["bb_low"].values

    kc_up  = ds["kc_up"].values
    kc_mid = ds["kc_mid"].values
    kc_low = ds["kc_low"].values

    rsi = ds["rsi30"].values
    rma = ds["rsi30_ma"].values

    atr = ds["kc_atr"].values if ("kc_atr" in ds.columns and stop_mode == "ATR") else None

    n = len(ds)
    idx = ds.index

    trades: list[dict] = []

    # ACCOUNT STATE
    equity = float(cash)     
    cur_day = None                 # current UTC date
    daily_loss_pct_acc = 0.0       # accumulated daily return in %

    # Active position
    pos = None  # dict with EntryBar, EntryPrice, Side, Size, etc.



    # helpers
    def band_ok(i: int) -> bool:
        p = close[i]
        touch_kc = (p >= kc_up[i]) if not math.isnan(kc_up[i]) else False
        touch_bb = (p >= bb_up[i]) if not math.isnan(bb_up[i]) else False
        if entry_band_mode == "KC":   return touch_kc
        if entry_band_mode == "BB":   return touch_bb
        if entry_band_mode == "Both": return touch_kc and touch_bb
        return touch_kc or touch_bb 

    def rsi_relation_ok(i: int) -> bool:
        if not use_rsi_relation:
            return True
        return _cmp(rsi[i], rma[i], rsi_relation)

    def compute_stop_price_short(entry_px: float, i: int) -> float | None:
        if not use_stop:
            return None
        if stop_mode == "Fixed %":
            return entry_px * (1.0 + stop_pct / 100.0)
        if stop_mode == "ATR":
            if atr is None or math.isnan(atr[i]) or stop_atr_mult is None:
                return None
            return entry_px + stop_atr_mult * atr[i]
        return None

    def update_trailing_stop_short(pos_dict, i: int):
        if not use_trailing:
            return
        cur_px = close[i]
        best_px = pos_dict.get("BestPrice", pos_dict["EntryPrice"])
        if cur_px < best_px:
            best_px = cur_px
            pos_dict["BestPrice"] = best_px
            pos_dict["StopPrice"] = best_px * (1.0 + trail_pct / 100.0)

    # MAIN LOOP
    for i in range(n):
        bar_ts = idx[i]
        day = bar_ts.date()

        # new day -> reset daily loss accumulator
        if cur_day is None or day != cur_day:
            cur_day = day
            daily_loss_pct_acc = 0.0

        # ENTRY (short only)
        if pos is None and equity > 0:
            # If daily loss limit hit, don't open new trades
            if (
                daily_loss_limit is not None
                and daily_loss_limit > 0
                and daily_loss_pct_acc <= -daily_loss_limit
            ):
                pass
            else:
                if (
                    band_ok(i)
                    and (not math.isnan(rsi[i]) and rsi[i] >= rsi_min)
                    and (not math.isnan(rma[i]) and rma[i] >= rsi_ma_min)
                    and rsi_relation_ok(i)
                ):
                    entry_px = float(close[i])
                    if entry_px <= 0:
                        continue

                    equity_before = equity

                    # SIZING: SPOT vs FUTURES
                    if "Simple" in trade_mode:
                        # SPOT MODE
                        # Always 1×, use all equity as notional
                        notional = equity_before
                        size = notional / entry_px
                        effective_leverage = 1.0
                        margin_util = float("nan")   # N/A in spot
                        risk_dollars = float("nan")  # N/A in spot
                        stop_price = compute_stop_price_short(entry_px, i)  # stop only affects exit

                    else:
                        # MARGIN / FUTURES MODE
                        if use_stop:
                            # stop-based sizing
                            stop_price = compute_stop_price_short(entry_px, i)
                            if stop_price is not None and not math.isnan(stop_price):
                                risk_per_unit = abs(stop_price - entry_px)
                            else:
                                # fallback if stop can't be computed
                                risk_per_unit = entry_px * 0.01
                                stop_price = None

                            if risk_per_unit <= 0:
                                continue

                            if risk_per_trade_pct > 0 and equity_before > 0:
                                risk_dollars = equity_before * (risk_per_trade_pct / 100.0)
                                size = risk_dollars / risk_per_unit
                            else:
                                size = 1.0
                                risk_dollars = 0.0

                            notional = size * entry_px

                            # leverage cap
                            if max_leverage is not None and max_leverage > 0:
                                notional_cap = equity_before * max_leverage
                                if notional > notional_cap and notional > 0:
                                    scale = notional_cap / notional
                                    size *= scale
                                    notional = size * entry_px
                        else:
                            # no stop -> size from max_leverage
                            effective_max_lev = max_leverage if (max_leverage is not None and max_leverage > 0) else 1.0
                            notional = equity_before * effective_max_lev
                            size = notional / entry_px
                            risk_dollars = float("nan") 
                            stop_price = None

                        # effective leverage & margin utilization
                        effective_leverage = notional / equity_before if equity_before > 0 else float("nan")
                        if max_leverage is not None and max_leverage > 0 and equity_before > 0:
                            margin_util = notional / (equity_before * max_leverage)
                        else:
                            margin_util = float("nan")

                        # optional margin-utilization cap (only in Margin / Futures mode)
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
                                continue

                    # OPEN POSITION
                    pos = {
                        "EntryBar":           i,
                        "EntryPrice":         entry_px,
                        "Side":               "Short",
                        "BarsInTrade":        0,
                        "Size":               size,
                        "NotionalEntry":      notional,
                        "RiskDollars":        risk_dollars,
                        "EquityBefore":       equity_before,
                        "BestPrice":          entry_px,
                        "EffectiveLeverage":  effective_leverage,
                        "MarginUtilAtEntry":  margin_util,
                        "LiqPrice":           None,   # set below for margin mode
                    }
                    pos["StopPrice"] = stop_price

                    # simple liquidation model (Margin / Futures only)
                    if (
                        trade_mode == "Margin / Futures"
                        and maintenance_margin_pct is not None
                        and maintenance_margin_pct > 0
                        and size > 0
                    ):
                        m = float(maintenance_margin_pct) / 100.0
                        EQ0 = equity_before
                        N = notional
                        # Equity(P) = EQ0 + (Entry - P)*size  (short)
                        # Liquidation when Equity(P) <= N * m
                        #   EQ0 + N - size*P = N*m  ->  P = (EQ0 + N - N*m) / size
                        liq_price = (EQ0 + N - N * m) / size
                        pos["LiqPrice"] = liq_price

                    continue  # move to next bar

        # EXIT / STOP / LIQUIDATION MANAGEMENT
        if pos is not None:
            pos["BarsInTrade"] += 1
            exit_reason = None
            exit_px = float(close[i])

            # 1) Update trailing stop
            if use_trailing:
                update_trailing_stop_short(pos, i)

            # 2) Hard stop
            sp = pos.get("StopPrice")
            if use_stop and sp is not None and not math.isnan(sp):
                if high[i] >= sp:
                    exit_reason = "stop_loss"
                    exit_px = sp

            # 3) Time stop
            if exit_reason is None and max_bars_in_trade is not None:
                if pos["BarsInTrade"] >= max_bars_in_trade:
                    exit_reason = "time_stop"
                    exit_px = float(close[i])

            # 4) Original BB/KC signal exit
            if exit_reason is None:
                if exit_channel == "BB":
                    thresh_arr = bb_mid if exit_level == "mid" else bb_low
                else:
                    thresh_arr = kc_mid if exit_level == "mid" else kc_low
                thresh = thresh_arr[i]
                if not math.isnan(thresh) and close[i] <= thresh:
                    exit_reason = "signal_exit"
                    exit_px = float(close[i])

            # 5) Liquidation check (Margin/Futures only, last resort)
            if (
                exit_reason is None
                and trade_mode == "Margin / Futures"
                and maintenance_margin_pct is not None
                and maintenance_margin_pct > 0
            ):
                liq_price = pos.get("LiqPrice")
                if liq_price is not None and not math.isnan(liq_price):
                    # short: liquidation when price spikes UP to liq_price
                    if high[i] >= liq_price:
                        exit_reason = "liquidation"
                        exit_px = liq_price

            # 6) If we have any exit reason, finalize trade + account update
            if exit_reason is not None:
                entry = pos["EntryPrice"]
                side  = pos["Side"]
                size  = pos["Size"]
                notional = pos["NotionalEntry"]
                risk_dollars = pos.get("RiskDollars", float("nan"))
                equity_before = pos.get("EquityBefore", equity)

                if side == "Short":
                    pnl_per_unit = entry - exit_px
                else:  
                    pnl_per_unit = exit_px - entry

                pnl = pnl_per_unit * size
                equity_after = equity_before + pnl

                # R-multiple (only meaningful if risk_dollars > 0)
                if not math.isnan(risk_dollars) and risk_dollars > 0:
                    R_mult = pnl / risk_dollars
                else:
                    R_mult = float("nan")

                # update daily loss accumulator
                ret_pct_equity = (pnl / equity_before * 100.0) if equity_before != 0 else 0.0
                daily_loss_pct_acc += ret_pct_equity

                trades.append({
                    "EntryBar":           pos["EntryBar"],
                    "ExitBar":            i,
                    "EntryPrice":         entry,
                    "ExitPrice":          exit_px,
                    "Side":               side,
                    "ExitReason":         exit_reason,
                    "Size":               size,
                    "NotionalEntry":      notional,
                    "RealizedPnL":        pnl,
                    "EquityAfter":        equity_after,
                    "R_multiple":         R_mult,
                    "EquityBefore":       equity_before,
                    "EffectiveLeverage":  pos.get("EffectiveLeverage", float("nan")),
                    "MarginUtilAtEntry":  pos.get("MarginUtilAtEntry", float("nan")),
                    "LiqPrice":           pos.get("LiqPrice"),
                })

                equity = equity_after
                pos = None

    # Force close any open position at the last bar
    if pos is not None:
        entry = pos["EntryPrice"]
        side  = pos["Side"]
        size  = pos["Size"]
        notional = pos["NotionalEntry"]
        risk_dollars = pos.get("RiskDollars", float("nan"))
        equity_before = pos.get("EquityBefore", equity)
        exit_px = float(close[-1])

        if side == "Short":
            pnl_per_unit = entry - exit_px
        else:
            pnl_per_unit = exit_px - entry

        pnl = pnl_per_unit * size
        equity_after = equity_before + pnl

        if not math.isnan(risk_dollars) and risk_dollars > 0:
            R_mult = pnl / risk_dollars
        else:
            R_mult = float("nan")

        ret_pct_equity = (pnl / equity_before * 100.0) if equity_before != 0 else 0.0
        daily_loss_pct_acc += ret_pct_equity

        trades.append({
            "EntryBar":           pos["EntryBar"],
            "ExitBar":            n - 1,
            "EntryPrice":         entry,
            "ExitPrice":          exit_px,
            "Side":               side,
            "ExitReason":         "close_at_end",
            "Size":               size,
            "NotionalEntry":      notional,
            "RealizedPnL":        pnl,
            "EquityAfter":        equity_after,
            "R_multiple":         R_mult,
            "EquityBefore":       equity_before,
            "EffectiveLeverage":  pos.get("EffectiveLeverage", float("nan")),
            "MarginUtilAtEntry":  pos.get("MarginUtilAtEntry", float("nan")),
            "LiqPrice":           pos.get("LiqPrice"),
        })

    return pd.DataFrame(trades)



def finalize_trades(
    ds: pd.DataFrame,
    raw_trades: pd.DataFrame,
    *,
    timeframe: str,
    time_convention: Literal["open","close"] = "close",
) -> pd.DataFrame:
    ds = _ensure_utc_index(ds)

    if raw_trades is None or raw_trades.empty:
        return pd.DataFrame(columns=[
            "EntryBar","ExitBar","EntryTimeUTC","ExitTimeUTC",
            "EntryPrice","ExitPrice","Side","ExitReason","EntryTimeConv","ReturnPct"
        ])

    t = raw_trades.copy()
    if "EntryBar" not in t.columns or "ExitBar" not in t.columns:
        raise ValueError("finalize_trades requires EntryBar and ExitBar in raw_trades")

    t["EntryBar"] = t["EntryBar"].astype(int)
    t["ExitBar"]  = t["ExitBar"].astype(int)

    n = len(ds)
    mask = (t["EntryBar"].between(0, n-1)) & (t["ExitBar"].between(0, n-1))
    t = t[mask].reset_index(drop=True)
    if t.empty:
        return pd.DataFrame(columns=[
            "EntryBar","ExitBar","EntryTimeUTC","ExitTimeUTC",
            "EntryPrice","ExitPrice","Side","ExitReason","EntryTimeConv","ReturnPct"
        ])

    t["EntryTimeUTC"] = [ _bar_time(ds, ei, timeframe, time_convention) for ei in t["EntryBar"] ]
    t["ExitTimeUTC"]  = [ _bar_time(ds, xi, timeframe, time_convention) for xi in t["ExitBar"]  ]

    # prices/side
    if "EntryPrice" not in t.columns:
        t["EntryPrice"] = [ float(ds["Close"].iloc[ei]) for ei in t["EntryBar"] ]
    if "ExitPrice" not in t.columns:
        t["ExitPrice"]  = [ float(ds["Close"].iloc[xi]) for xi in t["ExitBar"]  ]
    if "Side" not in t.columns:
        t["Side"] = "Short"

    # per-unit return (price move)
    t["ReturnPct"] = (t["EntryPrice"] - t["ExitPrice"]) / t["EntryPrice"]
    t["EntryTimeConv"] = time_convention

    if "ExitReason" not in t.columns:
        t["ExitReason"] = "signal_exit"

    base_cols = [
        "EntryBar","ExitBar","EntryTimeUTC","ExitTimeUTC",
        "EntryPrice","ExitPrice","Side","ExitReason","EntryTimeConv","ReturnPct"
    ]

    extra_cols = [c for c in t.columns if c not in base_cols]

    return t[base_cols + extra_cols]


# Simple stats
def compute_stats(trades: pd.DataFrame, ds: pd.DataFrame, params: Dict) -> pd.Series:
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
        })

    ret = trades["ReturnPct"].astype(float)
    wins = ret > 0
    pos_sum = float(ret[ret > 0].sum())
    neg_sum = float(ret[ret < 0].sum())
    td = _tf_delta(params.get("timeframe", "1h"))
    avg_bars = (trades["ExitBar"] - trades["EntryBar"]).mean() if len(trades) else 0

    # Overall equity return, based on EquityAfter if present
    initial_cash = float(params.get("initial_cash", 10_000.0))
    if "EquityAfter" in trades.columns and len(trades):
        final_equity = float(trades["EquityAfter"].iloc[-1])
        total_equity_return_pct = (final_equity / initial_cash - 1.0) * 100.0
    else:
        total_equity_return_pct = 0.0

    stats = {
        "trades": len(trades),
        "win_rate": float(wins.mean()) * 100.0,
        "avg_return_pct": float(ret.mean()) * 100.0,
        "median_return_pct": float(ret.median()) * 100.0,
        "best_return_pct": float(ret.max()) * 100.0,
        "worst_return_pct": float(ret.min()) * 100.0,
        "profit_factor": math.inf if neg_sum == 0 else (pos_sum / abs(neg_sum)),
        "avg_duration": avg_bars * td,
        "total_equity_return_pct": total_equity_return_pct,
    }

    return pd.Series(stats)


def run_backtest(
    df,
    timeframe,
    bb_len,
    bb_std,
    bb_basis_type,
    kc_ema_len,
    kc_atr_len,
    kc_mult,
    kc_mid_type,
    rsi_len_30m,
    rsi_ma_len,
    rsi_smoothing_type,
    rsi_ma_type,
    rsi_min,
    rsi_ma_min,
    use_rsi_relation,
    rsi_relation,
    entry_band_mode,
    exit_channel,
    exit_level,
    cash=10_000.0,
    commission=0.0005,

    trade_mode="Simple (1x spot-style)",
    use_stop=True,
    stop_mode="Fixed %",
    stop_pct=2.0,
    stop_atr_mult=2.0,
    use_trailing=False,
    trail_pct=1.0,
    max_bars_in_trade=100,
    daily_loss_limit=3.0,
    risk_per_trade_pct=1.0,

    max_leverage=None,
    maintenance_margin_pct=None,
    max_margin_utilization=None,
):
    df = _ensure_utc_index(df)

    ds = build_dataset(
        df,
        bb_len=bb_len, bb_std=bb_std, bb_basis_type=bb_basis_type,
        kc_ema_len=kc_ema_len, kc_atr_len=kc_atr_len, kc_mult=kc_mult, kc_mid_type=kc_mid_type,
        rsi_len_30m=rsi_len_30m, rsi_ma_len=rsi_ma_len,
        rsi_smoothing_type=rsi_smoothing_type, rsi_ma_type=rsi_ma_type,
    )

    trades_raw = run_strategy_loop(
        ds,
        # entry / exit
        rsi_min=rsi_min, rsi_ma_min=rsi_ma_min,
        use_rsi_relation=use_rsi_relation, rsi_relation=rsi_relation,
        entry_band_mode=entry_band_mode,
        exit_channel=exit_channel, exit_level=exit_level,
        # risk / stops / mode
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

    trades = finalize_trades(ds, trades_raw, timeframe=timeframe, time_convention="close")

    stats = compute_stats(trades, ds, params={"timeframe": timeframe, "initial_cash": cash})
    return stats, ds, trades