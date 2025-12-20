import math
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.utils import calculate_drawdown


def build_trades_table(trades_obj) -> pd.DataFrame:
    if trades_obj is None:
        return pd.DataFrame()

    t = pd.DataFrame(trades_obj).copy()
    if t.empty:
        return pd.DataFrame()

    required = [
        "EntryBar", "ExitBar", "EntryTimeUTC", "ExitTimeUTC",
        "EntryPrice", "ExitPrice", "Side", "ExitReason", "ReturnPct",
    ]

    missing = [c for c in required if c not in t.columns]
    if missing:
        raise ValueError(f"Engine trades missing required columns: {missing}")

    out = pd.DataFrame({
        "EntryTime": pd.to_datetime(t["EntryTimeUTC"], utc=True),
        "ExitTime": pd.to_datetime(t["ExitTimeUTC"], utc=True),
        "EntryBar": t["EntryBar"].astype("Int64"),
        "ExitBar": t["ExitBar"].astype("Int64"),
        "Side": t["Side"],
        "Price@Entry": pd.to_numeric(t["EntryPrice"]),
        "Price@Exit": pd.to_numeric(t["ExitPrice"]),
        "ExitReason": t["ExitReason"],
    })

    optional = [
        "Size", "NotionalEntry", "RealizedPnL",
        "EquityAfter", "EquityBefore", "R_multiple", "SkippedDailyLoss",
        "EffectiveLeverage", "MarginUtilAtEntry", "LiqPrice",
    ]
    for col in optional:
        if col in t.columns:
            out[col] = t[col]

    ret_pct = pd.to_numeric(t["ReturnPct"], errors="coerce")
    out["Price Move %"] = (ret_pct * 100).round(3)
    out["Duration"] = out["ExitTime"] - out["EntryTime"]
    out["PnL (per unit)"] = out["Price@Entry"] - out["Price@Exit"]

    return out


def build_entry_diagnostics(trades_table: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    if trades_table is None or trades_table.empty:
        return pd.DataFrame()

    time_col = "EntryTimeUTC" if "EntryTimeUTC" in trades_table.columns else (
        "EntryTime" if "EntryTime" in trades_table.columns else None
    )
    bar_col = "EntryBar" if "EntryBar" in trades_table.columns else None
    if bar_col is None:
        return pd.DataFrame()

    has_exit_reason = "ExitReason" in trades_table.columns
    margin_mode = False
    if "MarginUtilAtEntry" in trades_table.columns:
        if trades_table["MarginUtilAtEntry"].notna().any():
            margin_mode = True

    rows = []
    for _, r in trades_table.iterrows():
        ei = r.get(bar_col)
        try:
            ei = int(ei) if pd.notna(ei) else None
        except Exception:
            ei = None

        def _safe(col, idx):
            try:
                return float(ds[col].iloc[idx])
            except Exception:
                return float("nan")

        equity_before = float(r.get("EquityBefore")) if "EquityBefore" in trades_table.columns and pd.notna(r.get("EquityBefore")) else float("nan")
        equity_after = float(r.get("EquityAfter")) if "EquityAfter" in trades_table.columns and pd.notna(r.get("EquityAfter")) else float("nan")
        size = float(r.get("Size")) if "Size" in trades_table.columns and pd.notna(r.get("Size")) else float("nan")
        notional = float(r.get("NotionalEntry")) if "NotionalEntry" in trades_table.columns and pd.notna(r.get("NotionalEntry")) else float("nan")

        if equity_before and not math.isnan(equity_before) and equity_before > 0:
            effective_lev = notional / equity_before
        else:
            effective_lev = float("nan")

        if margin_mode and equity_before and not math.isnan(equity_before) and equity_before > 0:
            margin_util = r.get("MarginUtilAtEntry", float("nan"))
        else:
            margin_util = float("nan")

        if ei is None or ei < 0 or ei >= len(ds):
            row = {
                "EntryTime": r.get(time_col),
                "EntryBar": r.get(bar_col),
                "Price@Entry (ds)": float("nan"),
                "RSI@Entry": float("nan"),
                "RSI_MA@Entry": float("nan"),
                "TouchKC?": None,
                "TouchBB?": None,
                "BandOK?": None,
                "Size": size,
                "NotionalEntry": notional,
                "EquityBefore": equity_before,
                "EquityAfter": equity_after,
                "RealizedPnL": r.get("RealizedPnL"),
                "EffectiveLeverage": effective_lev,
            }

            if margin_mode:
                r_mult = r.get("R_multiple")
                liq = r.get("LiqPrice")

                row["R_multiple"] = r_mult
                row["MarginUtilAtEntry"] = margin_util
                row["HighMarginUtil(>80%)"] = (
                    isinstance(margin_util, (int, float))
                    and not math.isnan(margin_util)
                    and margin_util > 0.8
                )
                row["LiqPrice"] = liq

            if has_exit_reason:
                row["ExitReason"] = r["ExitReason"]
            rows.append(row)
            continue

        px = _safe("Close", ei)
        rsi_v = _safe("rsi30", ei) if "rsi30" in ds.columns else float("nan")
        rma_v = _safe("rsi30_ma", ei) if "rsi30_ma" in ds.columns else float("nan")
        bb_u = _safe("bb_up", ei) if "bb_up" in ds.columns else float("nan")
        kc_u = _safe("kc_up", ei) if "kc_up" in ds.columns else float("nan")

        touch_kc = (px >= kc_u) if pd.notna(kc_u) else False
        touch_bb = (px >= bb_u) if pd.notna(bb_u) else False

        row = {
            "EntryTime": r.get(time_col),
            "EntryBar": ei,
            "Price@Entry (ds)": px,
            "RSI@Entry": rsi_v,
            "RSI_MA@Entry": rma_v,
            "TouchKC?": bool(touch_kc),
            "TouchBB?": bool(touch_bb),
            "Size": size,
            "NotionalEntry": notional,
            "EquityBefore": equity_before,
            "EquityAfter": equity_after,
            "RealizedPnL": r.get("RealizedPnL"),
            "EffectiveLeverage": effective_lev,
        }

        if margin_mode:
            r_mult = r.get("R_multiple")
            liq = r.get("LiqPrice")

            row["R_multiple"] = r_mult
            row["MarginUtilAtEntry"] = margin_util
            row["HighMarginUtil(>80%)"] = (
                isinstance(margin_util, (int, float))
                and not math.isnan(margin_util)
                and margin_util > 0.8
            )
            row["LiqPrice"] = liq

        if has_exit_reason:
            row["ExitReason"] = r["ExitReason"]
        rows.append(row)

    return pd.DataFrame(rows)


def build_backtest_figure(
    ds: pd.DataFrame,
    trades_table: pd.DataFrame,
    equity_curve,
    params: dict,
    *,
    selected_trade: Optional[dict] = None,
    show_candles: bool = True,
    lock_rsi_y: bool = True,
):
    if ds is None or ds.empty:
        return go.Figure()

    ds_full = ds
    ds_plot = ds_full.copy()
    max_points = 5000
    if len(ds_plot) > max_points:
        step = max(1, len(ds_plot) // max_points)
        ds_plot = ds_plot.iloc[::step]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.50, 0.25, 0.25],
        subplot_titles=("Price + BB + KC", "RSI (30m) + MA", "Equity & Drawdown"),
    )

    if show_candles:
        fig.add_trace(
            go.Candlestick(
                x=ds_plot.index,
                open=ds_plot["Open"],
                high=ds_plot["High"],
                low=ds_plot["Low"],
                close=ds_plot["Close"],
                name="Price",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=ds_plot.index, y=ds_plot["Close"], mode="lines", name="Close"),
            row=1,
            col=1,
        )

    bb_basis_type = params.get("bb_basis_type", "sma")
    kc_mid_type = params.get("kc_mid_type", "ema")
    rsi_ma_type = params.get("rsi_ma_type", "sma")

    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["bb_mid"],
            mode="lines",
            name=f"BB mid ({bb_basis_type.upper()})",
            line=dict(width=1, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["bb_up"],
            mode="lines",
            name="BB upper",
            line=dict(color="gray", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["bb_low"],
            mode="lines",
            name="BB lower",
            line=dict(color="gray", width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["kc_mid"],
            mode="lines",
            name=f"KC mid ({kc_mid_type.upper()})",
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["kc_up"],
            mode="lines",
            name="KC upper",
            line=dict(color="blue", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["kc_low"],
            mode="lines",
            name="KC lower",
            line=dict(color="blue", width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["rsi30"],
            mode="lines",
            name="RSI(30m)",
            line=dict(color="purple"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ds_plot.index,
            y=ds_plot["rsi30_ma"],
            mode="lines",
            name=f"RSI MA ({rsi_ma_type.upper()})",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )

    rsi_min = params.get("rsi_min", 70)
    rsi_ma_min = params.get("rsi_ma_min", 70)
    fig.add_hline(y=rsi_min, line_dash="dot", annotation_text="RSI Min", row=2, col=1)
    fig.add_hline(y=rsi_ma_min, line_dash="dot", annotation_text="RSI MA Min", row=2, col=1)

    if equity_curve is not None and len(equity_curve) > 0:
        if len(ds_plot) < len(equity_curve):
            step = max(1, len(equity_curve) // len(ds_plot))
            eq_sampled = equity_curve[::step][: len(ds_plot)]
        else:
            eq_sampled = equity_curve[: len(ds_plot)]

        eq_sampled = eq_sampled[: len(ds_plot.index)]

        drawdown_pct, _ = calculate_drawdown(eq_sampled)

        fig.add_trace(
            go.Scatter(
                x=ds_plot.index[: len(eq_sampled)],
                y=eq_sampled,
                mode="lines",
                name="Equity",
                line=dict(color="royalblue", width=2),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ds_plot.index[: len(drawdown_pct)],
                y=-drawdown_pct,
                mode="lines",
                fill="tozeroy",
                name="Drawdown %",
                line=dict(color="crimson", width=1),
                fillcolor="rgba(220, 20, 60, 0.3)",
            ),
            row=3,
            col=1,
        )

    if selected_trade is None:
        if trades_table is not None and not trades_table.empty:
            fig.add_trace(
                go.Scattergl(
                    x=trades_table["EntryTime"],
                    y=trades_table["Price@Entry"],
                    mode="markers",
                    marker=dict(size=8, symbol="triangle-down", color="red"),
                    name="Short Entry",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scattergl(
                    x=trades_table["ExitTime"],
                    y=trades_table["Price@Exit"],
                    mode="markers",
                    marker=dict(size=8, symbol="circle-dot", color="green"),
                    name="Exit",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
    else:
        try:
            if "EntryBar" in selected_trade and pd.notna(selected_trade.get("EntryBar")):
                ei = int(selected_trade["EntryBar"])
                xi_raw = selected_trade.get("ExitBar", ei)
                xi = int(xi_raw) if (xi_raw is not None and str(xi_raw).isdigit()) else ei
            else:
                e_ts = pd.to_datetime(selected_trade.get("EntryTime"))
                x_ts = pd.to_datetime(selected_trade.get("ExitTime") or e_ts)
                ei = int(ds_full.index.get_indexer([e_ts], method="nearest")[0])
                xi = int(ds_full.index.get_indexer([x_ts], method="nearest")[0])

            pad = 30
            i0 = max(0, min(ei, xi) - pad)
            i1 = min(len(ds_full) - 1, max(ei, xi) + pad)
            x0, x1 = ds_full.index[i0], ds_full.index[i1]

            fig.update_xaxes(range=[x0, x1], row=1, col=1)
            fig.update_xaxes(range=[x0, x1], row=2, col=1)
            fig.update_xaxes(range=[x0, x1], row=3, col=1)

            window_high = float(ds_full["High"].iloc[i0 : i1 + 1].max())
            window_low = float(ds_full["Low"].iloc[i0 : i1 + 1].min())
            span = max(1e-9, window_high - window_low)
            pad_y = 0.07 * span
            fig.update_yaxes(range=[window_low - pad_y, window_high + pad_y], row=1, col=1)

            e_ts = pd.to_datetime(selected_trade.get("EntryTime")) if "EntryTime" in selected_trade else ds_full.index[ei]
            x_ts = pd.to_datetime(selected_trade.get("ExitTime")) if "ExitTime" in selected_trade else (ds_full.index[xi] if xi is not None else None)
            e_px = selected_trade.get("Price@Entry")
            x_px = selected_trade.get("Price@Exit")

            if pd.notna(e_ts) and pd.notna(e_px):
                fig.add_trace(
                    go.Scattergl(
                        x=[e_ts],
                        y=[e_px],
                        mode="markers",
                        marker=dict(size=16, symbol="triangle-down", color="crimson", line=dict(width=1)),
                        name="Selected Entry",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_vline(x=e_ts, line_width=1.5, line_dash="dot", line_color="crimson", row=1, col=1)

            if pd.notna(x_ts) and pd.notna(x_px):
                fig.add_trace(
                    go.Scattergl(
                        x=[x_ts],
                        y=[x_px],
                        mode="markers",
                        marker=dict(size=16, symbol="x", color="limegreen", line=dict(width=2)),
                        name="Selected Exit",
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_vline(x=x_ts, line_width=1.5, line_dash="dot", line_color="limegreen", row=1, col=1)
        except Exception:
            pass

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        height=700,
    )
    if lock_rsi_y:
        fig.update_yaxes(range=[0, 100], fixedrange=True, row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=2, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=3, col=1)
    fig.update_yaxes(title_text="Equity / DD%", row=3, col=1)

    return fig
