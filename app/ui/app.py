import streamlit as st
from core.data import fetch_ohlcv, fetch_ohlcv_range
from backtest.engine import run_backtest
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import datetime as dt
import math
import pandas as pd


# Helpers
def _cmp(a: float, b: float, op: str) -> bool:
    if op == "<":   return a <  b
    if op == "<=":  return a <= b
    if op == ">":   return a >  b
    if op == ">=":  return a >= b
    return True


# Build Trades Table from trades object
def build_trades_table(trades_obj) -> pd.DataFrame:
    if trades_obj is None:
        return pd.DataFrame()

    t = pd.DataFrame(trades_obj).copy()
    if t.empty:
        return pd.DataFrame()

    required = [
        "EntryBar","ExitBar","EntryTimeUTC","ExitTimeUTC",
        "EntryPrice","ExitPrice","Side","ExitReason","ReturnPct"
    ]

    missing = [c for c in required if c not in t.columns]
    if missing:
        raise ValueError(f"Engine trades missing required columns: {missing}")

    # Convert types
    out = pd.DataFrame({
        "EntryTime":    pd.to_datetime(t["EntryTimeUTC"], utc=True),
        "ExitTime":     pd.to_datetime(t["ExitTimeUTC"],  utc=True),
        "EntryBar":     t["EntryBar"].astype("Int64"),
        "ExitBar":      t["ExitBar"].astype("Int64"),
        "Side":         t["Side"],
        "Price@Entry":  pd.to_numeric(t["EntryPrice"]),
        "Price@Exit":   pd.to_numeric(t["ExitPrice"]),
        "ExitReason":   t["ExitReason"],
    })

    # Optional account-level fields — pass through if present
    optional = [
        "Size","NotionalEntry","RealizedPnL",
        "EquityAfter","EquityBefore","R_multiple","SkippedDailyLoss",
        "EffectiveLeverage", "MarginUtilAtEntry", "LiqPrice"
    ]
    for col in optional:
        if col in t.columns:
            out[col] = t[col]

    ret_pct = pd.to_numeric(t["ReturnPct"], errors="coerce")
    out["Price Move %"]       = (ret_pct * 100).round(3)
    out["Duration"]       = out["ExitTime"] - out["EntryTime"]
    out["PnL (per unit)"] = out["Price@Entry"] - out["Price@Exit"]

    return out


def build_entry_diagnostics(trades_table: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    if trades_table is None or trades_table.empty:
        return pd.DataFrame()

    time_col = "EntryTimeUTC" if "EntryTimeUTC" in trades_table.columns else (
        "EntryTime" if "EntryTime" in trades_table.columns else None
    )
    bar_col  = "EntryBar" if "EntryBar" in trades_table.columns else None
    if bar_col is None:
        return pd.DataFrame()

    has_exit_reason = "ExitReason" in trades_table.columns
    margin_mode = (trade_mode == "Margin / Futures")

    rows = []
    for _, r in trades_table.iterrows():
        # basic index resolution
        ei = r.get(bar_col)
        try:
            ei = int(ei) if pd.notna(ei) else None
        except Exception:
            ei = None

        # safe pulls from ds for entry bar
        def _safe(col, idx):
            try:
                return float(ds[col].iloc[idx])
            except Exception:
                return float("nan")

        # account-level fields from trades table
        equity_before = float(r.get("EquityBefore")) if "EquityBefore" in trades_table.columns and pd.notna(r.get("EquityBefore")) else float("nan")
        equity_after = float(r.get("EquityAfter")) if "EquityAfter" in trades_table.columns and pd.notna(r.get("EquityAfter")) else float("nan")
        size = float(r.get("Size")) if "Size" in trades_table.columns and pd.notna(r.get("Size")) else float("nan")
        notional = float(r.get("NotionalEntry")) if "NotionalEntry" in trades_table.columns and pd.notna(r.get("NotionalEntry")) else float("nan")

        # Effective leverage = notional / equity_before
        if equity_before and not math.isnan(equity_before) and equity_before > 0:
            effective_lev = notional / equity_before
        else:
            effective_lev = float("nan")

        # Margin util at entry (% of equity tied as margin)
        if (
            trade_mode == "Margin / Futures"
            and equity_before
            and not math.isnan(equity_before)
            and equity_before > 0
        ):
            if max_leverage is not None and max_leverage > 0:
                required_margin = notional / max_leverage
            else:
                required_margin = notional
            margin_util = (required_margin / equity_before) * 100.0
        else:
            margin_util = float("nan")

        # "no index" path: fill with NaNs but still include account fields
        if ei is None or ei < 0 or ei >= len(ds):
            row = {
                "EntryTime": r.get(time_col),
                "EntryBar": r.get(bar_col),
                "Price@Entry (ds)": float("nan"),
                "RSI@Entry": float("nan"),
                "RSI_MA@Entry": float("nan"),
                "TouchKC?": None, "TouchBB?": None, "BandOK?": None,
                f"RSI≥{rsi_min}?": None,
                f"RSI_MA≥{rsi_ma_min}?": None,
                (f"RSI{rsi_relation}RSI_MA?") if use_rsi_relation else "RSI relation (disabled)": None,
                "Size": size,
                "NotionalEntry": notional,
                "EquityBefore": equity_before,
                "EquityAfter": equity_after,
                "RealizedPnL": r.get("RealizedPnL"),
                "EffectiveLeverage": effective_lev,
            }

            # margin-only extras
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

        px   = _safe("Close", ei)
        rsiV = _safe("rsi30", ei)      if "rsi30" in ds.columns else float("nan")
        rmaV = _safe("rsi30_ma", ei)   if "rsi30_ma" in ds.columns else float("nan")
        bbU  = _safe("bb_up", ei)      if "bb_up" in ds.columns else float("nan")
        kcU  = _safe("kc_up", ei)      if "kc_up" in ds.columns else float("nan")

        touch_kc = (px >= kcU) if pd.notna(kcU) else False
        touch_bb = (px >= bbU) if pd.notna(bbU) else False

        if entry_band_mode == "KC":
            band_ok = touch_kc
        elif entry_band_mode == "BB":
            band_ok = touch_bb
        elif entry_band_mode == "Both":
            band_ok = touch_kc and touch_bb
        else:
            band_ok = touch_kc or touch_bb

        rsi_ok    = (rsiV >= rsi_min)    if pd.notna(rsiV) else False
        rsi_ma_ok = (rmaV >= rsi_ma_min) if pd.notna(rmaV) else False
        rel_ok    = _cmp(rsiV, rmaV, rsi_relation) if use_rsi_relation and pd.notna(rsiV) and pd.notna(rmaV) else True

        row = {
            "EntryTime": r.get(time_col),
            "EntryBar": ei,
            "Price@Entry (ds)": px,
            "RSI@Entry": rsiV,
            "RSI_MA@Entry": rmaV,
            "TouchKC?": bool(touch_kc),
            "TouchBB?": bool(touch_bb),
            "BandOK?": bool(band_ok),
            f"RSI≥{rsi_min}?": bool(rsi_ok),
            f"RSI_MA≥{rsi_ma_min}?": bool(rsi_ma_ok),
            (f"RSI{rsi_relation}RSI_MA?") if use_rsi_relation else "RSI relation (disabled)": bool(rel_ok),
            "Size": size,
            "NotionalEntry": notional,
            "EquityBefore": equity_before,
            "EquityAfter": equity_after,
            "RealizedPnL": r.get("RealizedPnL"),
            "EffectiveLeverage": effective_lev,
        }

        # margin-only extras
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
    



st.set_page_config(page_title="BB+KC+RSI Backtester", layout="wide")
st.title("BB + KC + RSI Short Strategy — Backtesting UI")


if "selected_trade" not in st.session_state:
    st.session_state.selected_trade = None
if "run_ready" not in st.session_state:
    st.session_state.run_ready = False
if "last_params" not in st.session_state:
    st.session_state.last_params = None
if "results" not in st.session_state:
    st.session_state.results = None
if "dirty_params" not in st.session_state:
    st.session_state.dirty_params = False



@st.cache_data(show_spinner=False)
def _cached_fetch(exchange, symbol, timeframe, start_ts, end_ts):
    df = fetch_ohlcv_range(exchange, symbol, timeframe, start_ts, end_ts)
    return df

@st.cache_data(show_spinner=False)
def _cached_backtest(df, params: dict):
    stats, ds, trades = run_backtest(
        df,
        timeframe=params["timeframe"],

        # indicators
        bb_len=params["bb_len"],
        bb_std=params["bb_std"],
        bb_basis_type=params["bb_basis_type"],
        kc_ema_len=params["kc_ema_len"],
        kc_atr_len=params["kc_atr_len"],
        kc_mult=params["kc_mult"],
        kc_mid_type=params["kc_mid_type"],
        rsi_len_30m=params["rsi_len_30m"],
        rsi_ma_len=params["rsi_ma_len"],
        rsi_smoothing_type=params["rsi_smoothing_type"],
        rsi_ma_type=params["rsi_ma_type"],

        # entry/exit
        rsi_min=params["rsi_min"],
        rsi_ma_min=params["rsi_ma_min"],
        use_rsi_relation=params["use_rsi_relation"],
        rsi_relation=params["rsi_relation"],
        entry_band_mode=params["entry_band_mode"],
        exit_channel=params["exit_channel"],
        exit_level=params["exit_level"],

        # capital & fees
        cash=params["cash"],
        commission=params["commission"],

        # trade mode & risk
        trade_mode=params["trade_mode"],
        use_stop=params["use_stop"],
        stop_mode=params["stop_mode"],
        stop_pct=params["stop_pct"],
        stop_atr_mult=params["stop_atr_mult"],
        use_trailing=params["use_trailing"],
        trail_pct=params["trail_pct"],
        max_bars_in_trade=params["max_bars_in_trade"],
        daily_loss_limit=params["daily_loss_limit"],
        risk_per_trade_pct=params["risk_per_trade_pct"],

        # margin/futures
        max_leverage=params["max_leverage"],
        maintenance_margin_pct=params["maintenance_margin_pct"],
        max_margin_utilization=params["max_margin_utilization"],
    )
    return stats, ds, trades



with st.sidebar:
    st.header("Data")
    exchange = st.selectbox("Exchange", ["coinbase", "kraken", "gemini", "bitstamp", "binanceus"], index=3)
    symbol = st.text_input("Symbol", "BTC/USD")
    timeframe = st.selectbox("Timeframe", ["30m", "1h", "4h", "1d"], index=0)

    today = dt.datetime.utcnow().date()
    default_start = dt.date(2023, 10, 1)
    date_range = st.date_input(
        "Date range (UTC)",
        value=(default_start, today),
        help="Backtest window. Data is sliced to this range after fetch."
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = default_start
        end_date = today


    st.header("Bollinger Bands")
    bb_len = st.number_input("BB length", 5, 200, 20, 1)
    bb_std = st.number_input("BB std dev", 1.0, 4.0, 2.0, 0.1)
    bb_basis_type = st.selectbox("BB basis type", ["sma", "ema"], index=0)

    st.header("Keltner Channel")
    kc_ema_len = st.number_input("KC EMA/SMA length (mid)", 5, 200, 20, 1)
    kc_atr_len = st.number_input("KC ATR length", 5, 200, 14, 1)
    kc_mult = st.number_input("KC ATR multiplier", 0.5, 5.0, 2.0, 0.1)
    kc_mid_is_ema = st.checkbox("KC mid uses EMA (uncheck = SMA)", value=True)
    kc_mid_type = "ema" if kc_mid_is_ema else "sma"

    st.header("RSI (30m resample)")
    rsi_len_30m = st.number_input("RSI length", 5, 100, 14, 1)
    rsi_smoothing_type = st.selectbox("RSI smoothing type", ["ema", "sma", "rma"], index=0)
    rsi_ma_len = st.number_input("RSI MA length", 2, 100, 10, 1)
    rsi_ma_type = st.selectbox("RSI MA smoothing type", ["sma", "ema"], index=0)

    st.header("Entry Conditions")
    rsi_min = st.number_input("RSI minimum (entry)", 0, 100, 70, 1)
    rsi_ma_min = st.number_input("RSI MA minimum (entry)", 0, 100, 70, 1)
    use_rsi_relation = st.checkbox(
        "Use RSI ↔ RSI MA relation?",
        value=True,
        help="If unchecked, the RSI vs RSI MA comparison is ignored."
    )
    rsi_relation = st.selectbox("RSI vs RSI MA relation", ["<", "<=", ">", ">="], index=3)

    entry_band_mode = st.selectbox(
        "Price must touch which top band?",
        ["Either", "KC", "BB", "Both"],
        index=0,
        help="Either = KC or BB; Both = touch both uppers at once."
    )

    st.header("Exit Conditions")
    exit_channel = st.selectbox("Exit channel", ["BB", "KC"], index=0)
    exit_level = st.selectbox("Exit level", ["mid", "lower"], index=0,
                              help="Exit when price ≤ selected level on chosen channel.")

    st.header("Chart Display")
    show_candles = st.checkbox("Candlesticks (unchecked = close line)", value=True)

    st.subheader("Y-axis controls")
    #enable_price_y_zoom = st.checkbox("Enable Price Y-axis zoom", value=False)
    #price_y_mode = st.radio(
    #    "Price Y mode",
    #    ["Auto", "Manual"],
    #    index=0,
    #    disabled=not enable_price_y_zoom,
    #    help="Auto = autorange; Manual = specify min/max. Both allow vertical zoom when enabled."
    #)

    #price_y_min = st.number_input("Price Y min", value=0.0, disabled=(not enable_price_y_zoom or price_y_mode != "Manual"))
    #price_y_max = st.number_input("Price Y max", value=0.0, disabled=(not enable_price_y_zoom or price_y_mode != "Manual"))

    lock_rsi_y = st.checkbox("Lock RSI to 0–100", value=True)

    st.header("Backtest")
    st.subheader("Capital & Fees")
    cash = st.number_input("Starting cash", 100, 1_000_000_000, 10_000, 100)
    commission = st.number_input("Commission (fraction)", 0.0, 0.01, 0.001, 0.0001)

    st.subheader("Trade Mode")
    trade_mode = st.selectbox(
        "Trade mode",
        ["Simple (1x spot-style)", "Margin / Futures"],
        index=0,
        help="Simple = current behavior with no leverage; Margin/Futures = leverage + margin & liquidation."
    )

    st.header("Risk Management")

    use_stop = st.checkbox("Enable stop loss", value=False)

    stop_mode = st.selectbox(
        "Stop type",
        ["Fixed %", "ATR"],
        index=0,
        help="Fixed % = stop offset from entry; ATR = stop based on volatility."
    )

    if stop_mode == "Fixed %":
        stop_pct = st.number_input(
            "Fixed stop % (per trade)",
            min_value=0.1,
            max_value=50.0,
            value=2.0,
            step=0.1
        )
        stop_atr_mult = None
    else:
        stop_atr_mult = st.number_input(
            "ATR stop multiplier",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Stop loss = EntryPrice + ATR * multiplier (for short trades)"
        )
        stop_pct = None

    use_trailing = st.checkbox("Enable trailing stop", value=False)

    trail_pct = st.number_input(
        "Trailing stop %",
        min_value=0.1,
        max_value=50.0,
        value=1.0,
        step=0.1,
        help="Distance from recent extreme (in %) to trail the stop."
    )

    max_bars_in_trade = st.number_input(
        "Max bars in trade (time stop)",
        min_value=1,
        max_value=500,
        value=100,
        step=1,
        help="Exit if trade is still open after this many bars."
    )

    daily_loss_limit = st.number_input(
        "Daily loss limit %",
        min_value=0.0,
        max_value=50.0,
        value=3.0,
        step=0.5,
        help="If equity drops this % from day's start, stop opening new trades for that day."
    )

    risk_per_trade_pct = st.number_input(
        "Risk per trade % of equity",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.5,
        help="Used to size positions so each stop-out risks only this % of equity."
    )

    # Margin / Futures–only settings
    if trade_mode == "Margin / Futures":
        st.header("Margin / Futures Settings")
        max_leverage = st.number_input(
            "Max leverage",
            min_value=1.0,
            max_value=125.0,
            value=5.0,
            step=0.5,
            help="Cap notional = equity * max_leverage."
        )

        maintenance_margin_pct = st.number_input(
            "Maintenance margin %",
            min_value=0.1,
            max_value=50.0,
            value=0.5,
            step=0.1,
            help="Approx. equity threshold below which positions are liquidated."
        )

        if use_stop:
            enable_max_margin_util = st.checkbox(
            "Limit margin utilization?",
            value=False,
            help="If enabled, skip new trades that would exceed this % of equity used as margin."
        )
            max_margin_utilization = st.number_input(
                "Max margin utilization %",
                min_value=10.0,
                max_value=100.0,
                value=70.0,
                step=5.0,
                help="Don't let required margin exceed this % of equity."
            )
        else:
            max_margin_utilization = None
    else:
        max_leverage = None
        maintenance_margin_pct = None
        max_margin_utilization = None
    
    run = st.button("Run Backtest")


def _snapshot_params():
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_ts": pd.Timestamp(start_date),
        "end_ts": pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),

        # indicators
        "bb_len": bb_len,
        "bb_std": bb_std,
        "bb_basis_type": bb_basis_type,
        "kc_ema_len": kc_ema_len,
        "kc_atr_len": kc_atr_len,
        "kc_mult": kc_mult,
        "kc_mid_type": kc_mid_type,
        "rsi_len_30m": rsi_len_30m,
        "rsi_ma_len": rsi_ma_len,
        "rsi_smoothing_type": rsi_smoothing_type,
        "rsi_ma_type": rsi_ma_type,

        # entry/exit
        "rsi_min": rsi_min,
        "rsi_ma_min": rsi_ma_min,
        "use_rsi_relation": use_rsi_relation,
        "rsi_relation": rsi_relation,
        "entry_band_mode": entry_band_mode,
        "exit_channel": exit_channel,
        "exit_level": exit_level,

        # fees / starting capital
        "cash": cash,
        "commission": commission,

        # trade mode
        "trade_mode": trade_mode,

        # risk management
        "use_stop": use_stop,
        "stop_mode": stop_mode,
        "stop_pct": stop_pct,
        "stop_atr_mult": stop_atr_mult,
        "use_trailing": use_trailing,
        "trail_pct": trail_pct,
        "max_bars_in_trade": max_bars_in_trade,
        "daily_loss_limit": daily_loss_limit,
        "risk_per_trade_pct": risk_per_trade_pct,

        # margin/futures (may be None in simple mode)
        "max_leverage": max_leverage,
        "maintenance_margin_pct": maintenance_margin_pct,
        "max_margin_utilization": max_margin_utilization,
    }

params_now = _snapshot_params()

if st.session_state.last_params is not None and st.session_state.last_params != params_now:
    st.session_state.dirty_params = True


# On click: compute, then flip run_ready
if run:
    with st.spinner("Fetching data and running backtest..."):
        df = _cached_fetch(
            params_now["exchange"], params_now["symbol"], params_now["timeframe"],
            params_now["start_ts"], params_now["end_ts"]
        )
        if df.empty:
            st.error("No data in the selected date range. Try another range/timeframe, or a different exchange.")
            st.stop()
        stats, ds, trades = _cached_backtest(df, params_now)
        st.session_state.results = {"df": df, "stats": stats, "ds": ds, "trades": trades}
        st.session_state.last_params = params_now
        st.session_state.dirty_params = False
        st.session_state.selected_trade = None
        st.session_state.run_ready = True



# main render path
if st.session_state.get("run_ready", False) and st.session_state.results is not None and not st.session_state.dirty_params:
    results = st.session_state.results
    df = results["df"]
    stats = results["stats"]
    ds = results["ds"]
    trades = results["trades"]

    st.subheader("Performance")
    st.dataframe(stats.to_frame())

    ds_full = ds

    trades_table = build_trades_table(trades)

    tz = "America/Los_Angeles"
    trades_view = trades_table.copy()
    if not trades_view.empty:
        trades_view["Entry (Local)"] = pd.to_datetime(trades_view["EntryTime"], utc=True)\
            .dt.tz_convert(tz).dt.strftime("%m/%d/%Y %H:%M")
        trades_view["Exit (Local)"] = pd.to_datetime(trades_view["ExitTime"],  utc=True)\
            .dt.tz_convert(tz).dt.strftime("%m/%d/%Y %H:%M")

    # Placehoilder for chart to keep it above TABLE
    chart_slot = st.container()

    # Trades
    st.subheader("Trades")
    if trades_table.empty:
        st.info("No trades for the selected settings / window.")
        selected_trade_now = None
    else:
        preferred_cols = [
            "Entry (Local)","Exit (Local)","EntryBar","ExitBar",
            "Side","Price@Entry","Price@Exit","Price Move %","Duration","PnL (per unit)"
        ]

        cols_present = [c for c in preferred_cols if c in trades_view.columns]
        df_grid = trades_view[cols_present].copy()

        gb = GridOptionsBuilder.from_dataframe(df_grid)

        # make sure the checkbox is on a visible column
        first_vis = cols_present[0] if cols_present else df_grid.columns[0]
        gb.configure_column(first_vis, checkboxSelection=True, header_name=first_vis)

        gb.configure_selection(selection_mode="single", use_checkbox=True)
        gb.configure_default_column(filter=True, sortable=True, resizable=True)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=True)

        gb.configure_grid_options(suppressRowClickSelection=True) 

        grid_options = gb.build()

        grid_resp = AgGrid(
            df_grid,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=False,
            height=300,
            theme="balham",
            fit_columns_on_grid_load=True
        )

        sel = grid_resp.get("selected_rows", [])
        selected_trade_now = sel[0] if sel else None
        if selected_trade_now is not None:
            st.session_state.selected_trade = selected_trade_now



    # Chart
    with chart_slot:
        st.subheader("Chart")
        ds_plot = ds_full.copy()
        MAX_POINTS = 5000
        if len(ds_plot) > MAX_POINTS:
            step = len(ds_plot) // MAX_POINTS
            ds_plot = ds_plot.iloc[::step]

        selected_trade = selected_trade_now or st.session_state.get("selected_trade")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.65, 0.35],
            subplot_titles=("Price + BB + KC", "RSI (30m) + MA")
        )

        # price traces
        if show_candles:
            fig.add_trace(go.Candlestick(
                x=ds_plot.index, open=ds_plot["Open"], high=ds_plot["High"],
                low=ds_plot["Low"], close=ds_plot["Close"],
                name="Price", showlegend=False
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["Close"], mode="lines", name="Close"), row=1, col=1)

        # bands
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["bb_mid"], mode="lines", name=f"BB mid ({bb_basis_type.upper()})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["bb_up"],  mode="lines", name="BB upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["bb_low"], mode="lines", name="BB lower"), row=1, col=1)

        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["kc_mid"], mode="lines", name=f"KC mid ({kc_mid_type.upper()})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["kc_up"],  mode="lines", name="KC upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["kc_low"], mode="lines", name="KC lower"), row=1, col=1)

        # rsi
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["rsi30"],    mode="lines", name="RSI(30m)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["rsi30_ma"], mode="lines", name=f"RSI MA ({rsi_ma_type.upper()})"), row=2, col=1)

        if selected_trade is None:
            # trade markers from the table
            if not trades_table.empty:
                fig.add_trace(go.Scattergl(
                    x=trades_table["EntryTime"], y=trades_table["Price@Entry"],
                    mode="markers", marker=dict(size=6, symbol="triangle-down"),
                    name="Short Entry", showlegend=True
                ), row=1, col=1)
                fig.add_trace(go.Scattergl(
                    x=trades_table["ExitTime"], y=trades_table["Price@Exit"],
                    mode="markers", marker=dict(size=6, symbol="x"),
                    name="Exit", showlegend=True
                ), row=1, col=1)
        # auto-zoom to current selection
        else:
            try:
                # figure out bars / times for x-zoom
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

                # horizontal zoom (both panels)
                fig.update_xaxes(range=[x0, x1], row=1, col=1)
                fig.update_xaxes(range=[x0, x1], row=2, col=1)

                # vertical zoom (price panel): use window highs/lows with padding
                window_high = float(ds_full["High"].iloc[i0:i1+1].max())
                window_low  = float(ds_full["Low"].iloc[i0:i1+1].min())
                span = max(1e-9, window_high - window_low)
                pad_y = 0.07 * span
                fig.update_yaxes(range=[window_low - pad_y, window_high + pad_y], row=1, col=1)

                e_ts = pd.to_datetime(selected_trade.get("EntryTime")) if "EntryTime" in selected_trade else ds_full.index[ei]
                x_ts = pd.to_datetime(selected_trade.get("ExitTime"))  if "ExitTime"  in selected_trade else (ds_full.index[xi] if xi is not None else None)
                e_px = selected_trade.get("Price@Entry")
                x_px = selected_trade.get("Price@Exit")

                # make entries more obvious
                if pd.notna(e_ts) and pd.notna(e_px):
                    fig.add_trace(
                        go.Scattergl(
                            x=[e_ts], y=[e_px], mode="markers",
                            marker=dict(size=16, symbol="triangle-down", color="crimson", line=dict(width=1)),
                            name="Selected Entry", showlegend=True
                        ),
                        row=1, col=1
                    )
                    fig.add_vline(x=e_ts, line_width=1.5, line_dash="dot", line_color="crimson", row=1, col=1)

                # make exits more obvious
                if pd.notna(x_ts) and pd.notna(x_px):
                    fig.add_trace(
                        go.Scattergl(
                            x=[x_ts], y=[x_px], mode="markers",
                            marker=dict(size=16, symbol="x", color="limegreen", line=dict(width=2)),
                            name="Selected Exit", showlegend=True
                        ),
                        row=1, col=1
                    )
                    fig.add_vline(x=x_ts, line_width=1.5, line_dash="dot", line_color="limegreen", row=1, col=1)

            except Exception as e:
                st.warning(f"Highlight/zoom failed: {e}")

        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified"
        )
        if lock_rsi_y:
            fig.update_yaxes(range=[0, 100], fixedrange=True, row=2, col=1)
        fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
        fig.update_xaxes(rangeslider=dict(visible=False), row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)


    # Summary metrics
    if not trades_table.empty:
        returns = pd.to_numeric(trades_table["Price Move %"], errors="coerce")
        wins = returns > 0
        win_rate = 100.0 * wins.mean() if len(returns) else 0.0
        avg_ret = returns.mean() if len(returns) else float("nan")
        med_ret = returns.median() if len(returns) else float("nan")
        best = returns.max() if len(returns) else float("nan")
        worst = returns.min() if len(returns) else float("nan")
        

        pos_sum = returns[returns > 0].sum()
        neg_sum = returns[returns < 0].sum()
        price_profit_factor = float("inf") if neg_sum == 0 else (pos_sum / abs(neg_sum))

        avg_dur = trades_table["Duration"].mean()

        # Total equity return % (from backend stats)
        total_equity_ret = stats.get("total_equity_return_pct", 0.0)

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        c1.metric("Trades", f"{len(trades_table)}")
        c2.metric("Win Rate", f"{win_rate:.1f}%")
        c3.metric("Avg Return", f"{avg_ret:.2f}%")
        c4.metric("Median", f"{med_ret:.2f}%")
        c5.metric("Best", f"{best:.2f}%")
        c6.metric("Worst", f"{worst:.2f}%")
        c7.metric("Price Profit Factor", "∞" if price_profit_factor == float("inf") else f"{price_profit_factor:.2f}")
        c8.metric("Total Equity Return", f"{total_equity_ret:.2f}%")

        st.caption(f"Average duration: {avg_dur}")

        st.download_button(
            "Download trades CSV",
            data=trades_table.to_csv(index=False).encode("utf-8"),
            file_name="backtest_trades.csv",
            mime="text/csv"
        )


    try:
        diag = build_entry_diagnostics(trades, ds)
        with st.expander("🔍 Entry condition diagnostics"):
            if diag.empty:
                st.info("No diagnostics available.")
            else:
                col_rsi   = [c for c in diag.columns if c.startswith("RSI≥")]
                col_rsima = [c for c in diag.columns if c.startswith("RSI_MA≥")]
                col_rel   = [c for c in diag.columns if "RSI" in c and "RSI_MA" in c and c.endswith("?")]
                rsi_ok    = diag[col_rsi[0]].astype(bool)   if col_rsi   else True
                rsima_ok  = diag[col_rsima[0]].astype(bool) if col_rsima else True
                rel_ok    = diag[col_rel[0]].astype(bool)   if col_rel   else True
                band_ok   = diag["BandOK?"].astype(bool)    if "BandOK?" in diag.columns else True

                suspect_mask = ~(rsi_ok & rsima_ok & rel_ok & band_ok)
                suspects = diag[suspect_mask].copy()

                st.dataframe(diag, use_container_width=True)
                csv_diag = diag.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download diagnostics CSV",
                    data=csv_diag,
                    file_name="entry_diagnostics.csv",
                    mime="text/csv"
                )

                if not suspects.empty:
                    st.warning(f"{len(suspects)} trade(s) appear to violate configured entry rules.")
                    st.dataframe(suspects, use_container_width=True)
                    csv_bad = suspects.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download suspect trades CSV",
                        data=csv_bad,
                        file_name="suspect_trades.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("All trades match the configured entry conditions.")
    except Exception as e:
        st.error(f"Diagnostics render failed: {e}")

else:
    if st.session_state.get("dirty_params", False):
        st.info("Parameters changed. Click **Run Backtest** to recompute.")
    else:
        st.info("Adjust parameters in the sidebar and click **Run Backtest**.")
