import streamlit as st
from core.data import fetch_ohlcv, fetch_ohlcv_range, fetch_ohlcv_range_cached
from core.utils import cmp, calculate_drawdown
from core.presets import STRATEGY_PRESETS, get_preset_names, DEFAULT_PRESET
from optimization.grid_search import (
    run_grid_search, create_custom_grid, analyze_results,
    DEFAULT_PARAM_GRID, QUICK_PARAM_GRID,
    ResultConstraints, WalkForwardConfig, run_walk_forward_grid_search
)
from backtest.engine import run_backtest
from discovery.database import DiscoveryDatabase, WinCriteria, BacktestRun
from discovery.engine import (
    run_discovery, run_discovery_parallel, DiscoveryConfig, 
    QUICK_DISCOVERY_GRID, FOCUSED_DISCOVERY_GRID, MARGIN_PARAM_GRID,
    create_discovery_grid, create_margin_discovery_grid,
    count_combinations, estimate_discovery_time, get_cpu_count,
    estimate_filtered_combinations
)
from discovery.rules import find_winning_patterns, get_rule_summary
from discovery.leaderboard import Leaderboard, WinningStrategy
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import datetime as dt
import math
import numpy as np
import pandas as pd
import os


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
    # Determine margin mode by checking if MarginUtilAtEntry column exists AND has valid data
    margin_mode = False
    if "MarginUtilAtEntry" in trades_table.columns:
        # Check if the column contains any non-NaN values
        if trades_table["MarginUtilAtEntry"].notna().any():
            margin_mode = True

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
            margin_mode
            and equity_before
            and not math.isnan(equity_before)
            and equity_before > 0
        ):
             # We don't have max_leverage easily accessible here without passing params, 
             # but we can try to infer or just use the pre-calculated one if available
             margin_util = r.get("MarginUtilAtEntry", float("nan"))
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
        
        # We need entry_band_mode here, but it's a global/param. 
        # For diagnostics, we'll just report the raw flags.
        
        row = {
            "EntryTime": r.get(time_col),
            "EntryBar": ei,
            "Price@Entry (ds)": px,
            "RSI@Entry": rsiV,
            "RSI_MA@Entry": rmaV,
            "TouchKC?": bool(touch_kc),
            "TouchBB?": bool(touch_bb),
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
st.title("BB + KC + RSI Short Strategy")


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
if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Custom"
if "preset_params" not in st.session_state:
    st.session_state.preset_params = DEFAULT_PRESET.copy()
if "discovery_running" not in st.session_state:
    st.session_state.discovery_running = False
if "discovery_results" not in st.session_state:
    st.session_state.discovery_results = None
if "selected_winning_strategy" not in st.session_state:
    st.session_state.selected_winning_strategy = None

# Widget key defaults - initialize from DEFAULT_PRESET
# These are used to properly sync widget state with session state
WIDGET_DEFAULTS = {
    # Data & Timeframe
    "w_exchange": "bitstamp",
    "w_symbol": "BTC/USD",
    "w_timeframe": "30m",
    "w_start_date": dt.date(2022, 1, 1),
    "w_end_date": dt.datetime.utcnow().date(),
    # Capital
    "w_cash": 10_000,
    "w_commission": 0.001,
    # Indicators - BB
    "w_bb_len": DEFAULT_PRESET.get("bb_len", 20),
    "w_bb_std": DEFAULT_PRESET.get("bb_std", 2.0),
    "w_bb_basis_type": DEFAULT_PRESET.get("bb_basis_type", "sma"),
    # Indicators - KC
    "w_kc_ema_len": DEFAULT_PRESET.get("kc_ema_len", 20),
    "w_kc_atr_len": DEFAULT_PRESET.get("kc_atr_len", 14),
    "w_kc_mult": DEFAULT_PRESET.get("kc_mult", 2.0),
    "w_kc_mid_is_ema": DEFAULT_PRESET.get("kc_mid_type", "ema") == "ema",
    # Indicators - RSI
    "w_rsi_len_30m": DEFAULT_PRESET.get("rsi_len_30m", 14),
    "w_rsi_smoothing_type": DEFAULT_PRESET.get("rsi_smoothing_type", "ema"),
    "w_rsi_ma_len": DEFAULT_PRESET.get("rsi_ma_len", 10),
    "w_rsi_ma_type": DEFAULT_PRESET.get("rsi_ma_type", "sma"),
    # Entry/Exit
    "w_rsi_min": DEFAULT_PRESET.get("rsi_min", 70),
    "w_rsi_ma_min": DEFAULT_PRESET.get("rsi_ma_min", 70),
    "w_use_rsi_relation": DEFAULT_PRESET.get("use_rsi_relation", True),
    "w_rsi_relation": DEFAULT_PRESET.get("rsi_relation", ">="),
    "w_entry_band_mode": DEFAULT_PRESET.get("entry_band_mode", "Either"),
    "w_exit_channel": DEFAULT_PRESET.get("exit_channel", "BB"),
    "w_exit_level": DEFAULT_PRESET.get("exit_level", "mid"),
    # Risk Management
    "w_trade_mode": DEFAULT_PRESET.get("trade_mode", "Simple (1x spot-style)"),
    "w_use_stop": DEFAULT_PRESET.get("use_stop", False),
    "w_stop_mode": DEFAULT_PRESET.get("stop_mode", "Fixed %"),
    "w_stop_pct": float(DEFAULT_PRESET.get("stop_pct", 2.0) or 2.0),
    "w_stop_atr_mult": float(DEFAULT_PRESET.get("stop_atr_mult", 2.0) or 2.0),
    "w_use_trailing": DEFAULT_PRESET.get("use_trailing", False),
    "w_trail_pct": float(DEFAULT_PRESET.get("trail_pct", 1.0)),
    "w_max_bars_in_trade": int(DEFAULT_PRESET.get("max_bars_in_trade", 100)),
    "w_daily_loss_limit": float(DEFAULT_PRESET.get("daily_loss_limit", 3.0)),
    "w_risk_per_trade_pct": float(DEFAULT_PRESET.get("risk_per_trade_pct", 1.0)),
    # Margin
    "w_max_leverage": float(DEFAULT_PRESET.get("max_leverage", 5.0) or 5.0),
    "w_maintenance_margin_pct": float(DEFAULT_PRESET.get("maintenance_margin_pct", 0.5) or 0.5),
    "w_max_margin_utilization": float(DEFAULT_PRESET.get("max_margin_utilization", 70.0) or 70.0),
}

# Initialize widget keys in session state if not present
for key, default_val in WIDGET_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


def sync_widgets_to_preset(preset_params: dict):
    """Sync widget session state keys to match preset values."""
    key_mapping = {
        "bb_len": "w_bb_len",
        "bb_std": "w_bb_std",
        "bb_basis_type": "w_bb_basis_type",
        "kc_ema_len": "w_kc_ema_len",
        "kc_atr_len": "w_kc_atr_len",
        "kc_mult": "w_kc_mult",
        "kc_mid_type": ("w_kc_mid_is_ema", lambda v: v == "ema"),
        "rsi_len_30m": "w_rsi_len_30m",
        "rsi_smoothing_type": "w_rsi_smoothing_type",
        "rsi_ma_len": "w_rsi_ma_len",
        "rsi_ma_type": "w_rsi_ma_type",
        "rsi_min": "w_rsi_min",
        "rsi_ma_min": "w_rsi_ma_min",
        "use_rsi_relation": "w_use_rsi_relation",
        "rsi_relation": "w_rsi_relation",
        "entry_band_mode": "w_entry_band_mode",
        "exit_channel": "w_exit_channel",
        "exit_level": "w_exit_level",
        "trade_mode": "w_trade_mode",
        "use_stop": "w_use_stop",
        "stop_mode": "w_stop_mode",
        "stop_pct": "w_stop_pct",
        "stop_atr_mult": "w_stop_atr_mult",
        "use_trailing": "w_use_trailing",
        "trail_pct": "w_trail_pct",
        "max_bars_in_trade": "w_max_bars_in_trade",
        "daily_loss_limit": "w_daily_loss_limit",
        "risk_per_trade_pct": "w_risk_per_trade_pct",
        "max_leverage": "w_max_leverage",
        "maintenance_margin_pct": "w_maintenance_margin_pct",
        "max_margin_utilization": "w_max_margin_utilization",
    }
    
    for param_key, widget_key in key_mapping.items():
        if param_key in preset_params:
            value = preset_params[param_key]
            if isinstance(widget_key, tuple):
                # Special transform (e.g., kc_mid_type -> boolean)
                actual_key, transform = widget_key
                st.session_state[actual_key] = transform(value)
            else:
                # Handle None values with defaults
                if value is not None:
                    st.session_state[widget_key] = value

# Initialize discovery database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "discovery.db")
discovery_db = DiscoveryDatabase(DB_PATH)
discovery_db.initialize()


def get_param_value(key: str, default):
    """Get parameter value from preset or use default."""
    if st.session_state.selected_preset != "Custom":
        preset = STRATEGY_PRESETS.get(st.session_state.selected_preset, {})
        return preset.get(key, default)
    return st.session_state.preset_params.get(key, default)



@st.cache_data(show_spinner=False)
def _cached_fetch(exchange, symbol, timeframe, start_ts, end_ts):
    # Disk-backed cache dramatically speeds up subsequent cold starts by
    # incrementally filling missing history instead of re-downloading it all.
    df = fetch_ohlcv_range_cached(
        exchange, symbol, timeframe,
        start_ts=start_ts, end_ts=end_ts,
        repair_gaps=False
    )
    return df

@st.cache_data(show_spinner=False)
def _cached_backtest(df, params: dict):
    stats, ds, trades, equity_curve = run_backtest(
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
    return stats, ds, trades, equity_curve



with st.sidebar:
    st.header("Strategy Settings")
    
    # Top run button
    run_top = st.button("Run Backtest", type="primary", use_container_width=True, key="run_top")
    
    st.divider()

    # Preset Selector
    preset_options = ["Custom"] + list(STRATEGY_PRESETS.keys())
    preset_display_names = ["Custom (Manual Configuration)"] + [
        f"{STRATEGY_PRESETS[k]['name']}" for k in STRATEGY_PRESETS.keys()
    ]
    
    selected_preset_idx = st.selectbox(
        "Load Preset",
        range(len(preset_options)),
        format_func=lambda i: preset_display_names[i],
        key="preset_selector"
    )
    selected_preset = preset_options[selected_preset_idx]
    
    if selected_preset != st.session_state.selected_preset:
        st.session_state.selected_preset = selected_preset
        if selected_preset != "Custom":
            st.session_state.preset_params = STRATEGY_PRESETS[selected_preset].copy()
            # Sync widget session state keys to match preset values
            sync_widgets_to_preset(st.session_state.preset_params)
        st.session_state.dirty_params = True
    
    if selected_preset != "Custom":
        preset_info = STRATEGY_PRESETS[selected_preset]
        st.info(f"**{preset_info['name']}**: {preset_info['description']}")
    
    st.divider()
    
    # Data Settings
    with st.expander("Data & Timeframe", expanded=True):
        exchange_options = ["coinbase", "kraken", "gemini", "bitstamp", "binanceus"]
        exchange = st.selectbox(
            "Exchange", 
            exchange_options, 
            index=exchange_options.index(st.session_state.w_exchange) if st.session_state.w_exchange in exchange_options else 3,
            key="w_exchange"
        )
        symbol = st.text_input("Symbol", key="w_symbol")
        timeframe_options = ["30m", "1h", "4h", "1d"]
        timeframe = st.selectbox(
            "Timeframe", 
            timeframe_options, 
            index=timeframe_options.index(st.session_state.w_timeframe) if st.session_state.w_timeframe in timeframe_options else 0,
            key="w_timeframe"
        )

        today = dt.datetime.utcnow().date()
        # Use session state values as defaults to persist across reruns
        date_range = st.date_input(
            "Date range (UTC)",
            value=(st.session_state.w_start_date, st.session_state.w_end_date),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            # Persist selected dates in session state
            st.session_state.w_start_date = start_date
            st.session_state.w_end_date = end_date
        else:
            start_date = st.session_state.w_start_date
            end_date = st.session_state.w_end_date

    # Indicator Settings
    with st.expander("Indicators (BB, KC, RSI)"):
        st.caption("Bollinger Bands")
        bb_len = st.number_input("BB length", 5, 200, key="w_bb_len")
        bb_std = st.number_input("BB std dev", 1.0, 4.0, key="w_bb_std", step=0.1)
        bb_basis_options = ["sma", "ema"]
        bb_basis_type = st.selectbox(
            "BB basis type", bb_basis_options, 
            index=bb_basis_options.index(st.session_state.w_bb_basis_type),
            key="w_bb_basis_type"
        )
        
        st.caption("Keltner Channel")
        kc_ema_len = st.number_input("KC EMA/SMA length (mid)", 5, 200, key="w_kc_ema_len")
        kc_atr_len = st.number_input("KC ATR length", 5, 200, key="w_kc_atr_len")
        kc_mult = st.number_input("KC ATR multiplier", 0.5, 5.0, key="w_kc_mult", step=0.1)
        kc_mid_is_ema = st.checkbox("KC mid uses EMA (uncheck = SMA)", key="w_kc_mid_is_ema")
        kc_mid_type = "ema" if kc_mid_is_ema else "sma"

        st.caption("RSI (30m resample)")
        rsi_len_30m = st.number_input("RSI length", 5, 100, key="w_rsi_len_30m")
        rsi_smoothing_options = ["ema", "sma", "rma"]
        rsi_smoothing_type = st.selectbox(
            "RSI smoothing type", rsi_smoothing_options,
            index=rsi_smoothing_options.index(st.session_state.w_rsi_smoothing_type),
            key="w_rsi_smoothing_type"
        )
        rsi_ma_len = st.number_input("RSI MA length", 2, 100, key="w_rsi_ma_len")
        rsi_ma_options = ["sma", "ema"]
        rsi_ma_type = st.selectbox(
            "RSI MA smoothing type", rsi_ma_options,
            index=rsi_ma_options.index(st.session_state.w_rsi_ma_type),
            key="w_rsi_ma_type"
        )

    # Strategy Logic
    with st.expander("Entry & Exit Logic"):
        st.subheader("Entry")
        rsi_min = st.number_input("RSI minimum (entry)", 0, 100, key="w_rsi_min")
        rsi_ma_min = st.number_input("RSI MA minimum (entry)", 0, 100, key="w_rsi_ma_min")
        use_rsi_relation = st.checkbox(
            "Use RSI ↔ RSI MA relation?",
            key="w_use_rsi_relation",
        )
        rsi_relation_options = ["<", "<=", ">", ">="]
        rsi_relation = st.selectbox(
            "RSI vs RSI MA relation", rsi_relation_options,
            index=rsi_relation_options.index(st.session_state.w_rsi_relation),
            key="w_rsi_relation"
        )

        entry_band_options = ["Either", "KC", "BB", "Both"]
        entry_band_mode = st.selectbox(
            "Price must touch which top band?",
            entry_band_options,
            index=entry_band_options.index(st.session_state.w_entry_band_mode),
            key="w_entry_band_mode",
        )

        st.subheader("Exit")
        exit_channel_options = ["BB", "KC"]
        exit_channel = st.selectbox(
            "Exit channel", exit_channel_options,
            index=exit_channel_options.index(st.session_state.w_exit_channel),
            key="w_exit_channel"
        )
        exit_level_options = ["mid", "lower"]
        exit_level = st.selectbox(
            "Exit level", exit_level_options,
            index=exit_level_options.index(st.session_state.w_exit_level),
            key="w_exit_level",
        )

    # Risk Management
    with st.expander("Risk Management & Capital"):
        cash = st.number_input("Starting cash", 100, 1_000_000_000, 10_000, 100, key="w_cash")
        commission = st.number_input("Commission (fraction)", 0.0, 0.01, 0.001, 0.0001, key="w_commission")

        trade_mode_options = ["Simple (1x spot-style)", "Margin / Futures"]
        trade_mode = st.selectbox(
            "Trade mode",
            trade_mode_options,
            index=trade_mode_options.index(st.session_state.w_trade_mode) if st.session_state.w_trade_mode in trade_mode_options else 0,
            key="w_trade_mode",
        )

        use_stop = st.checkbox("Enable stop loss", key="w_use_stop")
        stop_mode_options = ["Fixed %", "ATR"]
        stop_mode = st.selectbox(
            "Stop type",
            stop_mode_options,
            index=stop_mode_options.index(st.session_state.w_stop_mode) if st.session_state.w_stop_mode in stop_mode_options else 0,
            key="w_stop_mode",
        )

        if stop_mode == "Fixed %":
            stop_pct = st.number_input(
                "Fixed stop % (per trade)",
                min_value=0.1, max_value=50.0,
                step=0.1,
                key="w_stop_pct"
            )
            stop_atr_mult = None
        else:
            stop_atr_mult = st.number_input(
                "ATR stop multiplier",
                min_value=0.1, max_value=10.0,
                step=0.1,
                key="w_stop_atr_mult"
            )
            stop_pct = None

        use_trailing = st.checkbox("Enable trailing stop", key="w_use_trailing")
        trail_pct = st.number_input(
            "Trailing stop %",
            min_value=0.1, max_value=50.0,
            step=0.1,
            key="w_trail_pct"
        )

        max_bars_in_trade = st.number_input(
            "Max bars in trade",
            min_value=1, max_value=500,
            step=1,
            key="w_max_bars_in_trade"
        )

        daily_loss_limit = st.number_input(
            "Daily loss limit %",
            min_value=0.0, max_value=50.0,
            step=0.5,
            key="w_daily_loss_limit"
        )

        risk_per_trade_pct = st.number_input(
            "Risk per trade % of equity",
            min_value=0.1, max_value=100.0,
            step=0.5,
            key="w_risk_per_trade_pct"
        )

        # Margin / Futures–only settings
        if trade_mode == "Margin / Futures":
            st.caption("Margin Settings")
            max_leverage = st.number_input("Max leverage", 1.0, 125.0, 
                                          step=0.5,
                                          key="w_max_leverage")
            maintenance_margin_pct = st.number_input("Maintenance margin %", 0.1, 50.0, 
                                                    step=0.1,
                                                    key="w_maintenance_margin_pct")

            if use_stop:
                enable_max_margin_util = st.checkbox("Limit margin utilization?", value=False, key="w_enable_max_margin_util")
                max_margin_utilization = st.number_input("Max margin utilization %", 10.0, 100.0, 
                                                        step=5.0,
                                                        key="w_max_margin_utilization")
            else:
                max_margin_utilization = None
        else:
            max_leverage = None
            maintenance_margin_pct = None
            max_margin_utilization = None
    
    st.divider()
    run_bottom = st.button("Run Backtest", type="primary", use_container_width=True, key="run_bottom")


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
if run_top or run_bottom:
    with st.spinner("Fetching data and running backtest..."):
        df = _cached_fetch(
            params_now["exchange"], params_now["symbol"], params_now["timeframe"],
            params_now["start_ts"], params_now["end_ts"]
        )
        if df.empty:
            st.error("No data in the selected date range. Try another range/timeframe, or a different exchange.")
            st.stop()
        stats, ds, trades, equity_curve = _cached_backtest(df, params_now)
        st.session_state.results = {"df": df, "stats": stats, "ds": ds, "trades": trades, "equity_curve": equity_curve}
        st.session_state.last_params = params_now
        st.session_state.dirty_params = False
        st.session_state.selected_trade = None
        st.session_state.run_ready = True



# main render path - show results even when params are dirty (with warning)
if st.session_state.get("run_ready", False) and st.session_state.results is not None:
    # Show warning banner if params have changed since last run
    if st.session_state.dirty_params:
        st.warning("⚠️ **Parameters have changed.** Results shown below are from the previous run. Click **Run Backtest** to update.")
    
    results = st.session_state.results
    df = results["df"]
    stats = results["stats"]
    ds = results["ds"]
    trades = results["trades"]
    equity_curve = results.get("equity_curve")

    # Layout Tabs
    tab_dashboard, tab_trades, tab_analysis = st.tabs(["📊 Dashboard", "📝 Trades & Diagnostics", "🔬 Analysis & Discovery"])

    trades_table = build_trades_table(trades)

    # --- TAB 1: Dashboard ---
    with tab_dashboard:
        # Metrics Row
        # Get metrics from backend stats
        total_equity_ret = stats.get("total_equity_return_pct", 0.0)
        max_drawdown = stats.get("max_drawdown_pct", 0.0)
        sharpe = stats.get("sharpe_ratio", 0.0)
        sortino = stats.get("sortino_ratio", 0.0)
        win_rate = stats.get("win_rate", 0.0)  # Already in percentage from engine
        profit_factor = stats.get("profit_factor", 0.0)
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Return", f"{total_equity_ret:.2f}%")
        c2.metric("Win Rate", f"{win_rate:.1f}%")
        c3.metric("Profit Factor", f"{profit_factor:.2f}")
        c4.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        c5.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c6.metric("Trades", f"{len(trades_table)}")

        st.divider()

        # Chart controls
        chart_col1, chart_col2 = st.columns([1, 1])
        with chart_col1:
            show_candles = st.checkbox("Show Candlesticks", value=True, key="show_candles")
        with chart_col2:
            lock_rsi_y = st.checkbox("Lock RSI Y-axis (0-100)", value=True, key="lock_rsi_y")

        # Chart
        ds_full = ds
        ds_plot = ds_full.copy()
        MAX_POINTS = 5000
        if len(ds_plot) > MAX_POINTS:
            step = len(ds_plot) // MAX_POINTS
            ds_plot = ds_plot.iloc[::step]

        selected_trade = st.session_state.get("selected_trade")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.50, 0.25, 0.25],
            subplot_titles=("Price + BB + KC", "RSI (30m) + MA", "Equity & Drawdown")
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
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["bb_mid"], mode="lines", name=f"BB mid ({bb_basis_type.upper()})", line=dict(width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["bb_up"],  mode="lines", name="BB upper", line=dict(color='gray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["bb_low"], mode="lines", name="BB lower", line=dict(color='gray', width=1)), row=1, col=1)

        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["kc_mid"], mode="lines", name=f"KC mid ({kc_mid_type.upper()})", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["kc_up"],  mode="lines", name="KC upper", line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["kc_low"], mode="lines", name="KC lower", line=dict(color='blue', width=1)), row=1, col=1)

        # rsi
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["rsi30"],    mode="lines", name="RSI(30m)", line=dict(color="purple")), row=2, col=1)
        fig.add_trace(go.Scatter(x=ds_plot.index, y=ds_plot["rsi30_ma"], mode="lines", name=f"RSI MA ({rsi_ma_type.upper()})", line=dict(color="orange")), row=2, col=1)
        
        # Add RSI thresholds
        fig.add_hline(y=rsi_min, line_dash="dot", annotation_text="RSI Min", row=2, col=1)
        fig.add_hline(y=rsi_ma_min, line_dash="dot", annotation_text="RSI MA Min", row=2, col=1)

        # Equity curve and drawdown (row 3)
        if equity_curve is not None and len(equity_curve) > 0:
            # Sample equity curve to match ds_plot indices
            if len(ds_plot) < len(equity_curve):
                step = len(equity_curve) // len(ds_plot)
                eq_sampled = equity_curve[::step][:len(ds_plot)]
            else:
                eq_sampled = equity_curve[:len(ds_plot)]
            
            # Ensure arrays match in length
            eq_sampled = eq_sampled[:len(ds_plot.index)]
            
            # Calculate drawdown
            drawdown_pct, _ = calculate_drawdown(eq_sampled)
            
            # Equity line
            fig.add_trace(go.Scatter(
                x=ds_plot.index[:len(eq_sampled)], 
                y=eq_sampled,
                mode="lines", 
                name="Equity",
                line=dict(color="royalblue", width=2)
            ), row=3, col=1)
            
            # Drawdown fill (negative values for visual effect)
            fig.add_trace(go.Scatter(
                x=ds_plot.index[:len(drawdown_pct)],
                y=-drawdown_pct,
                mode="lines",
                fill="tozeroy",
                name="Drawdown %",
                line=dict(color="crimson", width=1),
                fillcolor="rgba(220, 20, 60, 0.3)"
            ), row=3, col=1)

        if selected_trade is None:
            # trade markers from the table
            if not trades_table.empty:
                fig.add_trace(go.Scattergl(
                    x=trades_table["EntryTime"], y=trades_table["Price@Entry"],
                    mode="markers", marker=dict(size=8, symbol="triangle-down", color="red"),
                    name="Short Entry", showlegend=True
                ), row=1, col=1)
                fig.add_trace(go.Scattergl(
                    x=trades_table["ExitTime"], y=trades_table["Price@Exit"],
                    mode="markers", marker=dict(size=8, symbol="circle-dot", color="green"),
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

                # horizontal zoom (all panels)
                fig.update_xaxes(range=[x0, x1], row=1, col=1)
                fig.update_xaxes(range=[x0, x1], row=2, col=1)
                fig.update_xaxes(range=[x0, x1], row=3, col=1)

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
            hovermode="x unified",
            height=700 
        )
        if lock_rsi_y:
            fig.update_yaxes(range=[0, 100], fixedrange=True, row=2, col=1)
        fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
        fig.update_xaxes(rangeslider=dict(visible=False), row=2, col=1)
        fig.update_xaxes(rangeslider=dict(visible=False), row=3, col=1)
        fig.update_yaxes(title_text="Equity / DD%", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)


    # --- TAB 2: Trades ---
    with tab_trades:
        tz = "America/Los_Angeles"
        trades_view = trades_table.copy()
        if not trades_view.empty:
            trades_view["Entry (Local)"] = pd.to_datetime(trades_view["EntryTime"], utc=True)\
                .dt.tz_convert(tz).dt.strftime("%m/%d/%Y %H:%M")
            trades_view["Exit (Local)"] = pd.to_datetime(trades_view["ExitTime"],  utc=True)\
                .dt.tz_convert(tz).dt.strftime("%m/%d/%Y %H:%M")

        # Trades
        if trades_table.empty:
            st.info("No trades for the selected settings / window.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Trade Log")
                preferred_cols = [
                    "Entry (Local)","Exit (Local)","EntryBar","ExitBar",
                    "Side","Price@Entry","Price@Exit","Price Move %","Duration","PnL (per unit)"
                ]

                cols_present = [c for c in preferred_cols if c in trades_view.columns]
                df_grid = trades_view[cols_present].copy()

                gb = GridOptionsBuilder.from_dataframe(df_grid)
                
                # Configure selection
                gb.configure_selection(
                    selection_mode="single", 
                    use_checkbox=True,
                    pre_selected_rows=[] # Clean invalid prop
                )
                
                gb.configure_default_column(filter=True, sortable=True, resizable=True)
                gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=15)
                
                # We do NOT set suppressRowClickSelection in configure_grid_options to avoid the warning if it conflicts
                # Instead, we rely on standard behavior or pass it cleanly if supported
                grid_options = gb.build()
                grid_options["suppressRowClickSelection"] = True

                grid_resp = AgGrid(
                    df_grid,
                    gridOptions=grid_options,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    allow_unsafe_jscode=False,
                    height=400,
                    theme="balham",
                    fit_columns_on_grid_load=False
                )

                sel = grid_resp.get("selected_rows", [])
                if sel:
                    st.session_state.selected_trade = sel[0]
                    st.info(f"Selected Trade: {sel[0].get('Entry (Local)')} (Go to Dashboard tab to see on chart)")
            
            with col2:
                st.subheader("Entry Diagnostics")
                try:
                    diag = build_entry_diagnostics(trades, ds)
                    if diag.empty:
                        st.info("No diagnostics available.")
                    else:
                        st.dataframe(diag, use_container_width=True, height=400)
                        
                        csv_diag = diag.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Diagnostics",
                            data=csv_diag,
                            file_name="entry_diagnostics.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Diagnostics render failed: {e}")

    # --- TAB 3: Analysis ---
    with tab_analysis:
        st.header("Analysis & Discovery")
        
        mode = st.radio("Select Tool", ["Parameter Optimization", "Strategy Discovery", "Leaderboard", "Pattern Recognition"], horizontal=True, key="analysis_mode")
        
        if mode == "Parameter Optimization":
            st.markdown("""
            Run a grid search to find optimal parameter combinations. 
            This will test multiple parameter values and rank results by the selected metric.
            """)
            
            opt_col1, opt_col2 = st.columns(2)
            
            with opt_col1:
                st.subheader("Parameter Ranges")
                
                # RSI Range
                rsi_range = st.slider(
                    "RSI Minimum Range",
                    min_value=60, max_value=85,
                    value=(65, 75),
                    key="opt_rsi_range",
                    help="Range of RSI minimum values to test"
                )
                
                # Stop % Range
                stop_range = st.slider(
                    "Stop Loss % Range",
                    min_value=0.5, max_value=5.0,
                    value=(1.5, 3.0),
                    step=0.5,
                    key="opt_stop_range",
                    help="Range of stop loss percentages to test"
                )
                
                # Band multiplier range
                band_range = st.slider(
                    "Band Multiplier Range (BB std / KC mult)",
                    min_value=1.5, max_value=3.0,
                    value=(1.8, 2.2),
                    step=0.1,
                    key="opt_band_range",
                    help="Range for BB std dev and KC ATR multiplier"
                )
            
            with opt_col2:
                st.subheader("Optimization Settings")

                opt_validation_mode = st.radio(
                    "Validation mode",
                    ["In-sample (single window)", "Walk-forward (train/test)"],
                    horizontal=True,
                    key="opt_validation_mode"
                )
                
                # Optimization metric
                opt_metric = st.selectbox(
                    "Optimize for",
                    ["profit_factor", "sharpe_ratio", "sortino_ratio", "win_rate", "total_equity_return_pct"],
                    index=0,
                    key="opt_metric",
                    help="Select the metric to maximize"
                )
                
                # Grid density
                grid_steps = st.slider(
                    "Grid Steps",
                    min_value=2, max_value=5,
                    value=3,
                    key="opt_grid_steps",
                    help="Number of values to test within each range (more = slower but more thorough)"
                )
                
                # Include entry modes
                include_entry_modes = st.checkbox("Test all entry band modes", value=True, key="opt_include_entry_modes")
                include_exit_levels = st.checkbox("Test both exit levels (mid/lower)", value=False, key="opt_include_exit_levels")
                
                # Minimum trades filter
                min_trades = st.number_input(
                    "Minimum trades required",
                    min_value=1, max_value=50,
                    value=5,
                    key="opt_min_trades",
                    help="Skip configurations with fewer trades"
                )

                st.subheader("Hard Constraints (optional)")
                min_pf = st.number_input(
                    "Min Profit Factor (0 = off)",
                    min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                    key="opt_min_pf"
                )
                min_wr = st.number_input(
                    "Min Win Rate % (0 = off)",
                    min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                    key="opt_min_wr"
                )
                max_dd = st.number_input(
                    "Max Drawdown % (0 = off)",
                    min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                    key="opt_max_dd"
                )
                min_total_ret = st.number_input(
                    "Min Total Return % (0 = off)",
                    min_value=-100.0, max_value=1000.0, value=0.0, step=1.0,
                    key="opt_min_total_ret"
                )

                if opt_validation_mode == "Walk-forward (train/test)":
                    st.subheader("Walk-forward settings")
                    wf_train_days = st.number_input("Train window (days)", min_value=30, max_value=3650, value=180, step=30, key="wf_train_days")
                    wf_test_days = st.number_input("Test window (days)", min_value=7, max_value=365, value=30, step=7, key="wf_test_days")
                    wf_max_folds = st.number_input("Max folds (0 = all)", min_value=0, max_value=200, value=8, step=1, key="wf_max_folds")
            
            # Calculate expected combinations
            param_grid = create_custom_grid(
                rsi_range=rsi_range,
                stop_range=stop_range,
                band_mult_range=band_range,
                steps=grid_steps,
                include_entry_modes=include_entry_modes,
                include_exit_levels=include_exit_levels,
            )
            
            total_combos = 1
            for v in param_grid.values():
                total_combos *= len(v)
            
            constraints = ResultConstraints(
                min_trades=int(min_trades),
                min_profit_factor=(float(min_pf) if min_pf and min_pf > 0 else None),
                min_win_rate=(float(min_wr) if min_wr and min_wr > 0 else None),
                max_drawdown=(float(max_dd) if max_dd and max_dd > 0 else None),
                # Treat 0 as "off" to match UI label (even though 0 can be meaningful).
                min_total_return=(float(min_total_ret) if (min_total_ret is not None and float(min_total_ret) != 0.0) else None),
            )

            if opt_validation_mode == "Walk-forward (train/test)":
                # crude estimate of how many folds we can fit
                try:
                    df_days = max(0, (results["df"].index.max() - results["df"].index.min()).days)
                    est_folds = max(0, (df_days - int(wf_train_days)) // max(1, int(wf_test_days)))
                except Exception:
                    est_folds = 0
                st.info(
                    f"Grid size: **{total_combos:,}** per fold. "
                    f"Estimated folds: **{est_folds}** (capped by Max folds)."
                )
            else:
                st.info(f"**{total_combos:,} combinations** will be tested. Estimated time: ~{total_combos // 10} seconds.")
            
            # Run button
            if st.button("🚀 Run Optimization", type="primary"):
                # Build base params from current settings
                base_params = params_now.copy()
                
                progress_bar = st.progress(0, text="Starting optimization...")
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress, text=f"Testing configuration {current}/{total}")
                
                with st.spinner("Running grid search..."):
                    try:
                        if opt_validation_mode == "Walk-forward (train/test)":
                            wf_cfg = WalkForwardConfig(
                                train_days=int(wf_train_days),
                                test_days=int(wf_test_days),
                                step_days=None,
                                max_folds=(None if int(wf_max_folds) == 0 else int(wf_max_folds)),
                            )

                            def wf_progress(cur, tot):
                                progress = cur / tot if tot > 0 else 0.0
                                progress_bar.progress(progress, text=f"Walk-forward fold {cur}/{tot}")

                            fold_results, wf_summary = run_walk_forward_grid_search(
                                df=results["df"],
                                param_grid=param_grid,
                                base_params=base_params,
                                metric=opt_metric,
                                constraints=constraints,
                                wf=wf_cfg,
                                top_n_train=20,
                                progress_callback=wf_progress
                            )

                            progress_bar.empty()

                            if fold_results is None or fold_results.empty:
                                st.warning(f"Walk-forward produced no results. {wf_summary.get('error', '')}")
                            else:
                                st.success(f"✅ Walk-forward complete! OK folds: {wf_summary.get('folds_ok', 0)}/{wf_summary.get('folds_total', 0)}")
                                st.subheader("Fold Results (out-of-sample)")
                                st.dataframe(fold_results, use_container_width=True)

                                st.subheader("Out-of-sample Summary")
                                st.write(f"- Avg OOS Return: `{wf_summary.get('oos_avg_return', 0):.2f}%`")
                                st.write(f"- Median OOS Return: `{wf_summary.get('oos_median_return', 0):.2f}%`")
                                st.write(f"- Avg OOS Win Rate: `{wf_summary.get('oos_avg_win_rate', 0):.1f}%`")
                                st.write(f"- Avg OOS Profit Factor: `{wf_summary.get('oos_avg_profit_factor', 0):.2f}`")
                                st.write(f"- Avg OOS Max DD: `{wf_summary.get('oos_avg_max_drawdown', 0):.2f}%`")

                                st.subheader("Recommended (stable) parameters")
                                for k, v in (wf_summary.get("recommended_params") or {}).items():
                                    st.write(f"- {k}: `{v}`")

                                st.download_button(
                                    "📥 Download Walk-forward Results",
                                    data=fold_results.to_csv(index=False).encode("utf-8"),
                                    file_name="walk_forward_results.csv",
                                    mime="text/csv"
                                )
                        else:
                            opt_results = run_grid_search(
                                df=results["df"],
                                param_grid=param_grid,
                                base_params=base_params,
                                metric=opt_metric,
                                min_trades=min_trades,
                                constraints=constraints,
                                top_n=20,
                                progress_callback=update_progress
                            )
                            
                            progress_bar.empty()
                            
                            if opt_results.empty:
                                st.warning("No valid configurations found. Try relaxing constraints / minimum trades.")
                            else:
                                st.success(f"✅ Optimization complete! Found {len(opt_results)} valid configurations.")
                                
                                # Show top results
                                st.subheader("Top Configurations")
                                
                                # Format display columns
                                display_cols = list(param_grid.keys()) + [
                                    "profit_factor", "win_rate", "total_return", 
                                    "max_drawdown", "sharpe_ratio", "num_trades"
                                ]
                                display_cols = [c for c in display_cols if c in opt_results.columns]
                                
                                st.dataframe(
                                    opt_results[display_cols].style.format({
                                        "profit_factor": "{:.2f}",
                                        "win_rate": "{:.1f}%",
                                        "total_return": "{:.2f}%",
                                        "max_drawdown": "{:.2f}%",
                                        "sharpe_ratio": "{:.2f}",
                                    }),
                                    use_container_width=True
                                )
                                
                                # Analysis
                                analysis = analyze_results(opt_results)
                                
                                st.subheader("Analysis")
                                anal_col1, anal_col2 = st.columns(2)
                                
                                with anal_col1:
                                    st.markdown("**Best Configuration:**")
                                    for k, v in analysis.get("best_params", {}).items():
                                        st.write(f"- {k}: `{v}`")
                                
                                with anal_col2:
                                    st.markdown("**Performance:**")
                                    st.write(f"- Top Profit Factor: `{analysis.get('top_profit_factor', 0):.2f}`")
                                    st.write(f"- Avg PF (Top 10): `{analysis.get('avg_profit_factor_top_10', 0):.2f}`")
                                
                                # Download results
                                st.download_button(
                                    "📥 Download Optimization Results",
                                    data=opt_results.to_csv(index=False).encode("utf-8"),
                                    file_name="optimization_results.csv",
                                    mime="text/csv"
                                )
                            
                    except Exception as e:
                        progress_bar.empty()
                        st.error(f"Optimization failed: {e}")
        
        elif mode == "Strategy Discovery":
            st.markdown("""
            **Automatically discover winning strategies** by systematically testing thousands of parameter combinations.
            Results are saved to a database.
            """)
            
            # Performance settings at the top
            st.subheader("Discovery Settings")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                cpu_count = get_cpu_count()
                disc_n_workers = st.slider(
                    "Parallel Workers",
                    min_value=1, max_value=cpu_count,
                    value=max(1, cpu_count - 1),
                    key="disc_n_workers",
                )
            
            with perf_col2:
                disc_use_parallel = st.checkbox("Enable Parallel Processing", value=True, key="disc_use_parallel")
            
            with perf_col3:
                disc_skip_tested = st.checkbox("Skip tested combinations", value=True, key="disc_skip_tested")
            
            disc_col1, disc_col2, disc_col3 = st.columns(3)
            
            with disc_col1:
                st.caption("Strategy Parameters")
                disc_rsi_range = st.slider("RSI Min Range", 60, 82, (68, 74), key="disc_rsi_range")
                disc_rsi_ma_range = st.slider("RSI MA Min Range", 58, 80, (66, 72), key="disc_rsi_ma_range")
                disc_band_range = st.slider("Band Mult Range", 1.5, 2.8, (1.9, 2.1), key="disc_band_range")
            
            with disc_col2:
                st.caption("Margin/Risk")
                disc_include_margin = st.checkbox("Include Margin Mode", value=True, key="disc_include_margin")
                if disc_include_margin:
                    disc_leverage_options = st.multiselect("Leverage", [2.0, 3.0, 5.0, 10.0, 20.0], [2.0, 5.0, 10.0], key="disc_leverage_options")
                    disc_risk_options = st.multiselect("Risk %", [0.5, 1.0, 1.5, 2.0, 3.0], [0.5, 1.0, 2.0], key="disc_risk_options")
                else:
                    disc_leverage_options = []
                    disc_risk_options = [1.0]
                disc_include_atr = st.checkbox("Include ATR Stops", value=True, key="disc_include_atr")
                disc_include_trailing = st.checkbox("Include Trailing", value=True, key="disc_include_trailing")
            
            with disc_col3:
                st.caption("Win Criteria")
                disc_min_return = st.number_input("Min Return %", -100.0, 100.0, 0.0, key="disc_min_return")
                disc_max_dd = st.number_input("Max Drawdown %", 1.0, 50.0, 20.0, key="disc_max_dd")
                disc_min_trades = st.number_input("Min Trades", 5, 100, 10, key="disc_min_trades")
                disc_min_pf = st.number_input("Min Profit Factor", 0.5, 3.0, 1.0, key="disc_min_pf")
            
            # Build discovery grid based on settings
            if disc_include_margin:
                disc_grid = create_margin_discovery_grid(
                    rsi_range=(int(disc_rsi_range[0]), int(disc_rsi_range[1])),
                    rsi_ma_range=(int(disc_rsi_ma_range[0]), int(disc_rsi_ma_range[1])),
                    band_mult_range=disc_band_range,
                    leverage_options=disc_leverage_options if disc_leverage_options else [5.0],
                    risk_pct_options=disc_risk_options if disc_risk_options else [1.0],
                    include_atr_stops=disc_include_atr,
                    include_trailing=disc_include_trailing,
                    rsi_step=2,
                    mult_step=0.1,
                )
            else:
                disc_grid = create_discovery_grid(
                    rsi_range=(int(disc_rsi_range[0]), int(disc_rsi_range[1])),
                    rsi_ma_range=(int(disc_rsi_ma_range[0]), int(disc_rsi_ma_range[1])),
                    band_mult_range=disc_band_range,
                    stop_range=(1.5, 2.5),
                    rsi_step=2,
                    mult_step=0.1,
                    stop_step=0.5,
                    include_all_modes=True
                )
            
            # Get current database stats
            db_stats = discovery_db.get_statistics()
            tested_count = db_stats.get("total_runs", 0)
            
            # Estimate time with parallel workers
            effective_workers = disc_n_workers if disc_use_parallel else 1
            estimate = estimate_discovery_time(
                disc_grid, 
                tested_count if disc_skip_tested else 0,
                n_workers=effective_workers
            )
            
            st.info(f"**{estimate['remaining']:,} new to test** | Estimated time: **{estimate['human_readable']}**")
            
            if st.button("🔬 Run Strategy Discovery", type="primary", key="run_discovery_btn"):
                win_criteria = WinCriteria(
                    min_total_return=disc_min_return,
                    max_drawdown=disc_max_dd,
                    min_trades=disc_min_trades,
                    min_profit_factor=disc_min_pf
                )
                
                config = DiscoveryConfig(
                    param_grid=disc_grid,
                    win_criteria=win_criteria,
                    skip_tested=disc_skip_tested,
                    batch_size=50
                )
                
                progress_bar = st.progress(0, text="Starting discovery...")
                
                def discovery_progress(current, total, status):
                    progress = current / total if total > 0 else 0
                    progress_bar.progress(progress, text=status)
                
                try:
                    if disc_use_parallel:
                        disc_result = run_discovery_parallel(
                            df=results["df"],
                            base_params=params_now,
                            db=discovery_db,
                            config=config,
                            n_workers=disc_n_workers,
                            progress_callback=discovery_progress
                        )
                    else:
                        disc_result = run_discovery(
                            df=results["df"],
                            base_params=params_now,
                            db=discovery_db,
                            config=config,
                            progress_callback=discovery_progress
                        )
                    
                    progress_bar.empty()
                    st.success(f"✅ Discovery complete! Found {disc_result.winners_found} winners.")
                    st.session_state.discovery_results = disc_result
                    
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Discovery failed: {e}")

        elif mode == "Leaderboard":
            st.subheader("Winning Strategies Leaderboard")
            
            lb = Leaderboard(discovery_db)
            lb_stats = lb.get_stats()
            
            if lb_stats.total_winners == 0:
                st.info("No winning strategies found yet. Run Discovery to find winners!")
            else:
                # Stats row
                lb_c1, lb_c2, lb_c3, lb_c4 = st.columns(4)
                lb_c1.metric("Total Winners", lb_stats.total_winners)
                lb_c2.metric("Best Return", f"{lb_stats.best_return:.2f}%")
                lb_c3.metric("Avg Return", f"{lb_stats.avg_return:.2f}%")
                lb_c4.metric("Avg Drawdown", f"{lb_stats.avg_drawdown:.2f}%")
                
                # Sort options
                lb_sort_col1, lb_sort_col2 = st.columns(2)
                with lb_sort_col1:
                    lb_sort_by = st.selectbox(
                        "Sort by",
                        ["total_return", "profit_factor", "sharpe_ratio", "win_rate", "max_drawdown"],
                        key="lb_sort_by"
                    )
                with lb_sort_col2:
                    lb_top_n = st.selectbox("Show top", [10, 25, 50, 100], key="lb_top_n")
                
                # Get leaderboard
                top_strategies = lb.get_top(n=lb_top_n, sort_by=lb_sort_by)
                
                if top_strategies:
                    # Create DataFrame for display
                    lb_data = []
                    for s in top_strategies:
                        lb_data.append({
                            "Rank": s.rank,
                            "Return %": f"{s.total_return:.2f}",
                            "Max DD %": f"{s.max_drawdown:.2f}",
                            "Profit Factor": f"{s.profit_factor:.2f}",
                            "Win Rate %": f"{s.win_rate:.1f}",
                            "Sharpe": f"{s.sharpe_ratio:.2f}",
                            "Trades": s.num_trades,
                            "Hash": s.params_hash[:8],
                        })
                    
                    lb_df = pd.DataFrame(lb_data)
                    
                    # Build grid
                    gb = GridOptionsBuilder.from_dataframe(lb_df)
                    gb.configure_selection(selection_mode="single", use_checkbox=True)
                    gb.configure_default_column(filter=True, sortable=True)
                    gb.configure_column("Rank", checkboxSelection=True)
                    grid_options = gb.build()
                    
                    grid_response = AgGrid(
                        lb_df,
                        gridOptions=grid_options,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        height=300,
                        theme="balham",
                        fit_columns_on_grid_load=True
                    )
                    
                    # Show selected strategy details
                    selected = grid_response.get("selected_rows", [])
                    if selected:
                        selected_row = selected[0]
                        selected_hash = None
                        
                        # Find full hash
                        for s in top_strategies:
                            if s.params_hash.startswith(selected_row.get("Hash", "")):
                                selected_hash = s.params_hash
                                break
                        
                        if selected_hash:
                            selected_strategy = lb.get_strategy_by_hash(selected_hash)
                            if selected_strategy:
                                st.subheader(f"Strategy #{selected_strategy.rank} Parameters")
                                
                                param_col1, param_col2 = st.columns(2)
                                
                                params_list = list(selected_strategy.params.items())
                                mid = len(params_list) // 2
                                
                                with param_col1:
                                    for k, v in params_list[:mid]:
                                        st.write(f"**{k}:** `{v}`")
                                
                                with param_col2:
                                    for k, v in params_list[mid:]:
                                        st.write(f"**{k}:** `{v}`")
                                
                                # Load strategy button
                                if st.button("📥 Load This Strategy", key="load_strategy_btn"):
                                    # Update preset params with selected strategy
                                    st.session_state.preset_params.update(selected_strategy.params)
                                    st.session_state.selected_preset = "Custom"
                                    st.session_state.dirty_params = True
                                    st.session_state.selected_winning_strategy = selected_strategy
                                    # Sync widget session state keys to match loaded strategy
                                    sync_widgets_to_preset(selected_strategy.params)
                                    st.success("Strategy loaded! Click 'Run Backtest' to test it.")
                                    st.rerun()
                    
                    # Export button
                    export_data = lb_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Export Leaderboard CSV",
                        data=export_data,
                        file_name="winning_strategies.csv",
                        mime="text/csv"
                    )

        elif mode == "Pattern Recognition":
            st.subheader("Discovered Patterns & Rules")
            
            if st.button("🔄 Analyze Patterns", key="analyze_patterns_btn"):
                with st.spinner("Analyzing winning strategies..."):
                    rules = find_winning_patterns(discovery_db, min_confidence=0.3)
                    
                    if not rules:
                        st.warning("Not enough data to discover patterns. Run more discovery tests.")
                    else:
                        st.success(f"Found {len(rules)} patterns!")
            
            # Show existing rules
            existing_rules = discovery_db.get_rules(min_confidence=0.3)
            
            if not existing_rules:
                st.info("No patterns discovered yet. Run Discovery and then click 'Analyze Patterns'.")
            else:
                st.markdown(get_rule_summary(existing_rules))