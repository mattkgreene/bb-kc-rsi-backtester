"""
Strategy preset configurations for the BB+KC+RSI backtester.

This module defines pre-configured strategy presets that users can quickly
select in the UI. Each preset is optimized for different trading objectives
such as maximizing profit factor, win rate, or minimizing drawdown.

Presets are designed based on the short-only mean-reversion strategy using
Bollinger Bands, Keltner Channels, and RSI indicators.
"""

from typing import TypedDict, Optional, Literal, Dict


class StrategyPreset(TypedDict, total=False):
    """
    Type definition for a strategy preset configuration.
    
    All parameters correspond to the backtest engine settings.
    Optional fields use total=False to allow partial definitions.
    """
    # Metadata
    name: str
    description: str
    category: Literal["conservative", "aggressive", "balanced", "specialized"]
    
    # Bollinger Bands
    bb_len: int
    bb_std: float
    bb_basis_type: Literal["sma", "ema"]
    
    # Keltner Channel
    kc_ema_len: int
    kc_atr_len: int
    kc_mult: float
    kc_mid_type: Literal["ema", "sma"]
    
    # RSI
    rsi_len_30m: int
    rsi_ma_len: int
    rsi_smoothing_type: Literal["ema", "sma", "rma"]
    rsi_ma_type: Literal["ema", "sma"]
    
    # Entry Conditions
    rsi_min: int
    rsi_ma_min: int
    use_rsi_relation: bool
    rsi_relation: Literal["<", "<=", ">", ">="]
    entry_band_mode: Literal["Either", "KC", "BB", "Both"]
    
    # Exit Conditions
    exit_channel: Literal["BB", "KC"]
    exit_level: Literal["mid", "lower"]
    
    # Risk Management
    use_stop: bool
    stop_mode: Literal["Fixed %", "ATR"]
    stop_pct: Optional[float]
    stop_atr_mult: Optional[float]
    use_trailing: bool
    trail_pct: float
    max_bars_in_trade: int
    daily_loss_limit: float
    risk_per_trade_pct: float
    
    # Trade Mode (for margin/futures)
    trade_mode: Literal["Simple (1x spot-style)", "Margin / Futures"]
    max_leverage: Optional[float]
    maintenance_margin_pct: Optional[float]
    max_margin_utilization: Optional[float]


# =============================================================================
# Default Base Configuration
# =============================================================================

DEFAULT_PRESET: StrategyPreset = {
    "name": "Default",
    "description": "Balanced default configuration",
    "category": "balanced",
    
    # Bollinger Bands
    "bb_len": 20,
    "bb_std": 2.0,
    "bb_basis_type": "sma",
    
    # Keltner Channel
    "kc_ema_len": 20,
    "kc_atr_len": 14,
    "kc_mult": 2.0,
    "kc_mid_type": "ema",
    
    # RSI
    "rsi_len_30m": 14,
    "rsi_ma_len": 10,
    "rsi_smoothing_type": "ema",
    "rsi_ma_type": "sma",
    
    # Entry
    "rsi_min": 70,
    "rsi_ma_min": 70,
    "use_rsi_relation": True,
    "rsi_relation": ">=",
    "entry_band_mode": "Either",
    
    # Exit
    "exit_channel": "BB",
    "exit_level": "mid",
    
    # Risk
    # Defaulting to *no stop* while also defaulting to margin/futures is a
    # recipe for liquidation in trending markets. Keep the default safe.
    "use_stop": True,
    "stop_mode": "Fixed %",
    "stop_pct": 2.0,
    "stop_atr_mult": 2.0,
    "use_trailing": False,
    "trail_pct": 1.0,
    "max_bars_in_trade": 100,
    "daily_loss_limit": 3.0,
    "risk_per_trade_pct": 1.0,
    
    # Trade mode
    "trade_mode": "Simple (1x spot-style)",
    "max_leverage": None,
    "maintenance_margin_pct": None,
    "max_margin_utilization": None,
}


# =============================================================================
# Strategy Presets Dictionary
# =============================================================================

STRATEGY_PRESETS: Dict[str, StrategyPreset] = {
    
    # -------------------------------------------------------------------------
    # CONSERVATIVE STRATEGIES - Focus on high win rate and capital preservation
    # -------------------------------------------------------------------------
    
    "conservative": {
        "name": "Conservative",
        "description": "High win rate with tight risk controls. Requires strong signals (both bands + high RSI). Best for capital preservation.",
        "category": "conservative",
        
        # Indicators (standard)
        "bb_len": 20,
        "bb_std": 2.0,
        "bb_basis_type": "sma",
        "kc_ema_len": 20,
        "kc_atr_len": 14,
        "kc_mult": 2.0,
        "kc_mid_type": "ema",
        "rsi_len_30m": 14,
        "rsi_ma_len": 10,
        "rsi_smoothing_type": "ema",
        "rsi_ma_type": "sma",
        
        # Entry - Very strict conditions
        "rsi_min": 75,
        "rsi_ma_min": 72,
        "use_rsi_relation": True,
        "rsi_relation": ">=",
        "entry_band_mode": "Both",  # Must touch BOTH bands
        
        # Exit - Quick exit at mid
        "exit_channel": "BB",
        "exit_level": "mid",
        
        # Risk - Tight controls
        "use_stop": True,
        "stop_mode": "Fixed %",
        "stop_pct": 1.5,
        "stop_atr_mult": None,
        "use_trailing": False,
        "trail_pct": 1.0,
        "max_bars_in_trade": 50,  # Short holding period
        "daily_loss_limit": 2.0,
        "risk_per_trade_pct": 0.5,  # Small position size
        
        "trade_mode": "Simple (1x spot-style)",
        "max_leverage": None,
        "maintenance_margin_pct": None,
        "max_margin_utilization": None,
    },
    
    "low_drawdown": {
        "name": "Low Drawdown",
        "description": "Minimizes maximum drawdown with very conservative entries and strict daily limits. Sacrifices returns for stability.",
        "category": "conservative",
        
        "bb_len": 20,
        "bb_std": 2.2,  # Wider bands = fewer signals
        "bb_basis_type": "sma",
        "kc_ema_len": 20,
        "kc_atr_len": 14,
        "kc_mult": 2.2,
        "kc_mid_type": "ema",
        "rsi_len_30m": 14,
        "rsi_ma_len": 10,
        "rsi_smoothing_type": "ema",
        "rsi_ma_type": "sma",
        
        "rsi_min": 75,
        "rsi_ma_min": 73,
        "use_rsi_relation": True,
        "rsi_relation": ">=",
        "entry_band_mode": "Both",
        
        "exit_channel": "BB",
        "exit_level": "mid",
        
        "use_stop": True,
        "stop_mode": "Fixed %",
        "stop_pct": 1.0,  # Very tight stop
        "stop_atr_mult": None,
        "use_trailing": False,
        "trail_pct": 0.8,
        "max_bars_in_trade": 40,
        "daily_loss_limit": 1.5,  # Strict daily limit
        "risk_per_trade_pct": 0.3,  # Very small positions
        
        "trade_mode": "Simple (1x spot-style)",
        "max_leverage": None,
        "maintenance_margin_pct": None,
        "max_margin_utilization": None,
    },
    
    # -------------------------------------------------------------------------
    # AGGRESSIVE STRATEGIES - Focus on profit factor and larger gains
    # -------------------------------------------------------------------------
    
    "aggressive": {
        "name": "Aggressive",
        "description": "Higher profit factor with larger position sizing and wider stops. More trades, accepts lower win rate for bigger winners.",
        "category": "aggressive",
        
        "bb_len": 20,
        "bb_std": 2.0,
        "bb_basis_type": "sma",
        "kc_ema_len": 20,
        "kc_atr_len": 14,
        "kc_mult": 1.8,  # Tighter KC = more signals
        "kc_mid_type": "ema",
        "rsi_len_30m": 14,
        "rsi_ma_len": 10,
        "rsi_smoothing_type": "ema",
        "rsi_ma_type": "sma",
        
        "rsi_min": 65,  # Lower threshold = more entries
        "rsi_ma_min": 65,
        "use_rsi_relation": True,
        "rsi_relation": ">=",
        "entry_band_mode": "Either",  # Either band OK
        
        "exit_channel": "BB",
        "exit_level": "lower",  # Hold for bigger moves
        
        "use_stop": True,
        "stop_mode": "ATR",
        "stop_pct": None,
        "stop_atr_mult": 2.5,  # Wider ATR-based stop
        "use_trailing": True,
        "trail_pct": 2.0,
        "max_bars_in_trade": 100,
        "daily_loss_limit": 5.0,
        "risk_per_trade_pct": 2.0,  # Larger positions
        
        "trade_mode": "Simple (1x spot-style)",
        "max_leverage": None,
        "maintenance_margin_pct": None,
        "max_margin_utilization": None,
    },
    
    "high_profit_factor": {
        "name": "High Profit Factor",
        "description": "Optimized specifically for maximum profit factor. Uses ATR stops and trailing to let winners run while cutting losers.",
        "category": "aggressive",
        
        "bb_len": 20,
        "bb_std": 2.0,
        "bb_basis_type": "sma",
        "kc_ema_len": 20,
        "kc_atr_len": 14,
        "kc_mult": 2.0,
        "kc_mid_type": "ema",
        "rsi_len_30m": 14,
        "rsi_ma_len": 10,
        "rsi_smoothing_type": "ema",
        "rsi_ma_type": "sma",
        
        "rsi_min": 72,
        "rsi_ma_min": 70,
        "use_rsi_relation": True,
        "rsi_relation": ">=",
        "entry_band_mode": "Either",
        
        "exit_channel": "BB",
        "exit_level": "lower",  # Target lower band for bigger moves
        
        "use_stop": True,
        "stop_mode": "ATR",
        "stop_pct": None,
        "stop_atr_mult": 2.0,
        "use_trailing": True,
        "trail_pct": 1.5,  # Trailing to protect profits
        "max_bars_in_trade": 80,
        "daily_loss_limit": 4.0,
        "risk_per_trade_pct": 1.5,
        
        "trade_mode": "Simple (1x spot-style)",
        "max_leverage": None,
        "maintenance_margin_pct": None,
        "max_margin_utilization": None,
    },
    
    # -------------------------------------------------------------------------
    # SPECIALIZED STRATEGIES - Specific trading styles
    # -------------------------------------------------------------------------
    
    "scalping": {
        "name": "Scalping",
        "description": "Quick in-and-out trades targeting small moves. Tight stops, short holding period, exits at mid band. High frequency.",
        "category": "specialized",
        
        "bb_len": 15,  # Shorter period = more responsive
        "bb_std": 1.8,
        "bb_basis_type": "ema",  # EMA for faster response
        "kc_ema_len": 15,
        "kc_atr_len": 10,
        "kc_mult": 1.8,
        "kc_mid_type": "ema",
        "rsi_len_30m": 10,  # Shorter RSI
        "rsi_ma_len": 5,
        "rsi_smoothing_type": "ema",
        "rsi_ma_type": "ema",
        
        "rsi_min": 70,
        "rsi_ma_min": 68,
        "use_rsi_relation": True,
        "rsi_relation": ">=",
        "entry_band_mode": "KC",  # KC only for speed
        
        "exit_channel": "KC",
        "exit_level": "mid",  # Quick exit at mid
        
        "use_stop": True,
        "stop_mode": "Fixed %",
        "stop_pct": 1.0,  # Tight stop
        "stop_atr_mult": None,
        "use_trailing": False,
        "trail_pct": 0.5,
        "max_bars_in_trade": 20,  # Very short holding
        "daily_loss_limit": 3.0,
        "risk_per_trade_pct": 0.5,
        
        "trade_mode": "Simple (1x spot-style)",
        "max_leverage": None,
        "maintenance_margin_pct": None,
        "max_margin_utilization": None,
    },
    
    "swing": {
        "name": "Swing Trading",
        "description": "Longer-term trades targeting larger moves. Wider stops, longer holding period, exits at lower band. Fewer but bigger trades.",
        "category": "specialized",
        
        "bb_len": 25,  # Longer period = smoother
        "bb_std": 2.2,
        "bb_basis_type": "sma",
        "kc_ema_len": 25,
        "kc_atr_len": 20,
        "kc_mult": 2.2,
        "kc_mid_type": "ema",
        "rsi_len_30m": 14,
        "rsi_ma_len": 14,
        "rsi_smoothing_type": "rma",  # RMA for smoother signals
        "rsi_ma_type": "sma",
        
        "rsi_min": 68,
        "rsi_ma_min": 66,
        "use_rsi_relation": True,
        "rsi_relation": ">=",
        "entry_band_mode": "BB",  # BB for swing trades
        
        "exit_channel": "BB",
        "exit_level": "lower",  # Full reversion to lower
        
        "use_stop": True,
        "stop_mode": "ATR",
        "stop_pct": None,
        "stop_atr_mult": 2.5,  # Wide ATR stop
        "use_trailing": True,
        "trail_pct": 2.5,
        "max_bars_in_trade": 150,  # Long holding OK
        "daily_loss_limit": 5.0,
        "risk_per_trade_pct": 1.0,
        
        "trade_mode": "Simple (1x spot-style)",
        "max_leverage": None,
        "maintenance_margin_pct": None,
        "max_margin_utilization": None,
    },
    
    "momentum_burst": {
        "name": "Momentum Burst",
        "description": "Catches extreme overbought conditions with very high RSI requirements. Fewer trades but high-probability setups.",
        "category": "specialized",
        
        "bb_len": 20,
        "bb_std": 2.0,
        "bb_basis_type": "sma",
        "kc_ema_len": 20,
        "kc_atr_len": 14,
        "kc_mult": 2.0,
        "kc_mid_type": "ema",
        "rsi_len_30m": 14,
        "rsi_ma_len": 10,
        "rsi_smoothing_type": "ema",
        "rsi_ma_type": "sma",
        
        "rsi_min": 80,  # Very high RSI required
        "rsi_ma_min": 75,
        "use_rsi_relation": True,
        "rsi_relation": ">=",
        "entry_band_mode": "Either",
        
        "exit_channel": "BB",
        "exit_level": "mid",
        
        "use_stop": True,
        "stop_mode": "ATR",
        "stop_pct": None,
        "stop_atr_mult": 1.5,  # Tighter stop - high conviction
        "use_trailing": True,
        "trail_pct": 1.0,
        "max_bars_in_trade": 60,
        "daily_loss_limit": 3.0,
        "risk_per_trade_pct": 1.5,
        
        "trade_mode": "Simple (1x spot-style)",
        "max_leverage": None,
        "maintenance_margin_pct": None,
        "max_margin_utilization": None,
    },
    
    "mean_reversion": {
        "name": "Mean Reversion Classic",
        "description": "Classic mean reversion setup using standard BB parameters. No trailing stop, relies on price returning to mean.",
        "category": "balanced",
        
        "bb_len": 20,
        "bb_std": 2.0,
        "bb_basis_type": "sma",
        "kc_ema_len": 20,
        "kc_atr_len": 14,
        "kc_mult": 2.0,
        "kc_mid_type": "ema",
        "rsi_len_30m": 14,
        "rsi_ma_len": 10,
        "rsi_smoothing_type": "ema",
        "rsi_ma_type": "sma",
        
        "rsi_min": 70,
        "rsi_ma_min": 68,
        "use_rsi_relation": False,  # No RSI relation check
        "rsi_relation": ">=",
        "entry_band_mode": "BB",
        
        "exit_channel": "BB",
        "exit_level": "mid",  # Mean reversion to mid
        
        "use_stop": True,
        "stop_mode": "Fixed %",
        "stop_pct": 2.5,
        "stop_atr_mult": None,
        "use_trailing": False,  # No trailing for pure reversion
        "trail_pct": 1.0,
        "max_bars_in_trade": 100,
        "daily_loss_limit": 3.0,
        "risk_per_trade_pct": 1.0,
        
        "trade_mode": "Simple (1x spot-style)",
        "max_leverage": None,
        "maintenance_margin_pct": None,
        "max_margin_utilization": None,
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_preset_names() -> list[str]:
    """Return list of all preset names."""
    return list(STRATEGY_PRESETS.keys())


def get_preset(name: str) -> StrategyPreset:
    """
    Get a preset by name.
    
    Args:
        name: Preset name (e.g., 'conservative', 'aggressive')
    
    Returns:
        StrategyPreset configuration dictionary
    
    Raises:
        KeyError: If preset name doesn't exist
    """
    if name not in STRATEGY_PRESETS:
        raise KeyError(f"Preset '{name}' not found. Available: {get_preset_names()}")
    return STRATEGY_PRESETS[name]


def get_presets_by_category(category: str) -> Dict[str, StrategyPreset]:
    """
    Get all presets in a specific category.
    
    Args:
        category: One of 'conservative', 'aggressive', 'balanced', 'specialized'
    
    Returns:
        Dictionary of presets in that category
    """
    return {
        name: preset 
        for name, preset in STRATEGY_PRESETS.items() 
        if preset.get("category") == category
    }


def merge_with_defaults(preset: StrategyPreset) -> StrategyPreset:
    """
    Merge a partial preset with default values.
    
    Args:
        preset: Partial preset configuration
    
    Returns:
        Complete preset with defaults filled in
    """
    merged = DEFAULT_PRESET.copy()
    merged.update(preset)
    return merged


def preset_to_params(preset: StrategyPreset) -> dict:
    """
    Convert a preset to parameter dictionary for the backtest engine.
    
    Removes metadata fields (name, description, category) and returns
    only the parameters needed for run_backtest().
    
    Args:
        preset: Strategy preset configuration
    
    Returns:
        Dictionary of parameters for backtest engine
    """
    # Fields to exclude (metadata only)
    exclude = {"name", "description", "category"}
    return {k: v for k, v in preset.items() if k not in exclude}


def get_preset_summary() -> str:
    """
    Get a formatted summary of all presets for display.
    
    Returns:
        Multi-line string describing all presets
    """
    lines = ["Available Strategy Presets:", "=" * 40]
    
    for category in ["conservative", "aggressive", "balanced", "specialized"]:
        presets = get_presets_by_category(category)
        if presets:
            lines.append(f"\n{category.upper()}:")
            for name, preset in presets.items():
                lines.append(f"  • {preset['name']}: {preset['description']}")
    
    return "\n".join(lines)
