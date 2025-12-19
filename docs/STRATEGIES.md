# Strategy Presets Guide

This document describes the pre-configured strategy presets available in the BB+KC+RSI Backtester. Each preset is optimized for different trading objectives and market conditions.

## Overview

The backtester implements a **short-only mean-reversion strategy** that:
1. Enters short positions when price touches upper Bollinger Bands (BB) and/or Keltner Channels (KC) with high RSI
2. Exits when price reverts to the mid or lower band
3. Uses various stop loss and risk management techniques

## Quick Comparison

| Preset | Win Rate | Profit Focus | Risk Level | Trade Frequency |
|--------|----------|--------------|------------|-----------------|
| Conservative | High | Low | Low | Low |
| Low Drawdown | High | Low | Very Low | Very Low |
| Aggressive | Medium | High | High | High |
| High Profit Factor | Medium | Very High | Medium | Medium |
| Scalping | Medium | Medium | Medium | Very High |
| Swing | Medium | High | Medium | Low |
| Momentum Burst | High | Medium | Medium | Low |
| Mean Reversion | Medium | Medium | Medium | Medium |

---

## Conservative Strategies

### Conservative

**Focus:** High win rate with tight risk controls

**Description:** This preset prioritizes capital preservation over profit maximization. It requires strong confirmation signals (price must touch BOTH BB and KC upper bands) and uses tight stop losses.

**Key Parameters:**
- RSI Minimum: 75 (very overbought)
- RSI MA Minimum: 72
- Entry Band Mode: Both (must touch KC AND BB)
- Stop Loss: 1.5% fixed
- Max Bars: 50 (short holding period)
- Risk per Trade: 0.5% of equity

**When to Use:**
- In volatile markets where capital preservation is priority
- When you want consistent small wins over occasional large wins
- For accounts where drawdown must be minimized

**When to Avoid:**
- In trending markets (may miss continuation moves)
- When you need higher returns to meet objectives
- In low volatility environments (few signals)

**Expected Performance:**
- Win Rate: 60-70%
- Profit Factor: 1.0-1.3
- Max Drawdown: 5-10%

---

### Low Drawdown

**Focus:** Minimize maximum drawdown

**Description:** The most conservative preset, designed for traders who cannot tolerate significant drawdowns. Uses wider bands (fewer signals), very tight stops, and strict daily loss limits.

**Key Parameters:**
- BB Std / KC Mult: 2.2 (wider bands = fewer signals)
- RSI Minimum: 75
- Entry Band Mode: Both
- Stop Loss: 1.0% (very tight)
- Daily Loss Limit: 1.5%
- Risk per Trade: 0.3% of equity

**When to Use:**
- For conservative capital allocation
- When preserving capital is more important than growth
- In highly volatile market conditions

**When to Avoid:**
- When you need meaningful returns
- In stable, trending markets

**Expected Performance:**
- Win Rate: 65-75%
- Profit Factor: 1.0-1.2
- Max Drawdown: 3-7%

---

## Aggressive Strategies

### Aggressive

**Focus:** Higher profit factor with larger position sizing

**Description:** This preset accepts a lower win rate in exchange for larger winning trades. It uses looser entry conditions, ATR-based stops that allow more room, and trailing stops to capture extended moves.

**Key Parameters:**
- RSI Minimum: 65 (lower threshold = more entries)
- KC Multiplier: 1.8 (tighter bands = more signals)
- Entry Band Mode: Either (KC or BB)
- Exit Level: Lower (hold for bigger moves)
- Stop Loss: ATR × 2.5
- Trailing Stop: 2.0%
- Risk per Trade: 2.0% of equity

**When to Use:**
- In mean-reverting market conditions
- When you can tolerate higher drawdowns for higher returns
- For accounts with longer time horizons

**When to Avoid:**
- In strongly trending markets
- When drawdown tolerance is low
- For short-term trading needs

**Expected Performance:**
- Win Rate: 40-50%
- Profit Factor: 1.5-2.5
- Max Drawdown: 15-25%

---

### High Profit Factor

**Focus:** Maximize profit factor (gross profit / gross loss)

**Description:** Specifically optimized for the highest profit factor. Uses ATR-based stops and trailing stops to let winners run while cutting losers quickly. Targets the lower band for exits to capture full mean reversion.

**Key Parameters:**
- RSI Minimum: 72
- Entry Band Mode: Either
- Exit Level: Lower
- Stop Loss: ATR × 2.0
- Trailing Stop: 1.5%
- Risk per Trade: 1.5% of equity

**When to Use:**
- When maximizing risk-adjusted returns is the goal
- In markets with clear mean-reverting behavior
- For systematic trading approaches

**When to Avoid:**
- In trending markets
- When you need consistent, predictable returns

**Expected Performance:**
- Win Rate: 45-55%
- Profit Factor: 2.0-3.0
- Max Drawdown: 12-20%

---

## Specialized Strategies

### Scalping

**Focus:** Quick in-and-out trades targeting small moves

**Description:** Designed for high-frequency trading with short holding periods. Uses faster indicators, targets mid-band exits for quick profits, and tight stop losses.

**Key Parameters:**
- BB Length: 15 (shorter period)
- BB Basis: EMA (faster response)
- KC ATR Length: 10
- RSI Length: 10
- Exit Level: Mid (quick exit)
- Stop Loss: 1.0%
- Max Bars: 20 (very short)

**When to Use:**
- In range-bound, choppy markets
- When you can monitor positions closely
- For accounts with low transaction costs

**When to Avoid:**
- In trending markets
- With high commission structures
- When you can't actively monitor

**Expected Performance:**
- Win Rate: 50-60%
- Profit Factor: 1.2-1.8
- Trades: Many small trades

---

### Swing Trading

**Focus:** Larger moves over longer periods

**Description:** Designed for traders who prefer fewer, larger trades. Uses longer indicator periods for smoother signals and wider stops to allow for normal market fluctuations.

**Key Parameters:**
- BB Length: 25
- KC EMA Length: 25
- RSI Smoothing: RMA (smoother)
- Entry Band Mode: BB only
- Exit Level: Lower
- Stop Loss: ATR × 2.5
- Trailing Stop: 2.5%
- Max Bars: 150

**When to Use:**
- In markets with clear mean-reverting cycles
- When you prefer less active trading
- For accounts where you can hold longer

**When to Avoid:**
- In choppy, directionless markets
- When you need frequent trading activity

**Expected Performance:**
- Win Rate: 45-55%
- Profit Factor: 1.5-2.5
- Trades: Fewer but larger

---

### Momentum Burst

**Focus:** Catch extreme overbought conditions

**Description:** Waits for very high RSI readings (extreme overbought) before entering. Fewer trades but higher conviction setups. Uses tighter stops since extreme readings often lead to quick reversals.

**Key Parameters:**
- RSI Minimum: 80 (very high)
- RSI MA Minimum: 75
- Stop Loss: ATR × 1.5 (tighter)
- Trailing Stop: 1.0%

**When to Use:**
- In volatile markets with price spikes
- When you want fewer, higher-quality setups
- For momentum exhaustion plays

**When to Avoid:**
- In low volatility environments
- When you need consistent trade frequency

**Expected Performance:**
- Win Rate: 60-70%
- Profit Factor: 1.5-2.2
- Trades: Few but high conviction

---

### Mean Reversion Classic

**Focus:** Standard Bollinger Band mean reversion

**Description:** A classic implementation of BB mean reversion without trailing stops or complex exit logic. Relies purely on price returning to the mean (mid band).

**Key Parameters:**
- Standard BB/KC parameters (20/2.0)
- RSI Minimum: 70
- Use RSI Relation: No
- Entry Band Mode: BB only
- Exit Level: Mid
- No Trailing Stop

**When to Use:**
- As a baseline for comparison
- In stable, range-bound markets
- When you want simple, understandable logic

**When to Avoid:**
- In trending markets
- When more sophisticated risk management is needed

**Expected Performance:**
- Win Rate: 50-60%
- Profit Factor: 1.3-1.8
- Max Drawdown: 10-18%

---

## Customizing Presets

Presets are starting points. You can:

1. **Load a preset** as a base configuration
2. **Modify parameters** in the sidebar to fine-tune
3. **Run backtests** to compare performance
4. **Use optimization** to find better parameter combinations

### Tips for Customization

1. **Start conservative** - Begin with Conservative or Low Drawdown presets
2. **Test incrementally** - Change one parameter at a time
3. **Match market conditions** - Use Scalping in choppy markets, Swing in trending
4. **Monitor drawdown** - Always check max drawdown, not just returns
5. **Consider trade frequency** - More trades = better statistical significance

---

## Parameter Sensitivity

### Most Impactful Parameters

1. **RSI Minimum** - Higher = fewer but higher quality signals
2. **Entry Band Mode** - "Both" is most conservative, "Either" most aggressive
3. **Stop Loss %** - Tighter = more losses but smaller; wider = fewer but larger
4. **Exit Level** - "Mid" = quicker exits; "Lower" = larger targets

### Least Sensitive Parameters

1. **BB Basis Type** - SMA vs EMA has minor impact
2. **RSI MA Length** - Usually stable between 8-12
3. **KC Mid Type** - EMA vs SMA similar performance

---

## Backtesting Recommendations

1. **Test on sufficient data** - Minimum 6 months of historical data
2. **Check trade count** - Need at least 30+ trades for statistical significance
3. **Verify across markets** - Test on different symbols/timeframes
4. **Account for slippage** - Real fills may differ from backtested prices
5. **Include commissions** - Set realistic commission rates

---

## Next Steps

- See [OPTIMIZATION.md](OPTIMIZATION.md) for parameter optimization guide
- Review the main [README.md](../README.md) for setup instructions
- Use the UI optimization tool to find custom configurations
