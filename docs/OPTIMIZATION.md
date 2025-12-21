# Parameter Optimization Guide

This guide explains how to use the grid search optimization tool to find optimal parameter configurations for the BB+KC+RSI strategy.

## Overview

Parameter optimization systematically tests combinations of parameter values to find configurations that maximize a target metric (like profit factor). The tool uses **grid search**, which exhaustively evaluates all combinations within specified ranges.

## Quick Start

1. Run a backtest with default parameters first
2. Open the "Parameter Optimization" expander
3. Set your parameter ranges
4. Choose your optimization metric
5. Click "Run Optimization"
6. Review results and apply best configuration

---

## Understanding Grid Search

### How It Works

Grid search creates a "grid" of parameter combinations and tests each one:

```
RSI Min:   [65, 70, 75]     → 3 values
Stop %:    [1.5, 2.0, 2.5]  → 3 values
Entry Mode: [Either, Both]  → 2 values

Total combinations: 3 × 3 × 2 = 18 backtests
```

Each combination runs a full backtest, and results are ranked by your chosen metric.

### Optimization Metrics

| Metric | Description | Best For |
|--------|-------------|----------|
| **profit_factor** | Gross profit / Gross loss | Balanced risk/reward |
| **sharpe_ratio** | Risk-adjusted return (annualized) | Risk-conscious traders |
| **sortino_ratio** | Downside-adjusted return | Drawdown-sensitive strategies |
| **win_rate** | Percentage of winning trades | Psychological comfort |
| **total_equity_return_pct** | Total portfolio return | Maximum growth |

**Recommendation:** Start with `profit_factor` as it balances win rate and reward size.

---

## Using the UI Optimizer

### Step 1: Set Parameter Ranges

**RSI Minimum Range:**
- Conservative: 70-80 (stricter = fewer but better signals)
- Aggressive: 60-70 (looser = more signals)

**Stop Loss % Range:**
- Tight: 1.0-2.0% (quick stops, more losses)
- Wide: 2.0-4.0% (room to breathe, larger losses)

**Band Multiplier Range:**
- Narrow bands: 1.5-2.0 (more signals)
- Wide bands: 2.0-2.5 (fewer signals)

### Step 2: Configure Grid Density

**Grid Steps:** Controls how many values to test within each range

| Steps | Values per Range | Total Combinations (typical) |
|-------|------------------|------------------------------|
| 2 | Min, Max | ~50 |
| 3 | Min, Mid, Max | ~400 |
| 4 | Quartiles | ~2,000 |
| 5 | Quintiles | ~10,000 |

**Tip:** Start with 3 steps, then refine with 4-5 around promising areas.

### Step 3: Optional Toggles

- **Test all entry band modes:** Tests Either, KC, BB, Both (4x combinations)
- **Test both exit levels:** Tests mid and lower exits (2x combinations)

### Step 4: Set Minimum Trades

Filter out configurations with too few trades:
- **5 trades:** Shows all viable configs (may include noise)
- **10 trades:** More reliable results
- **20+ trades:** Higher statistical confidence

---

## Interpreting Results

### Results Table

The top configurations are displayed with:
- **Parameter values** used
- **Performance metrics** achieved
- **Trade count** for statistical reference

### What to Look For

1. **Consistency:** Are top results clustered around similar parameter values?
2. **Trade Count:** More trades = more reliable results
3. **Drawdown:** High return with high drawdown may not be desirable
4. **Profit Factor:** Anything above 1.5 is good, above 2.0 is excellent

### Analysis Section

The optimizer provides:
- **Best Configuration:** Top-ranked parameters
- **Common in Top 10:** Parameters that appear frequently in best results
- **Top Profit Factor:** Best PF achieved
- **Avg PF (Top 10):** Average PF of top 10 configs

---

## Avoiding Overfitting

### What is Overfitting?

Overfitting occurs when parameters are so precisely tuned to historical data that they fail on new data. Signs include:
- Very high backtested returns
- Very specific parameter values (e.g., RSI = 73.5)
- Results that don't generalize to other time periods

### Prevention Techniques

#### 1. Use Sufficient Data

| Timeframe | Minimum History |
|-----------|-----------------|
| 30m | 6+ months |
| 1h | 1+ year |
| 4h | 2+ years |
| 1d | 3+ years |

#### 2. Require Minimum Trades

- At least 30 trades for basic confidence
- 50+ trades for robust results
- 100+ trades for high confidence

#### 3. Out-of-Sample Testing

Split your data:
1. **Optimization period:** Use 70% of data for grid search
2. **Validation period:** Test best configs on remaining 30%

Example:
```
Full data: Jan 2022 - Dec 2024
Optimize on: Jan 2022 - Jun 2024
Validate on: Jul 2024 - Dec 2024
```

#### 4. Walk-Forward Analysis

For more robust validation:
1. Optimize on months 1-12
2. Test on month 13
3. Re-optimize on months 2-13
4. Test on month 14
5. Continue rolling forward

#### 5. Parameter Stability

Look for **stable regions** where nearby parameters produce similar results:

```
Good (stable):
  RSI 68 → PF 1.8
  RSI 70 → PF 1.9  ← Best
  RSI 72 → PF 1.7

Bad (fragile):
  RSI 68 → PF 0.9
  RSI 70 → PF 2.5  ← Suspicious
  RSI 72 → PF 1.0
```

#### 6. Simplicity Principle

When two configurations perform similarly, prefer:
- Fewer modified parameters
- Round numbers (RSI 70 vs 73)
- Wider parameter ranges

---

## Optimization Workflow

### Recommended Process

```
1. Baseline Test
   └── Run with default preset
   └── Note performance metrics

2. Coarse Grid Search
   └── Wide parameter ranges
   └── 3 steps
   └── Identify promising regions

3. Fine Grid Search
   └── Narrow ranges around best results
   └── 4-5 steps
   └── Find optimal configuration

4. Validation
   └── Test on different date range
   └── Test on different symbol
   └── Compare to baseline

5. Deployment
   └── Use validated configuration
   └── Monitor live performance
   └── Re-optimize periodically
```

### Example Session

**Step 1: Coarse Search**
```
RSI Range: 60-80
Stop Range: 1.0-4.0%
Steps: 3
→ Best RSI around 70, Stop around 2%
```

**Step 2: Fine Search**
```
RSI Range: 68-74
Stop Range: 1.5-2.5%
Steps: 4
→ Best: RSI 71, Stop 2.0%
```

**Step 3: Validate**
```
Test best config on different date range
→ If performance holds, use it
→ If not, return to Step 1 with different data
```

---

## Programmatic Usage

For advanced users, the grid search can be used programmatically:

```python
from optimization.grid_search import run_grid_search, create_custom_grid

# Create custom parameter grid
param_grid = create_custom_grid(
    rsi_range=(65, 75),
    stop_range=(1.5, 3.0),
    band_mult_range=(1.8, 2.2),
    steps=4,
    include_entry_modes=True,
    include_exit_levels=False,
    stop_mode=current_params.get("stop_mode", "Fixed %"),
)

# Run optimization
results = run_grid_search(
    df=ohlcv_data,
    param_grid=param_grid,
    base_params=current_params,
    metric="profit_factor",
    min_trades=10,
    top_n=20,
)

# Get best configuration
best_config = results.iloc[0].to_dict()
```

### Custom Parameter Grids

```python
# Manual grid definition
custom_grid = {
    "rsi_min": [65, 68, 70, 72, 75],
    "rsi_ma_min": [65, 68, 70],
    "entry_band_mode": ["Either", "Both"],
    "stop_pct": [1.5, 2.0, 2.5],
}

results = run_grid_search(df, custom_grid, base_params)
```

When `stop_mode` is `"ATR"`, use `stop_atr_mult` in the grid instead of `stop_pct`.

---

## Performance Tips

### Speed Optimization

1. **Start small:** Use QUICK_PARAM_GRID for initial tests
2. **Reduce steps:** 3 steps is usually sufficient for coarse search
3. **Limit parameters:** Test 3-4 parameters at a time
4. **Shorter date range:** Use 6 months instead of 2 years for initial tests

### Memory Management

- Grid search runs many backtests
- Clear results after analysis
- Use smaller date ranges for large grids

---

## Common Mistakes

### 1. Optimizing on Too Little Data
**Problem:** 3 months of data, 10 trades
**Solution:** Use more history, require 30+ trades

### 2. Ignoring Drawdown
**Problem:** 200% return with 60% drawdown
**Solution:** Include max_drawdown in evaluation criteria

### 3. Over-Tuning
**Problem:** Parameters like RSI = 71.34
**Solution:** Round to whole numbers, test stability

### 4. No Validation
**Problem:** Only testing on optimization period
**Solution:** Always validate on unseen data

### 5. Ignoring Trade Count
**Problem:** Amazing PF with 3 trades
**Solution:** Require minimum 20-30 trades

---

## Next Steps

- Review [STRATEGIES.md](STRATEGIES.md) for preset descriptions
- Start with a preset, then optimize around it
- Keep records of what you test and results
- Re-optimize periodically as market conditions change
