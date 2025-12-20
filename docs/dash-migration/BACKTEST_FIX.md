# Backtest Button Fix & Loading Indicators

## Problem
The "Run Backtest" button in the Dash UI was not working because:
1. `store-params` was not initialized with default values on page load
2. No loading indicators were shown during backtest execution
3. No clear status messages to inform the user what was happening

## Solution

### 1. Initialize store-params with defaults ([dash_app.py:302](app/ui/dash_app.py#L302))
```python
# Before:
dcc.Store(id="store-params"),

# After:
dcc.Store(id="store-params", data=_build_params(DEFAULTS)),
```

This ensures that clicking "Run Backtest" immediately after page load works without requiring the user to modify any input first.

### 2. Add loading indicator for Run Backtest button ([dash_app.py:320-324](app/ui/dash_app.py#L320-L324))
```python
html.Button("Run Backtest", id="run-backtest", n_clicks=0, className="btn-primary"),
dcc.Loading(
    id="loading-backtest",
    type="default",
    children=html.Div(id="run-status", style={"marginTop": "8px"}),
),
```

### 3. Add loading indicators for data tables ([dash_app.py:573-603](app/ui/dash_app.py#L573-L603))
Wrapped both the trades table and diagnostics table with `dcc.Loading` components to show spinners while data is loading.

### 4. Enhanced status messages
Updated the backtest callback to provide clear status messages:
- `✓ Backtest complete! X trades executed.` - Success
- `✓ Backtest loaded from cache.` - Cache hit
- `✗ No data in selected range.` - Error (no data)
- `✗ Backtest failed: <error>` - Error (exception)

### 5. Added debug logging
Added console logging throughout the backtest callback to help debug issues:
- `[BACKTEST] Starting backtest with X params`
- `[BACKTEST] Fetching data for <exchange> <symbol>...`
- `[BACKTEST] Fetched X rows`
- `[BACKTEST] Running backtest with X params...`
- `[BACKTEST] Complete! Trades: X, DS rows: X`
- `[DASHBOARD] update_dashboard called...`

## Testing

Created comprehensive test script `test_dash_backtest.py` that verifies:
- ✓ Params initialization works
- ✓ Data fetching works
- ✓ Backtest callback executes successfully
- ✓ Dashboard updates with results
- ✓ Trades table populates correctly

All tests pass successfully.

## Files Modified
- `app/ui/dash_app.py` - Main fixes and improvements
- `test_dash_backtest.py` - New test script

## How to Use
1. Start the Dash app: `.venv311/bin/python app/ui/dash_app.py`
2. Open browser to http://localhost:8050
3. Click "Run Backtest" - it will now work immediately!
4. Watch for loading indicators while backtest runs
5. Check the status message below the button for feedback

## Next Steps
Consider adding:
- Progress bar for long-running backtests
- Estimated time remaining
- Cancel button for running backtests
- Toast notifications for completion
