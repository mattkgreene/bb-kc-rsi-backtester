#!/usr/bin/env python
"""
Test script to verify the Dash backtest functionality works correctly.
"""
import sys
import datetime as dt
import pandas as pd

sys.path.insert(0, 'app')

from ui.dash_app import (
    _default_ui_values,
    _build_params,
    run_backtest_callback,
    update_dashboard,
    update_trades,
)
from core.data import fetch_ohlcv_range_db_cached

def test_params_initialization():
    """Test that default params are built correctly."""
    print("\n=== Testing Params Initialization ===")
    defaults = _default_ui_values()
    params = _build_params(defaults)

    assert params is not None, "Params should not be None"
    assert len(params) > 0, "Params should have keys"
    assert 'exchange' in params, "Params should have exchange"
    assert 'symbol' in params, "Params should have symbol"
    assert 'timeframe' in params, "Params should have timeframe"

    print(f"✓ Params initialized with {len(params)} keys")
    print(f"  Exchange: {params['exchange']}")
    print(f"  Symbol: {params['symbol']}")
    print(f"  Timeframe: {params['timeframe']}")
    return params

def test_data_fetch(params):
    """Test that data can be fetched."""
    print("\n=== Testing Data Fetch ===")

    # Use a smaller date range for testing
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=30)

    df = fetch_ohlcv_range_db_cached(
        params['exchange'],
        params['symbol'],
        params['timeframe'],
        start_ts=start,
        end_ts=end,
        db_path='data/market_data.db'
    )

    assert not df.empty, "DataFrame should not be empty"
    assert 'Close' in df.columns, "DataFrame should have Close column"

    print(f"✓ Fetched {len(df)} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")

def test_backtest_callback():
    """Test the backtest callback with minimal params."""
    print("\n=== Testing Backtest Callback ===")

    defaults = _default_ui_values()
    # Use smaller date range for faster test
    defaults['w_start_date'] = dt.date.today() - dt.timedelta(days=30)
    defaults['w_end_date'] = dt.date.today()

    params = _build_params(defaults)

    # Simulate button click
    n_clicks = 1

    print(f"Running backtest callback with n_clicks={n_clicks}...")
    results, last_params, dirty, selected_trade, status = run_backtest_callback(n_clicks, params)

    assert results is not None, "Results should not be None"
    assert not isinstance(results, dict) or not results.get('error'), f"Backtest failed: {results.get('error') if isinstance(results, dict) else 'unknown'}"

    print(f"✓ Backtest callback succeeded")
    print(f"  Status: {status}")

    if isinstance(results, dict):
        print(f"  Has df: {'df' in results}")
        print(f"  Has ds: {'ds' in results}")
        print(f"  Has trades: {'trades' in results}")
        print(f"  Has stats: {'stats' in results}")

    return results

def test_dashboard_update(results):
    """Test that dashboard updates correctly with results."""
    print("\n=== Testing Dashboard Update ===")

    message, metrics, figure = update_dashboard(results, ['on'], ['on'], None)

    assert message == "" or isinstance(message, str), "Message should be empty or string"
    assert isinstance(metrics, list), "Metrics should be a list"
    # Figure can be either dict or Figure object
    assert figure is not None, "Figure should not be None"

    print(f"✓ Dashboard update succeeded")
    print(f"  Message: {message}")
    print(f"  Metrics count: {len(metrics)}")
    print(f"  Figure type: {type(figure).__name__}")

def test_trades_update(results):
    """Test that trades table updates correctly."""
    print("\n=== Testing Trades Update ===")

    (trades_msg, trades_data, trades_cols,
     diag_msg, diag_data, diag_cols) = update_trades(results)

    print(f"✓ Trades update succeeded")
    print(f"  Trades message: {trades_msg}")
    print(f"  Trades rows: {len(trades_data)}")
    print(f"  Trades columns: {len(trades_cols)}")
    print(f"  Diagnostics rows: {len(diag_data)}")

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Dash Backtest Functionality Test")
        print("=" * 60)

        params = test_params_initialization()
        test_data_fetch(params)
        results = test_backtest_callback()
        test_dashboard_update(results)
        test_trades_update(results)

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
