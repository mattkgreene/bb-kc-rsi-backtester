# Compute entrypoints inventory

This project already has compute modules that can be invoked from a backend worker.

## Optimization
- `app/optimization/grid_search.py`
  - `run_grid_search(df, param_grid, base_params, ...)`
  - `run_walk_forward_grid_search(df, param_grid, base_params, ...)`

## Strategy discovery
- `app/discovery/engine.py`
  - `run_discovery(df, base_params, db, config, ...)`
  - `run_discovery_parallel(df, base_params, db, config, n_workers, ...)`

## Leaderboard
- `app/discovery/leaderboard.py`
  - `Leaderboard.get_top(n, sort_by, min_trades)`
  - `Leaderboard.get_stats()`

## Pattern recognition
- `app/discovery/rules.py`
  - `find_winning_patterns(db, min_confidence, min_occurrence_pct)` (persists `discovered_rules`)

## Market data persistence (SQLite cache)
- `app/core/data.py`
  - `fetch_ohlcv_range_db_cached(exchange, symbol, timeframe, start_ts, end_ts, db_path=...)`
- `app/core/ohlcv_cache.py`
  - `get_cache_bounds(exchange, symbol, timeframe, db_path=...)`

