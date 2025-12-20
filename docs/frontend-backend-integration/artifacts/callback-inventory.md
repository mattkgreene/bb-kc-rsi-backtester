# Callback inventory (heavy compute → backend jobs)

## Optimization
- Previous: `run_optimization()` ran `run_grid_search()` / `run_walk_forward_grid_search()` inline.
- Now:
  - `run_optimization()` enqueues `POST /v1/optimize` and stores `store-opt-job`.
  - `poll_optimization_job()` polls `GET /v1/jobs/{id}` and writes `store-opt-results` / `store-wf-results`.

## Strategy discovery
- Previous: `run_discovery_callback()` ran `run_discovery()` / `run_discovery_parallel()` inline.
- Now:
  - `run_discovery_callback()` enqueues `POST /v1/discover` and stores `store-disc-job`.
  - `poll_discovery_job()` polls and updates `disc-status` / `disc-results`.

## Leaderboard
- Previous: `update_leaderboard()` computed leaderboard by opening SQLite DB in the UI.
- Now:
  - `enqueue_leaderboard_job()` enqueues `POST /v1/leaderboard/refresh` and stores `store-lb-job`.
  - `poll_leaderboard_job()` polls and stores `store-lb-snapshot`.
  - `update_leaderboard()` renders from `store-lb-snapshot` (or latest snapshot loaded on tab entry).

## Pattern recognition
- Previous: `update_patterns()` ran `find_winning_patterns()` inline and read SQLite in the UI.
- Now:
  - `enqueue_patterns_job()` enqueues `POST /v1/patterns/refresh` and stores `store-pattern-job`.
  - `poll_patterns_job()` polls and refreshes `store-pattern-rules` via `GET /v1/patterns`.
  - `render_patterns()` renders summary from `store-pattern-rules`.

