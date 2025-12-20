# Execution Plan

> **Project Title:** Dash Migration for BB + KC + RSI Backtester
> **Owner:** Matthew Greene • **Date:** 2025-12-19 • **Version:** 0.1.0
> **Status:** In Progress

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

## 1) Objective & Scope
- **Objective:** Replace the Streamlit UI with a Dash UI while preserving feature parity and reusing existing backtest logic.
- **Out of Scope:** Strategy logic changes, new exchanges, optimizer changes, or data pipeline rewrites.

## 2) Acceptance Criteria (Go/No-Go)
- **AC-1:** Dash UI supports all current inputs/outputs and can run a backtest end-to-end.
- **AC-2:** Results from Dash match Streamlit for identical parameters (within tolerance for floating point).
- **Monitoring/Telemetry to validate:** Manual UI smoke tests, logs, and comparison of key stats.

## 3) Preconditions & Dependencies
- Preconditions: Investigation artifacts complete (H-001 to H-003 validated), approval to add Dash dependencies.
- Dependencies: `dash`, `plotly`, optional caching middleware (`flask-caching` or `diskcache`).
- Backout window: Keep Streamlit entrypoint intact for immediate rollback.
- Freeze rules: Avoid changes to strategy logic during UI migration.

## 4) Step-by-Step Plan (each is testable)
> **Format per step:** ID, goal, commands, expected result, test, evidence, rollback.

### S-101: Add Dash skeleton and entrypoint
- Goal: Stand up a minimal Dash app entrypoint (supports H-001, D-002).
- Commands/Actions:
  - Add `app/ui/dash_app.py` with minimal layout and server run block.
  - Add Dash dependency to requirements.
- Expected Result: Dash app starts and serves a page.
- **Test T-101:** Run `python app/ui/dash_app.py` and load `http://localhost:8050` (page renders).
- Evidence: `docs/dash-migration/artifacts/dash-skeleton.txt`
- Rollback: Remove `dash_app.py` and Dash dependency; confirm Streamlit still runs.

### S-102: Define UI defaults and state store
- Goal: Centralize default parameters and state handling (supports H-001, H-002).
- Commands/Actions:
  - Extract defaults into a shared module (e.g., `app/ui/ui_defaults.py`).
  - Add `dcc.Store` to persist UI state across callbacks.
- Expected Result: UI loads with defaults and state is accessible in callbacks.
- **Test T-102:** Load Dash UI and verify defaults appear in controls.
- Evidence: `docs/dash-migration/artifacts/dash-defaults.txt`
- Rollback: Revert new module and `dcc.Store` wiring; keep skeleton layout.

### S-103: Port sidebar controls and parameter mapping
- Goal: Recreate Streamlit input controls as Dash components (supports H-001).
- Commands/Actions:
  - Map each Streamlit widget to a Dash component and update state.
  - Implement callbacks to update state on user input.
- Expected Result: All inputs appear and update state without errors.
- **Test T-103:** Change each control; verify state updates and no callback errors.
- Evidence: `docs/dash-migration/artifacts/dash-controls.txt`
- Rollback: Revert layout/callback changes for the control panel.

### S-104: Wire backtest execution
- Goal: Trigger backtest runs from Dash with selected parameters (supports H-002).
- Commands/Actions:
  - Add a Run button callback to call core backtest functions.
  - Add loading indicator to avoid UI confusion during long runs.
- Expected Result: Clicking Run triggers backtest and returns stats/results.
- **Test T-104:** Run a backtest with a small date range; verify outputs are populated.
- Evidence: `docs/dash-migration/artifacts/dash-backtest-smoke.txt`
- Rollback: Disable the callback and return to a stub state.

### S-105: Render charts and results tables
- Goal: Display Plotly chart and results tables in Dash (supports H-003).
- Commands/Actions:
  - Reuse existing Plotly figure code.
  - Render stats/trades tables with Dash components.
- Expected Result: Chart renders and tables show results.
- **Test T-105:** Run backtest and verify chart/table contents are visible.
- Evidence: `docs/dash-migration/artifacts/dash-plotly-results.txt`
- Rollback: Remove chart/table components and related callbacks.

### S-106: Add caching for data and backtests
- Goal: Preserve performance benefits of Streamlit caching (supports H-002).
- Commands/Actions:
  - Implement caching around data fetch and backtest calls.
  - Log cache hits/misses for verification.
- Expected Result: Repeated runs are faster and cache hits occur.
- **Test T-106:** Run the same backtest twice; second run shows cache hit and faster response.
- Evidence: `docs/dash-migration/artifacts/dash-cache.txt`
- Rollback: Disable caching and rely on direct calls.

### S-107: Update docs and deployment config
- Goal: Document how to run Dash and preserve rollback path.
- Commands/Actions:
  - Update README and Docker/railway configs with Dash entrypoint.
  - Keep Streamlit path documented for rollback.
- Expected Result: Docs reference Dash run commands and deploy path.
- **Test T-107:** Review docs and confirm commands are present and accurate.
- Evidence: `docs/dash-migration/artifacts/docs-update.txt`
- Rollback: Revert documentation and config changes.

## 5) Risks & Mitigations (Execution)
| ID | Risk | Trigger | Mitigation | Owner |
|----|------|---------|------------|-------|
| R-201 | Callback complexity causes bugs | Frequent callback errors | Modularize callbacks and add logging | Codex |
| R-202 | Long backtests freeze UI | Run button locks up | Add loading state or async job | Codex |
| R-203 | Cache mismatch with Streamlit | Inconsistent results | Validate cache keys and bypass option | Codex |

## 6) Communication Plan
- Before: Confirm scope and acceptance criteria with owner.
- During: Update after each major step (S-101, S-104, S-107).
- After: Summarize results, gaps, and next steps.

## 7) Approval
- Change ticket: TBD
- Approvers: Matthew Greene
- Scheduled window: TBD
