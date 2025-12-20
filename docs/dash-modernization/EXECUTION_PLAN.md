# Execution Plan

> **Project Title:** Dash Modernization + Backtest Fix + OHLCV Cache
> **Owner:** Matthew Greene • **Date:** 2025-12-19 • **Version:** 0.2.0
> **Status:** In Progress

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

## 1) Objective & Scope
- **Objective:** Modernize the Dash UI, fix backtest execution, and add a 3+ year SQLite OHLCV cache for BTC with incremental updates.
- **Out of Scope:** Strategy logic changes, new indicators, optimizer algorithm changes.

## 2) Acceptance Criteria (Go/No-Go)
- **AC-1:** Dash backtest runs end-to-end with visible stats, chart, and tables.
- **AC-2:** OHLCV cache loads on startup and updates incrementally before each run.
- **AC-3:** UI updated with modern styling and responsive layout.
- **Monitoring/Telemetry to validate:** Manual UI smoke test; cache hit/miss logs.

## 3) Preconditions & Dependencies
- Preconditions: Access to CCXT network for initial data fill or a local cache file.
- Dependencies: SQLite, pandas, ccxt.
- Backout window: Streamlit remains available as rollback.
- Freeze rules: No changes to backtest engine logic.

## 4) Step-by-Step Plan (each is testable)
> **Format per step:** ID, goal, commands, expected result, test, evidence, rollback.

### S-101: Fix Dash backtest execution path
- Goal: Ensure run button triggers data fetch and backtest with validated params.
- Commands/Actions:
  - Add logging/validation in Dash callback.
  - Ensure required params (exchange/symbol/timeframe/start/end) exist.
- Expected Result: Backtest completes or returns clear error.
- **Test T-101:** Trigger callback with synthetic inputs and check result payload.
- Evidence: `docs/dash-modernization/artifacts/backtest-trace.txt`
- Rollback: Revert callback changes to previous version.

### S-102: Implement SQLite OHLCV cache
- Goal: Store 3+ years of BTC data and support incremental updates.
- Commands/Actions:
  - Add cache module with schema + upsert + range read.
  - Add range bounds query to decide missing data.
- Expected Result: Cache table exists and read/write passes.
- **Test T-102:** Insert sample OHLCV and read back range.
- Evidence: `docs/dash-modernization/artifacts/ohlcv-schema.md`
- Rollback: Remove new module and references.

### S-103: Integrate cache into Dash data fetch
- Goal: Use DB-first fetch on startup and before backtest runs.
- Commands/Actions:
  - On app start, warm BTC/USD 30m for last 3 years in background.
  - Before run, fetch missing edges and upsert into cache.
- Expected Result: Cache hit for existing data, incremental fetch when needed.
- **Test T-103:** Run twice; second run uses cache for same range.
- Evidence: `docs/dash-modernization/artifacts/ohlcv-cache-smoke.txt`
- Rollback: Switch back to direct CCXT fetch only.

### S-104: Modernize Dash UI
- Goal: Apply modern visual styling and improved layout.
- Commands/Actions:
  - Add `app/ui/assets/` CSS with typography, cards, gradients.
  - Add class names to layout containers.
  - Ensure responsive layout for mobile.
- Expected Result: UI visually updated and usable on desktop/mobile.
- **Test T-104:** Manual UI review on desktop/mobile widths.
- Evidence: `docs/dash-modernization/artifacts/ui-style-notes.md`
- Rollback: Remove assets and class names.

### S-105: Update documentation
- Goal: Document cache behavior and Dash UI usage.
- Commands/Actions:
  - Update README or relevant docs.
- Expected Result: Docs describe Dash run + cache behavior.
- **Test T-105:** Verify docs mention cache + run steps.
- Evidence: `docs/dash-modernization/artifacts/docs-update.txt`
- Rollback: Revert doc changes.

## 5) Risks & Mitigations (Execution)
| ID | Risk | Trigger | Mitigation | Owner |
|----|------|---------|------------|-------|
| R-201 | Cache fill takes too long | Long startup | Background thread + progress text | Codex |
| R-202 | CCXT failures stop UI | Network errors | Fallback to cached data | Codex |
| R-203 | CSS causes layout regressions | Mobile clipping | Responsive CSS + quick fixes | Codex |

## 6) Communication Plan
- Before: Confirm cache location and UI style direction.
- During: Report after S-102 and S-104.
- After: Summarize changes and validation status.

## 7) Approval
- Change ticket: TBD
- Approvers: Matthew Greene
- Scheduled window: TBD
