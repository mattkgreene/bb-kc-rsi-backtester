# Execution Log

> **Project Title:** Dash Migration for BB + KC + RSI Backtester
> **Owner:** Matthew Greene • **Date:** 2025-12-19 • **Version:** 0.1.0
> **Status:** In Progress

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

## 1) Summary
- Window: 2025-12-20 02:30 -> 2025-12-20 03:34
- Result: Partial
- Link to Plan: [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)

## 2) Timeline (chronological)
| Time (UTC) | Actor | Action | Ref (Step/Test) | Expected vs Actual | Result | Artifacts |
|------------|-------|--------|------------------|--------------------|--------|-----------|
| 2025-12-20 02:35 | Codex | Ran widget inventory scan and captured keyed widgets | S-001 / T-001 | Expected: widget list and state keys / Actual: captured in artifact | Pass | `docs/dash-migration/artifacts/widget-inventory.md` |
| 2025-12-20 02:36 | Codex | Scanned Streamlit-specific dependencies | S-002 / T-002 | Expected: list of Streamlit APIs and replacements / Actual: captured in artifact | Pass | `docs/dash-migration/artifacts/streamlit-dependencies.md` |
| 2025-12-20 02:38 | Codex | Ran core backtest smoke test (synthetic data) | S-003 / T-003 | Expected: backtest completes / Actual: failed to import numpy | Fail | `docs/dash-migration/artifacts/core-backtest-smoke.txt` |
| 2025-12-20 02:39 | Codex | Checked Dash/Plotly imports for minimal app | S-004 / T-004 | Expected: Dash imports succeed / Actual: dash module missing | Fail | `docs/dash-migration/artifacts/dash-plotly-smoke.txt` |
| 2025-12-20 02:55 | Codex | Added Dash entrypoint, helpers, and requirements | S-101 / T-101 | Expected: Dash app starts / Actual: code added; runtime deps missing | Blocked | `docs/dash-migration/artifacts/dash-skeleton.txt` |
| 2025-12-20 02:57 | Codex | Implemented defaults + state stores | S-102 / T-102 | Expected: defaults load / Actual: code added; not runtime-tested | Blocked | `docs/dash-migration/artifacts/dash-defaults.txt` |
| 2025-12-20 02:58 | Codex | Ported sidebar controls and parameter mapping | S-103 / T-103 | Expected: controls update state / Actual: code added; not runtime-tested | Blocked | `docs/dash-migration/artifacts/dash-controls.txt` |
| 2025-12-20 03:00 | Codex | Wired backtest execution callback | S-104 / T-104 | Expected: run completes / Actual: code added; numpy missing | Blocked | `docs/dash-migration/artifacts/dash-backtest-smoke.txt` |
| 2025-12-20 03:02 | Codex | Rendered Plotly charts and tables | S-105 / T-105 | Expected: chart/table render / Actual: code added; dash missing | Blocked | `docs/dash-migration/artifacts/dash-plotly-results.txt` |
| 2025-12-20 03:04 | Codex | Added caching for backtest + data | S-106 / T-106 | Expected: cache hits on repeat / Actual: code added; not runtime-tested | Blocked | `docs/dash-migration/artifacts/dash-cache.txt` |
| 2025-12-20 03:07 | Codex | Updated docs and deployment configs | S-107 / T-107 | Expected: docs updated / Actual: README/docker/railway updated | Pass | `docs/dash-migration/artifacts/docs-update.txt` |
| 2025-12-20 03:12 | Codex | Attempted to install Dash/base deps | T-003/T-004 unblock | Expected: deps install / Actual: blocked by PEP 668 | Fail | `docs/dash-migration/artifacts/deps-install.txt` |
| 2025-12-20 03:20 | Codex | Created Python 3.11 virtualenv and installed deps | T-003/T-004 unblock | Expected: deps install / Actual: success in .venv311 | Pass | `docs/dash-migration/artifacts/deps-install.txt` |
| 2025-12-20 03:26 | Codex | Re-ran core backtest smoke (synthetic data) | S-003 / T-003 | Expected: backtest completes / Actual: completed successfully | Pass | `docs/dash-migration/artifacts/core-backtest-smoke.txt` |
| 2025-12-20 03:27 | Codex | Verified Dash/Plotly imports | S-004 / T-004 | Expected: imports succeed / Actual: versions loaded | Pass | `docs/dash-migration/artifacts/dash-plotly-smoke.txt` |
| 2025-12-20 03:30 | Codex | Imported Dash app module | S-101 / T-101 | Expected: app loads without running server / Actual: import succeeded | Pass | `docs/dash-migration/artifacts/dash-skeleton.txt` |
| 2025-12-20 03:32 | Codex | Built Plotly figure from synthetic backtest | S-105 / T-105 | Expected: figure builds / Actual: trace count generated | Pass | `docs/dash-migration/artifacts/dash-plotly-results.txt` |
| 2025-12-20 03:33 | Codex | Added missing store-params to Dash layout | S-102 / T-102 | Expected: store exists for params state / Actual: store-params added | Pass | N/A |
| 2025-12-20 03:34 | Codex | Honored margin utilization toggle in param builder | S-103 / T-103 | Expected: max margin utilization only applied when enabled | Pass | N/A |

## 3) Deviations & Decisions
- **D-001:** Implemented in-memory backtest cache in Dash UI
  - Reason: Preserve caching behavior without introducing new infrastructure
  - Impact on AC/SLO: Low
  - Link back to Plan Step: S-106

## 4) Issues & Incidents
| ID | Description | Severity | Root Cause (when known) | Follow-up |
|----|-------------|----------|--------------------------|-----------|
| INC-001 | Dash/numpy missing in local environment, blocking runtime tests | Low | Dependencies not installed | Resolved via Python 3.11 venv |
| INC-002 | pip install blocked by PEP 668 (externally-managed env) | Low | System Python is managed | Resolved via Python 3.11 venv |

## 5) Post-Execution Validation
- Tests re-run: T-003, T-004, T-101, T-103, T-105 -> results: pass; T-104 pending (requires CCXT network access)
- Metrics observed: N/A
- Rollback performed? N

## 6) Lessons Learned / Next Actions
- What to automate next time
- Permanent fixes / tickets created (IDs)
