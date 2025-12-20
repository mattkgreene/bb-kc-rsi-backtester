# Investigation Plan

> **Project Title:** Dash Migration for BB + KC + RSI Backtester
> **Owner:** Matthew Greene • **Date:** 2025-12-19 • **Version:** 0.1.0
> **Status:** In Progress

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

## 1) Problem Statement
- **Context:** The current UI is built with Streamlit and we want a more robust Dash UI while keeping the backtest engine intact.
- **Desired Outcome:** A Dash UI plan that reaches feature parity and enables future UI expansion without changing strategy logic.
- **Constraints/Assumptions:**
  - Core backtest, data, and indicator modules under `app/` remain unchanged.
  - Plotly figures should be reused where possible.
  - Keep Streamlit entrypoint intact until Dash parity is verified.

## 2) Stakeholders & RACI
| Role | Person/Agent | Responsibility |
|------|--------------|----------------|
| R | Codex | Draft investigation artifacts and findings |
| A | Matthew Greene | Approve scope and decisions |
| C | TBD | UI/infra consultation if needed |
| I | TBD | Users/stakeholders |

## 3) Hypotheses
- **H-001:** Dash components can replace Streamlit widgets with equivalent UX and inputs.
  - Evidence to confirm/refute: widget inventory and mapping table.
- **H-002:** Core backtest/data code has no Streamlit dependencies and can be called from Dash callbacks.
  - Evidence to confirm/refute: static scan plus a standalone core backtest smoke test.
- **H-003:** Existing Plotly figures can be reused in Dash with minimal adjustments.
  - Evidence to confirm/refute: render an existing Plotly fig in a minimal Dash app.

## 4) Questions to Answer
- **Q-001:** Which Streamlit features (session state, caching, layout) need Dash replacements? → linked test(s): T-002
- **Q-002:** Should Dash be single-page or multi-page for this UI? → linked test(s): T-003
- **Q-003:** How should long-running backtests avoid blocking the UI? → linked test(s): T-004

## 5) Investigation Tasks (test-first)
- **S-001:** Inventory all Streamlit widgets, outputs, and state keys in `app/ui/app.py`.
  - Expected signals: complete list of inputs, outputs, and session_state keys.
  - Test T-001: `rg -n "st\\." app/ui/app.py` and document widgets/state keys.
  - Artifacts: `docs/dash-migration/artifacts/widget-inventory.md`
  - Risk R-001: Missing widget leads to feature loss.
- **S-002:** Identify Streamlit-only behaviors (caching, expander, downloads, session state).
  - Expected signals: list of Streamlit APIs to replace with Dash equivalents.
  - Test T-002: `rg -n "st\\.cache|session_state|st\\.download_button|st\\.expander" app/ui/app.py`
  - Artifacts: `docs/dash-migration/artifacts/streamlit-dependencies.md`
  - Risk R-002: Hidden dependencies require refactor.
- **S-003:** Validate core backtest pipeline can run without Streamlit imports.
  - Expected signals: backtest completes from a standalone script using core modules.
  - Test T-003: run a minimal script importing `app/backtest/engine.py` and `app/core/data.py` without Streamlit.
  - Artifacts: `docs/dash-migration/artifacts/core-backtest-smoke.txt`
  - Risk R-003: Tight coupling to Streamlit blocks migration.
- **S-004:** Prototype a minimal Dash app rendering an existing Plotly figure.
  - Expected signals: Plotly chart renders in Dash.
  - Test T-004: run a minimal Dash app with a sample fig; confirm render.
  - Artifacts: `docs/dash-migration/artifacts/dash-plotly-smoke.txt`
  - Risk R-004: Plotly rendering requires refactor.

## 6) Acceptance Criteria (for Investigation Completion)
- **IC-1:** Widget/state inventory and mapping to Dash components completed.
- **IC-2:** Streamlit-specific dependencies list with proposed Dash replacements completed.
- **IC-3:** Core backtest smoke test passes outside Streamlit.
- **IC-4:** Minimal Dash + Plotly smoke test passes.

## 7) Initial Risks
| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-001 | Feature parity gaps in widget mapping | High | Medium | Complete inventory and review with owner |
| R-002 | Long-running backtests block UI | High | Medium | Plan background job or loading state |
| R-003 | Cache behavior regression | Medium | Medium | Use caching middleware and measure |
| R-004 | Plotly/Dash rendering requires refactor | Medium | Low | Prototype early and adjust |

## 8) Decision Log (provisional)
- **D-001:** Keep core backtest/indicator logic unchanged; adapt only the UI layer.
  - Rationale: Reduce migration risk; core logic already tested.
  - Alternatives considered: Rewrite UI and engine together.
- **D-002:** Use Dash with Plotly for charts and `dcc.Store` for UI state.
  - Rationale: Leverage existing Plotly usage and keep UI state explicit.
  - Alternatives considered: Panel or Shiny for Python.

## 9) Exit Artifacts
- `docs/dash-migration/artifacts/widget-inventory.md`
- `docs/dash-migration/artifacts/streamlit-dependencies.md`
- `docs/dash-migration/artifacts/core-backtest-smoke.txt`
- `docs/dash-migration/artifacts/dash-plotly-smoke.txt`
