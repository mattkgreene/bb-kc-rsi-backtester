# Investigation Plan

> **Project Title:** Dash Modernization + Backtest Fix + OHLCV Cache
> **Owner:** Matthew Greene • **Date:** 2025-12-19 • **Version:** 0.2.0
> **Status:** In Progress

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

## 1) Problem Statement
- **Context:** The Dash UI is visually bland, backtest flows are failing, and we need a persistent local cache of historical BTC data for 3+ years with incremental updates.
- **Desired Outcome:** A modernized Dash UI, reliable backtest execution, and a SQLite OHLCV cache that warms on app start and updates before each run.
- **Constraints/Assumptions:**
  - Keep core strategy logic unchanged.
  - Cache should be local, deterministic, and resilient to partial updates.
  - Network access may be restricted; offline cache must work.

## 2) Stakeholders & RACI
| Role | Person/Agent | Responsibility |
|------|--------------|----------------|
| R | Codex | Investigate, implement, and document changes |
| A | Matthew Greene | Approve scope and UX direction |
| C | TBD | Data/infra consult if needed |
| I | TBD | Users/stakeholders |

## 3) Hypotheses
- **H-001:** Backtest failures are due to UI parameter/state or fetch pipeline issues, not core engine logic.
  - Evidence to confirm/refute: callback traces and synthetic backtest results.
- **H-002:** A Dash asset-based CSS layer can modernize the UI without breaking functionality.
  - Evidence to confirm/refute: updated layout + CSS preview.
- **H-003:** A SQLite OHLCV cache can store 3+ years of BTC data and incrementally update on demand.
  - Evidence to confirm/refute: cache schema + successful range read/write tests.

## 4) Questions to Answer
- **Q-001:** Where exactly does the Dash backtest flow fail (params, fetch, or results rendering)? → linked test(s): T-001
- **Q-002:** Should OHLCV cache live in `data/market_data.db` or reuse `data/discovery.db`? → linked test(s): T-002
- **Q-003:** Which UI visual direction should we implement? → linked test(s): T-003

## 5) Investigation Tasks (test-first)
- **S-001:** Trace Dash backtest callback inputs and confirm required params are present.
  - Expected signals: validated param dict + errors captured.
  - Test T-001: simulate callback with synthetic params and record results.
  - Artifacts: `docs/dash-modernization/artifacts/backtest-trace.txt`
  - Risk R-001: backtest still fails due to missing fields.
- **S-002:** Decide DB location and design OHLCV schema.
  - Expected signals: documented schema and DB path decision.
  - Test T-002: create table and validate read/write.
  - Artifacts: `docs/dash-modernization/artifacts/ohlcv-schema.md`
  - Risk R-002: schema mismatch with existing cache usage.
- **S-003:** Define UI style direction and layout structure.
  - Expected signals: design tokens and layout plan.
  - Test T-003: render HTML/CSS with new styles.
  - Artifacts: `docs/dash-modernization/artifacts/ui-style-notes.md`
  - Risk R-003: CSS changes break layout or responsiveness.

## 6) Acceptance Criteria (for Investigation Completion)
- **IC-1:** Root cause(s) of Dash backtest failure identified.
- **IC-2:** OHLCV cache schema and DB path defined.
- **IC-3:** UI design direction documented with tokens/structure.

## 7) Initial Risks
| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-001 | Backtest failure persists after fixes | High | Medium | Add logging and unit checks | 
| R-002 | Cache blocks startup due to large fetch | Medium | Medium | Warm in background + chunked writes |
| R-003 | UI regressions on mobile | Medium | Medium | Add responsive CSS and test breakpoints |

## 8) Decision Log (provisional)
- **D-001:** Use a dedicated `data/market_data.db` for OHLCV cache.
  - Rationale: keep discovery DB focused on strategy runs.
  - Alternatives considered: reuse `data/discovery.db`.
- **D-002:** Use a “fintech terminal” visual style (dark graphite, high-contrast accents, card layout).
  - Rationale: modern but readable for dense analytics.
  - Alternatives considered: minimalist light theme.

## 9) Exit Artifacts
- `docs/dash-modernization/artifacts/backtest-trace.txt`
- `docs/dash-modernization/artifacts/ohlcv-schema.md`
- `docs/dash-modernization/artifacts/ui-style-notes.md`
