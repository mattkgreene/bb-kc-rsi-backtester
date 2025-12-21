> **Project Title:** Repo Cleanup + Test Baseline
> **Owner:** mkg • **Date:** 2025-12-20 • **Version:** 0.1.0
> **Status:** Done

### Project Links
- Investigation Plan → [INVESTIGATION.md](./INVESTIGATION.md)
- Execution Plan → [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)
- Execution Log → [EXECUTION_LOG.md](./EXECUTION_LOG.md)

# Execution Plan

## 1) Objective & Scope
- **Objective:** Ensure the repo compiles and unit tests pass reliably, and tighten cancellation semantics for queued jobs.
- **Out of Scope:** Adding a new test framework or enforcing heavyweight deps in CI.

## 2) Acceptance Criteria (Go/No-Go)
- **AC-1:** `py_compile` passes on the key modules.
- **AC-2:** `unittest discover` passes with explicit skips (not import failures).
- **AC-3:** A canceled job remains canceled even if worker attempts to mark success/failure.

## 3) Preconditions & Dependencies
- Preconditions: Python 3 available.
- Dependencies: none beyond stdlib for the baseline suite.

## 4) Step-by-Step Plan (each is testable)

### S-101: Establish compile baseline
- Goal: Catch syntax/indent errors quickly.
- Commands/Actions:
  - Run `python3 -m py_compile` across the touched backend/frontend modules.
- Expected Result: exit code 0 and no syntax errors.
- **Test T-101:** `python3 -m py_compile frontend/api_client.py backend/db/jobs.py backend/db/leaderboard.py backend/api/main.py backend/worker/main.py frontend/ui/dash_app.py`
- Evidence: `docs/repo-cleanup-tests/artifacts/py_compile.out`
- Rollback: Revert offending file changes; re-run T-101.

### S-102: Ensure unit test discovery is resilient
- Goal: Make tests runnable even without optional deps.
- Commands/Actions:
  - Use lazy imports in packages where heavy deps are optional.
  - Skip dash/pandas/fastapi-dependent tests when not installed.
- Expected Result: `unittest discover` succeeds with skips instead of failures.
- **Test T-102:** `python3 -m unittest discover -s . -p 'test_*.py'`
- Evidence: `docs/repo-cleanup-tests/artifacts/unittest.out`
- Rollback: Revert test changes; re-run T-102.

### S-103: Make cancellation terminal (race-safe)
- Goal: Prevent worker from overwriting a canceled job’s status.
- Commands/Actions:
  - Guard `mark_succeeded`/`mark_failed` to only update `status='running'`.
  - Have worker check cancellation and avoid marking success/failure after cancel.
  - Add a unit test proving canceled stays canceled.
- Expected Result: cancel remains terminal and visible to the frontend queue UI.
- **Test T-103:** `python3 -m unittest test_backend_jobs_db.JobsDbTests`
- Evidence: `docs/repo-cleanup-tests/artifacts/unittest_after_cancel_fix.out`
- Rollback: Revert job DB/worker changes; re-run T-103.

### S-104: Final verification sweep
- Goal: Ensure no regressions.
- Commands/Actions:
  - Run full unit test discovery again after cancellation changes.
- Expected Result: suite passes.
- **Test T-104:** `python3 -m unittest discover -s . -p 'test_*.py'`
- Evidence: `docs/repo-cleanup-tests/artifacts/unittest_final.out`
- Rollback: Revert latest change set; re-run T-104.

## 5) Risks & Mitigations (Execution)
| ID | Risk | Trigger | Mitigation | Owner |
|----|------|---------|------------|-------|
| R-201 | Hidden dependency import-time failures | discover crashes | add lazy import + skip tests | Codex |
| R-202 | Cancel semantics still racy | status flips after cancel | add DB guards + worker checks + test | Codex |

## 6) Communication Plan
- Before: N/A
- After: Share artifact paths and test summary.

## 7) Approval
- Change ticket: N/A
- Approvers: mkg
- Scheduled window: N/A
