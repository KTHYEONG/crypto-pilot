---
name: check
description: Perform Regression Testing, Coverage Auditing, and Error Triage.
---

# Skill: Check (Integrated Validation & Triage Gatekeeper)

## Purpose
Unify static contract compliance, dynamic regression test execution, and structural test coverage auditing. Perform systematic triage and error diagnosis strictly upon failure.

## Execution Rules

### 1. Unified Validation Pipeline
Perform the following validation steps sequentially:
1. **Static Spec Matching**: Verify that the designed scenarios (Happy Path, Edge Cases, Error Handling) from `docs/specs/[feature].md` are structurally present in the test files (`tests/...`).
2. **Decoupled Execution (Plan Submission)**: 
   - Prohibit running multiple interactive commands step-by-step.
   - Package all verification commands (e.g., `uv run ruff check`, `uv run mypy`, regression pytest, coverage target) into a single batch execution request to the system runner.
   - Terminate the turn (sleep) immediately after submitting the execution plan.
3. **Dynamic Verification (Happy Path)**:
   - **[CRITICAL] Scope Isolation**: `[regression_target]` MUST be the explicit test file paths mapped 1:1 from the current spec blueprint (e.g., `tests/unit/domain/futures/signals/test_my_feature.py`). NEVER use bare `tests/` or `tests/unit/` without a precise path.
   - **[CRITICAL] Out-of-scope Error Ban**: If collection errors occur from files OUTSIDE `[regression_target]`, do NOT fix them. Report as a separate issue and proceed with the targeted scope only (use `--ignore=<path>` if needed).
   - Regression: Ensure surrounding modules are intact (`uv run pytest [regression_target]`).
   - Coverage: Enforce a coverage metric $\ge$ 90% for core logic, $\ge$ 70% for adapters/boilerplate (`uv run pytest --cov=[module_path] --cov-report=term-missing [regression_target]`).


### 2. Failure Triage & Loop Circuit Breaker
If any step in the validation pipeline fails:
- **Triage (Error Diagnostics)**: Read only the compiled error logs and failure lines retrieved from the execution runner.
- **Diagnostics**: Analyze whether the failure is a design error (requires `spec` rollback) or code bug (requires `implement` rollback).
- **Circuit Breaker**: If regression fails for **3 consecutive cycles**, STOP and request human intervention.

## Verdicts & Routing
- **PASS**: Meets all static/dynamic criteria and coverage $\ge$ 90%. ➔ **Transition to `sync`**
- **FAIL**: Discrepancies found or tests failed. ➔ **Transition to `implement` (or `spec`)** with a clear Gap Analysis.

## Output Format
```md
### 🏁 Verification Result: [PASS / FAIL]

#### 📊 Regression & Coverage Summary
- **Regression Status:** [Passed / Failed count]
- **Target Coverage %:** [e.g., 92%]

#### 🔍 Gap Analysis & Diagnosis (Required ONLY if FAIL)
*Systematic analysis of error logs. Max 3 bullet points.*
- **Error Line:** `[file:line]`
- **Diagnosis:** [Design discrepancy | Implementation Bug]
- **Action Plan:** [Return to implement | Return to spec | Human Intervention]
```

