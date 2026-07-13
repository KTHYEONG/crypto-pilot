---
name: check
description: Perform Regression Testing, Coverage Auditing, and Error Triage with Minimal Token Consumption.
---

# Skill: Check (Ultra-Lean Validation & Triage Gatekeeper)

## Purpose
Unify static contract compliance, dynamic regression test execution, and structural test coverage auditing. Maintain strict token efficiency by minimizing command output logs and isolating verification scope.

## Execution Rules

### 1. Minimal Output Validation Pipeline (Token Saving)
Run the integrated check script targeting the modified files to avoid multiple command runs and massive terminal logs:
- **Command:** `uv run python scripts/lean_check.py --files [modified_files]`
- **Diagnostic Exception:** If the coverage target is missed or tests fail, inspect the minimal summary printed by the script. Avoid running raw pytest or cov commands independently on large paths.

### 2. Verification Scope

#### 2a. Target Limits
- **Precise Paths**: Always specify the exact target test file path (e.g. `tests/unit/domain/.../test_x.py`). Never run pytest on broad directories like `tests/` to prevent execution overhead and massive terminal outputs.

#### 2b. Coverage Thresholds
- Compare against target thresholds using the printed coverage summary (Apply Tolerance Buffer):
  - **Core Logic (Domain, Signal, Sizing, Portfolio):** Target >= 90% (Accept **85% ~ 89%** as a **Conditional PASS** if all unit tests pass).
  - **Adapters/Runners/DTOs/Boilerplate:** Target >= 70% (Accept **65% ~ 69%** as a **Conditional PASS**).
- Measure coverage exclusively on files created or modified by the current task.

### 3. Triage & Circuit Breaker (On Failure)
If any step fails, retrieve ONLY the specific failing assertion line or traceback summary:
- **Diagnostics**: Determine if it requires a design change (`spec` rollback) or a bug fix (`implement` rollback).
- **Circuit Breaker**: If regression fails for **3 consecutive cycles**, STOP and request human intervention.

## Output Format
Write check results strictly in the compact format below:

### 🟢 PASS Format:
🟢 PASS | All checks passed (Cov [value]%)

### 🔴 FAIL Format:
🔴 FAIL | [brief failure summary]
- 🔍 Cause: [file path:line number] - [error reason]
- 🛠️ Fix: Apply [modification action] to [function/class] in [target file]
