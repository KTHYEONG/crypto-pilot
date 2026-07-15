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
- **Strict Constraint (1:1 Mapping)**: You **MUST** include both the source file and its corresponding test file (e.g. `src/x.py` and `tests/unit/test_x.py`) in `--files` to prevent test bypass.
- **Strict Constraint (No print)**: Raw `print()` is strictly prohibited. Ensure all outputs use `logging` following the tag schema in `logging.md`.
- **Diagnostic Exception**: If the coverage target is missed or tests fail, inspect the minimal summary printed. You are exceptionally allowed to run a targeted pytest with `--tb=short` once to view detailed traceback if the cause is not clear from the summary.

### 2. Verification Scope

#### 2a. Target Limits
- **Precise Paths**: Always specify the exact target test file path (e.g. `tests/unit/domain/.../test_x.py`). Never run pytest on broad directories like `tests/` to prevent execution overhead and massive terminal outputs.

#### 2b. Coverage Thresholds
- Compare against target thresholds using the printed coverage summary.
- **SSOT Directive**: The exact coverage targets (e.g., Domain/Signal >= 90%, Adapter >= 70%) and their respective Tolerance Buffers (Conditional PASS ranges) are defined exclusively in [testing.md](file:///.agents/rules/testing.md). You MUST reference and adhere to those limits; do not hardcode or verify static numbers here.
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
