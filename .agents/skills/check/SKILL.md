---
name: check
description: Perform Regression Testing, Coverage Auditing, and Error Triage.
---

# Skill: Check (Regression & Coverage Auditor)

## Purpose
Empirically verify the integrity of the codebase after TDD implementation. Run regression test suites, audit coverage metrics, and triage unresolved failures.

## Execution Rules
1. **Regression Testing**:
   - TDD unit tests are verified during the `implement` phase. In the `check` phase, verify the surrounding module or package to ensure no existing logic is broken.
   - Run `uv run pytest [module_test_file]` or `uv run pytest` on the affected test path.
2. **Coverage Audit (Target >= 90%)**:
   - Run coverage analysis for the modified code: `uv run pytest --cov=[module_path] --cov-report=term-missing`.
   - Identify any paths or exception blocks that were missed by the TDD test cases.
3. **Circuit Breaker & Triage**:
   - **3-Strike Rule**: If a regression bug persists for **3 consecutive cycles**, STOP and request human intervention.
   - **Error Triage (Routing Loop)**:
     - **Scenario A (Interface Broken / Logic Gap)**: Route back to `spec`.
     - **Scenario B (Minor Bug / Syntax Mistake)**: Route back to `implement`.
4. **Single Responsibility (DO NOT OVERSTEP)**:
   - You are ONLY the Regression Tester/Triager. Stop immediately after regression check and coverage calculation.

## Output Format
```md
### ✅ Regression Testing & Coverage: [PASS / FAIL]

#### 📊 Regression Results
- **Command:** `uv run pytest [regression_target] ...`
- **Status:** [Total Passed / Total Failed]
- **Unbroken Status:** [Yes/No] (Confirmed existing features still work)

#### 📉 Coverage Report
- **Target Module:** `[module_path]`
- **Coverage %:** [e.g., 94%]
- **Missing Lines:** [e.g., L45-48 (Exception Handler)]
```
