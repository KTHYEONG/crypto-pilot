---
name: check
description: Perform Regression Testing, Coverage Auditing, and Error Triage.
---

# Skill: Check (Final Regression & Coverage Gatekeeper)

## Purpose
Empirically verify the code's execution integrity after passing static `audit`. Run dynamic regression test suites, measure test coverage, and perform final compliance checks.

## Execution Rules
1. **Trigger**: Run ONLY after the code successfully passes the `audit` phase.
2. **Regression Testing**:
   - Run the surrounding module's tests to guarantee existing functionality remains unbroken: `uv run pytest [regression_target]`.
3. **Coverage Audit (Target >= 90%)**:
   - Execute coverage metric checks on modified paths: `uv run pytest --cov=[module_path] --cov-report=term-missing`.
4. **Triage & Routing Loop**:
   - **3-Strike Rule**: If regression tests fail for **3 consecutive cycles**, STOP and request human intervention.
   - **Triage Decision**:
     - Spec Logic / Design Error $\rightarrow$ Route back to `spec`.
     - Minor Syntax / Bug in Implementation $\rightarrow$ Route back to `implement` (skipping `audit` to re-implement and re-verify).
5. **Pass Transition**: If all checks pass, proceed to `sync` (Documentation Synchronization & Clean Up).

## Output Format
```md
### ✅ Regression Testing & Coverage: [PASS / FAIL]

#### 📊 Regression Results
- **Command:** `uv run pytest [regression_target]`
- **Status:** [Total Passed / Total Failed]
- **No Refactoring Side-effects:** [Yes/No]

#### 📉 Coverage Report
- **Target Module:** `[module_path]`
- **Coverage %:** [e.g., 94%]
- **Missing Lines:** [e.g., L45-48]
- **Next Phase:** [Proceed to `sync` | Return to `implement` | Human Intervention Required]
```

