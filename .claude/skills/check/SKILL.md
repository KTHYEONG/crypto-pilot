---
name: check
description: Run tests, perform Surgical Hotfixes, and Error Triage.
---

# Skill: Check

## Purpose
Empirically verify the implementation via `pytest` and perform "Error Triage" or "Hotfixes" to maintain velocity.
## Execution Rules
1. **Targeted L2 Verification:** Do NOT run the entire test suite unless necessary. Run `uv run pytest [specific_test_file]` or use `-k [keyword]` to filter tests related to the modified code.
   - **Diagnostic Assistance**: If a test fails, use Serena MCP (`get_diagnostics` or `find_implementations`) to localize issues. Limit raw output analysis to save tokens.
2. **Circuit Breaker (Anti-Loop)**:
   - **3-Strike Rule**: If the same test failure or logic gap persists for **3 consecutive routing cycles** (e.g., check -> implement -> check), STOP all automated attempts.
   - **Action**: Summarize the blockage clearly and request **Human Intervention**. Do not waste more tokens on repetitive failures.
3. **Surgical Hotfix (Fast-Track - EFFICIENCY)**:
...
   - If a failure is caused by a **minor, obvious error** (e.g., missing import, simple typo, obvious off-by-one), fix it directly using the `replace` tool.
   - **Limit**: Max 1 Hotfix attempt per file. If it fails, stop and route immediately.
3. **Error Triage (Routing Loop)**:
   - **Scenario A: Logic/Test Failure**: Route back to `spec`.
   - **Scenario B: Implementation Error**: Route back to `implement`.
4. **Single Responsibility (DO NOT OVERSTEP):**
   - You are ONLY the Tester/Triager. Stop immediately after tests pass or routing is decided. Do not add new features.

## Output Format
```md
### ✅ Testing & Triage: [PASS / FAIL]

#### 🚀 Executive Summary
- [1-sentence summary: e.g., "All 5 tests passed for the new indicator logic" or "1 test failed due to a dimension mismatch."]

#### 📊 Test Results (Token-Optimized)
- **Command:** `uv run pytest [target] ...`
- **Status:** [Total Passed / Total Failed]
- **Failures (if any):** [List only the name of failing test cases and the final error line. Avoid full tracebacks.]
```
