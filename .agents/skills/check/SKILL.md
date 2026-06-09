---
name: check
description: Run tests, perform Surgical Hotfixes, and Error Triage.
---

# Skill: Check

## Purpose
Empirically verify the implementation via `pytest` and perform "Error Triage" or "Hotfixes" to maintain velocity.

## Execution Rules
1. **L2 Verification:** Run `uv run pytest`.
2. **Surgical Hotfix (Fast-Track - EFFICIENCY)**:
   - If a failure is caused by a **minor, obvious error** (e.g., missing import, simple typo, obvious off-by-one in a test assertion), you MAY fix it directly using the `replace` tool.
   - After a Hotfix, re-run tests. If it passes, proceed to AUDIT.
   - **Limit**: Max 1 Hotfix attempt per file. If it doesn't fix the issue, stop and route.
3. **Error Triage (Routing Loop)**:
   - **Scenario A: Logic/Test Failure** (Complex failure, edge case, or Hotfix failed): **Route back to `spec`**.
   - **Scenario B: Implementation Error** (Code deviates significantly from Spec or has non-obvious bugs): **Route back to `implement`**.
   - **Scenario C: Regression**: Breaking existing tests. **Route back to `spec`**.
4. **Spec Alignment**: Ensure all `Test Scenario Design` points from the Spec are covered.

## Output Format
```md
### ✅ Testing & Triage: [PASS / FAIL]

**1. Results**
- **Command:** `uv run pytest ...`
- **Output:** `[Raw Summary]`

**2. Hotfix / Triage & Routing**
- **Hotfix applied?**: [Yes (Details) / No]
- **Diagnosis:** [Why did it fail?]
- **Next Step:** [Proceed to AUDIT / Return to SPEC (Logic) / Return to IMPLEMENT (Coding)]
```
