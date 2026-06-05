---
name: check
description: Confirm implementation completeness based on Spec acceptance criteria and verification scripts.
---

# Skill: Check

## Purpose
Empirically verify mechanical integrity (tests, linting, type-checking) and confirm that changes meet the technical criteria defined in the Spec.

## Execution Rules
1. **Systemic Integrity (L2):** Prioritize proving the code *works* (tests) and doesn't break other parts of the system.
2. **Functional Focus:** Focus on `pytest` execution. Do NOT redundantly run file-specific `ruff` or `mypy` if they were already confirmed in the `implement` phase, unless a project-wide regression is suspected.
3. **Spec Alignment:** Read `Verification Snippet` and `Acceptance Criteria` in the relevant `docs/specs/` file before verifying.
4. **Standard Commands:** Use `uv run pytest` with appropriate filters (e.g., `-k`, `--cov`).
5. **Raw Output:** Include raw tool output summaries (e.g., `5 passed, 2 failed`) without paraphrasing.
6. **Handoff:** If mechanical checks pass but logical complexity is high, clearly state that deep logic auditing is deferred to the `audit` skill.

## Verification Checklist
- [ ] **Acceptance:** Does it satisfy all Acceptance Criteria in the Spec?
- [ ] **Functional:** Does the Verification Snippet output match the Expected Output?
- [ ] **Regression:** Do related existing tests still pass?

## Output Format
```md
### ✅ Testing & Verification: [PASS / FAIL / PARTIAL]

**1. Verification Summary**
- **Source Spec:** `docs/specs/filename.md`
- **Execution Command:** `[Commands executed]` (e.g., pytest)
- **Raw Result Summary:** `[Raw output summary]`

**2. Detailed Checklist**
- [ ] **Criteria 1:** [Pass/Fail] - [Brief note]
- [ ] **Criteria 2:** [Pass/Fail]

**3. Issues & Findings**
- [Details of failed tests or regressions]

**4. Next Step**
- [Fix direction or Next Step]
```
