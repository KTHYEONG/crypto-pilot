---
name: verify
description: Confirm implementation completeness based on Spec acceptance criteria and verification scripts.
---

# Skill: Verify

## Purpose
Empirically verify mechanical integrity (tests, linting, type-checking) and confirm that changes meet the technical criteria defined in the Spec.

## Execution Rules
1. **Mechanical Focus:** Prioritize proving the code *works* and is *valid*. Use tests and static analysis.
2. **Spec Alignment:** Read `Verification Snippet` and `Acceptance Criteria` in the relevant `docs/specs/` file before verifying.
3. **Standard Commands:** Use project standard commands (e.g., pytest, ruff, mypy) for verification.
4. **Raw Output:** Include raw tool output summaries (e.g., `5 passed, 2 failed`) without paraphrasing.
5. **Handoff:** If mechanical checks pass but logical complexity is high, clearly state that deep logic auditing is deferred to the `review` skill.

## Verification Checklist
- [ ] **Acceptance:** Does it satisfy all Acceptance Criteria in the Spec?
- [ ] **Functional:** Does the Verification Snippet output match the Expected Output?
- [ ] **Quality:** No regressions in static analysis (Lint/Type) or existing tests?

## Output Format
```md
### ✅ Verification: [PASS / FAIL / PARTIAL]

**1. Execution Overview**
- **Target Spec:** `docs/specs/filename.md`
- **Commands:** `[Commands executed]`
- **Result:** `[Raw output summary]`

**2. Detailed Results**
- [ ] **Criteria 1:** [Pass/Fail] - [Brief note]
- [ ] **Criteria 2:** [Pass/Fail]

**3. Findings/Gaps**
- [Details of failed tests or lint errors]

**4. Next Steps**
- [Fix direction or Next Step (e.g., Request review)]
```
