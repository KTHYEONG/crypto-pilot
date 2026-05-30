---
name: verify
description: Confirm implementation completeness based on Spec acceptance criteria and verification scripts.
---

# Skill: Verify

## Purpose
Empirically verify mechanical integrity (tests, linting, type-checking) and confirm that changes meet the technical criteria defined in the Spec.

## Execution Rules
1. **Systemic Integrity (L2):** Prioritize proving the code *works* (tests) and doesn't break other parts of the system.
2. **Functional Focus:** Focus on `pytest` execution. Do NOT redundantly run file-specific `ruff` or `mypy` if they were already confirmed in the `implement` phase, unless a project-wide regression is suspected.
3. **Spec Alignment:** Read `Verification Snippet` and `Acceptance Criteria` in the relevant `docs/specs/` file before verifying.
4. **Standard Commands:** Use `uv run pytest` with appropriate filters (e.g., `-k`, `--cov`).
5. **Raw Output:** Include raw tool output summaries (e.g., `5 passed, 2 failed`) without paraphrasing.
6. **Handoff:** If mechanical checks pass but logical complexity is high, clearly state that deep logic auditing is deferred to the `review` skill.

## Verification Checklist
- [ ] **Acceptance:** Does it satisfy all Acceptance Criteria in the Spec?
- [ ] **Functional:** Does the Verification Snippet output match the Expected Output?
- [ ] **Regression:** Do related existing tests still pass?

## Output Format
```md
### ✅ 검증 및 테스트: [PASS / FAIL / PARTIAL]

**1. 검증 요약**
- **기준 설계 문서:** `docs/specs/filename.md`
- **실행 명령어:** `[Commands executed]` (e.g., pytest)
- **결과 요약:** `[Raw output summary]`

**2. 세부 검증 내역**
- [ ] **완료 기준 1:** [Pass/Fail] - [Brief note]
- [ ] **완료 기준 2:** [Pass/Fail]

**3. 문제점 및 발견 사항**
- [Details of failed tests or regressions]

**4. 다음 단계**
- [Fix direction or Next Step]
```
