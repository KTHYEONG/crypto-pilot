---
name: verify
description: Verify implemented changes using the blueprint's verification snippet and acceptance criteria.
---

# verify

## Purpose
Run and report checks to ensure technical integrity. You must prove that the implementation meets the specific standards defined in the architect's blueprint.

## Rules
1.  **Blueprint Priority:** You MUST read the relevant blueprint file in `.agents/specs/` before verifying.
2.  **Execute Snippets:** Prioritize running the `Verification Snippet` exactly as defined in the blueprint.
3.  **Actual Summaries:** Always include the raw summary line from the tool (e.g., `5 passed, 2 failed in 0.5s`). Do not paraphrase.
4.  **Side-Effect Awareness:** If shared logic was changed, you MUST run tests for at least one dependent module.

## Verification Checklist
- [ ] **Acceptance Criteria:** Check each item from the blueprint's `Acceptance Criteria`.
- [ ] **Snippet Success:** Did the `Verification Snippet` produce the `Expected Output`?
- [ ] **Static Analysis:** Run `ruff` and `mypy` as standard sanity checks.

## Output
```md
<verify>

### ✅ Verification: [PASS / FAIL / PARTIAL]
- **Blueprint Reference:** `.agents/specs/[feature_name].md`
- **Summary:** `[Actual tool output summary]`
- **Commands Run:** `[Exact commands from snippet or manual]`
- **Acceptance Criteria Check:**
  - [ ] [Criterion 1 from Blueprint]
  - [ ] [Criterion 2 from Blueprint]
- **Passed:** [Specific tests/checks]
- **Failed/Skipped:** [Specific failures with brief error]
- **Next:** `[Next Step]`

</verify>
```
