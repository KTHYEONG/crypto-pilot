---
name: review
description: Critically audit compliance with Specs and Project Standards to determine approval.
---

# Skill: Review

## Purpose
Act as the final logical gatekeeper. Critically review code changes to ensure they perfectly align with the design (Spec) and fulfill all Acceptance Criteria.

## Audit Checklist
1. **Logic Integrity:** Do the changes fulfill the "Why" and "How" of the Spec? Check for subtle logical drift or missing edge cases.
2. **Spec Alignment:** Do interfaces, types, and logic match `docs/specs/*.md` 100%?
3. **Surgical Precision:** No unnecessary file modifications or "just-in-case" logic outside `Target Files`?
4. **Standards Compliance:** Adherence to project rules (e.g., Python 3.11+, Strict Typing, Logging, Docstrings)?
5. **Final Acceptance:** Verify that the implementation achieves the measurable outcomes defined in the Spec's Acceptance Criteria.

## Verdicts
- **PASS**: Perfect alignment with Spec and Rules.
- **PASS WITH RISKS**: Alignment achieved, but potential edge cases or minor improvements identified.
- **FAIL**: Spec mismatch, rule violation, or logic errors found.

## Output Format
```md
### 🏁 Review Verdict: [PASS / PASS WITH RISKS / FAIL]

**1. Core Audit Results**
- **Spec Alignment:** [Pass/Fail] (Ref: `docs/specs/filename.md`)
- **Surgical Precision:** [Pass/Fail] (Unintended changes check)
- **Standards Compliance:** [Pass/Fail] (Project rules check)

**2. Findings & Risks**
- [Issue 1: Detail and location]
- [Issue 2: Recommendations]

**3. Handoff**
- [Specific fixes required or Next Step]
```
