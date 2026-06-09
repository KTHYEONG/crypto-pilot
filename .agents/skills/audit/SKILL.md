---
name: audit
description: Intent Alignment & Core Logic Verification.
---

# Skill: Audit (Lightweight Intent Gatekeeper)

## Purpose
Verify that the implementation accurately reflects the business logic requirements defined in the Spec. Exclude style or syntax checks to minimize token consumption.

## Execution Rules

### 1. Targeted Diff Review (Token Efficiency)
- **Focus**: Do not re-analyze the entire existing codebase. Concentrate on comparing the `git diff` or changed code snippets directly against the `Algorithmic Flow` in the Spec.
- **Efficiency**: Minimize internal thinking steps and judge only whether the core logic changes align with the Spec's goals.

### 2. Intent & Logic Alignment
- **Quant/Financial Core**: Verify that formulas, vectorization (NumPy/Pandas), and trading logic intent match the Spec.
- **Critical Failure Points**: Briefly check for performance bottlenecks (e.g., unnecessary loops) or fatal logical flaws.

### 3. Skip Mechanical Checks
- **Delegate to Linter**: Pass off mechanical conventions like Pydantic settings, Logging vs Print, and type hinting to the `check` phase (ruff/mypy).
- **No Documentation**: Do not perform architecture document updates or ADR logging here. (Delegate to the `sync` skill).

## Verdicts
- **PASS**: Core logic is accurately implemented according to the Spec's intent.
- **FAIL**: Missing logic, incorrect algorithm usage, or severe performance degradation found. -> **Return to `implement`** (or `spec`) with clear feedback.

## Output Format
```md
### 🏁 Professional Audit: [PASS / FAIL]

**1. Intent Alignment**
- [ ] Spec Core Logic vs Code Implementation
- [ ] Quant/Financial Vectorization & Efficiency

**2. Expert Feedback & Next Step**
- [Brief and sharp review comments on the core logic]
- [Next Step: Proceed to SYNC / Return to IMPLEMENT]
```
