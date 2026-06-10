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
- **Serena Support (Architecture & Dependency Safety)**:
  - **Structure Check**: Use Serena MCP `get_symbols_overview` on modified files to verify that public API signatures match Spec definitions and do not break architecture constraints.
  - **Impact Analysis**: Use `find_implementations` to detect external usages of modified components and ensure no downstream modules are broken.
- **Efficiency**: Minimize internal thinking steps and judge only whether the core logic changes align with the Spec's goals. Use standard file reads strictly as a fallback.

### 2. Intent & Logic Alignment
- **Quant/Financial Core**: Verify that formulas, vectorization (NumPy/Pandas), and trading logic intent match the Spec.
- **Critical Failure Points**: Briefly check for performance bottlenecks (e.g., unnecessary loops) or fatal logical flaws.

### 3. Skip Mechanical Checks
- **Delegate to Linter**: Pass off mechanical conventions like Pydantic settings, Logging vs Print, and type hinting to the `check` phase (ruff/mypy).
- **No Documentation**: Do not perform architecture document updates or ADR logging here. (Delegate to the `sync` skill).

### 4. Single Responsibility (DO NOT OVERSTEP)
- You are ONLY the Intent Reviewer. Do not write code (`implement`), do not run tests (`check`), and do not update documents (`sync`). Your only job is to compare the code against the Spec logic.

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
- [Next Step: ✅ `sync` is ready.]
```
