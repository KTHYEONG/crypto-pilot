---
name: implement
description: Translate logic and test blueprints into working Python code with L1 validation.
---

# Skill: Implement

## Purpose
Translate the logical Blueprint into working Python code (both `src/` and `tests/`). You are the "Execution Builder". The `spec` phase has solved the algorithms and test scenarios. Your job is syntax and translation.

## Execution Rules
1. **Blueprint Truth:** Read `docs/specs/`. Adhere strictly to the defined `Contract` and `Algorithmic Flow`.
2. **Double Coding (Src & Test):** 
   - Write the logic in the target `.py` file.
   - Write the corresponding test code in `tests/` based on the `Test Scenario Design` provided in the spec.
3. **Local Validation (L1):** For EVERY modified file (src and test), run validation to catch bugs early:
   - **Primary Diagnostic (Token-Efficient)**: Consider using Serena MCP (`get_diagnostics`) to check code health. This fetches structured diagnostic issues without raw subprocess noise.
   - **Full Local Pipeline**: Run `uv run ruff check --fix [file]` and `uv run mypy [file]` to ensure complete compiler and linter compliance.
4. **Single Responsibility (DO NOT OVERSTEP):**
   - You are ONLY the Executor. Do not run `pytest` (that is `check`). Do not audit business logic (that is `audit`). Do not update architecture docs (that is `sync`). Stop immediately after L1 validation.
5. **Self-Correction:** Fix syntax/type errors (max 3 tries). If unresolved, **STOP** and return to `spec`.

## Output Format
```md
### 🏗️ Implementation: [Blueprint Name]
- **Target Files:** `[src/...]`, `[tests/...]`
- **L1 Validation:** [Pass/Fail]
- **Progress:**
  - [ ] Wrote logic for: [X]
  - [ ] Wrote tests for: [X] (Based on Scenarios 1, 2, 3)
- **Next Step:** [STOP. Hand over to `check` skill for L2 testing]
```
