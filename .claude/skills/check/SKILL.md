---
name: check
description: Independently audit contract compliance, typing, regressions, coverage, and test validity.
---

# Check Protocol

Independent audit gate completing the main development loop (`spec` -> `implement` -> `check`). Performs code review, strict quality checks, and regression verification.

## Directives

1. **Identify Modified Scope**:
   - Inspect modified files using `git status` or `git diff --name-only`.

2. **Standard Audit Execution**:
   - Run token-efficient audit runner (auto-detects modified `.py` files and `docs/specs/*_contract.json` via git/filesystem):
     ```bash
     uv run python tools/agent_skills/lean_check.py
     ```
   - If a specific contract is targeted, explicitly pass `--spec`:
     ```bash
     uv run python tools/agent_skills/lean_check.py --spec docs/specs/<feature>_contract.json
     ```
   - Fallback (if script fails):
     - Code Style: `uv run ruff check`
     - Strict Typing: `uv run mypy`
     - Target Tests: `uv run pytest`


3. **Strict Audit Gate (No Code Mutation)**:
   - Perform auditing independently. Do NOT modify source code during the check pass.
   - Verify non-vacuous tests and contract compliance against `contract.json`.
   - If audit fails, report the exact failure diagnosis clearly for resolution in `/implement` or `/spec`.

## Output

Do NOT add any intro, summary, or explanations. Print output format directly:

- **PASS** (Single-line only):
  ✅ PASS: <Audit Target>

- **FAIL** (Compact format):
  ❌ FAIL: <Audit Target> | Root: <Cause> | Impact: <Scope> | Fix: <Action>

