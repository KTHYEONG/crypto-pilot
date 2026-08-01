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
   - Run token-efficient audit runner (auto-detects modified `.py` files via git):
     ```bash
     uv run python tools/agent_skills/lean_check.py
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

Provide a clear, concise summary with emojis. Example:

### 🔍 [CHECK] <Audit Target>

- **Status**: ✅ PASS (or ❌ FAIL)
- **Checks**:
  - ⚙️ Contract Alignment: <PASS/FAIL>
  - 🛡️ Strict Mypy & Ruff: <PASS/FAIL>
  - 🧪 Regression & Tests: <PASS/FAIL>





