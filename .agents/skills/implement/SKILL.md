---
name: implement
description: Implement an approved spec mechanically with focused TDD and integration verification.
---

# Implement Protocol

Fast-execution model protocol for mechanical feature implementation based on frozen spec contracts.

## Directives

1. **Strict Contract Compliance (Zero Guesswork)**:
   - Treat `contract.json` as absolute input. Do not invent new parameters, change signatures, or alter thresholds.

2. **Mechanical Scenario Translation**:
   - Directly translate `scenarios` and `python_assertion` from `contract.json` into concrete `pytest` test cases.
   - Implement source logic at exact specified `target_file` and `wiring` anchors.

3. **Surgical Code Modifications**:
   - MUST use targeted block/line edits (`replace_file_content` / `multi_replace_file_content`) to prevent code loss or unintended rewrites.

4. **Verification & Escalation Loop**:
   - Run verification via `uv run ruff check .` and `uv run pytest <target_test_file>`.
   - If contract conflicts with codebase realities or tests fail due to bad spec logic, STOP and escalate to `/spec`.

## Output

Provide a clear, concise summary with emojis. Example:

### 🔨 [IMPLEMENT] <Task Title>

- **Status**: ✅ COMPLETE (or ❌ INCOMPLETE)
- **Modified**: <Count> files
- **Verification**:
  - 🧪 Pytest: <Passed>/<Total> passed
  - 🧹 Ruff / Mypy: <PASS/FAIL>





