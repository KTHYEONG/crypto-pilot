---
name: implement
description: Implement an approved spec mechanically with focused TDD and integration verification.
---

# Implement Protocol

Fast-execution protocol for mechanical code implementation based strictly on frozen specs (`_spec.md` or `contract.json`).

## Execution Principles

Operate as a deterministic translator turning the specification into code and passing tests:
1. Append the test suite directly into target test files.
2. Implement clean production logic satisfying the spec's invariants and docstring.
3. Wire the caller at the specified anchor.
4. Verify with `lean_check.py`.

## Directives

1. **Scaffolding Exclusion**:
   - Production code must contain only finalized code and docstrings.
   - Do not paste or leave temporary spec directives, step numbers, or placeholder comments in code or docstrings.

2. **Fidelity & Anti-Defensive Sprawl**:
   - Treat the spec as truth. Do not invent unrequested parameters or speculative abstraction layers.
   - Do not leave stubs (`pass`, `...`, `NotImplementedError`, placeholder returns).
   - Avoid speculative `try-except` blocks or unrequested null checks that are not required by spec invariants.

3. **Direct Implementation Workflow (Invariant-Driven)**:
   - **Phase 1 (Production Logic & Wiring)**:
     - Implement clean, minimal production logic satisfying the spec contracts and invariants.
     - Wire caller snippet at `- Anchor: <anchor>`.
     - Run: `uv run ruff check <target_file> <caller_file>`.
   - **Phase 2 (Invariant Guard Tests)**:
     - Implement targeted guard tests directly in `<target_test_file>` verifying the spec's Invariant Scenarios.
     - Verify domain conservation laws, edge boundaries, and fail-closed error paths. Never test trivial internal getters or mocks.
     - Run targeted test: `uv run pytest <target_test_file> -q`.
   - **Phase 3 (Verification & Pruning)**:
     - Run: `uv run python tools/agent_skills/lean_check.py` (pass `--spec <spec_file>` if tracking spec).
     - Confirm all checks pass with 100% diff coverage.

4. **Diff Coverage Resolution (Pruning Over Bloat)**:
   - If diff coverage reports untested lines:
     1. Evaluate if it is speculative defensive code (unrequested `try-except`, unreachable branches): **Prune and delete the code**.
     2. If required domain logic lacks coverage, add the missing boundary scenario.

## Output

### 🔨 [IMPLEMENT] <Task Title>

- **Status**: ✅ COMPLETE (or ❌ ESCALATED)
- **Modified**: <Count> files
- **Verification**:
  - 🧪 Pytest: <Passed>/<Total> passed
  - 🧹 Ruff / Mypy: <PASS/FAIL>
  - 🛡️ Scaffolding & Diff Coverage: <PASS/FAIL>
