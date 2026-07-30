---
name: implement
description: Implement an approved spec mechanically with focused TDD and integration verification.
---

# Implement

Fast-execution model protocol (GPT-5.6 Luna, Sonnet 5, etc.). Execute frozen contract mechanically with zero architectural redesign.

## Rigid Rules & Constraints

- **STRICT BOUNDARY**: Contract is frozen input. Do NOT rename symbols, alter thresholds, or edit contract JSON.
- **ESCALATION GATE**: If contract disagrees with codebase or explicit `python_assertion` fails, STOP immediately and report precise conflict.
- **CLI PREFIX**: Execute ALL verification commands using `uv run`.

## Deterministic 3-Phase Checklist

### Phase 1: Contract Surface & Assertions
- [ ] Read spec, contract, and caller/test files.
- [ ] Confirm symbols/paths with `rg`.
- [ ] Add/update signatures, dataclasses, and validation.
- [ ] Implement contract tests using exact scenario names and verify `python_assertion` expressions.
- [ ] Run verification: `uv run ruff check .` and `uv run pytest <target_test_file>`.

### Phase 2: Core Logic
- [ ] Implement pure logic preserving numerical & causal rules specified in spec.
- [ ] Write concrete assertions for boundaries, failures, and regressions (No vacuous/dummy tests allowed).
- [ ] Run verification: `uv run ruff check .`, `uv run mypy <file>`, and `uv run pytest <target_test_file>`.

### Phase 3: Wiring & Integration
- [ ] Wire production caller at specified anchor.
- [ ] Add integration scenarios with exact names/files.
- [ ] Run full checks: `uv run ruff check .`, `uv run mypy <file>`, and `uv run pytest`.

## Failure Escalation Matrix

- `Contract / Signature / Assertion Conflict` -> **STOP & ESCALATE** (Do not edit spec/contract).
- `Logic Failure` -> Fix source code, rerun smallest affected test.
- `Coverage Miss` -> Add missing edge-case test, rerun once.

## Output Format

Return ONLY this 2-line status summary:

🛠️ `[IMPLEMENT COMPLETE] <Blueprint>`
`Files: <modified_count> | Tests: <PASS_count>/<FAIL_count> | Ruff: <PASS/FAIL> | Mypy: <PASS/FAIL> | Conflicts: <None/Brief Description>`


