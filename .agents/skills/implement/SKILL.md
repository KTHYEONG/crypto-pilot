---
name: implement
description: Implement an approved spec mechanically with focused TDD and integration verification.
---

# Implement

The contract is frozen input. Do not redesign, rename, move public symbols, relax thresholds, or edit the
contract to accommodate an implementation choice. Escalate an actual contract conflict to the Spec model.

## Before coding

1. Read the spec, contract, rules, and existing caller/tests.
2. Copy the exact scenario names and target files into a local checklist.
3. Confirm every `file_hint`, symbol, and wiring edge with `rg`.
4. If any item disagrees with the repository, stop with a precise conflict report.

## Three phases

### A. Contract surface

- Add or update signatures, dataclasses, validation, and imports.
- Add minimal contract tests using the frozen scenario names.
- Run the affected tests and ruff with `uv run`.

### B. Logic

- Implement pure logic first; preserve the stated causal and numerical rules.
- Cover the selected hypothesis, boundaries, failures, and regression symptom with concrete assertions.
- Run affected tests, strict mypy on touched interfaces, and `--spec-only` when literal assertions are
  ready.

### C. Wiring

- Connect the production caller at the specified anchor.
- Add every integration scenario using its exact name and target file.
- Check production callers, stale fields, and retired branches.
- Run ruff, strict mypy, affected tests, and final `--spec-only` before requesting `/check`.

## Failure routing

- Contract/file/name/signature conflict: stop and report; do not edit the spec.
- Logic failure: fix source and rerun the smallest affected test.
- Fixture failure: correct only the fixture while preserving expected properties.
- Coverage failure: collect all changed-line misses, add the complete edge-test set, then rerun once.
- Vacuous assertion or shape-only computation test: replace it with a concrete numerical or structural
  property.

Keep the change within the manifest. Architecture, public API, and threshold decisions belong to the Spec
model. Use the normal project command prefix (`uv run`) for all execution.

## Handoff

Report modified files, focused test result, ruff/mypy status, spec-only status, and any unresolved conflict.

## Output

Use a short handoff only:

`🛠️ [IMPLEMENT COMPLETE] <Blueprint>`
`Modified | Tests | Ruff | Mypy | Spec-only | Unresolved`
