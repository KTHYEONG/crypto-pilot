---
name: implement
description: Translate logic and test blueprints into working Python code following TDD with L1 validation.
---

# Skill: Implement (TDD Executor & Auto-Chaining)

## Purpose
Translate the logical Blueprint (`docs/specs/*.md`) into working Python code using a **TDD (Test-Driven Development)** cycle. Adhere strictly to the defined contract and test matrix in the spec.
Optimize for **low-cost models (Doer)** by enforcing strict mechanical execution and automatic pipeline triggering.

## Execution Rules

### 1. Read the Blueprint
- Read `docs/specs/*.md` to extract:
  - Exact file locations and target function/class contracts (signatures).
  - The **Skeleton Mock Boilerplate** and scenario descriptions.
- If a `[feature]_contract.json` exists, load its `scenarios[]` list now. Every `scenarios[].name` is the literal test function name to use — unit scope included, not just integration. Do not invent a more descriptive name; `/check`'s spec-compliance phase matches these as exact strings, and a semantically-equivalent rename reads as a missing test.

### 2. Three-Phase Implementation Cycle
Phase boundaries are defined by goals, not by pass/fail of tests. Do NOT skip directly to a later phase.

#### Phase A — Contract
Goal: source code compiles (importable, no SyntaxError) and the new public API is minimally callable.

1. Write new dataclasses/classes/functions in `src/...` with **full signature, docstring, `__post_init__` validation, and error_policy** from the spec.
2. Write 1-2 minimal tests that verify the contract:
   - Constructor succeeds with valid args (if applicable).
   - Constructor raises `ValueError` for each invalid arg (if applicable).
   - Function returns correct type on empty/trivial input.
3. Run `uv run pytest -k [target] --no-header -q` — it MUST pass before proceeding.
4. If contract changes (adds/removes fields, renames, changes signature):
   - Annotate with `# CONTRACT:` in the source.
   - **Propagate**: grep all callers and `mocker.Mock(spec=...)` references → update them.
   - Existing tests MUST remain green after propagation.

#### Phase B — Core Logic
Goal: each new function produces correct output for its primary algorithm.

1. Implement the core algorithm in pure functions (no I/O, no global state).
2. Write tests:
   - **P0** (2-3 tests): Regression guard — covers the specific bugs/symptoms the spec aims to fix. Use realistic fixtures, strict value assertions.
   - **P1** (2-3 tests): Happy path + error path for each new function. One test per distinct failure mode.
   - **Naming**: if a scenario in the contract's `scenarios[]` covers what a P0/P1 test would otherwise cover, use that scenario's exact `name` and `target_test_file` instead of writing an ad-hoc name — even for `"scope": "unit"` entries. Only free-name a test when no contract scenario matches its intent.
   - Write tests and source **in the same pass**. Do NOT run tests until both are ready.
3. Run `uv run pytest -k [target] --no-header -q`. If a test fails:
   - Classify the root cause into one of:
     - **Logic bug** → fix source, counts as 1 self-healing attempt.
     - **Test data / fixture problem** → fix test data only, does NOT count toward self-healing budget.
     - **Environment / non‑deterministic** (floating‑point tolerance, numpy version, import caching) → loosen tolerance or skip that check, does NOT count.
4. Run `uv run ruff check [modified_files]` — fix any errors.

#### Phase C — Wiring & Integration
Goal: the new code is called by existing pipeline code, and existing tests for caller modules still pass.

1. Wire into caller modules per the spec's `Integration & Connection Plan`.
2. **Orphaned-implementation self-check (do this before running any test):** for every new function/class the spec's `contracts` list names, run
   `grep -rn "\bName\b" src/ | grep -v "/tests/\|def Name"` and confirm at least one hit outside its own definition and outside `tests/`. A function that only appears in its own `def` line and in a test file is not wired — go back and call it from the anchor the spec's `wiring[].invocation_symbol` describes. This is the highest-cost defect to leave for `/check` to find (it means the entire behavioral change did nothing in production), so catch it here, not later.
   - If this rewrite retires an old branch/cascade, grep the old branch's config fields too (`\bold_field_name\b` across `src/`, excluding its own dataclass line) — a field with zero remaining readers must be deleted in this same phase, along with any test fixtures that only exist to pass it in.
3. Run `uv run pytest -k [caller_module] --no-header -q` — existing caller tests MUST pass.
4. If caller uses `mocker.Mock(spec=...)` for the new contract, ensure the mock includes all new default fields.
5. Add **1 test per `scenarios[]` entry marked `"scope": "integration"`** in the spec's contract, written in the exact `target_test_file` with the exact `name` given — not an ad-hoc new file. This is the same literal-naming rule from step 1 and Phase B, restated here because integration scenarios are the ones most often deferred to this phase and forgotten. `/check`'s spec-compliance phase matches every scenario's name literally regardless of scope; a renamed function in a different file reads as a missing test even when it passes.

### 3. Coding Conventions

- **No static dummies**: Every function body must contain real business logic. `return {}`, `return True`, `return None` stubs are forbidden unless the spec explicitly requires that value for an empty/error case.
- **Strict value assertions**: Test assertions must target exact numerical values, exception messages, or shape contracts. `assert result is not None` alone is prohibited.
- **No conditional escape hatches**: Never write `assert real_condition or fallback_condition_that_is_almost_always_true` (e.g. `... or total == 0`, `... or guard is not None`) or a bare `assert x >= 0` on a value that can't be negative. Before locking in an assertion for a regression/gate-reachability test, run the fixture standalone (`uv run python -c "..."`) and print the value the assertion depends on — assert the value you actually observed, not a shape that would also pass if the code path never ran.
- **No unsolicited refactoring**: Do not rename, restructure, or extract shared utilities unless the spec explicitly calls for it.
- **Type discipline**: `NDArray` annotation dtype MUST match the actual array dtype passed at runtime.

### 4. Local Verification

- Run `uv run ruff check --fix [modified_files]` before running pytest.
- Run `uv run pytest -k [target_name] --no-header -q`.
- Run `uv run mypy --strict --no-error-summary [modified_files]` in **advisory mode** before Phase C close. Treat errors as warnings; fix only those that indicate a real contract violation (wrong argument count, missing attribute). Cosmetic type issues (e.g., `Any` vs `float`) are low-priority and can be deferred.
- Remove `print()` / logging debug statements before final output.

### 5. Self-Healing Budget

- If local tests or linting fail after all three phases, the model is allowed a maximum of **3 consecutive auto-correction attempts** to fix the errors.
- **Exception — classified non‑logic failures do not count**: failures whose root cause is identified as test data error or environment/non‑deterministic mismatch (floating-point tolerance, import caching, numpy version differences) are excluded from the 3-attempt budget. In such cases, fix the fixture or loosen the tolerance and continue.
- If it still fails after 3 counted attempts, **STOP execution immediately** and report the error logs to the user for human triage. Include the failure classification in the report.

## Output Format
```md
🛠️ **[IMPLEMENT COMPLETE]** `[Blueprint Name]`
- **Modified**: `[src/...]`, `[tests/...]`
- **Status**: Pytest Green ✅ | Cov [value]% | Ruff Fixed ✅
```
