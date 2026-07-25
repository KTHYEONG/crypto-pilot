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

### 2. Strict Mechanical TDD Cycle (NO ARCHITECTURAL REFACTORING)
- **Step 1: Stub Registration & Locked Signature Guard**
  - Read `contract.json` to identify public APIs and classes. These are **LOCKED** contracts.
  - Create only the stub matching the signature in the source file (`src/...`).
  - Do **NOT** modify any locked method/class signature. If a design change is required, **STOP execution immediately** and escalate to the human or high-reasoning model.
- **Step 2: Single-Pass Contract & Test Synthesis (Combined Red-Green)**
  - Do **NOT** waste token cycles executing failing tests just to prove a Red phase.
  - Read `contract.json` to extract `contracts`, `assertions`, and `wiring`.
  - Write both the source logic (`src/...`) and the corresponding unit/scenario tests (`tests/...`) simultaneously, adhering strictly to the contract.json spec.
  - **Do NOT copy-paste dummy mock templates or return static dummy values (e.g. `return {}`, `return True`, or logger-only calls). Full concrete business logic must be implemented.**
  - **Strict Value Assertions**: Write test cases targeting the exact `assertions` defined in `contract.json`. Loose assertions like `assert result is not None` are strictly prohibited.
  - Wire the new logic into the caller module as defined in the `wiring` section of `contract.json`. The caller module specified in `wiring.target_file` MUST be included in the modified files list, ensuring both `import_symbol` and `invocation_symbol` (actual call/instantiation) are implemented in the caller module context.
  - Limit high-level scenario tests strictly to the 3-4 scenarios defined in the spec.
- **Step 3: Local Verification & Green Enforcement**
  - Run pytest locally (`uv run pytest -k [target_name]`) to verify that the synthesized code is functional.
  - Ensure that unit tests include strict assertions (verifying exact values, mathematical outputs, and exception types, not just `is not None`) to guarantee semantic correctness.
  - **Do NOT perform design refactoring.** Keep code changes minimal. Refactoring is restricted to basic syntax cleanups and lint compliance.
- **Step 4: L1.5 Local Verification & Clean Handshake**
  - Run pytest locally (`uv run pytest -k [target_name] --cov`) to ensure tests pass and coverage targets (Core >=85%, Adapter >=65%) are met.
  - Run `uv run ruff check --fix [modified_files]` to ensure format/style compliance.
  - Remove any temporary `print()` debugging statements.
  - **STOP execution and output the concise Implementation summary.** Do NOT auto-trigger check skill.

### 3. Self-Healing Budget (Max 3 Loops)
- If local tests or linting fail, the low-cost model is allowed a maximum of **3 consecutive auto-correction attempts** to fix the errors.
- If it still fails after the 3rd attempt, **STOP execution immediately** and report the error logs to the user for human triage.

## Output Format
```md
🛠️ **[IMPLEMENT COMPLETE]** `[Blueprint Name]`
- **Modified**: `[src/...]`, `[tests/...]`
- **Status**: Pytest Green ✅ | Cov [value]% | Ruff Fixed ✅
```
