---
name: implement
description: Translate logic and test blueprints into working Python code following TDD with L1 validation.
---

# Skill: Implement (TDD Executor)

## Purpose
Translate the logical Blueprint (`docs/specs/*.md`) into working Python code using a **TDD (Test-Driven Development)** cycle. Adhere strictly to the defined contract and test matrix in the spec.

## Execution Rules

### 1. Read the Blueprint
- Read `docs/specs/*.md` to extract:
  - Exact file locations
  - Target function/class contracts (signatures)
  - The **TDD Test Scenario Matrix**

### 2. The TDD Cycle (CRITICAL - DO NOT SKIP STEPS)
- **Step 1: Stub/Interface Registration & Self-Gate (Compilation Pass)**
  - Open the target source file (`src/...`) and create only the stub of the function/class matching the signature.
  - Return dummy values or raise `NotImplementedError`.
  - **[Self-Gate]**: Run `ruff check` and `uv run mypy [stub_file]` to guarantee API signatures and type hints match the `spec` contract 100% before coding tests.
- **Step 2: Write Failing Tests (Red Phase)**
  - Open or create the test file (`tests/...`).
  - Write test cases matching **Scenario 1, 2, 3, and Scenario 4 (Integration/Wiring)** from the spec blueprint (using the templates provided in spec).
  - **Coverage Gap Exception**: If the spec's scenarios do not achieve the coverage target (Domain >= 90%, Adapter >= 70%), you MUST write supplementary test cases targeting the uncovered lines.
  - Run the test command: `uv run pytest -k "test_name"` to confirm the tests **FAIL** (showing `NotImplementedError`, AssertionError, or mock failing assertions for integration).
- **Step 3: Implement Logic & Integration (Green Phase)**
  - Write the implementation logic in the target source file (`src/...`).
  - **Integration Wiring**: You MUST modify the parent calling module/pipeline to connect and activate the new logic as planned in the spec's `Integration & Connection Plan`. Do not leave the new module isolated.
  - Run `uv run pytest -k "test_name"` until all unit and integration tests (Scenario 4) **PASS**.
  - **Loop Limit:** Limit this trial-and-error cycle to **max 3 iterations**. If pytest continues to fail after 3 attempts, **STOP** and return to the `spec` phase to refine the design.
- **Step 4: Refactor & L1.5 Local Gate (Green & Clean Gate)**
  - Clean up code duplication, optimize local variables, and ensure docstrings match standards.
  - **[L1.5 Local Gate]**: Run the lightweight sanity check command targeting exclusively the modified files to verify local correctness:
    `uv run ruff check [modified_files] && uv run pytest -k [target_test_name] --tb=short`
    If any check fails, resolve it immediately. **Do not exit this phase until the L1.5 Gate is 100% Green.**
    *(Deep type checking via mypy and coverage audit are deferred to the check phase to prevent duplicate test runs.)*

### 3. Single Responsibility
- Stop immediately after the L1.5 Local Gate passes; submit results to the `check` phase for full regression and coverage auditing.

### 4. Constraints (Strictly Prohibited)
- **No Scripts Directory Modifications**: Do NOT create, modify, or delete any files inside the `scripts/` directory. The `scripts/` directory is reserved exclusively for validation/sync tooling. All production logic and helpers must be created in the `src/` directory.


## Output Format
```md
### 🏗️ TDD Implementation: [Blueprint Name]
- **Target Files:** `[src/...]`, `[tests/...]`
- **TDD Verification & L1.5 Gate:**
  - [ ] Wrote failing tests based on scenarios (Red)
  - [ ] Implemented minimal code & passed tests (Green)
  - [ ] Passed L1.5 Local Gate (Ruff + Mypy + Targeted Test) (🟢 PASS)
- **Next Phase:** Proceed to `check` (Regression & Coverage Review)
```
