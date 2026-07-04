---
name: implement
description: Translate logic and test blueprints into working Python code following TDD with L1 validation.
---

# Skill: Implement (TDD Executor)

## Purpose
Translate the logical Blueprint (`docs/specs/*.md`) into working Python code using a **TDD (Test-Driven Development)** cycle. You are the "Execution Builder". Adhere strictly to the defined contract and test matrix in the spec.

## Execution Rules

### 1. Read the Blueprint
- Read `docs/specs/*.md` to extract:
  - Exact file locations
  - Target function/class contracts (signatures)
  - The **TDD Test Scenario Matrix**

### 2. The TDD Cycle (CRITICAL - DO NOT SKIP STEPS)
- **Step 1: Stub/Interface Registration & Self-Gate (Compilation Pass)**
  - Open the target source file (`src/...`) and create only the stub of the function/class matching the signature.
  - Return dummy values or raise `NotImplementedError` (to bypass unused-argument lint warnings).
  - **[Self-Gate]**: Immediately run `ruff check` on the stub file to ensure API signatures and parameter type hints match the `spec` blueprint 100% before coding tests or logic.
- **Step 2: Write Failing Tests (Red Phase)**
  - Open or create the test file (`tests/...`).
  - Write test cases matching **Scenario 1, 2, and 3** from the spec blueprint (using the mock templates provided in spec).
  - Run the test command: `uv run pytest -k "test_name"` to confirm the tests **FAIL** (showing `NotImplementedError` or AssertionError).
- **Step 3: Implement Logic (Green Phase)**
  - Write the minimum code required in the source file (`src/...`) to make the failing tests pass.
  - Run `uv run pytest -k "test_name"` until they all **PASS**.
  - **Loop Limit:** Limit this trial-and-error cycle to **max 3 iterations**. If pytest continues to fail after 3 attempts, **STOP** and return to the `spec` phase to refine the design.
- **Step 4: Refactor (Refactor Phase)**
  - Clean up code duplication, optimize local variables, and ensure docstrings match standards while maintaining a green test suite.
  - **Refactor Limits (CRITICAL)**: Do NOT modify any public signatures, interfaces, or module dependencies during this phase. Focus strictly on internal clean-up and resolving Ruff/Mypy compliance.

### 3. Local L1 Validation
- For modified files (both src and test), run:
  - `uv run ruff check --fix [file]`
  - `uv run mypy [file]`
- Maximize compiler/linter compliance before concluding.

### 4. Single Responsibility (DO NOT OVERSTEP)
- Stop immediately after L1 validation and test-first passing. Do not perform regression analysis across the entire project. Submit results to the `check` phase for dynamic regression and coverage verification.

## Output Format
```md
### 🏗️ TDD Implementation: [Blueprint Name]
- **Target Files:** `[src/...]`, `[tests/...]`
- **TDD Verification:**
  - [ ] Created interface stub & verified signature (Self-Gate)
  - [ ] Wrote failing tests based on scenarios (Red)
  - [ ] Implemented minimal code & passed tests (Green)
- **L1 Validation (Ruff & Mypy):** [Pass/Fail]
- **Next Phase:** Proceed to `check` (Regression & Coverage Review)
```

