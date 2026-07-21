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

### 2. Strict Mechanical TDD Cycle (DO NOT REFACTOR ARCHITECTURE)
- **Step 1: Stub Registration & Verification**
  - Create only the stub matching the signature in the source file (`src/...`).
  - Return dummy values or raise `NotImplementedError`.
  - Verify signatures using local type-checking: `uv run ruff check [stub_file]` and `uv run mypy [stub_file]`.
- **Step 2: Write Tests using Skeleton Mocks (Red Phase)**
  - Copy the **Skeleton Mock Boilerplate** from the spec into the test file (`tests/...`).
  - Implement test functions matching Scenario 1, 2, 3, and 4.
  - Run `uv run pytest -k "test_name"` to confirm the tests **FAIL** (proving Red phase).
- **Step 3: Implement minimal logic & Connection (Green Phase)**
  - Write the minimal code to satisfy the tests.
  - Integrate/wire the new logic into the parent calling module as defined in the Spec's Connection Plan.
  - Run pytest locally until the tests pass.
- **Step 4: L1.5 Gate & Auto-Chain to Check**
  - Once local tests pass, run: `uv run ruff check [modified_files]` to ensure lint compliance.
  - **[Auto-Chain Execution]**: Immediately proceed to call the **Check** tool. **Note: `[modified_files]` must include both the new source/test files and the modified parent wiring file(s).**
    `uv run python scripts/lean_check.py --files [modified_files] --spec docs/specs/[feature]_contract.json --skip-lint --skip-mypy`
  - **Do NOT stop or ask for user permission between Implement and Check.** 

### 3. Self-Healing Budget (Max 3 Loops)
- If the integration check (`lean_check.py`) fails, the low-cost model is allowed a maximum of **3 consecutive auto-correction attempts** to fix the errors.
- If it still fails after the 3rd attempt, **STOP execution immediately** and report the error logs to the user for human triage or escalation to the high-reasoning model.

## Output Format (Only output when the entire Pipeline is Green)
```md
### 🏗️ TDD Implementation Pipeline: [Blueprint Name]

#### [IMPLEMENT PHASE]
- **Target Files:** `[src/...]`, `[tests/...]`
- **TDD Cycle (Red-Green-Refactor):**
  - [x] Stub Registration & Verification (ruff/mypy)
  - [x] TDD Test Implementation (Skeleton Mock)
  - [x] Local Tests Passing (Green)

#### [CHECK PHASE]
- **Gatekeeper Validation (lean_check.py):**
  - [x] Passed Auto-Chained L2 Gate (🟢 PASS)
  - [x] Coverage Metric: Cov [value]%
```
