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
  - Copy the **Skeleton Mock Boilerplate** from the spec into the test file (`tests/...`).
  - Write both the source logic (`src/...`) and the corresponding unit/scenario tests (`tests/...`) simultaneously, adhering strictly to the contract.json spec.
  - Wire the new logic into the parent calling module as defined in the Connection Plan, ensuring both `import_symbol` and `invocation_symbol` (actual call/instantiation) are implemented.
  - Limit high-level scenario tests strictly to the 3-4 scenarios defined in the spec.
- **Step 3: Local Verification & Green Enforcement**
  - Run pytest locally (`uv run pytest -k [target_name]`) to verify that the synthesized code is functional.
  - Ensure that unit tests include strict assertions (verifying exact values, mathematical outputs, and exception types, not just `is not None`) to guarantee semantic correctness.
  - **Do NOT perform design refactoring.** Keep code changes minimal. Refactoring is restricted to basic syntax cleanups and lint compliance.
- **Step 4: L1.5 Gate & Auto-Chain to L2 Check**
  - Once local tests pass, perform local mechanical checks:
    - Run `uv run ruff check --fix [modified_files]` to ensure format/style compliance.
    - Check for and remove any temporary `print()` debugging statements.
  - **[Auto-Chain Execution]**: Immediately trigger the **Check** (L2) gate. Do NOT stop or ask for user permission.
    `uv run python scripts/lean_check.py --files [modified_files] --spec docs/specs/[feature]_contract.json --skip-lint`
  - **Note: Do NOT pass `--skip-mypy` during auto-chaining. Let the Check gate perform strict Mypy and Spec integrity checks.** 

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
