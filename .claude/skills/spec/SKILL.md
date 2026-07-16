---
name: spec
description: Actionable implementation blueprint. Logic, Contracts, and Test Scenarios for TDD.
---

# Skill: Spec (Unified Architecture & TDD Blueprint)

## Purpose
Merge core architecture conceptualization (Why/What) and detailed TDD interfaces (How/API Contract) into a single blueprint document (`docs/specs/[feature].md`).
Leverage **high-reasoning models** for creative architectural design, while outputting absolute, deterministic specifications for **low-reasoning models** to implement and integrate mechanically with zero token waste.

## Execution Rules

### 1. Pre-process (Context Discovery)
- **Decisions Context**: Load `docs/decisions/decisions.md` (align history).
- **Rule Constraints**: Read and strictly adhere to `performance.md` (WSL resource limits, GPU limits) and `quant.md` (look-ahead bias, timezone isolation).
- **Dependency Topology Scan**: 
  - Inspect the codebase to identify the target calling module and dependency flow.
  - Ensure the new imports do not violate layering rules or create circular dependencies.
- **Data & API Discovery**: 
  - To verify API payloads, database schemas, or external library behavior, you are **encouraged** to create and execute temporary exploration scripts in the `scratch/` folder. Use literal data collection rather than guessing.

### 2. High-Reasoning Architectural Thought (High Autonomy)
- **Alternatives & Trade-offs**: Contrast multiple design options. Justify why the chosen design is selected and state its design trade-offs.
- **Scale-to-Fit Specifying**: For simple helper functions or utility modules, scale down the detailing. Do NOT force strict `[PERF-xx]` budgeting unless the module involves signal calculations, concurrent routines, or heavy IO.
- **Quant & System Resilience**: For trading signals, execution modules, or data collectors, explicitly plan for resilience: network time-outs, database state mismatch, and recovery flows.
- **Algorithmic & Logical Modeling**: Define mathematical models, data structures, state machines, and state transition rules.
- **System Flow Visualization**: Draw text-based Mermaid sequence/data flow diagrams showing module interactions.
- **Constraints & Boundaries**: 
  - Identify edge cases, performance bottlenecks, and algorithmic limitations, tagging each with a unique label (`[LIMIT-01]`, etc.).
  - **Performance & Resource Budgeting**: For resource-intensive components, define hardware constraints using `[PERF-xx]` tags (RSS limit, GPU VRAM, CPU workers matching WSL constraints in `performance.md`).
 
### 3. Low-Reasoning Implementation Specifications (Deterministic Constraints)
To ensure low-reasoning models can build and integrate the code without guessing:
- **Exact Contract changes**: Define class/function signatures with Python 3.11+ type hints.
- **Integration Spec (Wiring & Data Flow)**:
  - **Connection Point**: Define the exact file path, class, method, and local context anchor (e.g., `Class.method` -> "right before returning") for invocation.
  - **State Mutability & Side Effects**: Specify whether the invocation modifies existing state (`Mutable`) or behaves as a pure function (`Immutable`).
  - **Data Diff**: Express payload modifications using a compact representation: `{"+new_field": "Type", "-deprecated_field": "Type"}`.
- **TDD Scenario Matrix**:
  - Map each LIMIT and PERF tag directly to a concrete test scenario.
  - Scenarios:
    - **Scenario 1 (Happy Path)**: Unit input/output.
    - **Scenario 2 (Edge Cases)**: `[LIMIT-xx]` boundary conditions.
    - **Scenario 3 (Error Handling)**: Expected Exceptions and Error logs.
    - **Scenario 4 (Integration Verification)**: Asserting the correct trigger and connection inside the parent module.
- **Skeleton Mock Boilerplate**: Provide the structural test setup and mock boundary logic (focus on verification assertion points rather than verbose syntax completeness).
 
### 4. Machine-Readable Contract (`docs/specs/[feature]_contract.json`)
Generate a JSON contract alongside the spec markdown for automated compliance checking in the check phase:
```json
{
  "contracts": [
    {"kind": "class|function", "name": "ExactName", "file_hint": "src/domain/x.py"}
  ],
  "scenarios": [
    {"id": 1, "scope": "unit", "name": "test_exact_name_happy_path"},
    {"id": 2, "scope": "unit", "name": "test_exact_name_edge_case"},
    {"id": 3, "scope": "unit", "name": "test_exact_name_error"},
    {"id": 4, "scope": "integration", "name": "test_parent_module_wiring"}
  ],
  "wiring": [
    {"file": "src/application/parent.py", "anchor": "ExactAnchorSymbol"}
  ]
}
```
- `contracts`: every public class/fn with `file_hint` (target src/ file).
- `scenarios`: 1:1 with the TDD Scenario Matrix (id 1-4), `name` matching the exact test function name.
- `wiring`: integration connection — parent file path + anchor symbol to search for.

## Constraints (Strictly Prohibited)
- **No Production Code Modifications**: Do NOT create, touch, or modify any `.py` source (`src/`) or official test (`tests/`) files during the `spec` phase. (Exploratory scripts inside `scratch/` are fully allowed).
- **No Scripts Directory Design/Modifications**: Do NOT design, suggest, or write code paths inside the `scripts/` directory. The `scripts/` directory is reserved exclusively for validation/sync tooling. All production logic, auxiliary scripts, and helpers MUST reside in the `src/` directory.
- **No Quality Verification Execution**: Never execute quality-check loops such as `lean_check.py`, `pytest`, `ruff`, or `mypy` during this phase. (Simple `python scratch/temp.py` runs for data collection are fully allowed).
- **Immediate Pause (STOP)**: Once the `docs/specs/[feature].md` file is generated, stop tool execution immediately and wait for user feedback. Do not proceed to `check` or run tests.

## Output Format
Create a markdown file at `docs/specs/[feature].md`:

```md
# 🎯 Goal & Architecture
- **Goal**: 1-sentence capability.
- **Alternatives & Trade-offs**: Brief comparison of alternative design paths and reasons for the chosen design.
- **Mermaid Diagram**: Text-based sequence showing system integration context.

# ⚡ Performance & Resource Budget
*(Note: Can be simplified/omitted for trivial helper modules)*
- **Complexity**: Time & Space Complexity (Big-O) for core logic.
- **Limits**: `[PERF-01] RSS Limit (e.g. RSS < 4GB)`
- **Concurrency**: `[PERF-02] Concurrency Limit (e.g. max_workers <= 4)`
- **Hardware Acceleration**: `[PERF-03] GPU/VRAM Limit (e.g. VRAM < 2GB or CPU-only)`

# ⚙️ Logical Rules, State Machine & Resilience
- Logical rules, state transition tables, tagged constraints (`[LIMIT-01]`, etc.), and resilience/recovery flow.

# 🔌 Integration & Connection Plan
- **Target Location**: `path/to/file.py` > `ClassName.method_name` (anchor context, e.g., "before return")
- **State Impact**: `Mutable (Side-effects)` | `Immutable (Pure Function)`
- **Data Schema Diff**: `{"+new_field": "type", "~modified_field": "new_type"}`
- **Error Behavior**: `Propagate` | `Suppress` (how errors affect the caller)

# ✍️ Contract Changes
- Exact imports, class definitions, function signatures, and return types (100% complete syntax).

# 🧪 TDD Test Scenario Matrix
- **Scenario 1 (Happy Path)**: Input -> Expected Output.
- **Scenario 2 (Edge Cases)**: [LIMIT-xx] boundary / [PERF-xx] resource verification.
- **Scenario 3 (Error Handling)**: Expected Exceptions.
- **Scenario 4 (Integration)**: Assertion verifying the connection inside the parent module.
- **Mock & Integration Boilerplate**: Structural test template demonstrating mock boundaries and assertions.
```


