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

### 2. High-Reasoning Architectural Thought (High Autonomy)
- **Algorithmic & Logical Modeling**: Define mathematical models, data structures, state machines, and state transition rules.
- **System Flow Visualization**: Draw text-based Mermaid sequence/data flow diagrams showing module interactions.
- **Constraints & Boundaries**: 
  - Identify edge cases, performance bottlenecks, and algorithmic limitations, tagging each with a unique label (`[LIMIT-01]`, etc.).
  - **Performance & Resource Budgeting**: Explicitly design the hardware budget using `[PERF-xx]` tags. Define target memory (RSS) caps, GPU VRAM limits, and parallel core configurations conforming to the WSL constraints in `performance.md`.

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
- **Copy-Pasteable Mock Boilerplate**: Provide raw, ready-to-run Python test templates with literal mocks.

## Constraints (Strictly Prohibited)
- **No Python Modifications**: Do NOT create, touch, or modify any `.py` source or test files during the `spec` phase.
- **No Verification Execution**: Never execute `lean_check.py`, `pytest`, `ruff`, or `mypy` during this phase.
- **Immediate Pause (STOP)**: Once the `docs/specs/[feature].md` file is generated, stop tool execution immediately and wait for user feedback. Do not proceed to `check` or run tests.

## Output Format
Create a markdown file at `docs/specs/[feature].md`:

```md
# 🎯 Goal & Architecture
- **Goal**: 1-sentence capability.
- **Mermaid Diagram**: Text-based sequence showing system integration context.

# ⚡ Performance & Resource Budget
- **Complexity**: Time & Space Complexity (Big-O) for core logic.
- **Limits**: `[PERF-01] RSS Limit (e.g. RSS < 4GB)`
- **Concurrency**: `[PERF-02] Concurrency Limit (e.g. max_workers <= 4)`
- **Hardware Acceleration**: `[PERF-03] GPU/VRAM Limit (e.g. VRAM < 2GB or CPU-only)`

# ⚙️ Logical Rules & State Machine
- Logical rules, state transition tables, and tagged constraints (`[LIMIT-01]`, etc.).

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
- **Mock & Integration Boilerplate**: Copy-pasteable test snippet with literal mocks.
```


