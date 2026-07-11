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
- **Dependency Topology Scan**: 
  - Inspect the codebase to identify the target calling module and dependency flow.
  - Ensure the new imports do not violate layering rules or create circular dependencies.

### 2. High-Reasoning Architectural Thought (High Autonomy)
- **Algorithmic & Logical Modeling**: Define mathematical models, data structures, state machines, and state transition rules.
- **System Flow Visualization**: Draw text-based Mermaid sequence/data flow diagrams showing module interactions.
- **Constraints & Boundaries**: Exhaustively identify edge cases, performance bottlenecks, and algorithmic limitations, tagging each with a unique label (`[LIMIT-01]`, etc.).

### 3. Low-Reasoning Implementation Specifications (Deterministic Constraints)
To ensure low-reasoning models can build and integrate the code without guessing:
- **Exact Contract changes**: Define class/function signatures with Python 3.11+ type hints.
- **Integration Spec (Wiring & Data Flow)**:
  - **Connection Point**: Define the exact file path, class, method, and local context anchor (e.g., `Class.method` -> "right before returning") for invocation.
  - **State Mutability & Side Effects**: Specify whether the invocation modifies existing state (`Mutable`) or behaves as a pure function (`Immutable`).
  - **Data Diff**: Express payload modifications using a compact representation: `{"+new_field": "Type", "-deprecated_field": "Type"}`.
- **TDD Scenario Matrix**:
  - Map each LIMIT tag directly to a concrete test scenario.
  - Scenarios:
    - **Scenario 1 (Happy Path)**: Unit input/output.
    - **Scenario 2 (Edge Cases)**: `[LIMIT-xx]` boundary conditions.
    - **Scenario 3 (Error Handling)**: Expected Exceptions and Error logs.
    - **Scenario 4 (Integration Verification)**: Asserting the correct trigger and connection inside the parent module.
- **Copy-Pasteable Mock Boilerplate**: Provide raw, ready-to-run Python test templates with literal mocks.

## Output Format
Create a markdown file at `docs/specs/[feature].md`:

```md
# 🎯 Goal & Architecture
- **Goal**: 1-sentence capability.
- **Mermaid Diagram**: Text-based sequence showing system integration context.

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
- **Scenario 2 (Edge Cases)**: [LIMIT-xx] boundary.
- **Scenario 3 (Error Handling)**: Expected Exceptions.
- **Scenario 4 (Integration)**: Assertion verifying the connection inside the parent module.
- **Mock & Integration Boilerplate**: Copy-pasteable test snippet with literal mocks.
```


