---
name: spec
description: Actionable implementation blueprint. Logic, Contracts, and Test Scenarios for TDD.
---

# Skill: Spec (Unified Architecture & TDD Blueprint)

## Purpose
Merge core architecture conceptualization (Why/What) and detailed TDD interfaces (How/API Contract) into a single blueprint document (`docs/specs/[feature].md`).
This skill is optimized to leverage **high-reasoning models** for creative architectural design, while outputting absolute, deterministic specifications for **low-reasoning models** to implement mechanically.

## Execution Rules

### 1. Pre-process (Context Discovery)
- **Decisions Context**: Load `docs/decisions/decisions.md` to align with architectural history.
- **Dependency Scan**: Statically inspect the codebase to trace adjacent modules, import structures, and database/DTO schemas.

### 2. High-Reasoning Architectural Thought (High Autonomy)
- **Algorithmic Modeling**: Define the core mathematical models, signal logics, state transitions, and concurrency safety.
- **System Visualization**: Draw clear, text-based Mermaid sequence diagrams or data flows illustrating module interactions.
- **Edge Cases & Guardrails**: Exhaustively list algorithmic edge cases (e.g., Look-ahead bias, race conditions, rounding errors) and assign a unique tag (e.g., `[LIMIT-01]`, `[LIMIT-02]`) to each.

### 3. Low-Reasoning Implementation Specifications (Deterministic Constraints)
To ensure low-reasoning models (in the `implement` phase) can build the code without guessing:
- **Exact Contract changes**: Define class/function signatures with Python 3.11+ type hints and Docstrings.
- **TDD Scenario Matrix**:
  - Map each LIMIT tag from Step 2 directly to a concrete test scenario.
  - Divide scenarios into **Scenario 1 (Happy Path)**, **Scenario 2 (Edge Cases)**, and **Scenario 3 (Error Handling)**.
- **Copy-Pasteable Mock Templates**: Provide raw, ready-to-run Python code for mocking external APIs/DB calls. Ensure mock return values use explicit, literal data (dicts, lists, primitives) rather than dynamic factories.

## Output Format
Create a markdown file at `docs/specs/[feature].md`:

```md
# 🎯 Goal & Architecture
- **Goal**: 1-sentence capability.
- **Mermaid Diagram**: Text-based sequence or data flow.

# ⚙️ Algorithmic Rules & State Machine
- Mathematical formulas, state transition tables, and tagged constraints (`[LIMIT-01]`, etc.).

# ✍️ Contract Changes
- Exact imports, class definitions, function signatures, and return types (100% complete syntax).

# 🧪 TDD Test Scenario Matrix
- **Scenario 1 (Happy Path)**: Input -> Expected Output.
- **Scenario 2 (Edge Cases)**: [LIMIT-xx] boundary conditions.
- **Scenario 3 (Error Handling)**: Expected Exceptions and Error logs.
- **Mock Boilerplate Snippet**: Complete, copy-pasteable Python test snippet with literal mocks.
```

