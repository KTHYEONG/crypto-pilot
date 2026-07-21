---
name: spec
description: Actionable implementation blueprint. Logic, Contracts, and Test Scenarios for TDD.
---

# Skill: Spec (Unified Architecture & TDD Blueprint)

## Purpose
Merge core architecture conceptualization (Why/What) and detailed TDD interfaces (How/API Contract) into a single blueprint document (`docs/specs/[feature].md`).
Leverage **high-reasoning models (Thinker)** for creative architectural design, while outputting absolute, deterministic specifications for **low-cost models (Doer)** to implement mechanically with zero token waste.

## Execution Rules

### 1. Pre-process & Dynamic Interview (/grill-me)
- **Decisions Context**: Load `docs/decisions/decisions.md` (align history).
- **Rule Constraints**: Read and strictly adhere to `performance.md` and `quant.md` (timezone isolation, look-ahead bias).
- **Determine Workflow Tier**:
  - **Tier 1 (Light)**: Minor refactor, simple fix. *Directly skip Spec and proceed to implement.*
  - **Tier 2 (Standard)**: Medium complexity. Generate lightweight Markdown spec + `[feature]_contract.json`.
  - **Tier 3 (Architectural)**: Complex module, trading logic. Requires full Markdown spec + `[feature]_contract.json`.
- **Dynamic Interview (/grill-me)**:
  - For **Tier 3** tasks, before creating the spec document, analyze the user's prompt for design ambiguities.
  - Output exactly **3 targeted questions** to clarify architecture, DB schema, API interfaces, or edge-case handling.
  - **STOP immediately** and wait for user answers. Do NOT write the spec until the user answers.

### 2. High-Reasoning Architectural Thought (High Autonomy)
- **Alternatives & Trade-offs**: Contrast multiple design options. Justify why the chosen design is selected.
- **Constraints & Boundaries**: Identify edge cases, performance bottlenecks, and algorithmic limitations, tagging each with a unique label (`[LIMIT-01]`, etc.).
- **Quant & System Resilience**: Explicitly plan for network timeouts, timezone normalization, and look-ahead bias prevention.

### 3. Low-Reasoning Implementation Specifications (Deterministic Constraints)
To ensure low-reasoning models can build and integrate the code mechanically:
- **Exact Contract changes**: Define class/function signatures with Python 3.11+ type hints.
- **Wiring & Connection Plan**: Define the exact file path, class, method, and local context anchor line.
- **Skeleton Mock Boilerplate (CRITICAL)**:
  - Provide a **100% syntactically correct test setup and skeleton mock logic**.
  - Since low-cost models cannot design mocks from scratch, you must output copy-pasteable test skeletons matching the scenarios.
- **TDD Scenario Matrix**:
  - Map each LIMIT and PERF tag directly to a concrete test scenario:
    - **Scenario 1 (Happy Path)**: Unit input/output.
    - **Scenario 2 (Edge Cases)**: `[LIMIT-xx]` boundary conditions.
    - **Scenario 3 (Error Handling)**: Expected Exceptions and Error logs.
    - **Scenario 4 (Integration Verification)**: Asserting the correct trigger inside the parent module.

### 4. Machine-Readable Compliance Contract (`docs/specs/[feature]_contract.json`)
*(Mandatory for Tier 2 and Tier 3)*
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
    {"file": "src/application/parent.py", "anchor": "ExactAnchorSymbol", "import_symbol": "ExactName"}
  ]
}
```

## Output Format
Create a markdown file at `docs/specs/[feature].md`:

```md
# 🎯 Goal & Architecture
- **Goal**: 1-sentence capability.
- **Alternatives & Trade-offs**: Brief comparison.
- **Mermaid Diagram**: Text-based sequence showing system integration context.

# ⚡ Performance & Resource Budget
- Complexity, limits, concurrency.

# ⚙️ Logical Rules, State Machine & Resilience
- Logical rules, state transition tables, tagged constraints (`[LIMIT-01]`, etc.), and resilience/recovery flow.

# 🔌 Integration & Connection Plan
- Target Location, state impact, error behavior.

# ✍️ Contract Changes
- Exact imports, class definitions, function signatures, and return types.

# 🧪 TDD Test Scenario Matrix & Mocks
- Scenario 1-4 descriptions.
- **Skeleton Mock Boilerplate**: Complete mock testing file template.
```

## First Response Protocol after Spec Creation (Token-Efficient)
Do NOT repeat the technical contents. Present a highly-condensed summary in **Korean** within 8 lines:
1. **🎯 핵심 목표**: [Business goal]
2. **🔄 로직 흐름**: [A -> B -> C flow]
3. **⚖️ 설계 핵심 결정**: [Reasoning for design]
4. **👉 후속 대기**: [Wait for user review. Type `Proceed` to start Implement pipeline]
