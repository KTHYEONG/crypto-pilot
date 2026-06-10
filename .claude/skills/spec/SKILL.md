---
name: spec
description: Actionable implementation blueprint. STRICTLY FOR LOGIC AND CONTRACT DEFINITION.
---
name: spec
description: Actionable implementation blueprint. Logic, Contracts, and Test Scenarios.
---

# Skill: Spec (Implementation Blueprint)

## Purpose
Produce high-precision technical specifications and logic guidelines. 
**CRITICAL:** You are the "High-Reasoning Architect". You solve logic and **design the test strategy**. DO NOT write full Python code blocks. Provide strict pseudo-code and exact function signatures.

## Output 1: Blueprint File (Save to `docs/specs/*.md`)
- **# 🎯 Objective**: 1-sentence goal.
- **Contract Changes**: EXACT function signatures (with Python typing), data models, and return types.
- **Surgical Plan**: 
  - `[FILE_PATH]`
  - `[TARGET_FUNCTION_OR_CLASS]`
  - `[ALGORITHMIC_FLOW]`: Step-by-step logic, formulas, and constraints. Use pseudo-code.
- **# 🧪 Test Scenario Design (CRITICAL)**:
  *Purpose: Define 'What to test' so the implementer can write the Python test code.*
  - **Scenario 1 (Happy Path)**: Given [Input] -> When [Action] -> Then [Expected Output/State].
  - **Scenario 2 (Edge Case)**: e.g., Empty data, Zero division, Timeout.
  - **Scenario 3 (Error Handling)**: Expected exceptions and messages.
- **Verification**: Precise `uv run` commands (e.g., `pytest -k ...`).

## Output 2: Chat Summary
(Same as before: Q&A, Architecture Delta, Scope, and Approval Request)

## Reasoning Constraints
1. **Test Ownership**: You must design the test scenarios. If you don't define it, the implementer won't build it correctly.
2. **No Blind Coding**: Delegate Python syntax to `implement`. Focus on the "What" and "How".
3. **Contextual Awareness**: Check `docs/decisions/` and `docs/architecture/` as SSOT.
4. **Token Optimization & Contract Inspections**:
   - **Primary**: Utilize Serena MCP (`get_symbols_overview`, `find_implementations`) to inspect interface layouts, class boundaries, and inheritance lines without fetching file bodies.
   - **Secondary**: Use `view_file` or `read_file` with precise line ranges (e.g., `StartLine` and `EndLine` parameters) only to study actual implementation details if pseudo-code design requires it. Avoid reading whole files.
