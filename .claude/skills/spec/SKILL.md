---
name: spec
description: Actionable implementation blueprint. Logic, Contracts, and Test Scenarios for TDD.
---

# Skill: Spec (TDD & Interface Architect)

## Purpose
Produce high-precision technical specifications, strict interface contracts, and exhaustive test scenario designs.
**CRITICAL:** You are the "High-Reasoning Architect". You solve the logic, define boundaries, and **design the test suite**. DO NOT write full Python code blocks. Provide exact function signatures and mock requirements. You build the blueprint; a lower-reasoning model (Executor) will implement it following TDD.

## Execution Rules

### 1. Pre-process (Scan & Context Alignment)
- **Decisions Context**: Start by loading `docs/decisions/decisions.md` (the cumulative sliding window decisions log) to align with recent architectural decisions and prevent drift.
- **In-Memory Batch Scan**: Do not run multiple turn-by-turn search commands. Generate a batch search plan containing patterns and directories, then invoke a single tool call (e.g., `grep_search` with multi-patterns or `Serena MCP` dependency query) to retrieve all target file candidates in one turn.

## Output 1: Blueprint File (Save to `docs/specs/*.md`)
**[AI-Optimized Format]** Create a deterministic, machine-readable blueprint. It MUST be perfectly structured so the `implement` skill can execute it mechanically in a TDD fashion without needing deep reasoning or asking questions.

- **# 🎯 Objective**: 1-sentence goal.
- **# 📦 Context & Dependencies (CRITICAL for Handoff)**:
  - **Imports**: Exact import statements required.
  - **Data Shapes & Types**: Strict types, Models (Pydantic, etc.), TypedDicts, or NDArray shape definitions.
- **# ✍️ Contract Changes**: EXACT class and function signatures (with full Python 3.11 type hints) and return types.
- **# 🧪 TDD Test Scenario Matrix (CRITICAL for Test-First)**:
  - **Test Environment & Fixtures**: Existing fixtures to use, mock paths, and decorators.
  - **Mock Boilerplate Snippet (CRITICAL)**: If the test requires complex mocks, provide a direct **Raw Python Code** snippet of the setup. Do not just describe it. This ensures the implementer can copy-paste and verify immediately.
  - **Scenario 1 (Happy Path - Test Setup)**:
    - Input: [Exact python expression/data structure]
    - Expected Output: [Expected return or state change]
    - Test function name suggestion: `test_[function_name]_success`
  - **Scenario 2 (Edge Cases - Validation/Bounds)**:
    - Inputs: Zero values, empty data, out-of-bound ranges.
    - Expected Output: Defaults or graceful fallbacks.
  - **Scenario 3 (Error Handling - Exceptions)**:
    - Inputs: Bad parameters, network timeouts.
    - Expected Exception: E.g., `ValueError`, `ConnectionError` with expected message patterns.
- **# 🛠️ Algorithmic Plan**:
  - **Target Location**: `[FILE_PATH]` -> `[TARGET_FUNCTION_OR_CLASS]`
  - **Anchor**: Provide 2-3 lines of the *existing* code exactly as it appears for surgical replacement.
  - **Logic Flow**: Step-by-step logic, formulas, and constraints in pseudo-code format.

## Output 2: Chat Summary (Human-Friendly Briefing)
**[Human-Optimized Format]** Present the result in the chat using clear, non-technical language. Do NOT dump the markdown file content.
Provide a clean summary with:
1. **🚀 Executive Summary (TL;DR):** What is changing and why? (1-2 sentences).
2. **🧩 Key Changes:** High-level bullet points explaining the logic or architecture change.
3. **✅ TDD Scope:** Brief summary of the designed test scenarios.

## Reasoning Constraints
1. **Test Ownership**: You must design the test scenarios. If you do not define the test matrix, the implementer cannot write the tests first.
2. **No Blind Coding**: Delegate Python syntax to `implement`. Focus on the "What" and "How".
3. **Contextual Awareness**: Check `docs/decisions/` and `docs/architecture/` as SSOT.
4. **Token Optimization**: Utilize the single-turn batch scan results. Avoid reading whole files; use precise line ranges in `view_file` only if required.
