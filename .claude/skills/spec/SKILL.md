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
**[AI-Optimized Format]** Create a deterministic, machine-readable blueprint. It MUST be perfectly structured so the `implement` skill or Codex can execute it flawlessly without re-reading files or asking questions.

- **# 🎯 Objective**: 1-sentence goal.
- **# 📦 Context & Dependencies (CRITICAL for Handoff)**:
  - **Imports**: Exact import statements required (e.g., `from typing import Mapping`).
  - **Data Shapes**: Briefly define the structure of custom objects passed to the logic (e.g., `Config(BaseModel) has .symbol(str) and .qty(float)`).
- **# ✍️ Contract Changes**: EXACT function signatures (with Python typing) and return types.
- **# 🛠️ Surgical Implementation Plan**: 
  - **Target**: `[FILE_PATH]` -> `[TARGET_FUNCTION_OR_CLASS]`
  - **Anchor**: Provide 2-3 lines of the *existing* code exactly as it appears. This acts as a search anchor so the implementer knows exactly *where* to inject or replace code.
  - **Algorithmic Flow**: Step-by-step logic, formulas, and constraints. Use strict pseudo-code.
- **# 🧪 Test Scenario Design & Mocks**:
  - **Test Environment**: Existing fixtures or mocks to utilize (e.g., `Use @patch('src.core.exchange.Client')`).
  - **Scenario 1 (Happy Path)**: Given [Input] -> When [Action] -> Then [Expected Output/State].
  - **Scenario 2 (Edge Case)**: e.g., Empty data, Zero division, Timeout.
  - **Scenario 3 (Error Handling)**: Expected exceptions and messages.
- **# 🛡️ Verification**: Precise `uv run` commands (e.g., `pytest -k ...`).

## Output 2: Chat Summary (Human-Friendly Briefing)
**[Human-Optimized Format]** You MUST present the result in the chat using clear, non-technical language. Do NOT just dump the markdown file content. The user should not have to ask "Explain this easily."

Provide a clean summary with:
1. **🚀 Executive Summary (TL;DR):** What is changing and why? (1-2 sentences).
2. **🧩 Key Changes:** High-level bullet points explaining the logic or architecture change without code blocks.
3. **✅ Expected Impact:** How this solves the problem or improves the system.

## Reasoning Constraints
1. **Test Ownership**: You must design the test scenarios. If you don't define it, the implementer won't build it correctly.
2. **No Blind Coding**: Delegate Python syntax to `implement`. Focus on the "What" and "How".
3. **Contextual Awareness**: Check `docs/decisions/` and `docs/architecture/` as SSOT.
4. **Token Optimization & Contract Inspections**:
   - **Primary**: Utilize Serena MCP (`get_symbols_overview`, `find_implementations`) to inspect interface layouts, class boundaries, and inheritance lines without fetching file bodies.
   - **Secondary**: Use `view_file` or `read_file` with precise line ranges (e.g., `StartLine` and `EndLine` parameters) only to study actual implementation details if pseudo-code design requires it. Avoid reading whole files.
