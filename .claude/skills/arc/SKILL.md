---
name: arc
description: Core system logic, data shapes, and algorithm guidelines.
---

# Skill: Arc (Core System Designer)

## Purpose
Define the core logical boundaries, mathematical formulas, state transitions, and concurrency safety guidelines for the task. Focus purely on "Why" and "What" using minimal tokens.

## Execution Rules

### 1. Pre-process (Context Alignment)
- **Decisions Context**: Start by loading `docs/decisions/decisions.md` (the cumulative sliding window decisions log) to align with recent architectural decisions and prevent drift.
- **Context Scan**: Retrieve target file candidates or schemas to understand the structural dependencies.

### 2. Core Logic Design
- Write the algorithmic blueprint and mathematical models.
- **Visualize Design (CRITICAL)**: Draw a clean text-based sequence diagram or data flow using Mermaid syntax to explain state transitions or data routing.
- Define exact data schemas/shapes (e.g., Pydantic or TypedDict schemas).
- List potential edge cases (e.g., Look-ahead bias, race conditions, latency trade-offs).
- Prohibit writing full Python code blocks or detail-level mock specifications here.

## Output Format
Create a markdown file at `docs/specs/[feature]_core.md`:

- **# 🎯 Goal**: 1-sentence description of the target capability.
- **# 🧩 Core Data Shapes**: Exact fields, types, and model specifications.
- **# ⚙️ Algorithmic Rules & State Machine**: Critical logic flows, state change transitions, and mathematical formulas.
- **# ⚠️ Constraints & Edge Cases**: Execution safety rules, concurrency protection, and quant-specific constraints (e.g., Look-ahead prevention).
