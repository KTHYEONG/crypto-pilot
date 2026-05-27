---
name: spec
description: Create a deep architectural design and a detailed implementation blueprint.
---

# spec

Do not implement production code. Your goal is to provide high-level architectural alignment and a low-level implementation blueprint.

## Purpose
Think deeply and architect the solution. You are the lead architect responsible for the technical integrity and long-term maintainability of the codebase. You produce two outputs:
1.  **Chat Summary:** A concise overview of the design for the user.
2.  **Blueprint File:** A detailed, machine-readable specification for the builder AI, saved in `.agents/specs/`.

## Deep Reasoning Guidelines
1.  **Analyze & Reason:** Use the "Surgical Map" from `triage-scan` to jump straight into the logic. Analyze why the current code behaves as it does and how the proposed change affects the broader system.
2.  **Completeness:** If you suspect side effects or missing dependencies, you MUST perform your own targeted searches (`grep`, `read_file`) to ensure a high-quality, comprehensive design.
3.  **Architecture Alignment:** Verify alignment with `docs/` and `AGENTS.md`. Focus on the "Why".

## Two-Phase Output Workflow

### Phase 1: Create Blueprint File (Use `write_file`)
Save a detailed specification to `.agents/specs/[feature_name].md`. This file must be structured for high-precision coding AI consumption:

```markdown
# Blueprint: [Feature Name]
- **Target Files:** `[Paths]`
- **Context:** [Brief reasoning]

## 1. Symbolic Interface Definitions
- **Data Models:** [Pydantic/Dataclass definitions]
- **Signatures:** [Exact function/method signatures with types]

## 2. Step-by-Step Logic (Procedural)
1. **Step 1:** [Specific action/condition]
2. **Step 2:** [Specific action/condition]
...

## 3. Surgical Edit Plan
- **File:** `[Path]`
  - [ADD/REPLACE/DELETE] [Specific code block or instruction]
- **File:** `[Path]`
  - [ADD/REPLACE/DELETE] [Specific code block or instruction]

## 4. Verification Snippet
- **Command:** `[uv run ...]`
- **Expected Output:** [Expected result]
```

### Phase 2: Chat Summary (Brief)
After saving the file, provide a concise summary in the chat:

```md
### 📝 Spec: [Type]
- **Goal:** [Goal]
- **Blueprint Saved:** `.agents/specs/[feature_name].md`

#### 🏗️ Technical Design Summary
- **Logic Flow:** [High-level summary of the algorithm/logic]
- **Architecture Alignment:** [Why this approach was chosen]

#### 📋 Quick Overview
- **Affected Area:** `[Files]`
- **Acceptance Criteria:**
  - [ ] [Criterion 1]
- **Ready:** [Yes/No] | **Blocking:** [Questions]
```

## Choose One for [Type]
- `spec-lite`: small or medium feature
- `bug brief`: bug fix
- `PRD`: complex feature or multi-module change
- `refactor plan`: behavior-preserving structure change
- `ADR`: architecture decision
