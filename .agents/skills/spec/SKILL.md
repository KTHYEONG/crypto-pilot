---
name: spec
description: Create the smallest sufficient specification before implementation.
---

# spec

Do not implement production code.

## Purpose
Think deeply and architect the solution. You are the lead architect responsible for the technical integrity and long-term maintainability of the codebase.

## Deep Reasoning Guidelines
1.  **Analyze & Reason:** Use the "Surgical Map" from `triage-scan` to jump straight into the logic. Analyze why the current code behaves as it does and how the proposed change affects the broader system.
2.  **Self-Directed Search (High Quality):** While `triage-scan` provides the primary targets, you are responsible for **completeness**. If you suspect side effects or missing dependencies, you MUST perform your own targeted searches (`grep`, `read_file`) to ensure a high-quality, comprehensive design.
3.  **Architecture Alignment:** Verify that your design aligns with the principles in `docs/` and `AGENTS.md/CLAUDE.md`. Don't just patch; improve the structural health.
4.  **Business Logic Depth:** Focus on the "Why". Ensure the solution solves the root cause, not just the symptom.

## Choose One
- `spec-lite`: small or medium feature
- `bug brief`: bug fix
- `PRD`: complex feature or multi-module change
- `refactor plan`: behavior-preserving structure change
- `ADR`: architecture decision

## Required Content
Always include:
- **Goal:** Clear objective.
- **Proposed Solution:** Detailed technical design, reasoning, and logic flow. **This is your core contribution.**
- **Affected Area & Side Effects:** List all files to be modified and potential impacts on other modules.
- **Documentation Impact:** Which `docs/` files must be updated.
- **Acceptance Criteria:** Measurable outcomes.
- **Verification Plan:** How the builder (`implement`) and inspector (`verify`) should prove success.

## Output
```md
### 📝 Spec: [Type]
- **Goal:** [Goal]

#### 🏗️ Technical Design & Reasoning
- **Logic Flow:** [How it works]
- **Architecture Alignment:** [Why this is the right way]
- **Discovery (Self-Search):** [Any additional files/logic found during deep analysis]

#### 📋 Execution Details
- **Affected Area:** `[Files]` | **Docs Sync:** `[Files]`
- **Acceptance Criteria:**
  - [ ] [Criterion 1]
- **Verification Plan:** [Step-by-step for the builder]

- **Ready:** [Yes/No] | **Blocking:** [Questions]
```les to update]`
- **Acceptance Criteria:**
  - [ ] [Criterion 1]
- **Verification Plan:** [Plan]
- **Ready:** [Yes/No] | **Blocking:** [Questions]
```