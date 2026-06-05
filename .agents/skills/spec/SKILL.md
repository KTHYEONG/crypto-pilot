---
name: spec
description: Actionable code implementation blueprint. STRICTLY FOR CODE CONVERSION.
---

# Skill: Spec (Implementation Blueprint)

## Purpose
Produce high-precision technical specifications for the `implement` agent. 
**CRITICAL:** This is an EPHEMERAL WORK ORDER. It must clearly communicate the **Reasoning (Why/Direction)** to the User for approval, and the **Action (How)** to the Implementer for execution.

## Output 1: Blueprint File (Save to `docs/specs/*.md`)
*Purpose: Zero ambiguity instruction set for the builder AI.*

- **# 🎯 Objective**: 1-sentence goal (e.g., "Improve ML Gate logic to reduce false positives").
- **# 💡 Strategy**: A brief summary of *what* was found during analysis and *how* it will be improved (The "Logic" behind the change).
- **Target Files**: List of exact files to be modified.
- **Contract Changes**: Exact function signatures, types, and data models to add/change.
- **Surgical Plan**: 
  - `[FILE_PATH]`
  - `[ACTION: CREATE / UPDATE / DELETE]`
  - `[TARGET_FUNCTION_OR_CLASS]`
  - `[EXACT_CODE_OR_INSTRUCTION]` (Provide exact code for complex logic).
- **Verification**: Precise `uv run` commands to prove success.

## Output 2: Chat Summary (Brief)
*Purpose: Rapid user alignment & approval. Minimal tokens.*
```md
### 📝 Design Approval Request: [Title]
> **Summary:** [1-sentence summary of the problem or need for improvement]

**1. Strategy**
- [Explain how the logic will change in 2-3 readable bullets]
- [e.g., "Change ML Gate thresholding from weighted average to EMA"]

**2. Scope of Work**
- **Spec Path:** `docs/specs/[filename].md`
- **Modified Files:** `[File names...]`

**3. Success Criteria**
- [ ] [Specific outcome or state expected after implementation]

**Status:** Should I proceed with this design? (Yes/No or feedback)
```

## Reasoning Constraints
1. **Analysis First**: Before writing the spec, you MUST explain to yourself "What is wrong with the current code?". This logic must be reflected in the **Strategy**.
2. **User-Centric**: The Chat Summary must be readable by a human. Avoid overly cryptic jargon where simple terms suffice.
3. **No Documentation Overlap**: Keep this file as a throw-away task list. Only record permanent rules in `documentation.md`.
