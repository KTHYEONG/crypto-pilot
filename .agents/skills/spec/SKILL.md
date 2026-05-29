---
name: spec
description: Architecture design & machine-readable blueprint for implementation.
---

# Skill: Spec

## Purpose
Produce high-precision technical specifications. High-reasoning models must focus on "Why" and "How", delegating "Action" to the Blueprint.

## Output 1: Blueprint File (Save to `docs/specs/*.md`)
*Purpose: Strict instruction set for the builder AI. Zero ambiguity.*
- **Target Files:** List of files to be modified.
- **Contract:** Exact function signatures, types, and data models.
- **Step-by-Step Logic:** Procedural logic in pseudo-code or bullet points.
- **Surgical Plan:** 
  - `[FILE_PATH]`
  - `[ACTION: ADD/REPLACE/DELETE]`
  - `[CODE_OR_INSTRUCTION]` (Provide exact code snippets for complex logic).
- **Verification:** Precise `uv run` command and expected outcome.

## Output 2: Chat Summary (Brief)
*Purpose: Rapid user alignment. Minimal tokens.*
```md
### 📝 작업 설계: [Type] | [Brief Goal]
- **설계 문서:** `docs/specs/[filename].md`
- **수정 대상:** `[Files changed]`

**1. 설계 방향**
- **핵심 구현 아이디어:** [1-2 sentences on core logic]
- **고려사항/제약조건:** [Why this way? - only if critical]

**2. 완료 기준 (Checklist)**
- [ ] [Measurable outcome 1]
- [ ] [Measurable outcome 2]

**3. 상태:** [Ready to Implement? / Questions for User]
```

## Reasoning Constraints
1. **Verification First:** Before designing, use `read_file` to confirm the FULL signatures and context of functions/classes identified in `triage-scan`. Never assume based on partial imports or 10-line "sniffs".
2. **No Hallucination:** If a path is unverified, run `grep/ls` first.
3. **Context Density:** Do not repeat the prompt. Focus on the Delta (what changes).
4. **Spec Types:** `spec-lite` (minor), `prd` (major), `bug-fix`, `refactor`.
