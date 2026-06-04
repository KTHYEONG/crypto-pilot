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
### 📝 설계 승인 요청: [Title]
> **검토 결과:** [기존 코드의 문제점이나 개선이 필요한 이유를 1줄로 요약]

**1. 개선 방향 (Strategy)**
- [어떤 로직을 어떻게 바꿀 것인지 사용자가 이해하기 쉽게 2-3개 불릿으로 설명]
- [예: "ml_gate의 임계값 계산 방식을 가중치 평균에서 지수 이동 평균으로 변경"]

**2. 작업 범위**
- **문서 위치:** `docs/specs/[filename].md`
- **수정 파일:** `[File names...]`

**3. 완료 기준**
- [ ] [구현 후 기대되는 구체적인 결과물이나 상태]

**상태:** 이 설계대로 진행할까요? (Yes/No 또는 의견)
```

## Reasoning Constraints
1. **Analysis First**: Before writing the spec, you MUST explain to yourself "What is wrong with the current code?". This logic must be reflected in the **Strategy**.
2. **User-Centric**: The Chat Summary must be readable by a human. Avoid overly cryptic jargon where simple terms suffice.
3. **No Documentation Overlap**: Keep this file as a throw-away task list. Only record permanent rules in `documentation.md`.
