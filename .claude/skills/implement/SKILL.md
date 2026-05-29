---
name: implement
description: Apply blueprint changes with precision, focusing on 100% fidelity and efficiency.
---

# Skill: Implement

## Purpose
Translate the Blueprint into working code with surgical precision. Follow instructions exactly as defined by the architect.

## Execution Rules
1. **Blueprint Truth:** Read the blueprint in `docs/specs/` using `read_file` before acting. Follow `Contract` and `Logic` strictly.
2. **Surgical Tools:** ALWAYS use `replace` instead of overwriting files whenever possible to minimize token waste and preserve context.
3. **No Drift:** Do NOT perform unrelated refactoring or "clean-up" outside the `Surgical Plan`.
4. **Fallback:** If the Blueprint conflicts with the actual code or if the `replace` tool fails due to context mismatch, **STOP immediately**. Report the conflict and wait for clarification. Do not force the implementation.
5. **Iteration Limit:** Max 3 attempts to fix `verify` failures (lint/test). If unresolved, hand off to `review` or `spec` for re-evaluation.

## Output Format
```md
### 🏗️ 코드 구현: [Blueprint Name]
- **작업 파일 목록:** `[Paths]`
- **작업 진행률:**
  - [ ] Step 1: [Short name]
  - [ ] Step 2: [Short name]
- **현재 상태 및 이슈:** [Status / Conflicts noted]
```
