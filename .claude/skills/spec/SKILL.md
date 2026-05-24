---
name: spec
description: Create the smallest sufficient specification before implementation.
---

# spec

Do not implement production code.

## Purpose
Create only the amount of specification needed.

## Choose One
- `spec-lite`: small or medium feature
- `bug brief`: bug fix
- `PRD`: complex feature or multi-module change
- `refactor plan`: behavior-preserving structure change
- `ADR`: architecture decision

## Required Content
Always include:
- goal
- non-goals
- affected area
- acceptance criteria
- verification plan
- open questions, if blocking

## Output
```md
## Spec
- Type:
- Path:
- Goal:
- Non-goals:
- Acceptance Criteria:
- Verification Plan:
- Ready for Implementation:
- Blocking Questions:
```