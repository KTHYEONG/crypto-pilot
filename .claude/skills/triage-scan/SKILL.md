---
name: triage-scan
description: Classify tasks and map repository context efficiently before design.
---

# Skill: Triage-Scan

## Purpose
Quickly identify task type, strategy, and locate relevant code/tests with minimal token usage. Do not design or read full files.

## Scan Guidelines (Efficiency)
1. **Live Truth:** Use `ls`, `glob`, and `grep` only. Use live code as the source of truth.
2. **Pinpoint Grep:** Use `total_max_matches: 1` and minimal context to confirm existence, not deep logic.
3. **Index Mapping:** Map architecture via folder/file names. Read only the first 5-10 lines if the purpose is unclear.
4. **No Full Reads:** Never read >20 lines of any file. Defer deep analysis to the `spec` phase.

## Triage Classifications
- **Type:** bug | feature | refactor | quant | docs | config
- **Risk:** low | medium | high
- **Strategy:** direct patch | TDD | spike | regression-first

## Output Format
```md
### 🔍 작업 탐색: [Type]

**1. 작업 개요**
- **해결 방향:** `[Strategy]` (예상 위험도: `[Risk]`)
- **관련 모듈/계층:** `[Layer Name]`

**2. 영향 범위 및 참조**
- **수정할 주요 코드:** `[Path]:[Line]` (via [Keyword])
- **테스트 파일:** `[Path]`
- **의존성:** `[Key Imports/Classes]`
- **관련 파일:** `[Relevant Filenames]`

**3. 다음 단계**
- [Hand off verified paths to `spec` or `implement`]
```
