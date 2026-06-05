---
name: scan
description: Scout and map relevant file paths (code, tests, docs) BEFORE any design work.
---

# Skill: Scan (File Discovery & Mapping)

## Purpose
Act as a lightweight scout. Your ONLY goal is to locate the exact files, tests, and documentation related to the user's request. **DO NOT design solutions, analyze logic, or formulate strategies.** 

## Scout Guidelines (Strict Token Efficiency)
1. **Search Only:** Rely heavily on `grep_search`, `glob`, and `list_directory`. 
2. **Zero Logic Analysis:** Do not read entire files. If you find the target class/function via grep, record its path and line number, then immediately stop reading.
3. **Trace Ecosystem:** Always find the Holy Trinity for the target:
   - **Core Code:** Where is the logic defined?
   - **Tests:** Where is the `test_*.py` file for it?
   - **Docs:** Which `docs/domains/*.md` or `docs/architecture/*.md` governs this?
4. **No Guesses:** If you cannot find a related test or doc, explicitly state `None found`. Do not hallucinate paths.

## Output Format
```md
### 🔍 파일 탐색 완료

**1. 핵심 타겟 파일**
- `[File Path]` (Line: X) - *[짧은 설명: e.g., ML Gate 로직 정의부]*

**2. 연관 생태계 (Ecosystem)**
- **의존성/참조:** `[관련된 다른 모듈 Path]`
- **테스트 파일:** `[test_*.py Path]` (없을 경우 'None found')
- **관련 문서:** `[docs/*.md Path]` (없을 경우 'None found')

**3. 다음 단계**
- ➡️ 탐색된 경로를 바탕으로 `spec` 스킬로 넘어가 세부 설계를 진행합니다.
```
