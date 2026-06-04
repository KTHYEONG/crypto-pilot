---
name: review
description: Critically audit compliance with Specs and Project Standards to determine approval.
---

# Skill: Review

## Purpose
Act as the final logical gatekeeper and project maintainer. Critically audit code changes and ensure the system's permanent knowledge base is updated while cleaning up temporary task files.

## Audit Checklist
1. **Logic Integrity:** Do the changes fulfill the "Why" and "How" of the Spec? Check for subtle logical drift or missing edge cases.
2. **Spec Alignment:** Do interfaces, types, and logic match `docs/specs/*.md` 100%?
3. **Surgical Precision:** No unnecessary file modifications or "just-in-case" logic outside `Target Files`?
4. **Standards Review:** Manual check for adherence to project rules (e.g., Python 3.11+, Strict Typing, Logging, Docstrings).

## Finalization & Cleanup (Mandatory for PASS)
If the verdict is **PASS**, you MUST perform the following maintenance before finishing:
1. **Knowledge Promotion:** If the Spec introduced new business rules, invariants, or architectural changes, ensure they are merged into the relevant official documents in `docs/architecture/` or `docs/domains/` according to `documentation.md`.
2. **Garbage Collection:** Delete the temporary blueprint file in `docs/specs/` to prevent AI context pollution.
3. **Doc Sync:** Ensure the `last_verified` date in official documents is updated.

## Verdicts & Routing
- **PASS**: Perfect alignment. (Trigger Cleanup & Promotion).
- **PASS WITH RISKS**: Alignment achieved, but identify risks. (Trigger Cleanup & Promotion).
- **FAIL (Implementation Error)**: Spec is fine, but code has bugs/typos. -> **Handoff to `implement`**.
- **FAIL (Design Error)**: The Spec itself was flawed or didn't account for real-world code. -> **Handoff to `spec`**.

## Output Format
```md
### 🏁 최종 리뷰: [PASS / PASS WITH RISKS / FAIL]

**1. 주요 검토 결과**
- **설계 부합도:** [Pass/Fail] (Ref: `docs/specs/filename.md`)
- **작업 범위 준수:** [Pass/Fail]
- **표준 규격 준수:** [Pass/Fail]

**2. 지식 승격 및 정리 (PASS 시)**
- [ ] 공식 문서 업데이트 완료: `[Path]`
- [ ] 임시 Spec 파일 삭제 완료: `docs/specs/[filename].md`

**3. 발견 사항 및 리스크**
- [Issues/Recommendations]

**4. 후속 조치 (Routing)**
- [Next Step: implement / spec / close]
```
