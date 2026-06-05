---
name: audit
description: 설계 의도 부합 여부 및 시스템 지식 승격(Documentation Sync) 최종 감사.
---

# Skill: Audit (최종 논리 및 지식 감사)

## Purpose
단순한 버그나 오타 검사를 넘어, 변경 사항이 시스템의 아키텍처와 비즈니스 룰에 부합하는지 최종 판단합니다. 작업이 완료되면 임시 설계도(Spec)를 제거하고 공식 문서에 지식을 동기화하여 시스템의 영구적 지식(SSOT)을 유지합니다.

## Audit Checklist
1. **의도적 일관성 (Intent Consistency):**
   - **Spec 작업 시:** 코드 구현이 `docs/specs/*.md`의 "Why"와 "How"에서 벗어나지 않았는가?
   - **Spec-Less 작업 시:** 변경 사항이 사용자의 요청 사항을 논리적 비약 없이 충족하는가?
2. **논리적 견고함 (Logical Integrity):** `check` 단계(기계적 검증)에서 발견하지 못한 잠재적 레이스 컨디션, 잘못된 에러 핸들링 패턴, 또는 비효율적 로직이 없는가?
3. **지식 동기화 (Knowledge Sync):** 신규 비즈니스 규칙이나 아키텍처 변경점이 `docs/architecture/` 또는 `docs/domains/` 등 영구 문서에 정확히 반영되었는가? (`documentation.md` 규칙 준수)
4. **정리 및 마감 (Final Cleanup):** 임시 설계도(`docs/specs/`)를 삭제하여 AI 컨텍스트 오염을 방지했는가? (Spec-Less 작업 시 생략)

## Verdicts & Routing (Circuit Breaker)
- **PASS**: 논리 및 문서화 완벽. (작업 종료).
- **FAIL (Logic/Doc Error)**: 설계와 구현의 불일치 혹은 문서 미비. -> **재수정 요청 (최대 2회 한정)**.
- **CRITICAL FAIL**: 설계 자체의 근본적 결함 발견 또는 반복적 실패. -> **사용자 개입 요청 (Ask User)**.

## Output Format
```md
### 🏁 최종 감사 완료: [PASS / FAIL]

**1. 감사 요약**
- **대상 범위:** [작업된 주요 파일 및 로직]
- **설계 부합도:** [Pass/Fail] (기계적 검증 결과 `check` 스킬 데이터 인용)
- **지식 승격:** [업데이트된 공식 문서 경로 또는 'N/A']

**2. 논리적 완성도 및 품질**
- [ ] 비즈니스 룰 준수 여부
- [ ] 코드 패턴 및 유지보수성
- [ ] 에러 핸들링 및 예외 처리

**3. 지식 관리 및 정리**
- [ ] 공식 문서(`docs/`) 동기화 완료
- [ ] 임시 Spec 파일(`docs/specs/`) 삭제 완료 (해당 시)

**4. 후속 조치**
- [Next Step: Close / Handoff to Implement / Ask User]
```
