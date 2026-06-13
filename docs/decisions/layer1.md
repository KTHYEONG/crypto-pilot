# Layer 1 Architectural Decisions

## L1-ADR-001: L1 Nested SWF 통계 유의성 및 MDES 기반 신호 선정 개선 (2026-06-13)
- **Context**: 4/4 OUTER FOLDS가 `empty_opportunities`로 완전히 차단되던 통계적 소표본 병목 해결 목적.
- **Decision**: 단순 임계치 강하 대신 표본 크기에 연동되는 Student's t-distribution 임계값($t_{\text{crit}}$)과 검정력 $80\%$ 기준의 MDES 필터링 공식을 도입하여 소표본 노이즈를 제어하고 유효 신호 기회들을 복구함.
- **Status**: Accepted
