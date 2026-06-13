# Layer 1 Architectural Decisions

## L1-ADR-001: L1 Nested SWF 통계 유의성 및 MDES 기반 신호 선정 개선 (2026-06-13)
- **Context**: 4/4 OUTER FOLDS가 `empty_opportunities`로 완전히 차단되던 통계적 소표본 병목 해결 목적.
- **Decision**: 단순 임계치 강하 대신 표본 크기에 연동되는 Student's t-distribution 임계값($t_{\text{crit}}$)과 검정력 $80\%$ 기준의 MDES 필터링 공식을 도입하여 소표본 노이즈를 제어하고 유효 신호 기회들을 복구함.
- **Status**: Accepted

# Layer1 Signal Validation Restructure
- Context: nested SWF repeated fit cost and regime-based sample fragmentation kept Layer1 underpowered even after gate fixes.
- Decision: reuse causal prequential evidence snapshots keyed by `as_of_idx`, pool regime cells by default, and keep regime as risk overlay only.
- Decision: preserve `quality_weight` in ranking while keeping compatibility `qualified` tied to `hard_eligible and quality_weight > 0.0`.
- Consequence: production readiness now depends on `fold_cov`, `match_ratio`, `sym_count`, `fold_ratio`, and `probe_lcb_bps`; CPCV stays out of the production path.
- Status: Accepted
