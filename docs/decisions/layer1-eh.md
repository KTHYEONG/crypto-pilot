# Layer 1 Architectural Decisions

## L1-ADR-003: Prequential Evidence Grid Separation (2026-06-14)
- **Delta**: Evidence fold count decoupled from outer fold count: `ev_n_folds = min(outer_n × l1_evidence_grid_multiplier, l1_evidence_max_folds)` replacing `min(wf_n_folds, 3)`. IC `None → 0.000` render bug fixed to `n/a`.
- **Rationale**: Prior design produced identical grid spacing → fold 0 had 0 matured evidence pairs (starvation); fold 1 had single-fold evidence (`n_folds=1 < l1_pair_min_folds=2`) causing 100% qualification dropout despite 513 pairs. Multiplier=3 ensures ≥2 matured blocks before first outer OOS.
- **Edge Cases**: Multiplier enforces effective floor of 3 regardless of config (< config=2 would undercut min_folds+1 invariant). `l1_evidence_max_folds=32` caps compute explosion under large outer_n.

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
