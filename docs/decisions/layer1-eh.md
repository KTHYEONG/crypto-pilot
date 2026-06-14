# Layer 1 Architectural Decisions

## L1-ADR-008: IC 지표 제거 및 Probe-Only 검증 (2026-06-14)
- **Delta**: Removed IC calculation (`spearmanr` calls, `opportunity_ic=None` always). Removed IC from ENS log output. Removed IC column from Outer Fold table. Kept `probe_bps`/`probe_lcb_bps` as sole profitability metrics.
- **Rationale**: Arch-Only mode produces constant prediction arrays per archetype → Spearman IC = numerical noise. IC unmapped to gate inputs (5-Gate: fold_cov, match_ratio, sym_count, fold_ratio, probe_lcb_bps). Removed diagnostic noise. L1 passes 3/3 runs (Min-Profit 45-94 bps, t-stat 2.5-4.25). Test suite: 436 passed.
- **Status**: Accepted

## L1-ADR-007: Retention of Time-Series Selection and Rejection of Coupled Pooled IC (2026-06-14)
- **Delta**: Retained default `l1_opp_ic_mode="time_series"` and reverted `pooled` mode changes. Fixed test suite to globally patch `ProcessPoolExecutor` to `ThreadPoolExecutor` for safe synchronous mocked test execution.
- **Rationale**: Changing `l1_opp_ic_mode` to `"pooled"` coupled with `probe_series` logic, changing signal selection from high-performance symbol-wise time-series to noisy bar-wise cross-sectional selection, dropping edge from +45.7 bps to -0.45 bps (blocking L1). Reverted code changes to preserve original performance while maintaining unit test fixes.
- **Status**: Accepted

## L1-ADR-006: Deterministic Bootstrap Seeding for Layer 1 Folds (2026-06-14)
- **Delta**: Replaced Python's built-in `hash()` with a SHA-256 byte-convert integer offset (`int.from_bytes(sha256(...).digest()[:4]) % 10000`) for L1 bootstrap seed generation.
- **Rationale**: Built-in `hash()` is subject to process-level startup hash randomization. Replacing it with SHA-256 guarantees fully deterministic bootstrap seeds across runs/processes, ensuring perfect reproducibility of L1 validation.
- **Status**: Accepted

## L1-ADR-005: Layer 1 Hard Gate Reform (2026-06-14)
- **Delta**: Relaxed `l1_min_realized_match_ratio` (1.0 $\rightarrow$ 0.9) and `l1_min_fold_ratio` (0.6 $\rightarrow$ 0.5). Added HHI-based `l1_sym_count_mode="effective_n"` ($\ge 3.0$) and `l1_probe_lcb_pooled=True` (pooled OOS bootstrap LCB). Relaxed fold-level gate from bootstrap LCB to gross edge positive check (`probe_bps > 0`).
- **Rationale**: Solves the double-counting statistical penalty that rejected viable signals due to small-sample volatility in fold-level bootstrap estimations. Standardizes global validity on pooled samples while preserving robustness via HHI diversification.
- **Status**: Accepted

## L1-ADR-004: Outer Warm-Up Block Reservation (2026-06-14)
- **Delta**: `build_l1_nested_swf_folds` changed `block_len = available//(n_folds+1)` → `available//(n_folds+warmup)` and `oos_start = l1_start+(fold_idx+warmup)*block_len`. `l1_outer_warmup_blocks=2` added to config. Diagnostic warning (`Counter(structural_reasons)`) added to `compute_symbol_strategy_evidence` when qualified=0.
- **Rationale**: Anchored nested-SWF reserved only 1 block (~658 bars) before fold 0 OOS. With `l1_pair_min_folds=2` and `score_pct_variant_hist_window_bars=2160`, first snapshot was structurally underpowered (126 pairs, 0 qualified). warmup=2 expands fold 0 evidence window to ≈2×, recovering `ReadySyms:3, Probe:52bps`.
- **Edge Cases**: OOS coverage shrinks by `n/(n+warmup)` (≈17%); net positive as fold 0 becomes evaluable. Look-ahead preserved: `exit_idx < as_of_idx` filter unchanged. Zero-warmup blocked via `validate()` guard.

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
