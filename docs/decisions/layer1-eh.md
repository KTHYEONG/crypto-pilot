# Layer 1 Architectural Decisions

## L1-ADR-009: PIT state_cube 통합 + lifecycle gate + capacity clip (2026-06-19)
- **Delta:** `run_l1_nested_swf`에 `l2_start: date | None` 파라미터 추가. `SymbolLifecycleRecord` dataclass 도입(`fold_status`, `promotion_available_at`). `Layer1Result.symbol_lifecycle` 필드 추가. `active_mask`(state_cube 파생)로 per-symbol `promotion_available_at` 결정. `awf_sim` fit/OOS 루프에 `capacity_usdt` clip + 5 USDT min order.
- **Rationale:** PIT universe state_cube가 L1 fold에 반영되지 않으면 all-True active_mask로 look-ahead 노출. Lifecycle gate는 `promotion_available_at > l2_start` 심볼이 L2 `oos_stacked`에 포함되는 것을 차단. Capacity clip은 micro-position 거래비용 현실화.
- **Edge Cases:** `active_mask` all-True (stage6 경로 호환) → 모든 심볼 `promo_at = datetimes[l1_start_bars].date() <= l2_start` → 제외 없음(기존 동작 보존).

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

## L1-ADR-010: L1 Pipeline 성능 최적화 — Python loop 제거 및 vectorization (2026-06-18)
- **Delta**: 4개 파일 최적화: (1) `_by_q_values` backward loop → `np.minimum.accumulate`, (2) `_compute_incremental_bps` agg+join → transform, (3) `compute_symbol_strategy_evidence` 중복 .copy() 제거 + bool mask 단일화 + sort=False, (4) `select_candidate_events_for_portfolio` per-group sort → pre-sort + itertuples() → cumcount().
- **Rationale**: L1 PERF 실측 결과 (48.95s) 기반. `selection` 70-90%(1.1~3.7s/fold)와 `prep` 78-84%(0.6~1.6s/snapshot)가 주요 병목. 품질 훼손 없는 내부 알고리즘 최적화로 40~60% 단축 예상. + `--phase l1` 실행 시 `Layer1Result.labeled` 미존재로 인한 `AttributeError` 수정 (isinstance guard).
- **Files Changed**: `opt_main_futures.py`, `signal_selection.py`, `candidate_portfolio.py`.
- **Status**: Accepted

## L1-ADR-009: L1 PERF(15) 로그 계층적 타이밍 시스템 도입 (2026-06-18)
- **Delta**: `src.core.utils.utils.PERF=15` 로그 레벨을 모든 L1 서브페이즈에 적용. 기존 `logger.debug` → `logger.log(PERF)` 이관. 신규 마커 `[L1-CTX]`, `[L1-FOLD]`, `[CANDIDATE-FOLD]`, `[SIGNAL-EVIDENCE]`, `[AWF-PERF]` (`[L2-AWF-PROF]` 대체) 추가. `run_l1_nested_swf` evidence_snapshots 타이밍, outer fold per-fold 타이밍, candidate_workflow 워커 내 timing_profile PERF emit 추가. `signal_selection.py` evidence loop prep/stats/qualify 3단계 분해 타이밍. `awf_sim.py` DEBUG→PERF 마이그레이션 + per-fold 로그. `opt_main_futures.py` `"alo"/"full"` 레거시 phase 제거, L1-only 방어형 가드 `{"l2","l3"}` 도입.
- **Rationale**: L1 병목탐지가 DEBUG(10) 레벨에 흩어져 있어 운영 중 실시간 모니터링 불가. PERF(15)로 통일하여 `--phase l1` 실행 시 계층적 소요시간 로그만으로 병목지점 식별 가능하게 함. 실제 54심볼 측정 결과 `selection` 70-90%, `SIGNAL-EVIDENCE.prep` 78-84%가 주요 병목으로 확인됨.
- **Files Changed**: `pipeline.py`, `candidate_workflow.py`, `signal_selection.py`, `awf_sim.py`, `opt_main_futures.py`, `test_l1_determinism.py`.
- **Status**: Accepted

## L1-ADR-011: L1 성능 최적화 2차 — Numba JIT 도입 및 q-value 롤백 (2026-06-19)
- **Delta**: 
  - `candidate_dataset.py` 의 `_rolling_robust_z_1d`, `_rolling_robust_z_2d`, `_cross_sectional_robust_z_2d` 함수들을 Numba JIT (`@njit(cache=True)`)을 사용한 C레벨 고성능 연산 루프로 전면 재구현.
  - `signal_selection.py` 의 `_by_q_values` FDR 조정을 기존 numpy vectorization 버전에서 루프 기반 백프로파게이션 방식으로 롤백.
  - `pipeline.py` 의 tiered_workflow 로거에 `LOG_LEVEL=PERF` 동적 환경 변수 활성화 조건 추가.
- **Rationale**: 
  - 1차 성능 최적화 후에도 L1 Pipeline Total 소요시간이 47.90초로 큰 진전이 없었음.
  - 초정밀 타이밍 프로파일링 분석 결과, 전체 실행 시간의 58.3%가 `prime_aligned_feature_cache` (27.78초) 한 곳에서 발생했으며, 이는 Pandas rolling apply(Python lambda) 연산 오버헤드가 원인이었음.
  - 이를 Numba JIT로 가속하여 캐시 프라이밍 시간을 27.78초에서 **7.75초(72.1% 단축)**로 단축하고, L1 전체 소요시간을 47.64초에서 **25.17초(47.2% 단축)**로 단축시킴.
  - 또한, N<=200 이하 소표본 환경에서 numpy slicing 오버헤드로 회귀를 일으켰던 `_by_q_values`를 루프 기반으로 롤백하여 정합성 복구 및 가속 효과를 순증으로 돌려놓음.
- **Status**: Accepted

