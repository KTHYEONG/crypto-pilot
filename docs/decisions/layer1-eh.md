# Layer 1 Architectural Decisions

## L1-ADR-013: L1/Universe 파이프라인 루프 불변식 호이스팅 + 벡터화 (2026-06-20)
- **Delta:** (OPT-1) `align_data_maps` state_cube/readiness_cube 조인 루프에서 `np.searchsorted`/`positions`/`t_valid`/`p_valid` 를 심볼 루프 밖으로 호이스팅. pandas 3.0 bug fix: `calendar.as_unit("ns").asi8` 로 nanosecond epoch 강제. (OPT-2) `inject_membership_masks_into_maps` 진입 시 `_normalize_timeline()` 헬퍼로 timeline 1회 정규화 후 `build_membership_mask_bundle` 에 전달 (104회→1회). (OPT-3) `run_l1_nested_swf` 의 `volatility_2d` 컬럼 루프를 `pd.DataFrame.rolling().std(ddof=1)` 단일 행렬 호출로 대체. (OPT-6) `load_futures_data_maps_for_symbols` 말미 도달불가 중복 `return` 1줄 제거.
- **Rationale:** PERF 실측(52 syms, 617K events): searchsorted N=52→1, timeline 정규화 104→1. 수치 결과·공개 시그니처 불변. pandas 3.0.2 `.asi8` microsecond 버그는 pre-existing silent production bug였음(unit 미강제 시 1000× 스케일 오류).
- **Edge Cases:** pandas 3.0 호환: `as_unit("ns")` 없이 `.asi8` 사용 금지. `_normalize_timeline` 은 `norm_timeline=None` fallback으로 하위호환 유지. OPT-4(ablation twin diagnostics 5.66s) 기각 — bridge 3.4%, 게이트 정의 리스크 대비 절감 미미.

## L1-ADR-012: Promotion Summary 로그 PASS/FAIL 분리 + 우측절단 진단 (2026-06-19)
- **Delta:** `format_layer1_deployment_registry_table`에 `all_evidence` 파라미터 추가 → admit된 쌍은 `[L2-PASS] Q:hi/mid/lo` 전체 출력, 미admit 쌍은 `[NOT PROMOTED] N pairs | top: <reason>xN` 1줄 요약. `Layer1FoldReadiness`에 `dropped_by_maturity_count` 필드 추가 → Outer Fold 로그에 `[censored: N]` 노출. STATUS 어휘 PROMOTED/WATCH/REJECTED 제거 (오해 유발: "REJECTED"도 L2 전달됨). `pipeline.py` 호출부에 `all_evidence=deployment_evidence` 배선 완료.
- **Rationale:** PROMOTION SUMMARY의 STATUS는 q_value 버킷 표시 전용이며 실제 L2 전달 게이트(`build_qualified_signal_registry`)와 무관. "REJECTED" 심볼도 전부 L2로 유입 → 과대-전달 오독 방지. Fold #3 Edge 30bps가 실제 약세인지 우측절단 편향인지 분리 불가 문제 해소.
- **Edge Cases:** `all_evidence=()` 생략 시 기존 동작(FAIL 섹션 미출력) 보장. `dropped_by_maturity_count=0` 시 `[censored:]` 미노출.


## L1-ADR-010: IC 제거 완성 + mu_quality_shrinkage dead-code 제거 (2026-06-19)
- **Delta:** `predict_regime_conditional_ensemble`에서 `mu_quality_shrinkage` 블록(4줄) 제거. `mu_shrinkage_lambda` diagnostics 키 제거. `test_mu_quality_shrinkage_*` 테스트 2건 삭제. `test_auto_conditioning_exposes_diagnostics`에서 `mu_shrinkage_lambda` absent 검증으로 전환.
- **Rationale:** L1-ADR-008(Accepted)에서 IC 계산 제거 후 `validation_rank_ic=0.0`이 항상 기본값 → `lam=clip(0/0.05,0,1)=0.0` → mu가 단면평균으로 붕괴 (신호 차별성 전멸). `mu_quality_shrinkage_enabled=False` 기본값이라 현재 실행에서는 무영향이었으나 silent landmine. 완전 제거.
- **Edge Cases:** `mu_quality_shrinkage_enabled=True`로 켜던 외부 실험은 설정 무효화됨(기능 자체 삭제). `validation_rank_ic` 필드는 dataclass에 유지(진단용 0.0 기본값).

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


## L1-ADR-013: Ingestion & Alignment Performance Optimization (2026-06-20)
- **Delta:** 
  - Changed parallel symbol loading in `opt_data_utils.py` from `ProcessPoolExecutor` to `ThreadPoolExecutor`.
  - Added pre-conversion of meta columns (`pd.to_numeric`) at the initial ingestion phase in `load_single_symbol_data`.
  - Bypassed redundant `pd.to_datetime` calls in `_resolve_tradeable_scope` using a fast `datetime64` type check.
  - Optimized the meta column scanning loop in `align_data_maps` to scan NumPy-backed array views for non-NaN values, avoiding Series creation.
- **Rationale:** 
  - Loading 57+ symbols under `ProcessPoolExecutor` suffered from severe Python object serialization (pickling) overhead over IPC. Bypassing it via `ThreadPoolExecutor` yielded massive speedups since PyArrow/Pandas release the GIL during file decompression.
  - Doing `pd.to_numeric` on 12 meta columns inside the alignment loop caused `S * M` Series allocations. Pre-converting at ingestion and using raw NumPy scanning during alignment optimized critical latency paths while keeping 100% data fidelity and look-ahead safety.
- **Status**: Accepted

## L1-ADR-014: L1 데이터 준비 병목 최적화 — Numba 및 DatetimeIndex.isin 기반 최적화 (2026-06-20)
- **Delta:** 
  - `membership.py`에 Numba `@njit` 가속화된 `_calculate_warm_ready_numba`를 도입하여 Pandas groupby-cumsum 루프를 대체하였고, `build_membership_mask_bundle` 내 날짜비교를 `pd.Timestamp`와 `DatetimeIndex.isin` 기반 벡터화로 고속화하여 `datetime.date` 객체 생성 오버헤드를 우회함.
  - `opt_data_utils.py` 내 `_feature_group_coverage`에서 수치형 컬럼의 to_numeric 강제 변환을 우회하고, 1분 데이터 로딩 시 이미 monotonic/datetime64인 경우 정렬 및 타입 변환을 생략하도록 정돈함.
- **Rationale:** 
  - `[SKIPPED]` 로그 출력 후 `SYSTEM CONTEXT` 대시보드 도달 전까지의 극심한 데이터 딜레이를 유발하던 Pandas Object 배열 생성 및 groupby cumsum 병목과 피처 타입 변환 오버헤드를 우회하여 데이터 로딩 파이프라인의 Latency를 획기적으로 낮춤.
- **Status:** Accepted


