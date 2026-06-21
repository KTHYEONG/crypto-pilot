# Layer 1 Architectural Decisions

## L1-ADR-021: L1→L2 게이트 재설계 — LCB 경제성 하드게이트 + 바인딩 FDR + Q:hi/mid/lo 제거 (2026-06-21)
- **Delta:** `build_qualified_signal_registry` admit 조건을 4-condition 로 재정의. `lcb_net_bps` (block-bootstrap P5 of incremental gross) 필드 추가. `l1_breakeven_floor_bps=_DEFAULT_RT_BPS(≈7.5bps)`, `l1_fdr_hard_reject=True`, `l1_pair_fdr_alpha=0.10` 기본값 적용. `format_layer1_deployment_registry_table`에서 `Q:hi/mid/lo` 티어·stars·`[L2-PASS]` 라벨 제거 → `LCB(bps)|CONV|FOLDS|t(blk)` 컬럼으로 교체.
- **Rationale:** 기존 게이트는 P(μ>0)>0.5(동전던지기)가 유일 통계 문턱이었고, FDR은 soft-shrink(비binding)였으며, Q:hi/mid/lo는 다른 추정량(`naive IID t` vs `block t`)을 혼용한 표시전용 레이블이었다. 실질적 경제성·다중성 통제가 없어 노이즈 신호 통과. LCB 하드게이트로 "worst-plausible edge > round-trip cost" 보장.
- **Edge Cases:** `cfg=None` 시 LCB gate disable(backward compat). `l1_fdr_hard_reject=False`이면 soft-shrink 유지(하위 호환). `lcb_net_bps`는 peer-relative gross — 비용 이중차감 없음(백테스트 엔진이 별도 차감).

## L1-ADR-014: Archetype 라벨 의미 정합화 + Evidence Fold 스킵 로그 억제 (2026-06-21)
- **Delta:** `_log_ensemble_diagnostics`의 `standard_keys` 5종 → `archetype_labels` 7종으로 교체. `mean_rev→MRV`, `ts_mom→TMO`, `unwind→UNW`, `beta_neut→BTN`, `carry_rev→CRY`, `flow_rev→FLO` 명시(첫글자 fallback 의존 제거). `is_evidence_fold` 파라미터로 evidence fold warm-up skip WARNING 억제.
- **Rationale:** `mean_rev↔ts_mom`이 `MOM↔MRV`로 의미 정반대 swap되어 관측성 결함 유발. Fold #0~1 skip WARNING은 evidence fold 정상 동작(warm-up)이므로 노이즈. 수치 불변, 라벨·로그만 수정.
- **Edge Cases:** 7종 외 신규 archetype은 첫글자 fallback 유지. `is_evidence_fold=False` 기본값으로 outer fold 동작 불변.

## L1-ADR-017: Flow-Aware Layer1 Signal Families 및 Cell-Level Imbalance 도입 (2026-06-21)
- **Delta:** `_safe_taker_imbalance_2d`를 추가했고, `taker_buy`/`volume` 정합성이 셀 단위로 검증되도록 만들었다. `build_rule_signal_panels`는 `flow_imbalance`, `flow_mean_6`, `flow_z_24`, `funding_z_96`, `funding_z_168`, `ret_1`, `ret_12`, `ret_z_48`를 공유 캐시로 재사용했고, `funding_flow_carry`, `funding_flow_unwind`, `flow_exhaustion_reversal` 패널을 노출했다. `config.py`의 candidate family 목록과 ensemble prior 목록에도 해당 family를 포함했다.
- **Rationale:** ENS 로그에서 일부 flow 계열 항목이 weak/fail 상태로 남았고, 기존 exhaustion 패턴은 deterministic fixture에서 희귀해서 unit test와 운영 신호가 drift를 일으켰다. same-bar exhaustion/reversal과 funding-flow confirmation을 분리해 신호 coverage를 넓히면서도 causal input과 compute budget을 유지했다.
- **Edge Cases:** taker data가 없거나 invalid이면 flow-dependent panel이 fail-closed 되었고, mixed-valid row는 전체 row 실패로 오인하지 않았다. `flow_exhaustion_reversal`은 same-bar confirmation으로 단순화되어 event testability가 유지되었다.

## L1-ADR-020: 저성과 신호 패밀리 제거 및 per-symbol ENS-DIAG 진단 도입 (2026-06-21)
- **Delta:** `trend_donchian`(donchian_18/36), `oi_volume_impulse`, `oi_volume_confirmed_breakout`, `oi_price_divergence`, `oi_breakout_confirm`, `basis_zscore_reversion`, `basis_momentum`, `taker_exhaustion_reversal` 8개 family 제거. panel 수 40→31. `_log_signal_symbol_diagnostics` 신규 함수: ENS-FINAL에서 per-(archetype×family×symbol) DEBUG 진단 로그 + absent archetype WARNING 방출. `CandidateSignalPanel.archetype` 기본값 `"mean_rev"`→`""`로 변경하여 family 휴리스틱 우선 동작 보장.
- **Rationale:** 8개 family 실증 p>0.34, possym<0.50으로 유의성 미달. flow 계열(`funding_flow_carry/unwind`, `flow_exhaustion_reversal`)로 대체하여 coverage 유지. per-symbol 진단으로 데이터와이어링 결함 조기 발견.
- **Edge Cases:** `_log_signal_symbol_diagnostics`는 DEBUG off 시 O(1) noop. `symbol` 컬럼 부재 시 family rollup만 출력, possym="n/a". 빈 family는 진단에서 skip되어 zero-division 방지.

## L1-ADR-019: MTF 패널 람다 클로저 late-binding 수정 (2026-06-21)
- **Delta:** `build_rule_signal_panels`의 G1~(mtf_trend_pullback), G2(mtf_breakout_retest), G10(vol_term_structure_gate) 내부 `_compute_*_htf` 함수들에서 루프 변수 `_n_htf`/`_vts_win`을 기본 파라미터(`span=_n_htf` 등)로 캡처하여 Python late-binding closure 버그 수정. `funding_zscore_carry`(F2) 캐시 재사용 경로 통일(`_zscore_2d`→`funding_z_96`/`funding_z_168`). `taker_imbalance_momentum`(G7) CVD slope → `flow_imbalance` z-score로 전환하여 공유 캐시 재사용.
- **Rationale:** 루프 내 정의 함수가 마지막 루프 값을 공유해 파라미터 오염 가능. flow 캐시 재사용으로 compute budget 절감.
- **Edge Cases:** 단일 variant family(G4/oi_breakout_confirm)는 제거되어 영향 없음. F2/funding_zscore_carry는 window=48 variant가 `_zscore_2d` 호출 유지(별도 캐시 없음).

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

## L1-ADR-016: L1 Parquet I/O 최적화 — baggage drop / numeric early exit / copy 제거 (2026-06-20)
- **Delta:**
  - (OPT-5) `DataCollector._load_cache`에서 `close_time`, `no_trades`, `ignore` 3개 Binance metadata 컬럼을 즉시 드롭. parquet 저장 전에 이미 정규화 완료되어 다시 string→numeric 변환이 필요 없는 컬럼.
  - (OPT-6) `DataCollector._normalize_df`에 early exit guard 도입: 모든 non-datetime 컬럼이 numeric dtype이면 string→numeric loop를 skip. cache-read path에서 O(N) dtype 체크만 수행.
  - (OPT-7) `DataCollector.collect_and_save(fetch_network=False)`에서 `.loc[mask].copy()`의 `.copy()` 제거 (boolean indexing은 항상 copy 반환).
- **Rationale:** PRE-OPT baseline 101.98s vs POST-OPT 97.22s (실측, 57 symbols, `--phase l1 --sync skip`). 데이터 로딩 sub-stage 기준 ~25% 절감. OPT-1~4 합산 약 15-24s 단축 (101.98s → 97.22s 전체 대비는 pipeline L1 fold/ensemble 지배적). 3개 변경 모두 데이터 품질 0% 영향: L1 determinism 통과, strategy layer에 baggage column 참조 전무.
- **Edge Cases:** `_load_cache` OPT-5는 `if c in df.columns` guard로 스키마 불일치 시 안전. `_normalize_df` OPT-6은 `if non_dt` guard로 datetime-only DataFrame 처리. `collect_and_save` OPT-7은 `fetch_network=True` 경로에 영향 없음 (해당 경로는 `.copy()` 사용 안 함).
- **Status:** Accepted

## L1-ADR-015: L1 Post-SKIP 병목 제거 — raw_df.copy/audit/coverage CPU 최적화 (2026-06-20)
- **Delta:**
  - (OPT-1) `load_single_symbol_data` — `raw_df.copy()`를 `needs_merge` 조건으로 게이트. funding/metrics 모두 None이면 copy 생략 (view 참조). merge 필요 시 copy 경로 유지.
  - (OPT-2) `_to_unix_ms` 호출을 TF당 1회로 축소. `needs_merge` 블록 내에서 1회 계산 후 `df["timestamp"]`로 재사용 (기존: funding/metrics merge 각각 1회씩 2회 중복).
  - (OPT-3) `_append_stage_integrity("merged")`에서 `summarize_ohlcv_collection_integrity` 대신 `{rows, cols}`만 저장. downstream `load_futures_data_maps_for_symbols`의 `audit_df.groupby`는 존재하는 컬럼만 조건부 선택하여 `KeyError` 방지.
  - (OPT-4) `_feature_group_coverage`에 `_COL_GROUP_CACHE` 모듈-레벨 캐시 도입. `(tf_label, frozenset(col_lower))` 키로 column→group 매핑을 TF 세션당 1회 pre-compute. 동일 컬럼셋 심볼에 대해 O(C×P)→O(C) 단축.
  - (Hotfix) `_to_unix_ms` pandas 3.x 호환: tz-aware datetime에 `tz_localize(None)` guard 후 `.astype("datetime64[ns]")` 적용 (`DataCollector._normalize_df`가 tz-aware UTC 반환).
- **Rationale:** `[SKIPPED]` ledger sync 로그 후 `SYSTEM CONTEXT` 대시보드 도달까지 평균 ~40-50s 소요. 실측 trace 결과 `load_futures_data_maps_for_symbols`가 95% 이상 점유. 4개 OPT로 50심볼 기준 약 10-19s 단축 예상 (~20-30s). 잔여 20-30s는 PyArrow parquet I/O (hard floor).
- **Edge Cases:** OPT-3 merged audit에서 `audit_df` groupby가 integrity 컬럼(NaN%, gap, duplicate) 부재 시 `KeyError` 발생 — 존재하는 컬럼만 선택하도록 조건부 처리 완료. pandas 3.x `to_unix_ms` tz-aware guard 없으면 `astype("datetime64[ns]")` 실패 — `tz_localize(None)` 도입으로 해결.
- **Status:** Accepted

## L1-ADR-014: L1 데이터 준비 병목 최적화 — Numba 및 DatetimeIndex.isin 기반 최적화 (2026-06-20)
- **Delta:** 
  - `membership.py`에 Numba `@njit` 가속화된 `_calculate_warm_ready_numba`를 도입하여 Pandas groupby-cumsum 루프를 대체하였고, `build_membership_mask_bundle` 내 날짜비교를 `pd.Timestamp`와 `DatetimeIndex.isin` 기반 벡터화로 고속화하여 `datetime.date` 객체 생성 오버헤드를 우회함.
  - `opt_data_utils.py` 내 `_feature_group_coverage`에서 수치형 컬럼의 to_numeric 강제 변환을 우회하고, 1분 데이터 로딩 시 이미 monotonic/datetime64인 경우 정렬 및 타입 변환을 생략하도록 정돈함.
- **Rationale:** 
  - `[SKIPPED]` 로그 출력 후 `SYSTEM CONTEXT` 대시보드 도달 전까지의 극심한 데이터 딜레이를 유발하던 Pandas Object 배열 생성 및 groupby cumsum 병목과 피처 타입 변환 오버헤드를 우회하여 데이터 로딩 파이프라인의 Latency를 획기적으로 낮춤.
- **Status:** Accepted

## L1-ADR-021: CRY/FLO 안정화 신호 3종 추가 및 positioning_unwind warm-up barrier (2026-06-21)
- **Delta:**
  - `build_rule_signal_panels`에 신규 패널 3종(`funding_term_structure_carry`, `flow_trend_continuation`, `lsr_oi_regime_filter`)을 G9e/G9f/G9g로 추가.
  - `positioning_valid`에 `positioning_warm[:96]=False` barrier 도입.
  - `funding_ts_slope = funding_z_96 - funding_z_168`를 shared feature cache에 등록.
  - `CandidateStrategyConfig.candidate_families` 및 `ensemble_variant_prior_families`에 3종 family 등록.
- **Rationale:**
  - CRY 패널이 EVT 37K 구간에서 -35.5❌로 붕괴하는 원인이 funding 절대레벨 신호 단일 의존이었음. `funding_term_structure_carry`(가속도 기반)로 다각화하여 변동성 완화.
  - FLO 패널이 13.9~35.8 bps로 불안정: `flow_exhaustion_reversal`(반전)만 존재 → `flow_trend_continuation`(추세 연속) 추가로 패널 다양화.
  - UNW 패널이 EVT 37K 구간에서 -8.4❌: z-score window(42/96/168)가 채워지기 전 noise 진입. 96-bar warm-up barrier로 사전 차단.
  - `lsr_oi_regime_filter`는 LSR+OI 극단 국면을 식별하여 conditioning score로 전달, BTN 경로를 통해 TRD/MRV 간접 게이트 역할.
- **Status:** Accepted

## L1-ADR-022: FLO 회귀 수정 — flow_trend_continuation archetype 재분류 및 lsr_oi_regime_filter 활성화 (2026-06-21)
- **Delta:**
  - `flow_trend_continuation` archetype `flow_rev`→`ts_mom`, regime `flow_trend_continuation`→`flow_momentum_continuation`으로 변경.
  - `lsr_oi_regime_filter` side_hint 기본값 `0`→`np.where(_loi_regime_entry, -np.sign(lsr_log_z_42).astype(np.int8), 0)`, stop_atr_mult `0.0`→`1.5`, take_profit_atr_mult `0.0`→`2.0`으로 변경하여 conditioning gate에서 tradable signal로 전환.
  - `positioning_warm[:96]`→`[:168]`으로 warm-up barrier 확장.
  - FLO 결과: `+18.9 → +28.5 bps` (+9.6 bps), TOTAL `+54.5 → +54.3 bps` (noise).
- **Rationale:**
  - L1 최종 실행 결과 FLO가 +28.7→+12.1로 -16.6 bps 회귀. 원인은 `flow_trend_continuation`이 `flow_rev`(FLO) archetype에 배정되어 momentum 신호가 reversion archetype을 오염시킨 것. `ts_mom`(TMO)으로 재분류하여 reversion dilution 해소.
  - `lsr_oi_regime_filter` side_hint=0이면 BTN 신호 조건부 gate(score만 있고 방향 없음)로 이벤트 생성 불가 → -np.sign(lsr_log_z_42) 방향성 부여로 활성화.
  - `positioning_warm[:168]`: longest z-score window(168)로 정합, z-score pre-warm 완전 보장.
- **Edge Cases:**
  - archetype 변경으로 TMO `+52.7→+50.3 bps`(-2.4) 일부 dilution 발생, FLO 복구를 위한 자연스러운 trade-off.
  - `-np.sign(lsr_log_z_42)`: lsr_log_z_42=0일 때 sign=0 → side_hint=0 유지, 조건부 미활성 보존.
- **Status:** Accepted

## L1-ADR-023: Small-Sample ENS Stabilization — Bayesian Prior & Min Display Threshold (2026-06-21)
- **Delta:**
  - `_fit_cell_means`에 `prior_effective_n`, `prior_mean_bps` 파라미터 추가. 모든 _fit_cell_means 호출에 eb_fit_kwargs를 통해 전파.
  - archetype/archetype_regime 루프에서 JS shrinkage 전에 Bayesian prior 적용: `raw_mean = w_prior * raw_mean + (1-w_prior) * prior_mean_bps`, where `w_prior = n_eff / (n_eff + prior_effective_n)`.
  - `_log_ensemble_diagnostics`에 `min_display_events` 파라미터 추가. archetype event count가 threshold 미만이면 `insuf` 표시.
  - 신규 config: `l1_ens_prior_effective_n: float = 0.0`, `l1_ens_min_display_events: int = 0`.
- **Rationale:** 초기 `[ENS]` 로그에서 TRD -18.4❌는 단 3,432 events에서 추정된 noise. prior_n=100, prior_mean=0을 적용하면 n이 작을 때 edge가 0으로 수축 → 거짓 음성 기만 방지. n이 충분하면 prior 영향이 소멸되어 수렴 보존. min_display_events는 충분한 데이터가 쌓이기 전까지 archetype edge 자체를 숨겨 사용자 오독 방지.
- **Edge Cases:** prior_effective_n=0 (default) → 완전 backward compatible. 기존 동작에 영향 없음. prior는 JS 이전에 적용되므로 JS k가 0이어도 prior만 독립 적용됨.
- **Status:** Accepted

## L1-ADR-024: Adaptive Evidence Gate & Promotion Advisory Mode (2026-06-21)
- **Delta:**
  - `compute_symbol_strategy_evidence`에 `snapshot_index: int = -1` 파라미터 추가. early snapshot (index < `l1_evidence_early_snapshots`)에서 `l1_pair_min_effective_obs` / `l1_pair_min_folds` threshold를 relaxed 값으로 대체.
  - `build_l1_prequential_evidence_snapshots`에서 `snapshot_offset`을 `snapshot_index`로 전달.
  - `apply_variant_promotions`의 fail-closed(empty 반환)를 advisory pass-through(원본 반환)로 변경. warning → info 로그 레벨 변경.
  - 신규 config: `l1_evidence_early_snapshots: int = 0`, `l1_pair_min_effective_obs_early: float = 2.0`, `l1_pair_min_folds_early: int = 1`.
- **Rationale:** fold 0은 심볼 21개, fold당 이벤트 수가 적어 strict gate(eff_obs≥5, folds≥2) 통과가 어려움. early snapshot에서 gate를 완화하면 적은 fold 데이터로도 registry 진입이 가능해지고, 이후 `quality_weight`가 probability_positive 기반으로 자연 필터링하므로 noise가 L2까지 도달하지 않음. Promotion filter는 L1 SWF의 evidence computation과 중복 필터링이므로 advisory로 전환하여 혼선 제거.
- **Edge Cases:** l1_evidence_early_snapshots=0 (default) → disabled, 항상 strict gate 사용. deployment-level compute_symbol_strategy_evidence 호출 (run_l1_nested_swf L1053)은 snapshot_index 없이 호출 → -1 default → strict gate 적용. promotion filter advisory 모드에서 caller(ablation.py/bridge.py)의 `labeled.empty` guard는 pass-through 시 full data를 받으므로 pipeline 정상 진행.
- **Status:** Accepted

