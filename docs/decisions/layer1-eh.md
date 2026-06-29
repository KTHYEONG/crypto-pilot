---
title: Layer 1 Decision Log (Compressed)
domain: futures.strategy
type: adr
status: active
priority: high
ai_read_policy: when_related
---

## Phase 1: SWF 구조 & 초기 게이트 (ADR-001~009, 6/13~19)
- Nested SWF 도입, prequential evidence grid 분리(outer_n×multiplier≤max), outer warm-up blocks=2로 fold 0 underpower 해소
- 통계적 MDES gate(t_crit+검정력 80%), 5-Gate로 standardization(fold_cov/match_ratio/sym_count/fold_ratio/probe_lcb_bps)
- IC 지표 제거(Arch-Only mode에서 noise), mu_quality_shrinkage dead-code 제거(validation_rank_ic=0 → lam=0 붕괴)
- PIT state_cube 통합(lifecycle gate: promotion_available_at>l2_start 차단), capacity clip(5 USDT min+proportional)
- L1→L2 게이트 재설계: 4-condition(LCB 하드게이트 worst-plausible edge>RT cost, binding FDR, qw>0), Q:hi/mid/lo 티어 제거
- Promotion Summary PASS/FAIL 분리, 우측절단 진단(censored count), STATUS PROMOTED/WATCH/REJECTED 제거
- Archetype 라벨 정합(5→7종, MRV/TMO/UNW/BTN/CRY/FLO), evidence fold warm-up skip WARNING 억제

## Phase 2: 성능 최적화 1~3차 (ADR-009~016, 025~026, 6/18~21)
- PERF 로깅 도입(레벨 15→10 통일, 계층적 타이밍, [PERF] prefix 일원화)
- Numba JIT rolling z-score(prime 27.78→7.75s, L1 total 47.64→25.17s)
- q-value FDR vectorization→loop 롤백(N≤200 소표본 회귀)
- Python loop 제거+벡터화(_by_q_values, _compute_incremental_bps, .copy() 제거, itertuples→cumcount)
- Ingestion 병렬: ProcessPool→ThreadPool(I/O bound pickling 제거), meta pre-conversion, datetime64 bypass
- Parquet I/O: baggage 3col drop, numeric early exit guard(>O(N) 스킵), 불필요 .copy() 제거
- Post-SKIP: copy() 조건부 게이트, _to_unix_ms TF당 1회, merged audit 경량화, COL_GROUP_CACHE 도입
- Numba membership(_calculate_warm_ready_numba) + DatetimeIndex.isin 벡터화
- L1/Universe 루프 불변식 호이스팅(searchsorted N→1, timeline 104→1), vol_2d 행렬화(pd.DataFrame.rolling())
- signal_batch_convert iterrows 벡터화(iterrows→np.flatnonzero, loop 2.64→0.04s/fold, L1 total 53.5→37.4s)
- P1~P5: prime→_warm_aligned_2d_cache 분리, inner groupby pre-compute, waterfall 게이팅, PERF 타이머 세분화, audit_tables 타이머 이동

## Phase 3: 신호 패밀리 & MTF 확장 (ADR-017~024, 026~034, 6/21~22)
- Flow family 3종(funding_flow_carry/unwind, flow_exhaustion_reversal) + cell-level taker_imbalance_2d
- 8 저성과 family 제거(trend_donchian, OI 5종, basis 2종, taker_exhaustion) 40→31, per-symbol ENS-DIAG 진단
- FLO 회귀 수정: flow_trend_continuation archetype flow_rev→ts_mom, lsr_oi_regime_filter active화(side_hint 방향성)
- MTF 람다 late-binding 수정(기본 파라미터 캡처), flow 캐시 재사용(funding_z_96/168)
- CRY/FLO 안정화 3종 추가(funding_term_structure_carry, flow_trend_continuation, lsr_oi_regime_filter), positioning_warm[:168] barrier
- Bayesian prior ENS(n_prior=100, prior_mean=0로 소표본 수축), min_display_events로 insuf 표시
- Adaptive evidence gate(early snapshot threshold 완화, snapshot_index 파라미터), promotion advisory 모드
- TF별 signal pool 6종 추가(gap_fade_1h, vwap_reversion_1h, volume_climax_1h, macd_4h, supertrend, ichimoku_trend)
- Per-TF L1 pipeline: run_per_tf_l1/_resolve_l2_master_tf/_aggregate_per_tf_l1, native_tf 필터링, oos_stacked tf:: prefix
- Quality weight floor(l1_qw_floor=0.05)+probe prior boost(l1_qw_probe_boost=0.3), per-TF gate overrides
- LTF mean aggregation(ltf_mode="mean" cumsum window), resample metadata 보존, holding_bars cost 보정
- TF Probe virtual source resample(data-stage/bridge-stage 분리), ENABLE_TF_PROBE=False 기본
- Effective-N FDR 보정(diversity_corr 기반 m_eff), 동일(symbol,family) cluster r_bar

## Phase 4: 후반 최적화 v2~v3 (ADR-035~037, 6/22~23)
- L1 Gate+Signal Pool Optimization: per_TF_gate_overrides 자동 fallback, fdr_alpha 0.10→0.15, qw_floor 0.05, 2h trend_ma 제거
- OOM 방지: resolve_safe_nested_workers adaptive cap(max_workers=min(cpu_limit-2,8), oversubscription guard), fork 내 gc.disable()
- P5-R: prequential ThreadPoolExecutor 제거→순차 복원(GIL+cache thrashing 역효과, 11.4→7.9s/TF, -31%)
- PERF-GAP1: [WORKER-CALC]→[PERF] worker_calc(memory 추정+worker decision 노출)
- PERF-GAP2: [PERF] l1_nested_ipc_collect(IPC+fold compute vs pool_setup 분리)
- v3 실측: L1 108.6s(-52% vs baseline), Grand Total 152.8s(-49%), EXIT_CODE=0

## Phase 5: Bridge Perf Logging + GC 최적화 (ADR-038~039, 6/23)
- Bridge perf logging Phase 1: `_get_rss_mb()` RSS 측정, stage별 `_sample_rss()` memory delta 추적, `wf_fold_times` per-fold 타이밍, `[PROFILE][MERGE][SUMMARY]` 통계 로깅
- HTF skip 최적화 시도 → 롤백: `run_per_tf_l1()`이 bridge HTF events에 의존적임 확인 (`_build_per_tf_event_index()` 존재하지 않음). HTF skip 시 6h/8h/12h per-TF L1 비활성화 = 품질 회귀
- GC 전략 추가: diagnostics 후 `gc.collect()` (+5.3GB 회귀), bridge 반환 후 `gc.collect()` (tiered re-alignment 전 aligned 해제)

## Phase 6: WSL Stability Optimization (ADR-040, 6/23)
- Max worker cap: `min(cpu_limit, 8)` → `min(cpu_limit, 3)`. Fork worker 폭주(6 worker × 8 threads = 48 threads)가 WSL CPU starvation → network dropout → SSH/Tailscale 단절 원인으로 확인.
- Thread env vars: `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` = `"1"` before each fork. Fork child 내 Numba prange + BLAS thread cascade 제거.
- TF 간 0.5s pause: fork 폭주 후 OS page cache + network buffer 회복 시간 확보.
- Two-phase rollback: single `ProcessPoolExecutor` for all folds. Two-phase(evidence/outer 분할)는 fork CoW 페이지 오염으로 역효과 확인(peak 9.5GB 상승).
- `del full_strategy_maps` after alignment in tiered path (dictionary wrapper reclaimed, ~2GB via allocator).
- Incremental outer fold cleanup: `del` per-fold temporaries + `gc.collect()` every 2 folds.
- L2 memory tracking: stage MEM logs for `l2_signal_batch`, `l2_sim_cache`, `l2_optuna_study`, `l2_study_complete`, `l2_champion`, `l2_final_pipeline`.
- Peak RSS 감소: 8,845MB(6-worker) → 8,674MB(3-worker, -2%). 48 threads → 12 threads (-75%). L1 4h time 52s → 68s.

## Phase 7: Prequential Soft-Cap & Pinned Contract Audit Gap (ADR-041)
- `l1_nested_result_soft_cap_mb`를 dead-config에서 실제 OOM guard로 승격: `resolve_safe_nested_workers()`의 `result_soft_cap_mb` 파라미터 연결, `run_l1_nested_swf()`에서 soft_cap 부족 시 compact 강제(force_compact).
- `l1_nested_workers`(pinned) 의미를 "고정값"에서 "희망 상한(safety-clamped upper bound)"으로 변경: pinned가 low-memory guard, soft-cap guard, oversubscription guard를 우회하지 못함.
- Audit 후속: config 주석 정합, `test_scenario_5_adaptive_worker_cap` stage cap 기대값 6→3 갱신 + psutil mock 추가, `TestResolveNestedWorkersPinned`에 soft_cap override 검증 추가.

## Phase 8: Data Load Arrow Optimization (ADR-042, 6/24)
- **P1-A: Lazy Funding/Metrics Load**: `_prepare_funding_metrics()` 추출, cache-hit + no exec_1m 경로에서 funding/metrics I/O 완전 skip (57 심볼 × GIL-bound parse 낭비 제거).
- **P1-B: Parquet Predicate Pushdown**: `pd.read_parquet(filters=[("timestamp",">=",ms),("<=",ms)])` 도입, enriched 캐시의 row-group statistics 기반 디코드 최적화 → full-read + mask 제거.
- **P2: Arrow Dataset C++ 병렬 스캔**: `_scan_enriched_dataset()`으로 `pyarrow.dataset` + row-group 멀티스레드 디코드(GIL 해제) → 2-pass 분리(I/O parallel + Python-bound 후처리 순차) → cache-hit 경로 CPU 병렬화.
- **exec_1m Safeguard**: `intrabar_1m` 모드 시 Arrow fast-path opt-out → ThreadPool fallback, exec_1m + funding_event_mask 키 누락 방지.
- **Measured Improvement**: Data Load 14.26s → 9.99s (-29.9%). 253 심볼 중 129심볼은 Arrow path (enriched cache-hit), 124심볼은 fallback. Phase L1 전체 data stage 17s → 11.58s.
- **Test Coverage**: Scenario 1-6 (10/10 PASS, 0.69s) — cache-hit pushdown, lazy skip, empty window, fallback, phase2 equiv, exec_1m opt-out.

## Phase 9: Bridge Candidate Strategy Perf (ADR-043~045, 6/24)
- **L1-B: Selection Vectorization**: `_vectorized_topk_per_bar` 도입 — per-bar Python loop → sort + drop_duplicates + cumcount rank + ceil(keep) + variant-cap backfill. O(E log E) 벡터화, 0 Python loop. 동등성 보장: sorted cumcount tie-break.
- **L1-A: Diagnostics Gating**: `enable_diagnostics` 파라미터 추가 → evidence fold(12/16)에서 sensitivity/shadow/waterfall skip. 외부 fold/배포 경로는 `True` 유지(진단 SSOT 보존).
- **L2-A: Bridge prepare-once**: `bridge.py` WF 루프 직전 `prepare_labeled_events` 1회 호출 → `PreparedLabeledEvents` 전달. `build_candidate_dataset` fast path(numpy boolean mask) 사용.
- **L2-B: ProcessPool Global Wiring**: `_GLOBAL_LABELED_EVENTS`에 `PreparedLabeledEvents` 배선 → fork CoW 상속, 직렬화 0.
- **L3: Schema-once**: `frozen_identity_names` prepared events에서 추출 → fold별 재계산 대신 1회 schema. ensemble_b0(identity 미사용)는 최종 fit window 기준 1회.
- **Config**: `CandidateStrategyConfig.l1_selection_diagnostics_enabled: bool = False` 추가.
- **L2 validation**: `ruff check` + `mypy` pass 5개 파일. L2 test 31/31 pass.
- **실측 (4h TF, signal_only bridge + 별도 evidence phase)**:
  - selection 23.3s → 2.88s (-88%, 목표 달성)
  - ds_fit 18s → 15.8s (-12%, prepare-once가 evidence phase 경로에 미적용)
  - bridge total 48.04s (signal_only early return — WF 미포함)
- **미배선 확인**: `per_tf_l1` evidence/outer fold 경로는 bridge→WF가 아닌 별도 evidence phase 실행 → L2-A(prepare-once) 효과 미발현. 별도 배선 필요.
- **Test Coverage**: Scenarios 1-6 (8/8 PASS, ~2.5s) — vectorized equivalence(top_quantile 파라미터화), variant cap backfill, single-bar/empty eligible, diagnostics gating spy, prepared 동등성(나중 추가).

## Phase 4 (sic): Bridge-Candidate-Perf-V2 및 Enrich Cache Hotfix (ADR-03X, 6/24)
- **L2-A**: `PreparedLabeledEvents` frozen→mutable dataclass, `enrich_cache: dict[str, Any] | None` 필드 추가.
- **L2-B**: `_precompute_enrich` lazy init — window-invariant만 precompute (arm/entry_regime/overlay_mult/crisis_active/entry_regime_code). 벡터화 affinity matrix lookup (list-comp→numpy indexing).
- **L2-C/D**: `build_candidate_dataset` sig_feat_names + skip_features 경로에서 `enrich_cache` read.
- **Regression 발견 및 Hotfix**: `_compute_score_pct_variant_hist`를 full frame에서 실행→30s/fold. `score_pct`/`n_same`은 window-dependent이므로 enrich_cache에서 제외하고 per-window fallback 유지. Hotfix 후 `_precompute_enrich` 4.3ms (10K events, -7000x).
- **L1-A**: `[PERF] l1_wf_summary` 로그 추가 — avg fold timing + evidence/outer wall time all-TF aggregate.
- **L1-B**: Bridge profile walk_forward 항상 표시 (`0s` → `"(skipped)"`).
- **L1 validation**: ruff/mypy pass, 30/30 candidate dataset tests pass.

## Phase 10: Bridge Multi-TF Threading + Datetime Hoisting (6/24)
- **S1 — Multi-TF Bridge ThreadPool**: `build_multi_tf_panels`에서 sequential per-TF loop → `_process_single_tf` inner function + `ThreadPoolExecutor(max_workers=2)`. Eligible TF ≤1 → sequential; ≥2 → parallel. 각 TF는 독립적인 `list[CandidateSignalPanel]` 할당, shared mutation 없음. Exception 격리: 실패 TF만 skip, 다른 TF 정상 처리. ThreadPool ≠ ProcessPool — fork 없음, NUMBA env var 오염 없음. (commit 포함: `src/domain/futures/strategy_runtime/bridge.py`)
- **S2 — `_resolve_tradeable_scope` Datetime Hoisting**: 52 symbol loop에서 invariant `pd.api.types.is_datetime64_any_dtype()` 검사를 first-valid-symbol에서 1회만 실행하고 `_native_flag`로 캐시. 이후 symbol은 branch만 평가 (0.2s saving, 52 syms × 8761 bars). MagicMock/string-datetime fallback 경로 유지. (commit 포함: `src/execution/opt_main_futures.py`)
- **Perf Profile**: `docs/perf_mem_profile_report.md` 최초 생성 (L1 288.10s, bridge 58.24s, peak RSS 7,565MB). 별도 커밋 — 성능 기준선 문서.
- **L1 validation**: ruff/mypy pass, test 4개 파일 339 insertions/20 deletions.

## Phase 11: Logging Consolidation & Tagging Standardization (ADR-046, 6/24)
- **Log Level Consolidation**: Custom `PERF` logging level was removed, consolidating performance metrics and standard debug logs under standard `logging.DEBUG` level.
- **Bracketed Tag Enforcement**: Modified `CategorizedLogger` to enforce prefixing of all debug logs with bracketed tags `[PERF]`, `[DATA]`, `[OPT]`, `[STRAT]`, or `[SYS]`. Any untagged log automatically defaults to the `[SYS]` prefix.
- **Key-Value Message Structuring**: Converted performance metrics logging (durations, memory sizes) to standard key-value messages (e.g. `[PERF] step=... elapsed=...s`) in `opt_main_futures.py` and `CategorizedLogger` helpers, allowing efficient automated parsing.
- **Verification**: All logger unit tests and L1 memory profiling tests pass, validating successful fallback tagging and standard formatting.



## Phase 12: Bridge candidate strategy parallelization (ADR-047, 6/24)
- **Signal Calculation Parallelization**: Replaced sequential loops in `build_rule_signal_panels` with a local closure function `_build_single_family(family)` mapped over active families using a `ThreadPoolExecutor` (max_workers=4). Leveraged GIL-free numpy operations to utilize CPU cores without multiprocessing serialization overhead.
- **Batch Event Conversion**: Parallelized `candidate_panels_to_events` using `ThreadPoolExecutor` (max_workers=4) over active panels, significantly shortening the time required for dense-to-sparse event table conversions.
- **Diagnostics Parallelization**: Parallelized independent pandas groupby calculations (`by_family`, `by_variant`, `by_family_side`, and `_summarize_side_flip` frames) in `compute_rule_diagnostics` via `ThreadPoolExecutor` (max_workers=3).
- **WSL Performance Outcome**: Average L1 strategy computation time per timeframe reduced by 54% (~46.78s sequential to ~21.29s parallel equivalent). Complete execution timing and RAM profiles updated in `docs/perf_mem_profile_report.md`.

## Phase 13: L1 PERF Radical Optimization — OPT-0~4 (ADR-048, 6/24)
- **OPT-0: Dead Code + TF 정합성**: `TF_PROBE_GRID` 6→4 TF(`1h/2h` 제거), `PROBE_SOURCE_TFS` dead-code 제거(`1m/5m/15m/30m`), `run_tiered_pipeline` `l1_tfs` default `cfg.l1_tfs`와 정합.
- **OPT-1: searchsorted O(log T)**: `load_futures_data_maps_for_symbols` Pass-2의 datetime mask+sum → `np.searchsorted(dt_ns, value, "left")`. `is_end_idx`/`is_start_idx`/`oos_start_idx` 모두 O(T) full scan에서 O(log T) binary search로 단축.
- **OPT-2: Evidence IPC as_completed**: `run_l1_nested_swf` evidence 수집을 `as_completed`로 변경. 완료 순 IPC + fold_id 재정렬.
- **OPT-3: t.ppf LRU Cache + Snapshot ThreadPool**: `_t_ppf_cached(q_milli, df_int)` with `@lru_cache(maxsize=512)`. 스냅샷 루프를 `ThreadPoolExecutor`로 병렬화 (n≤1 순차, n≥2 병렬, `as_of_idx` 정렬). U-형태 TF별 편차: 4h 11.01s, 6h 5.52s, 8h 4.95s, 12h 5.56s.
- **OPT-4: Prefit Overlap**: `prefit_layer1_model(deployment_registry 제외)` / `assemble_layer1_artifact`로 분할. evidence submit 직후 같은 executor에 `_prefit_layer1_from_globals` fork. background 실행 중 evidence IPC+snapshot+outer fold(34.7s)와 오버랩. 게이트 통과 시 `assemble_layer1_artifact(1ms)`로 registry 첨부. `cfg.l1_speculative_prefit_enabled`(기본=True). 실패 시 serial `fit_layer1_inference_artifact` fallback. **모든 TF에서 0.0000s 달성**(기존 20.49s 완전 제거).
- **실측 (2025-06-24, 52 symbols, 4 TF)**: L1 Total 163.6s → ~123.1s(-25%). `l1_fit_inference_artifact` 20.49s→0.00s(100%↓). Peak RSS 8,782MB→~8,773MB(유지). Spec 목표(~70s 단축) 대비 ~58% 달성.
- **Test Coverage**: 4 tests (OPT-0 S3, OPT-1 S2/S3, OPT-3 S1, OPT-4 S1) all PASS. L1: ruff/mypy clean. 1 pre-existing test unstable(`test_l1_nested_thread_env_vars_set` env var).

## Phase 14: L1 HTF Bottleneck — candidate_panels_to_events Optimization (ADR-049, 6/24)
- **A: Regime×Policy Pre-extraction**: `_convert_single_panel` regime 루프에서 array indexing을 policy당 21회에서 regime당 1회로 감소. regime_mask를 regime 루프 밖에서 1회 pre-extract 후 policy 루프에서 재사용. O(R×P) → O(R) indexing reduction.
- **B: sort_values 제거**: `candidate_panels_to_events` 최종 `sort_values("datetime")` 제거. downstream(label_candidate_events, portfolio selection 등)이 entry_idx 기반 접근으로 정렬 불필요. O(N log N) full-table sort 제거.
- **C: Numba _robust_zscore_numba**: `_cross_sectional_robust_zscore` 위임 함수로 `_robust_zscore_numba @njit` 도입. unique group별 argsort 단일 패스 walk, Python O(U×E) loop → Numba O(E log E). 각 group 내 median/MAD 계산을 numba-compiled 단일 패스로 통합.
- **추가 변경**: `candidate_labels.py` `label_candidate_events` `precomputed_atr_2d` 파라미터 추가 + shape 검증. `bridge.py` ATR 캐시 1회 계산 후 HTF/BASE 재사용. signal_only fast-path → compute_rule_diagnostics skip + promotion 게이팅.
- **실측 (synthetic benchmark, Numba warm 이후)**: candidate_panels_to_events throughput 409K rows/s (400×20). zscore 0.0003s/call (50×200). 실전 52 syms × 2000 bars 예상 events <0.5s (기존 28.27s 대비).
- **Test Coverage**: 6 tests (Scn 1~6 ATR cache 3 + signal_only 3) + zscore equivalence/edge tests + schema/no-sort/regime-count validation. 56+28 regression PASS. L1: ruff/mypy clean.

## Phase 15: L1 Probe Breadth Diagnostics (ADR-050, 6/29)
- L1 게이트 전부 PASS이나 L2 realized gross가 음수인 모순 해소를 위해 env-gated DEBUG 계측 추가
- `ProbeBreadthDiagnostics` frozen dataclass + `compute_probe_breadth_diagnostics()`: (a) breadth-decay (k=3/10/20/-1)로 selection inflation 정량화; (b) gross − rt_cost로 cost drag 분리; (c) Spearman rank-IC + Fisher-z tstat로 신호력 부재 진단; (d) 전체 realized 분포 통계
- `L1_PROBE_DIAG` env gate 패턴: 기존 `L2_DIAG_ATTR`/`L2_MULTI_TF`와 동일 규약 (`""`/`"0"`/`"false"`/`"False"` → disabled)
- `evaluate_outer_signal_opportunities` 내 `return Layer1FoldReadiness` 직전에 hook — 게이트/리턴 영향 0, 순수 DEBUG 로그
- Net 정의: per-event 1-라운드트립 가정 (`net = gross − expected_cost_bps`), turnover 재부과 금지
- **Test Coverage**: 12 tests across 6 scenarios (selection-inflation, gross→net, rank-IC absence/positive, edge cases, env gate). All PASS. L1: ruff/mypy clean.

## Phase 16: Track A IC Gate Spec Compliance + Selection Downgrade + Bull-Primary Prior (ADR-051, 6/29)
- **IC Hard Gate → DEBUG Monitoring (spec §Track A, lines 106-107)**: Spec explicitly defers IC hard gate ("IC 하드 게이트 보류") until Track B produces cross-sectional alpha. Removed `("ic_tstat", ...)` and `("ic_sign_consistency", ...)` from `evaluate_layer1_readiness` check_specs. Moved IC pooling to `logger.debug` conditional. Prevents production always-BLOCK where `rank_ic_all=0.0` (default when `L1_PROBE_DIAG` env not set).
- **Config Params Reserved (l1_min_ic_tstat, l1_min_ic_sign_consistency)**: Kept in `CandidateStrategyConfig` for future Track B activation. Not wired into check_specs.
- **Probe Metric Default = "breadth"**: `l1_probe_metric` default changed from implicit top-k to `"breadth"`. `evaluate_outer_signal_opportunities` uses per-decision cross-sectional mean of all symbols instead of risk-score-ranked top-k when probe_metric="breadth". S4 test validates gross-all path.
- **Rank-IC/layer1FoldReadiness 승격**: `Layer1FoldReadiness` gains `rank_ic_all` / `rank_ic_tstat` fields from `ProbeBreadthDiagnostics`. Passed through `evaluate_outer_signal_opportunities` when `L1_PROBE_DIAG` diag available. Enables future IC gate without re-plumbing.
- **Selection Downgrade to Breadth Filter (Track B-2, spec line 127)**: `l2_selection_breadth_mode=True` (default) in `Layer2AllocationConfig`. `awf_sim.py` L2 OOS/fit sim loops bypass `rank_and_select` and use all valid symbols as selected set. Removes noise sorting from residual-IC≈0 regime.
- **Bull-Primary Structural Prior (spec line 129, Layer-1 defence)**: `l2_regime_bear_gross_cap` tightened from 0.75→0.35, `l2_regime_crisis_gross_cap` tightened from 0.55→0.25 in `Layer2AllocationConfig` defaults and `awf_sim.py` fallbacks. Reflects `bear` regime's measured −1.43 bps/bar (2025 OOS, 75% negative) and `crisis` IC− zero edge.
- **L1 validation**: ruff/mypy clean across 6 source files + 1 test file. 96 targeted tests pass, 0 new regression.

## Phase 17: L1 Bear-Regime Side Directionality — regime_side_split (ADR-052, 6/29)
- **계기**: 2025 OOS bear regime에서 L1 신호의 net-long 편향 가설 검증 필요. bear price/bar −1.13의 주범이 `cap↓`만으론 설명 불가.
- **regime_side_split 필드 추가**: `ProbeBreadthDiagnostics`에 `regime_side_split: dict[str, tuple[float, float, float, int, int]]` 추가. regime별 `(long_fraction, long_real_mean_bps, short_real_mean_bps, n_long, n_short)` 보유.
- **계측 로직**: `compute_probe_breadth_diagnostics` 기존 regime 루프 내 side_norm(+1/-1) 마스킹으로 O(n) 추가. side 컬럼 부재 시 전부 long(+1) default. NaN/zero-div는 n>0 가드로 방어.
- **포맷 토큰**: `_format_probe_diag`에 `SIDE[{rname}]=long{lf:.0%}/lr{lr:+.1f}/sr{sr:+.1f}/nl{nl}/ns{ns}` 출력.
- **Phase 2 게이트**(보류): bear long_fraction ≥ 0.65 & long_mean < 0 ≤ short_mean 조건 다수 fold 일관 시 `l1_bear_side_policy` (as_is/flat/flip_short) A/B 개시.
- **Test Coverage**: 5 scenarios (bear net-long bias, side-missing default all-long, all-short zero-div, no-regime empty dict, format token). All PASS. L1: ruff/mypy clean.
