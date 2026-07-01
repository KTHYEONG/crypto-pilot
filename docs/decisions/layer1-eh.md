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
- (Compressed...)

## Phase 2: 성능 최적화 1~3차 (ADR-009~016, 025~026, 6/18~21)
- PERF 로깅 도입(레벨 15→10 통일, 계층적 타이밍, [PERF] prefix 일원화)
- Numba JIT rolling z-score(prime 27.78→7.75s, L1 total 47.64→25.17s)
- q-value FDR vectorization→loop 롤백(N≤200 소표본 회귀)
- (Compressed...)

## Phase 3: 신호 패밀리 & MTF 확장 (ADR-017~024, 026~034, 6/21~22)
- Flow family 3종(funding_flow_carry/unwind, flow_exhaustion_reversal) + cell-level taker_imbalance_2d
- 8 저성과 family 제거(trend_donchian, OI 5종, basis 2종, taker_exhaustion) 40→31, per-symbol ENS-DIAG 진단
- FLO 회귀 수정: flow_trend_continuation archetype flow_rev→ts_mom, lsr_oi_regime_filter active화(side_hint 방향성)
- (Compressed...)

## Phase 4: 후반 최적화 v2~v3 (ADR-035~037, 6/22~23)
- L1 Gate+Signal Pool Optimization: per_TF_gate_overrides 자동 fallback, fdr_alpha 0.10→0.15, qw_floor 0.05, 2h trend_ma 제거
- OOM 방지: resolve_safe_nested_workers adaptive cap(max_workers=min(cpu_limit-2,8), oversubscription guard), fork 내 gc.disable()
- P5-R: prequential ThreadPoolExecutor 제거→순차 복원(GIL+cache thrashing 역효과, 11.4→7.9s/TF, -31%)
- (Compressed...)

## Phase 5: Bridge Perf Logging + GC 최적화 (ADR-038~039, 6/23)
- Bridge perf logging Phase 1: `_get_rss_mb()` RSS 측정, stage별 `_sample_rss()` memory delta 추적, `wf_fold_times` per-fold 타이밍, `[PROFILE][MERGE][SUMMARY]` 통계 로깅
- HTF skip 최적화 시도 → 롤백: `run_per_tf_l1()`이 bridge HTF events에 의존적임 확인 (`_build_per_tf_event_index()` 존재하지 않음). HTF skip 시 6h/8h/12h per-TF L1 비활성화 = 품질 회귀
- GC 전략 추가: diagnostics 후 `gc.collect()` (+5.3GB 회귀), bridge 반환 후 `gc.collect()` (tiered re-alignment 전 aligned 해제)

## Phase 6: WSL Stability Optimization (ADR-040, 6/23)
- Max worker cap: `min(cpu_limit, 8)` → `min(cpu_limit, 3)`. Fork worker 폭주(6 worker × 8 threads = 48 threads)가 WSL CPU starvation → network dropout → SSH/Tailscale 단절 원인으로 확인.
- Thread env vars: `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` = `"1"` before each fork. Fork child 내 Numba prange + BLAS thread cascade 제거.
- TF 간 0.5s pause: fork 폭주 후 OS page cache + network buffer 회복 시간 확보.
- (Compressed...)

## Phase 7: Prequential Soft-Cap & Pinned Contract Audit Gap (ADR-041)
- `l1_nested_result_soft_cap_mb`를 dead-config에서 실제 OOM guard로 승격: `resolve_safe_nested_workers()`의 `result_soft_cap_mb` 파라미터 연결, `run_l1_nested_swf()`에서 soft_cap 부족 시 compact 강제(force_compact).
- `l1_nested_workers`(pinned) 의미를 "고정값"에서 "희망 상한(safety-clamped upper bound)"으로 변경: pinned가 low-memory guard, soft-cap guard, oversubscription guard를 우회하지 못함.
- Audit 후속: config 주석 정합, `test_scenario_5_adaptive_worker_cap` stage cap 기대값 6→3 갱신 + psutil mock 추가, `TestResolveNestedWorkersPinned`에 soft_cap override 검증 추가.

## Phase 8: Data Load Arrow Optimization (ADR-042, 6/24)
- **P1-A: Lazy Funding/Metrics Load**: `_prepare_funding_metrics()` 추출, cache-hit + no exec_1m 경로에서 funding/metrics I/O 완전 skip (57 심볼 × GIL-bound parse 낭비 제거).
- **P1-B: Parquet Predicate Pushdown**: `pd.read_parquet(filters=[("timestamp",">=",ms),("<=",ms)])` 도입, enriched 캐시의 row-group statistics 기반 디코드 최적화 → full-read + mask 제거.
- **P2: Arrow Dataset C++ 병렬 스캔**: `_scan_enriched_dataset()`으로 `pyarrow.dataset` + row-group 멀티스레드 디코드(GIL 해제) → 2-pass 분리(I/O parallel + Python-bound 후처리 순차) → cache-hit 경로 CPU 병렬화.
- (Compressed...)

## Phase 9: Bridge Candidate Strategy Perf (ADR-043~045, 6/24)
- **L1-B: Selection Vectorization**: `_vectorized_topk_per_bar` 도입 — per-bar Python loop → sort + drop_duplicates + cumcount rank + ceil(keep) + variant-cap backfill. O(E log E) 벡터화, 0 Python loop. 동등성 보장: sorted cumcount tie-break.
- **L1-A: Diagnostics Gating**: `enable_diagnostics` 파라미터 추가 → evidence fold(12/16)에서 sensitivity/shadow/waterfall skip. 외부 fold/배포 경로는 `True` 유지(진단 SSOT 보존).
- **L2-A: Bridge prepare-once**: `bridge.py` WF 루프 직전 `prepare_labeled_events` 1회 호출 → `PreparedLabeledEvents` 전달. `build_candidate_dataset` fast path(numpy boolean mask) 사용.
- (Compressed...)

## Phase 4 (sic): Bridge-Candidate-Perf-V2 및 Enrich Cache Hotfix (ADR-03X, 6/24)
- **L2-A**: `PreparedLabeledEvents` frozen→mutable dataclass, `enrich_cache: dict[str, Any] | None` 필드 추가.
- **L2-B**: `_precompute_enrich` lazy init — window-invariant만 precompute (arm/entry_regime/overlay_mult/crisis_active/entry_regime_code). 벡터화 affinity matrix lookup (list-comp→numpy indexing).
- **L2-C/D**: `build_candidate_dataset` sig_feat_names + skip_features 경로에서 `enrich_cache` read.
- (Compressed...)

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
- (Compressed...)

## Phase 14: L1 HTF Bottleneck — candidate_panels_to_events Optimization (ADR-049, 6/24)
- **A: Regime×Policy Pre-extraction**: `_convert_single_panel` regime 루프에서 array indexing을 policy당 21회에서 regime당 1회로 감소. regime_mask를 regime 루프 밖에서 1회 pre-extract 후 policy 루프에서 재사용. O(R×P) → O(R) indexing reduction.
- **B: sort_values 제거**: `candidate_panels_to_events` 최종 `sort_values("datetime")` 제거. downstream(label_candidate_events, portfolio selection 등)이 entry_idx 기반 접근으로 정렬 불필요. O(N log N) full-table sort 제거.
- **C: Numba _robust_zscore_numba**: `_cross_sectional_robust_zscore` 위임 함수로 `_robust_zscore_numba @njit` 도입. unique group별 argsort 단일 패스 walk, Python O(U×E) loop → Numba O(E log E). 각 group 내 median/MAD 계산을 numba-compiled 단일 패스로 통합.
- (Compressed...)

## Phase 15: L1 Probe Breadth Diagnostics (ADR-050, 6/29)
- L1 게이트 전부 PASS이나 L2 realized gross가 음수인 모순 해소를 위해 env-gated DEBUG 계측 추가
- `ProbeBreadthDiagnostics` frozen dataclass + `compute_probe_breadth_diagnostics()`: (a) breadth-decay (k=3/10/20/-1)로 selection inflation 정량화; (b) gross − rt_cost로 cost drag 분리; (c) Spearman rank-IC + Fisher-z tstat로 신호력 부재 진단; (d) 전체 realized 분포 통계
- `L1_PROBE_DIAG` env gate 패턴: 기존 `L2_DIAG_ATTR`/`L2_MULTI_TF`와 동일 규약 (`""`/`"0"`/`"false"`/`"False"` → disabled)
- (Compressed...)

## Phase 16: Track A IC Gate Spec Compliance + Selection Downgrade + Bull-Primary Prior (ADR-051, 6/29)
- **IC Hard Gate → DEBUG Monitoring (spec §Track A, lines 106-107)**: Spec explicitly defers IC hard gate ("IC 하드 게이트 보류") until Track B produces cross-sectional alpha. Removed `("ic_tstat", ...)` and `("ic_sign_consistency", ...)` from `evaluate_layer1_readiness` check_specs. Moved IC pooling to `logger.debug` conditional. Prevents production always-BLOCK where `rank_ic_all=0.0` (default when `L1_PROBE_DIAG` env not set).
- **Config Params Reserved (l1_min_ic_tstat, l1_min_ic_sign_consistency)**: Kept in `CandidateStrategyConfig` for future Track B activation. Not wired into check_specs.
- **Probe Metric Default = "breadth"**: `l1_probe_metric` default changed from implicit top-k to `"breadth"`. `evaluate_outer_signal_opportunities` uses per-decision cross-sectional mean of all symbols instead of risk-score-ranked top-k when probe_metric="breadth". S4 test validates gross-all path.
- (Compressed...)

## Phase 17: L1 Bear-Regime Side Directionality — regime_side_split (ADR-052, 6/29)
- **계기**: 2025 OOS bear regime에서 L1 신호의 net-long 편향 가설 검증 필요. bear price/bar −1.13의 주범이 `cap↓`만으론 설명 불가.
- **regime_side_split 필드 추가**: `ProbeBreadthDiagnostics`에 `regime_side_split: dict[str, tuple[float, float, float, int, int]]` 추가. regime별 `(long_fraction, long_real_mean_bps, short_real_mean_bps, n_long, n_short)` 보유.
- **계측 로직**: `compute_probe_breadth_diagnostics` 기존 regime 루프 내 side_norm(+1/-1) 마스킹으로 O(n) 추가. side 컬럼 부재 시 전부 long(+1) default. NaN/zero-div는 n>0 가드로 방어.
- (Compressed...)

## Phase 18: L1 Cross-Sectional Alpha — 4 XS Families (2026-06-30)
- **계기**: result.md fold#1 −17.1%·CAGR 6.1%≪30%의 근본 원인이 L1 횡단면 alpha 부재로 진단됨(next.md §4). 30개 family 전부 per-symbol 시계열 변환 → rank IC≈0. "발화 자체가 횡단면"인 진정한 XS alpha 필요.
- **신규 helper 2종**: `_cross_sectional_rank_signed_2d` (per-timestamp rank → signed score [-1,1] + tercile side {-1,0,1}, min_cross_section guard), `_beta_residual_return_2d` (BTC-beta rolling residual, rolling_sum over lookback). 기존 Numba/import 변경 0.
- **신규 family 4종**: `xs_momentum`(beta-residual ret L12/48), `xs_carry`(-funding_z 96/168), `xs_flow`(flow_z_24), `xs_oi_skew`(-oi_build_z_42*sign(lsr_log_z_42)). 전부 `_cross_sectional_rank_signed_2d` 변환, `metadata={"archetype": "xs_alpha"}`.
- (Compressed...)

## Phase 19: L1 XS Factor Spread Diagnostics — env-gated pre-promotion 계측 (ADR-053, 2026-06-30)
- **계기**: XS factor(`xs_alpha` families)는 승격 게이트(per-pair incremental 검정)에서 배제됨. 기존 `compute_probe_breadth_diagnostics`는 `merged`(승격된 registry 신호)만 사용 → XS 부재. `rank_ic −0.108~+0.112`는 trend pair만의 잔차 IC로 XS factor 자체의 스프레드 엣지는 미계측. per-pair 게이트가 실제 portfolio-level XS alpha를 가리는지 판정 불가.
- **신규 dataclass + 함수 3종 + rank-IC helper**: `XsFactorSpreadDiagnostics` frozen dataclass + `compute_xs_factor_spread_diagnostics()` + `_l1_xs_spread_diag_enabled()` + `_format_xs_spread_diag()` + `_xs_rank_ic()` helper. 소스는 `realized_event_results`(pre-promotion 전체 candidate, XS 포함). side-adjusted 실현값으로 per-bar tercile long-short 스프레드 직접 산출.
- **계측 항목**: per-XS-factor `(n_bars, n_events, spread_mean_bps, spread_std_bps, spread_sharpe, spread_lcb_bps, rank_ic, rank_ic_tstat, long_frac)`. Bootstrap LCB via `moving_block_bootstrap_mean`. rank-IC는 per-bar Spearman ρ + Fisher-z tstat (≥3 cross-section).
- (Compressed...)

## Phase 20: residual_reversion(beta_neut) Regime Gate Hard Masking (ADR-054, 2026-07-01)
- **계기**: Phase 0 measure-first residual_reversion@R0(bull_quiet)만 breakeven 상회(4/4 fold LCB>0), R2/R3/R4는 명확히 음수임을 확인. 기존 `_allowed_regimes_for_archetype`의 beta_neut fallback(`("bull_quiet","bear_quiet","transition")`)은 `_attach_signal_context`의 게이트 조건이 `mean_rev_gating_enabled=True and archetype=="mean_rev"`만 검사하여 **도달 불가능한 죽은 코드**였음. spec의 sizing-multiplier 대안(discrete 증거 불일치 + 전역 blast radius) 기각 후 하드 마스킹 채택.
- **변경점** (3 source files):
  - `config.py`: `beta_neut_gating_enabled: bool = False` 필드 추가 (opt-in, 기본 무회귀 보장).
  - `rules.py` / `rule_signals.py` (mirror): `_allowed_regimes_for_archetype`에 `beta_neut → ("bull_quiet",)` 명시적 분기 추가. `_attach_signal_context` 게이트 조건에 `or (cfg.beta_neut_gating_enabled and archetype == "beta_neut")` 추가.
- **회귀 방지**: 기본값 `False` → 기존 champion 동작 무변경. 4개 신규 테스트(`test_beta_neut_gated_*`)로 gate 활성화/차단/기본값/전역오버라이드 검증 완료. L1 validation: ruff 0 error, mypy 0 new error, regression tests 58/58 pass.
- (Compressed...)

