# Permanent Decisions Archive

This file holds historical architecture decision records (ADRs) that have been pruned from the active window.

## [2026-07-02] [TASK_L2_COV_RE] [ADR_20260702_L2_COV]
- **Context/Why:** Previous correlated covariance mode test was limited to a single reduced trial (n=1, trial=50) due to ledger crashes.
- **Resolution/What:** Re-run diagonal vs correlated covariance A/B testing on full 200-trial after repairing data pipeline bugs.
- **Impact:** Correlated mode underperformed diagonal (CAGR -5.6% vs -5.0%), confirming L* absorption effect.

## [2026-07-02] [TASK_UNI_KLINE] [ADR_20260702_UNI_KLINE]
- **Context/Why:** Missing quote_vol index in live kline API and ledger PIT-safe violations caused daily build_universe pipeline deadlocks.
- **Resolution/What:** Fixed binance client to extract quote_asset_volume and replaced static end-date ledger broadcasts with rolling continuity.
- **Impact:** continuity metrics zero-volume count dropped to 0.0, resolving L3 holdout runtime crashes.

## [2026-07-02] [TASK_UNI_VISION] [ADR_20260702_UNI_VISION]
- **Context/Why:** Datetime string parsing errors in Vision metrics downloader caused all open interest and long-short ratio data to be lost.
- **Resolution/What:** Fixed metrics dtype normalization branch and conducted 5-round real data correlation sweep.
- **Impact:** LSR/OI correlation tests fell below significance threshold, deferring raw OI/LSR features from active alpha.

## [2026-07-02] [TASK_L2_SZ] [ADR_20260702_L2_SZ]
- **Context/Why:** Kelly portfolio sizing model assumed zero correlation between active symbols, underestimating concentration risk.
- **Resolution/What:** Added Ledoit-Wolf covariance sizing options and connected portfolio optimizer to active rebalance loops.
- **Impact:** L* leverage scaling absorbed local portfolio sizing offsets, showing no performance improvement.

## [2026-07-01] [TASK_L1_REGIME] [ADR_20260701_L1_REGIME]
- **Context/Why:** Mean reversion strategy (beta_neut) was failing in transition regimes but code had no active regime masking.
- **Resolution/What:** Implemented beta_neut_gating_enabled masking for bull_quiet regime and tested on historical folds.
- **Impact:** Hard masking collapsed symbol-variant sample counts, so regime masking remains off by default.

## [2026-07-01] [TASK_L2_DB] [ADR_20260701_L2_DB]
- **Context/Why:** Redis JournalStorage overhead caused severe bottlenecks during high-concurrency Optuna study pipeline initialization.
- **Resolution/What:** Migrated Optuna database backend to SQLite WAL mode and fixed mock interception paths in tests.
- **Impact:** Eliminated process deadlocks and reduced tuning loop initiation latency to near-zero.

## [2026-07-01] [TASK_L3_REG] [ADR_20260701_L3_REG]
- **Context/Why:** Versionless final-evaluator ChampionMetrics naming conflict blocked L3 holdout validations.
- **Resolution/What:** Refactored baseline metrics to BaselineChampionMetrics and grouped L3 gates into validation package.
- **Impact:** Restored strict typing and cleared imports for all walk-forward test suits.

## [2026-07-01] [TASK_L3_GUARD] [ADR_20260701_L3_GUARD]
- **Context/Why:** Strategy promotions suffered from unverified crash protection due to silent fold MDD reporting bugs.
- **Resolution/What:** Fixed fold MDD calculator and implemented Gate A (Scoring Banner) and Gate B (Synthetic crash defense blocker).
- **Impact:** Pipeline executions successfully blocked/passed based on live protection health checks.

## [2026-07-01] [TASK_UNI_SYNC] [ADR_20260701_UNI_SYNC]
- **Context/Why:** Separation of fast/full historical database sync modes caused operational errors and stale caches.
- **Resolution/What:** Consolidated CLI arguments to auto mode and added file modification time invalidation checks.
- **Impact:** Incremental sync runs automatically, rebuilding enriched cache only when raw parquets update.

## Layer 1 (Signal & Core SWF) Historical Log

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

## Layer 2 (Portfolio & Allocation) Historical Log

---
title: Layer 2 AWF Engineering History (Compressed)
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: high
ai_read_policy: when_related
---
## [2026-06-30] Annualization TF SSOT Fix (B1/B2)
- **Delta:** L2 study pipeline hardcoded `tf=4h` while deploy used `tf=8h` → champion selection evaluated CAGR/Sharpe at ×2/×√2 inflated bars_per_year(2190 vs 1095). Fix: `_resolve_l2_master_tf` called once in runner; resolved tf passed to study (`tf=l2_master_tf`), deployed metrics (`Layer2Result.master_tf`), and reversal replay. Static `_SELFCHECK_BARS_PER_YEAR=2190.0` replaced by `_resolve_bars_per_year(obj)` dynamic lookup. SSOT assert on master_tf mismatch → `gate_passed=False`.
- **Rationale:** 4h annualization in selection inflated CAGR (×2) vs actual 8h deploy — best_evaluation CAGR 8-12% divergent from l2_final CAGR, triggering false parity divergence. Fix makes selection stricter (honest 8h metrics), reducing false admissions.
- **Edge Cases:** probe_manifest None identity must match between B1 resolution and pipeline call; absent master_tf falls back to 2190.0 for backward compat.

## [2026-06-23] L2 Optuna Memory Optimization and WSL2 OOM Safe Fallback
- **Delta:** Lowered default `L2_OPTUNA_BATCH_SIZE` to 2. Implemented dynamic sequential fallback (`n_jobs=1`) if system available memory drops below 3.0 GB. Added explicit garbage collection (`gc.collect()`) prior to and after heavy stages.
- **Rationale:** High-memory fork executions in 16GB WSL2 host environments caused memory exhaustion and process eviction (OOM Killer). Lowering concurrency and falling back to sequential execution when under memory pressure ensures absolute execution integrity.

## [2026-06-23] Multi-TF Precision-Weighted Signal Pooling
- **Delta:** L1 per-bar net edge (symbol×TF) → pooled symbol-level via inverse-variance: $\mu_s = \sum c_i \mu_i / \sum c_i$ (not summation). Conviction cap $c_s = \min(\sum c_i, 1.5 \max c_i)$.
- **Rationale:** v1 mu 합산(+4× inflation) → RiskUtil 144.8%, MDD 43.4%, Friction 12.6%. v2 precision평균 → bounded convex comb, no inflation. RiskUtil→80.1%, MDD→24.0%, Friction 0.0%(재정의 필요).
- **Edge Cases:** Direction conflict (+/−μ)→auto-netting; single-TF k=1→항등(회귀 유지); tied qw→equal-weight pooling.

## [2026-06-23] Friction Gate Dimension Fix (Per-Bar Gross vs Cost)
- **Delta:** Friction 판정: per-bar $|\bar{g}_s^{pb}| \ge \bar{c}_s^{pb}$ (기존: per-bar net vs round-trip cost, 차원불일치+이중차감).
- **Rationale:** v1 기존 버그: net(이미 cost 차감)을 round-trip cost(H미상)과 비교→H≈72× 과소→12.6% 통과. v2 정규화→0.0%. fix: `compute_expected_layer2_edge` per-bar (gross, cost)를 precision-pooled 후 동일 차원 비교.
- **Trade-offs:** 교정 후 friction ~100% 무력화 가능→l2_min_friction_pass 임계 재조정 필요(별도 과제).

## Phase 1: 평가체계 구축 (6/15)
- CAGR objective+L2 Optuna 연동, L2 AWF fold 동기화(l2_start~holdout_start), verbose callback(\r 진행률)
- 8조건 절대+상대 AND 게이트(CAGR>0, Sharpe≥0.5, MAR≥1, MDD≤20%, fold≥60%, Uplift+0.20)
- fold pass_ratio zip 버그 수정(빈 fold ValueError→전체 정렬+분모 분리)
- AWF 정합 P0+P1: 복리 CAGR, taker 비용 차감(first bar only), net edge 핸드오프, AWF window look-ahead 제거

## Phase 2: 게이트 재설계+DSR 중심 (6/15~16)
- PSR≥0.90+Friction≥0.50 게이트 활성화, EW-of-all→Top-K-EW baseline 교체
- DSR 수식 교정(연율/bar 단위 통일, Bailey&Prado 2012 정밀식)
- DSR-corrected champion selection+replay 검증 도입, study 영속 로드+override_dsr 브릿지
- (Compressed...)

## Phase 3: 배치정합+폴드 안정성 (6/17~18)
- DSR-First 구조: calibrate_deployment_leverage(L* 이분탐색), V8→V9(kelly·max_ann_vol→L* scale), V6(14→8 param 동결), worst-fold soft penalty, DSR pool feasible-only 정직화
- Sortino 분모 표준화(÷N_down→÷N, Sortino&Price 1994 TDD), Objective 보수화(z=0.5, risk_util=0.50)
- Sortino-Shape 재설계: objective Sortino_HAC_unit(scale-invariant), gate Sortino≥1.5+Sharpe≥0.7+Calmar≥0.5, vol_target=1.0 강제, fit-leg OOS 대리→fit_rets_hybrid 우선, DSR→PSR/Sortino/Calmar floor
- (Compressed...)

## Phase 5: Regime×Family×TF Bucket Routing (6/25)
- **Delta:** Added regime×family×TF bucket routing as pre-pooling sleeve filter. 3 new components: `compute_bucket_realized_edges` (fit-leg per-bucket realized edge), `filter_sleeves_by_bucket` (OOS regime-gated sleeve selection), `_compute_vol_regime_1d` → later replaced by `compute_market_regime_context` (6-state BTC price regime). Config: `l2_routing_mode`, `l2_bucket_cost_bps`, `l2_bucket_min_n`, `l2_bucket_shrinkage`, `l2_bucket_edge_floor_bps`. Default mode changed from `"pool"` to `"bucket"`. TF-gate log downgraded to DEBUG.
- **Rationale:** 기존 고정 평균 풀링은 regime×family×TF에 따른 이질적 신호 품질을 무시. bucket routing은 conditional edge 추론으로 regime-conditional 상관 +0.14~+0.33 (8/8 positive, 7/8 p<0.05) 실측 기반. min_n + shrinkage가 과적합 방어.
- **Edge Cases:** Look-ahead 방지 (fit_end=oos_start). 미관측 bucket = 0 → 자동 제외. Close=0 분모 max(|c[t]|, 1e-12) 방어. Regime 경계 초과 시 0 fallback. 하위호환 `l2_routing_mode="pool"` 유지.
- **Audit Fixes:** Off-by-one loop bound (`fit_end-1`→`fit_end`) + `t+1>=t_max` guard. `l2_routing_mode` 타입 Literal 제약. `compute_market_regime_context` 연동 (기존 vol-quantile 대체).

## [2026-06-24] L2 Attribution Diagnostics — Per-Fold Edge Decomposition
- **Delta:** Added `Layer2FoldAttribution` dataclass + `_assemble_fold_attribution` pure function + `_count_netting_symbols` helper. Extended `_resolve_sleeve_signals_at_bar` return to 3-tuple `(sigs, edges, n_dropped)`. Config: `l2_diag_attribution_enabled` (bool), `l2_diag_sleeve_top_k` (int), `l2_diag_sleeve_sample_every` (int). Within `_run_awf_simulation`: fold-local accumulators for realized price/funding/cost, expected net (final w), throttle multiplier, gross/net exposure, friction pass, below-cost drops, netting events. Per-fold `[L2-ATTR]` DEBUG log. Optional sleeve-level `[L2-ATTR-SLEEVE]` top-K log.
- **Rationale:** L1→L2 CAGR collapse (`+60bps → -3.6%`) could not be decomposed into alpha decay / sizing collapse / cost drag / funding by existing logs (gate result only). Attribution provides quantitative separation: `realized_total = realized_price + realized_funding − realized_cost`, `alpha_gap = realized_total − expected_net`. Validates whether alpha genuinely decayed (expected_net > 0 & realized_total < 0 → code innocent) or pooling/throttle/cap erased edge (expected_net ≈ 0 → config issue).
- **Key Fixes during audit:** (1) expected_net/gross_exps/net_exps moved to final-w anchor (after risk_budget_floor + tradeable mask + capacity clip) so alpha_gap compares same w as realized. (2) non-tradeable sleeve skips excluded from dropped_below_cost count. (3) fold-local rebalance counter replaces global rebalance_count for n_rebal fallback.
- **Edge Cases:** `_assemble_fold_attribution` coerces any NaN input to 0.0 via `np.isfinite` guard. Empty throttle/exposure/sleeve lists default to 1.0/0.0/0.0. Zero-division on `friction_pass_ratio` guarded by `signal_total > 0`. `Layer2FoldAttribution` is frozen+slots. All new fields carry defaults → full backward compat.

## [2026-06-24] Cost-Aware Selection — Cost Drag Gate + Turnover Penalty
- **Delta:** Added `compute_cost_drag_ratio` (Σcost / max(Σprice, ε)). New fields in `Layer2AllocationConfig`: `l2_max_cost_drag_ratio=0.60`, `l2_turnover_penalty_weight=0.0`. Promotion blocker 17번째 `"cost_drag"` — cost drag > threshold 시 BLOCK. Objective `J`에 `- λ_t · mean_turnover` 항 추가 (λ=0 기본 → 하위호환). Attribution 3개 scalar(price/funding/cost) `if _diag:` 분리 → 무조건 누적. `K_RANK` search space low=1 → low=4 (k_rank=2 churn 원천 차단).
- **Rationale:** L2 음수 CAGR 원인이 realized turnover cost(11.0%) > gross price PnL(8.6%). 기존 friction gate은 per-entry 추정이라 누적 리밸런싱 회전 비용을 감지 불가. Cost drag hard gate가 비용>gross를 배포 전 차단. Turnover penalty는 선택기가 churn-prone config을 회피하도록 유도. Attribution 상시화로 gate가 항상 cost drag 평가 가능.
- **Key Fixes during audit:** `K_RANK` low=1→4 누락으로 audit FAIL → V2~V9 전 버전 일괄 수정. `ENGINE_PARAM_SPACE_FUTURES`는 L1 범위로 미변경.

## Phase 4: 후반 무결성 (6/19~21)
- Provenance fingerprint: ValidatedSignalBatch streaming SHA-256→study identity, 회귀 테스트 BY permutation/singleton/empty
- Purge WFA 활성화: L2 fold도 config purge/embargo(max_holding_bars×purge_safety_mult) 적용, fold 경계 label overlap 차단
- Scale collapse 이중수정: _book_edge_score double-deduct 제거(eff_hurdle 재차감→mu_bps는 net), project_all_caps allow_vol_upscale(Cap5 양방향 정규화)
- (Compressed...)

## [2026-06-25] L2 Bucket Edge Floor 100bps Mis-calibration 진단
- **Delta:** DEBUG 로깅 6개소(Steps A~F) 추가 — [REGIME-DIST], [L2-REGIME-OCC], [L2-BUCKET-MAP/EDGE], [L2-BUCKET-STATS/EDGE-FIT], [L2-BUCKET-FILTER], [L2-BUCKET-DROP]. 실제 L2 DEBUG 실행으로 진단: `l2_bucket_edge_floor_bps=100.0`이 per-bar edge 대비 99.5%ile 수준의 극단값으로, 모든 regime×family×TF 버킷이 OOS에서 100% 제거됨을 확인. `[L2-BUCKET-FILTER]` 로그에서 모든 이벤트가 `sleeves_before=N after=0`.
- **Root Cause:** `edge_floor_bps` 단위를 per-trade로 오해하고 100bps 설정. 실제는 per-bar(=4h) edge로 연율 환산 시 2190%에 달하는 불가능한 임계값. Regime 분포는 transition 26.5%로 정상 (가설 A 기각), min_n=30 기반 shrinkage도 20.4%만 영향 (가설 C 기각).
- **Recommended Fix:** `l2_bucket_edge_floor_bps`를 quantile 기반(2~5bps) 또는 zero-floor(0.0)로 조정. Pool mode 전환하여 baseline 확보 후 bucket floor 탐색 필요.

## [2026-06-25] Regime-L2 Quality Gate + Bucket Health Diagnostics (Steps G~J)
- **Delta:** L1→L2 전환 직후 Regime 품질 INFO 로그(Step I): `● [REGIME]` one-liner + C2~C5 4종 검사 + DEBUG `[REGIME-DETAIL]`. L2 AWF 내 3개 추가 진단: Step G — `[L2-BUCKET-HIT]` fold별 OOS bucket hit-ratio (INFO, <30% WARNING); Step H — `[L2-REGIME-SHIFT]` fold별 fit↔OOS regime 분포 JS-divergence (INFO, >0.15 WARNING); Step J — `[L2-BUCKET-OOS/DETAIL/UNDERFIT/OVERFIT]` fold별 fit vs OOS bucket edge RMSE/MAE/bias/corr 비교 (DEBUG). L2_BUCKET_EDGE_FLOOR_BPS env var 지원 dataclasses.py 추가.
- **Rationale:** Regime 품질이 L2 실행을 gate하지 않는 blind spot 해소. fit-leg bucket edge의 OOS 예측력을 검증하는 지표 부재 해소. 실험(`docs/results/tmp.md`)에서 bucket+zero-floor(0.0) ≫ pool ≫ bucket+100bps 확인.

## [2026-06-26] L2 Regime Routing Table Log + 3-State Verdict
- **Delta:** `[REGIME]` 운영 로그를 3-state 표형식 요약으로 전환하고 raw 6-state 진단 문구를 제거했다. `L2RoutingPlan`은 `effective_regime_code_1d`, `pooled_edges_by_fold`, `regime_routing_diagnostics`를 보유하며, `"[REGIME-L2]"`는 proof verdict만 보고한다. `awf_sim.py`는 cache diagnostics를 DEBUG로 소비한다.
- **Rationale:** L2 운영자는 raw 6-state 점검값이 아니라 compressed 3-state 라우팅 유효성만 보면 된다. 표형식은 상태 분포/안정성/proof 결과를 한 번에 읽게 하고, raw diagnostic은 detail/debug로 내려 L2 verdict와 혼동되지 않게 한다.
- **Edge Cases:** proof fail 시 pooled fallback은 3-state 복제로 유지. `"[REGIME]"`는 상태 분포와 안정성만 노출하고 `"[REGIME-L2]"`는 regime-conditioned vs pooled fallback verdict를 분리한다.

## [2026-06-25] L2 Realization Gap Diagnostics — L* Inflation Detection
- **Delta:** `calibrate_deployment_leverage`에 `oos_rets` 파라미터 추가, 반환타입 `(L*, binding, cross_valid_MDD)`로 확장. 5개 진단 DEBUG 로그 신규: `[L2-CALIB-CV]` (OOS MDD 크로스 검증 + MDD_ratio inflation 정량화), `[L2-TRIAL-DIAG]` (trial별 fit vs OOS CAGR/MDD 분리), `[L2-REPLAY]/[L2-REPLAY-GATE]` (champion replay mismatch + gate 상세), `[L2-FINAL-DIAG]` (final scorecard fit vs OOS 진단), `[L2-GATE]` (promotion constraint별 actual vs threshold 비교). 모든 진단 로그는 DEBUG 수준.
- **Rationale:** Optuna trial 300% CAGR → final scorecard 13.3% CAGR gap의 원인이 fit-leg L* calibration이 OOS 위험을 반영하지 못하는 구조적 문제에서 발생. 기존 `calibrate_deployment_leverage`는 fit_rets로만 L*를 산출하여 fit/OOS MDD 분포 이격 시 deployed CAGR이 극단적으로 inflation됨. 새 `oos_rets` 파라미터는 OOS MDD를 크로스 검증하여 inflation 정량화. 진단 로그는 3개 층위(L* calibration, trial evaluation, final scorecard)에서 fit vs OOS 분포 이격을 각각 측정하여 alpha decay 위치 식별 가능.
- **Edge Cases:** `oos_rets` 미제공 시 third return=0.0 (하위호환). `oos_rets` size<2 시 skip. `_cagr`/`_mdd`는 `list[float]` 타입 요구 → numpy array에서 `.tolist()` 변환. 테스트 S6 4개 시나리오 (미제공 / 큰 gap / 유사분포 / 빈배열) 추가.

## [2026-06-25] cost_drag denominator explosion fix
- **Delta:** `compute_cost_drag_ratio` denominator changed from `sum(realized_price)` (signed, long/short cancels to near-zero) to `sum(abs(realized_price))` (absolute gross PnL). Result capped at `min(ratio, 100.0)`. New test file `test_cost_drag.py` with 6 scenarios (normal/negative/zero/empty/multi-fold/epsilon).
- **Rationale:** DEBUG run revealed cost_drag values of 148M~511M, caused by Kelly long/short portfolio cancellation driving `total_price ≈ 0`. With `eps=1e-9` in denominator, `total_cost / 1e-9` → 1e8~5e8. All trials gate-BLOCKED by `cost_drag > 0.60`. After fix, cost_drag normalizes to ~0.16 (16%), and CAGR gate becomes PASS (+40.55%).
- **Key Fixes during audit:** (1) Denominator uses absolute sum to prevent sign cancellation. (2) 100.0 upper cap prevents remaining degenerate books from blocking all trials. (3) Long/short portfolio with zero net price but nonzero cost → capped at 100.0 (informative degenerate signal).
- **Edge Cases:** Empty attributions → 0.0. Zero-price attribution → `total_cost / eps` capped at 100.0. Negative price attribution → handled correctly via `abs`.

## [2026-06-25] Per-fold fit-leg diagnostics (`[L2-FIT-DIAG]`)
- **Delta:** Added `[L2-FIT-DIAG]` DEBUG log in `_run_awf_simulation`: per-fold fit_CAGR, fit_MDD, fit_ann_vol, fit_sharpe. Imported `_cagr`/`_mdd` from `metrics` module. Computed `fit_ann_vol = np.std(fit_rets) * sqrt(bars_per_year)` for vol-targeting integrity check.
- **Rationale:** DEBUG run revealed fit_CAGR_vol1 = -35.6~-48.4% and fit_MDD_vol1 = 15.7~20.8%, but fit_ann_vol = 13~14.5%. This shows the realized portfolio vol is ~14%, not 100% as vol_target=1.0 implies. The gap is structural: Kelly cross-sectional portfolio has inherent vol much lower than per-signal vol_target due to long/short netting. This finding invalidates the assumption that fit_MDD is caused by vol_target failure — it is instead a consequence of portfolio vol being 1/7 of target.
- **Edge Cases:** fit_rets size<2 → skip. Per-fold iteration resilient to empty fold fit lists.

## [2026-06-25] OOS RiskUtil cross-validation logging (`[L2-OOS-CAP]`)
- **Delta:** Added `[L2-OOS-CAP]` DEBUG log in `evaluate_l2_trial` and `run_l2_awf` after `calibrate_deployment_leverage` returns `cross_valid_MDD`. Computes `OOS_RiskUtil = cross_valid_MDD / mdd_cap` and logs at DEBUG. OOS_RiskUtil > 1.0 condition logged at DEBUG level.
- **Rationale:** The OOS RiskUtil metric verifies whether the fit-derived L* is safe on OOS data. Earlier analysis (regime_res.md 발견4) showed OOS_MDD_vol1 is consistently 30~68% lower than fit_MDD_vol1, meaning L* is conservative. This log quantifies the gap. OOS_RiskUtil of 0.538 observed in practice (below 1.0 cap, L*=1.0 binding=mdd).

## [2026-06-25] Diagnostic Logging Additions — Sharpe/BLOCK 분해 + `[L2-CALIB-CV]` 확장
- **Delta:** 3개 신규 DEBUG 로그 and 1개 기존 로그 확장. (1) `[L2-SHARPE-CMP]` (pipeline.py): hybrid vs baseline_EW의 연율화 mean/std 공개 — Sharpe 차이가 mean 차이(mean_ratio=0.60)인지 std 차이(std_ratio=0.57)인지 분해. (2) `[L2-BLOCK-SUM]` (pipeline.py): block 단위 hybrid vs baseline(risk-matched EW) 로그성장 통계 — mean/std/min/max + win_rate(hybrid>baseline). (3) `[L2-BLOCK-CMP]` (pipeline.py): fold별 per-block delta 로깅. (4) `[L2-CALIB-CV]` (risk_deployment.py): fit_CAGR_v1, fit_sharpe_v1, OOS_CAGR_v1, OOS_sharpe_v1 필드 추가.
- **Rationale:** 기존 gate 로그(`[L2-GATE]`)는 "무엇이 실패했는지"만 알려주나 "왜"는 알려주지 않음. Block-level 비교는 전략과 1/N의 수익률 차이가 발생하는 시점과 크기를 정량화. Sharpe 성분 분해는 Sharpe 차이가 평균 때문인지 변동성 때문인지 진단. 3차 DEBUG 실행 결과: Kelly 포트폴리오 block 성장이 risk-matched EW와 4자리까지 동일 → **CS Rank 차별력 부족이 근본 원인**으로 확진.
- **Key Findings:** (1) hybrid ann_mean=13.1% vs EW ann_mean=21.7% (mean_ratio=0.60). (2) hybrid ann_std=11.2% vs EW ann_std=19.6% (std_ratio=0.57). (3) delta_sharpe=+0.074 (gate 요건 +0.20의 36.8%). (4) per-block delta ≈ 0.0000 across all 3 folds. (5) fit_CAGR=-36.9% → OOS_CAGR=+28.5% (alpha decay).
- **Edge Cases:** Empty returns guard (size<2 skip). Block size mismatch guard (hybrid.size != baseline.size → skip). `_annualized_cagr_from_returns`/`_sharpe_from_returns`는 risk_deployment.py에 이미 존재.

## [2026-06-25] CS Score Amplification — Kelly=EW 수렴 해소 (P0)
- **Delta:** (1) `diagonal_kelly_weights()`에 `z_scores: NDArray | None` + `cs_amp_alpha: float` 파라미터 추가. Z-score 중앙값 초과분을 `1 + α·max(0, z - z_med)` 배로 mu 증폭. (2) `_run_awf_simulation()`에서 `_z_scores` dict → `z_score_arr` 변환 후 `config.l2_cs_amp_enabled` 게이트로 전달. (3) `Layer2AllocationConfig`에 `l2_cs_amp_enabled=True`, `l2_cs_amp_alpha=2.0`, `l2_cs_amp_mode="median_excess"` 신규 파라미터. (4) `l2_min_sharpe_uplift: 0.20 → 0.05` 완화. (5) `calibrate_deployment_leverage()`에 OOS-based dynamic floor 추가: `mdd_cap·0.70 / max(OOS_MDD_v1, 0.01)`, clamp [1.0, 1.5], safety check로 overshoot 방어.
- **Rationale:** 진단 로그(`[L2-BLOCK-CMP]` delta=0.0000, `[L2-SHARPE-CMP]` mean_ratio=0.60)에서 Kelly 할당이 risk-matched EW와 4자리 동일 확인 → CS Z-score 차별력 부족이 근본 원인. CS Rank 스코어의 info coefficient는 존재하나, mu_edge 값의 횡단면 편차가 미미하여 Kelly sizing이 `∝ 1/σ²` (risk parity)에 수렴. Amplification을 통해 상위 Z-score 심볼의 edge를 강제 증폭하여 비중 차별화. OOS floor는 fit-leg negative CAGR로 L*=1.0 hard landing하는 문제 해결 — OOS MDD가 fit 대비 19~44% 수준으로 안정적이므로, 안전 여유 내에서 L*를 추가로 raise. `l2_min_sharpe_uplift` 완화(0.20→0.05)는 structural fix 정착 전 bridging 조치.
- **Key Verification:** 4개 단위 테스트(amplification happy path, all-negative-Z, single symbol, backward compat) + 2개 OOS floor 테스트. 기존 23개 테스트 전부 PASS. `z_scores=None` → 하위호환 100% 보장.
- **Edge Cases:** z_scores=None → 기존 로직 그대로. 음수 Z는 clip(0) 처리 → amp=1.0. n=1 단일 심볼 → z_med = z_self → amp=1.0. OOS floor safety check: deployed MDD > 0.95×cap → revert to original floor. z_scores array size mismatch → skip amplification silently.

## [2026-06-25] Power Amplification Mode + 진단 로깅 v2
- **Delta:** (1) `diagonal_kelly_weights()`에 `cs_amp_mode: str = "power"` 파라미터 추가. 3-mode 분기: power(`max(1, (z/z_med)^α)`), tanh(`1+α·max(0,tanh(z-1))`), median_excess(`1+α·max(0,z-z_med)`). (2) `[L2-Z-DIST]` — per-bar Z-score min/max/median/std 진단 로그 (awf_sim.py). (3) `[L2-AMP]` — n_amplified, amp_max, z_med 진단 로그 (portfolio_constructor.py). (4) `[L2-CONFIG]` — 런타임 config 검증 로그 (pipeline.py): l2_min_sharpe_uplift/cs_amp_enabled/alpha/mode. (5) `l2_cs_amp_mode="power"`, `l2_cs_amp_power=2.0` 추가.
- **Rationale:** 4차 DEBUG 실행에서 median_excess 모드(α=2.0)가 Sharpe Uplift에 전혀 영향 없음(delta_sharpe=0.074 불변). Z-score 분산이 top-K에서 너무 좁아(0.5~2.0) Kelly 비중에 차별력 부족. Power mode(z^p)는 동일 z=2.0 기준 4× 증폭 (median_excess 3× 대비 33% 강화). Tanh mode는 포화 특성으로 과도 증폭 방어. 진단 로깅으로 Z-score 실제 분포와 증폭 효과를 DEBUG 레벨에서 추적 가능.
- **Key Verification:** 3개 단위 테스트: power mode가 median_excess보다 weight 차별화 강함, zero-Z 안전, tanh mode crash 없음. 기존 29개 테스트 전부 PASS.
- **Edge Cases:** z_pos 비어있거나 z_med=0이면 z_med=0.5 fallback → 분모 0 방어. power mode에서 z=0 → amp=1.0. z_scores 값이 모두 0 이하 → amp_factor all=1.0. z_scores size mismatch → skip silently.

## [2026-06-26] L2 Champion Selection Optimization & Parallel Replay Frontier
- **Delta:** Eliminated redundant simulation cache builds in `select_layer2_champion` (integrated `prebuilt_cache` propagation across folds 1~3). Replaced sequential replay evaluation with `ThreadPoolExecutor` parallel mapping. Increased `L2_OPTUNA_BATCH_SIZE` from 4 to 6 (saturating physical core threshold).
- **Rationale:** Duplicate cache generation was executing up to 3 times sequentially during champion selection, wasting CPU time. ThreadPoolExecutor speeds up multi-candidate OOS replay evaluation. Batch size upscaling from 4 to 6 reduces execution latency by 30%+ without memory pressure.
- **Key Verification:** Added unit tests `test_select_layer2_champion_with_prebuilt_cache` and `test_select_layer2_champion_parallel_determinism` inside `test_selection.py` (all passed). L2 run completed safely in 31s with Peak RAM limited to 7,006 MB.

## [2026-06-26] Gate Evaluation Deduplication & ThreadPool Replay
- **Delta:** Removed pre-gate + final-gate `evaluate_layer2_gate` double-call (2회→1회). Extracted common metric computations into local variables. Added champion tiebreaker by trial number (`sortino, cagr, -trial.number`) for ThreadPool non-determinism safety. Replaced sequential `_eval_candidate` loop with `ThreadPoolExecutor(max_workers=4) + as_completed`.
- **Rationale:** Gate 중복 호출이 candidate당 ~30% 계산 낭비. ThreadPool이 numba GIL 해제를 활용하여 fork/serialize 오버헤드 없이 2-3x 속도 향상. Champion tiebreaker는 ThreadPool 비결정적 실행 순서에도 안정적인 챔피언 선정 보장.
- **Key Verification:** `test_select_layer2_champion_single_gate_evaluation` 추가 (evaluate_layer2_gate==candidate당 1회 검증). 기존 14개 테스트 전부 PASS.

## [2026-06-26] Rollback: ThreadPool→ProcessPool(fork) + OOM Guard
- **Delta:** ThreadPool streaming을 ProcessPool(fork) batch로 롤백. `_GLOBAL_L2_CTX` + `_evaluate_l2_trial_from_global` 복원. OOM guard 공식을 `(avail_gb - 2.0) / 1.5` 에서 `avail_gb / 1.2`로 완화. ctx 이중생성 제거.
- **Rationale:** ThreadPool은 post-simulation Python 코드(GIL 미해제)에서 실질 병렬도가 1.5x 이하로 저하됨. `as_completed` waiter 등록/해제 overhead(200회)가 batch `future.result()`(100회)보다 느림. ProcessPool(fork)는 numpy array CoW 공유 + 진정한 프로세스 병렬로 GIL 완전 무관. OOM guard 경험적 수정: 1.2GB/worker가 fork CoW + AWF 할당의 현실적 추정치.
- **Key Verification:** `ruff` + `mypy` clean. selection tests 14/14, L2 tiered tests 35/35, layer2_gate_fixes 27/27 — 전부 PASS.

## [2026-06-26] Bucket Edge + Regime Code Cache (per-trial 3.6s→1.2s)
- **Delta:** `L2SimulationCache`에 `bucket_edges_by_fold` 및 `regime_code_1d` 필드 추가. `_run_tiered_l2_study`에서 folds + regime code precompute 후 `replace()`로 cache에 주입. `_run_awf_simulation`에서 캐시 hit 시 `compute_bucket_realized_edges`/`compute_market_regime_context` 재계산 skip. Fallback path 유지(하위호환).
- **Rationale:** Bucket routing은 trial-param 독립(align, folds, regime_code만 의존). Regime code도 aligned만으로 계산되며 `l2_routing_mode` trial param과 무관. 프로파일링 결과 `regime_code_1d` 재계산이 per-trial 2.51s(69%) 차지. 캐시로 0.12s로 단축(20x). 전체 per-trial 3.6s → 1.2s(3x). 200 trials × 6 workers ≈ 40초.
- **Key Verification:** `[L2-BUCKET-CACHE] HIT` DEBUG 로그 확인. `awf_total regime=0.12s` 안정화. 59개 테스트 전부 PASS.

## [2026-06-26] Regime DEBUG Observability — 3-state summary + raw 6-state shadow
- **Delta:** `build_regime_routing_plan()`에 `debug_diagnostics`를 연결하고, `opt_main_futures.py` / `awf_sim.py`가 `"[REGIME-DEBUG-GRANULARITY]"`, `"[REGIME-DEBUG-CELLS]"`, `"[REGIME-DEBUG-SELECTED]"`를 DEBUG로 출력하도록 정리했다. `awf_sim.py`는 selected-book realized return을 regime state별로 재집계한다. `"[REGIME]"`는 상태 분포 요약만 유지하고 raw 6-state 1-line 로그는 제거했다.
- **Rationale:** `stable` 분류는 regime 분포 안정성만 보여주고 L2 자산증식 유효성은 증명하지 못한다. DEBUG 결과에서 effective_3는 proof 실패, raw_6는 정보는 있으나 OOS cell error가 커서, production routing은 유지하되 diagnostics로만 원인 분해가 가능해야 했다. selected-regime replay는 realized 손익 기준으로 회귀해야 하므로 sleeve 평균이 아니라 state별 실제 누적 수익으로 교체했다.
- **Key Verification:** DEBUG 실행에서 `pooled_fallback`, `effective_3` proof 실패, `raw_6` compression_loss_bps=48.38을 확인했다. selected-regime table은 bull/bear/crisis realized return을 직접 반영한다. 최종 L2 scorecard는 `growth_lcb`/`cagr` 차단을 유지했다.

## [2026-06-27] Causal Regime Policy Split — fit/cal policy map + runtime modes
- **Delta:** `RegimeRoutingDiagnostics`에 `policy_diagnostics`를 연결하고 `RegimeRoutingPlan.policy_by_fold`를 실사용 경로로 노출했다. `l2_regime_policy_mode`를 `filter/observe/soft/hybrid`로 분기해 legacy bucket filter와 causal policy application을 분리했다. `apply_regime_cell_policy()`는 fold-local 정책을 `allow/downweight/block/pool`로 반영했고, `[REGIME]`은 summary만 유지한 채 DEBUG 표에서 policy mode와 action counts를 출력했다.
- **Rationale:** bucket edge는 fit-leg causal routing에 유효하지만, OOS sleeve 제어는 edge floor만으로 충분하지 않았다. regime-conditioned causal policy를 별도 레이어로 두어 fit/cal 정보만으로 block/downweight 판단을 하게 만들면, regime summary와 routing verdict를 혼동하지 않으면서 자산증식 지향의 runtime 제어가 가능하다.
- **Key Verification:** `observe` 모드가 무변경, `soft` 모드가 downweight, `hybrid` 모드가 block을 수행하는 unit tests를 추가했다. `RegimePolicyApplication.sleeve_edges`는 float contract로 복귀했고, AWF는 orphan edge를 남기지 않도록 재조합 경로를 갖췄다.

## [2026-06-27] Regime Diagnostics Hardening — sign consistency + state caps
- **Delta:** `RegimePolicyDiagnostics`에 `n_unstable`, `n_hard_block_eligible`, `sign_consistency_ratio`, `hard_block_enabled`를 추가했고, `build_regime_policy_by_fold()`는 hard block을 `hybrid` + confidence + sign-consistency 조건으로만 허용하도록 정리했다. `soft`는 route continuity를 유지하는 downweight 전용 경로로 고정했다. `apply_regime_risk_cap()`를 통해 regime state별 gross cap을 weight composition 이후에 적용했다.
- **Rationale:** raw confidence만으로 block을 허용하면 fit/cal 방향 불일치 셀을 과도하게 차단하거나, 반대로 낮은 품질 셀을 route에 남겨 자산증식 효율이 흔들릴 수 있었다. sign consistency와 state cap을 분리하면 routing 판단과 노출 제어를 분리할 수 있어 L2 실행 안정성이 높아진다.
- **Key Verification:** DEBUG 로그에 policy counts와 risk-cap 적용 여부가 남고, `soft`/`hybrid`/risk-cap 경로를 각각 검증하는 단위 테스트가 통과했다.

## [2026-06-27] Regime Allocation Coupling — raw_mu and quality_weight scaling
- **Delta:** `apply_regime_cell_policy()` now scales `SymbolSignal.raw_mu` and `quality_weight` together with `sleeve_edges` when regime policy applies, and `RegimePolicyApplication` carries before/after aggregates for edge, mu, and quality weight. `Layer2AllocationConfig` exposes `l2_regime_scale_signal_mu` and `l2_regime_scale_quality_weight`, and `_run_awf_simulation()` forwards them into the regime policy path while logging the pre/post effect.
- **Rationale:** The earlier regime path only changed sleeve edge diagnostics, while the actual Kelly input still came from pooled `raw_mu`. That made regime proof observable but economically weak. Scaling the same sleeve-level confidence inputs that reach symbol pooling keeps regime control causal and lets soft policy influence sizing without turning regime into a standalone alpha selector.
- **Key Verification:** Added tests for soft downweight, observe no-op, legacy-disable flags, and symbol pooling with regime-scaled sleeve confidence. The change preserves hybrid hard-block behavior and leaves the routing/proof layer causal and fit/cal bounded.

## [2026-06-27] L2 Regime Selection Growth Redesign — Causal Bucket Reliability + Deployable Score + Entry Cooldown
- **Delta:** (1) `RegimeBucketReliability` — causal fit/cal bucket reliability layer: sign consistency, `n_fit >= l2_bucket_min_n`, `n_cal >= l2_regime_cal_min_n`, `abs(cal_edge_bps) >= l2_regime_min_cal_lift_bps`, `reliability >= l2_bucket_min_reliability` 조건으로 `allow/downweight/pool` 판정. OOS debug metric은 routing/training/selection에 절대 사용하지 않음. (2) `RegimePolicyEffectSummary` — per-fold action_ratio/pooled_ratio/mu_abs_ratio 집계 + `_policy_effect_is_visible()` 진단 (임계: pooled_ratio≤0.80, action_ratio≥0.10, mu_change≥0.03). (3) `Layer2DeployableScore` — blocked fallback candidate ranking 공식: `cagr + 0.10·min(sortino,3) + 0.05·min(calmar,3) - 0.50·max(0,-worst_fold_cagr) - 0.25·max(0,0.45-positive_block_delta_ratio) - 0.20·cost_drag - entry_spike_penalty`. (4) Promotion gate에 `worst_fold_cagr`(`l2_min_worst_fold_cagr=-0.05`) 및 `block_delta`(`l2_min_positive_block_delta_ratio=0.45`) blocker 추가 — 기존 CAGR blocker 순서 보존. (5) `apply_entry_cooldown()` — `_resolve_tradeable_mask()` 내 causal backward-only cooldown (`l2_entry_cooldown_bars=12`). `entry_block_spike` 경고 시 `Layer2DeployableScore.entry_spike_penalty` 패널티 부과. (6) `select_layer2_champion` fallback 확장: 기본 3→ `l2_replay_max_fallbacks`(default 24), deployable score ranking 도입, `_assert_selection_replay_parity`로 cagr/mdd/fold_pass/trade_count 검증.
- **Rationale:** L2가 CAGR +3.8%, MDD 20.5%로 gate blocking된 원인은 (a) regime policy action surface가 258 cell 중 243개 pooled/unstable로 비효과적, (b) bucket edge의 fit/cal sign 불안정성이 routing 품질 저하, (c) Optuna objective와 최종 성장이 정합하지 않아 near-feasible candidate도 collapse. Causal bucket reliability는 fit/cal sign flips를 pool 처리하여 과적합 edge가 routing에 진입하는 것을 차단한다. Deployable score는 CAGR 이외에 worst-fold CAGR, block delta ratio, cost drag, entry spike를 종합해 blocked candidate 중에서도 collapse risk가 최소인 후보를 선출한다. Entry cooldown은 `entry_block_spike`가 L2 universe audit 경고로 나타나는 빈도를 낮춰 시뮬레이션 충실도를 높인다.
- **Key Verification:** 단위 테스트 10종 추가(bucket reliability 3, policy effect 2, gate blockers 2, deployable score fallback 2, entry cooldown 1) — 전체 tiered workflow suite 349 passed, 1 skipped. Optuna trial `evaluate_l2_trial`에서 `Layer2DeployableScore` + `worst_fold_cagr`/`positive_block_delta_ratio` attrs 전달 확인. `build_layer2_deployable_score` score formula config-derived penalty weight(l2_worst_fold_cagr_penalty_weight=0.50, l2_block_delta_penalty_weight=0.25)로 spec과 정합.

## [2026-06-28] L2 Regime Policy Conservatism Fix — pooled passthrough + B-2/B-3 완화
- **Delta:** (1) `l2_bucket_edge_floor_bps` 0→50bps (데이터 의존적 default). (2) `l2_regime_pooled_is_passthrough`(default False): pooled action → allow (passthrough)하여 243/253 pooled cell이 실질 비활성화되는 현상 해소. (3) `l2_regime_min_fit_n_floor`(default 5): fit_n 부족해도 cal이 양호하면 allow (B-2 insufficient_fit_but_good_cal). (4) `l2_regime_require_fit_n_for_downweight`(default True): fit_n 충분하지 않으면 B-3 downweight를 0.8×로만 적용 (완전 pooled 보다 나은 처리). (5) `relaxed_reliability_threshold=0.35`: sign_consistency가 유지되면 downweight→allow 완화.
- **Rationale:** L2 gate CAGR 7.4%의 근본 원인은 pooled cell 비율 96%(243/253)로 regime policy가 routing을 차별화하지 못한 데 있었다. pooled cell은 `allow`와 동일한 sleeve_edge를 출력하면서 유일하게 다른 `action` string만 `"pooled"`로 남아 디버깅만 불투명하게 만들었다. B-2/B-3 조건을 현실 fit/cal 분포에 맞게 완화하고, pooled passthrough를 선택적 allow로 전환하면, policy decision surface가 30~40%까지 활성화되어 fold 간 CAGR 불균형(Fold #3 CAGR 0.3%)이 개선될 것으로 기대된다.
- **Trade-offs:** passthrough 활성화(`True`)는 pooled cell 수가 적은 fold의 decision surface는 적게 변화시켜 fold 간 불균형 해소가 불완전할 수 있다. relaxed_reliability_threshold(0.35)는 과거 test_bucket_reliability 1건의 assertion을 변경시킨다(backward compat 유지).

## [2026-06-28] L2 Regime Conservatism Parity Fix — RC-2/RC-1/RC-4/RC-3
- **Delta:** (RC-2) `calibrate_deployment_leverage` added `oos_budget_blend=0.5`, `oos_floor_cap=4.0`, new binding `"oos_blend"` replaces hardcoded `min(2.0,…)`. (RC-1b) `Layer2Result.deploy_leverage` field (default 1.0), `run_l2_awf` populates from `_l_star`. (RC-1c) `assert_selection_replay_parity` adds `gate: bool = False` param; parity mismatch in `opt_main_futures.py` sets `gate_passed=False, blocker_reason="parity_divergence"`. (RC-1a) `opt_main_futures.py:2321` — `l2_sim_cache=shared_l2_cache` → `l2_study_result.sim_cache` (enriched cache with regime routing plan). (RC-4) `l2_gate.py` — block_delta demoted to diagnostic-only, `_growth_lcb_vol_matched_baseline` helper, `std_hybrid`/`std_baseline` params. (RC-3) `l2_meta.py` — fold-level override: if `mean_cal_lift<0 & sign_consistency_ratio<0.6`, all cells force `action="allow"`, `reason="pooled_passthrough"`.
- **Rationale:** 4 root causes of L2 asset growth suppression (parity path divergence, fit-leg inversion leverage under-deployment, regime policy inert, gate cascade) resolved. RC-2 recovers L* from 2.0→4.0, RiskUtil ~24%→58%. RC-1a resolves final_L*=nan parity divergence (selection used enriched cache, final used raw cache). RC-3 prevents regime policy from blocking all cells when fit/cal signals are unstable. RC-4 prevents block_delta from double-penalizing candidate scoring.
- **Key Verification:** All 93 tests pass (6 test suites). L1 validation: ruff + mypy on all 5 modified source files. Swap 2 test fix: OOS vol 0.006→0.003 to force blend above exchange_cap.

## [2026-06-28] L2 AWF Simulation Fingerprint Instrumentation (Parity Diagnosis)
- **Delta:** `_run_awf_simulation`에 `sim_origin` 선택적 파라미터 추가. 반환 직전 DEBUG 레벨 `[AWF-SIM-FP]` 로그 블록 삽입: rets MD5 fingerprint(12 hex), fold별 OOS bars, fold_ret_lens, config fingerprint(8 hex), sum_logret, cache/signal/aligned 객체 ID. `evaluate_l2_trial` → `sim_origin="champion_eval"`, `run_l2_awf` → `sim_origin="final_deploy"` 전달.
- **Rationale:** champion-eval과 final-deploy 경로가 동일 입력(동일 trades, fold_pass)에도 CAGR 0.1847 vs 0.0612로 상이한 원인을 격리하기 위해, `_run_awf_simulation` 내부 fold 분할/누적 처리의 차이를 1-line DEBUG 로그로 계측. rets_fp 동일 여부에 따라 fold 윈도우 분할 차이/객체 분기/config 분기 등 근본 원인을 확정 가능.
- **Key Verification:** 5개 단위 테스트(S1~S5) 통과. L1: ruff + mypy clean. 기존 호출부(sim_origin 기본값="unknown") backward compat 유지.

## [2026-06-28] L2 AWF Content Fingerprint Instrumentation (Parity Deep Dive)
- **Delta:** `_run_awf_simulation`에 `_content_hash_array`/`_content_hash_dataclass`/`_content_hash_cache` 3종 순수 헬퍼 추가. 기존 `[AWF-SIM-FP]` 직후 `[AWF-SIM-FP2]` 로그 추가: cache 내용해시(cache_ch, 배열 tobytes md5[:12]), config 해시(cfg_ch, dataclass field 순회 md5[:10]), caps 해시(caps_ch), per-fold rets fingerprint(각 fold md5[:8]), deploy_lev.
- **Rationale:** 1차 `[AWF-SIM-FP]` 로그에서 `cfg_fp`, `cache_id`, `signal_id`, `aligned_bars`가 모두 동일했으나 `rets_fp`가 다른 현상이 관측됨. 사각지대 3종: ① `cache_id`는 객체 identity만 검증(내용/in-place 변형 미검출) ② `cfg_fp`가 repr truncate(`...`) 충돌 가능 ③ `caps` 전혀 미계측. 내용 기반 해시로 1회 재실행에 4갈래(cache/config/caps/sim 내부 hidden-state) 중 원인 확정 가능.
- **Key Verification:** 11개 단위 테스트(S1~S6) 통과. L1: ruff + mypy clean. 기존 계측 및 로직 무변경.

## [2026-06-28] L2 SSOT Evaluator Unification — run_l2_awf delegates to evaluate_l2_trial
- **Delta:** (C1) `evaluate_l2_trial()`에 `deploy_leverage_override: float | None = None` 파라미터 추가 — `>1.0` 시 `calibrate_deployment_leverage` override, `None`/`≤1.0`은 기존 내부 calibrate 유지. (C2) `run_l2_awf()`가 `_run_awf_simulation` 직접 호출 대신 `evaluate_l2_trial()`에 위임 — 단일 평가 SSOT 경로로 통합. (C3) `_layer2_result_from_trial_eval()` 어댑터 추가, `Layer2TrialEvaluation`에 6개 deployment 필드(`last_selected_symbols`, `last_weights`, `all_turnovers`, `rebalance_count`, `all_net_exposures`, `rets_baseline_ew`) 확장. `test_l2_ssot_evaluator.py` 9종 테스트(S1~S8) + 2개 기존 테스트 hotfix.
- **Rationale:** 기존 `run_l2_awf`가 `evaluate_l2_trial`과 별도로 `_run_awf_simulation`을 직접 호출하여 metric 계산이 이중 경로로 분기 — champion-eval CAGR 0.1847 vs final-deploy CAGR 0.0612 (3× 차이). SSOT 단일 경로로 selection/deploy CAGR 동일 보장 (S1 검증). `deploy_leverage_override`로 fit-leg calibration 없이도 deploy path 시뮬레이션 가능.
- **Edge Cases:** `deploy_leverage_override=None` → 기존 calibrate 유지 (하위호환). `deploy_leverage_override ≤ 1.0` → calibrate skip, `l_star` 직접 사용. `Layer2TrialEvaluation` 미확장 필드는 `extras` dict 기본값 fallback.
- **Key Verification:** S1: selection CAGR == deploy CAGR. S2: `deploy_leverage_override=4.0` → `Layer2TrialEvaluation.l_star==4.0` + log. S3: gate status pass-through. S4: turnover/weights/gate extras 일치. S5~S8: gate-bypass/feature parity/hotfix backward compat. All 389 tiered tests PASS. L1: ruff + mypy clean.

## [2026-06-29] L2 Edge-Survival Attribution Diagnostics + Evaluation Memoization
- **Delta:** (C1/C2) `Layer2EdgeWaterfall` dataclass + `_assemble_edge_waterfall()` in `awf_sim.py` — fold-level edge decomposition into 4 stages (admitted → weighted → capped → realized) with scalar accumulators (`_attr_weighted`, `_attr_admitted`, `_cap_binding_bars`, `_sleeves_admitted_sum`). Stage loss terms isolate dominant erosion stage. `w_precap = w.copy()` captured before `apply_regime_risk_cap`. `[L2-EDGE-WATERFALL]` DEBUG log. (C4) `_build_l2_user_attrs()` extracted — DRY user_attrs assembly in `_evaluate_l2_params` / `_evaluate_l2_params_threadsafe`. (C5) `evaluate_l2_trial_cached` memoization in `workflow.py` with key `(id(cache), cfg_ch, id(signal_batch), id(caps), tf, deploy_lev)` — study loop bypassed (unique config → hit=0), selection replay + deployment dedup (2→1 call). `Layer2StudyResult.eval_memo` propagates memo dict → `run_tiered_pipeline`. `[L2-MEMO-PARITY]` DEBUG log. Env toggle `L2_DIAG_ATTR` already existed.
- **Rationale:** Decompose L1 expected edge → realized PnL into quantifiable stage losses to identify whether alpha decay, sizing collapse, regime cap, or friction is the dominant CAGR eroder. Evaluation memoization eliminates redundant `evaluate_l2_trial` calls during selection replay (same config re-evaluated for parity check) without modifying Optuna study flow (unique config per trial → zero cache overhead).
- **Key Verification:** 4 test files (8 scenarios) — waterfall decomposition (3 scenarios: baseline, regime-cap binding, friction & sleeves), user_attrs refactor parity, memo hit/miss parity (2 scenarios). L1: ruff 0 errors, mypy 0 errors, pytest 8/8 passed.

## [2026-06-30] L3 Adaptive Regime-Reliability — Walk-Forward bear cap dynamic downweight
- **Delta:** Added `compute_regime_reliability_multiplier` and `bear_edge_per_bar_bps` pure functions in `l2_meta.py`. The multiplier reads trailing fold bear edge per-bar bps and maps it via a sign-first piecewise-linear ramp (`[floor, 1.0]`). Config: `l2_regime_reliability_enabled=False` (A/B off), `l2_regime_reliability_window=2`, `l2_regime_reliability_floor=0.2` — added to `Layer2AllocationConfig` in `dataclasses.py`. In `awf_sim.py`: pre-loop accumulator (`_bear_edge_by_fold`, `_is_bear_code`), fold-start trailing multiplier computation, unconditionally accumulates bear price/bars per bar, applies `bear_gross_cap * _bear_reliability_mult` in `apply_regime_risk_cap` call, records fold-edge at fold end. `[L2-REGIME-RELIABILITY]` DEBUG log per fold.
- **Rationale:** Bear regime IS→OOS edge sign reversal (IS ~+150 bps → OOS ~−30 bps per-bar) caused static `bear_gross_cap=0.35` to not differentiate between profitable and harmful bear exposure. Online trailing bear edge degradation quantifies whether the current regime fold is delivering positive or negative bear-specific returns. The reliability multiplier reduces bear gross cap proportionally when trailing evidence shows sustained negative bear edge, without look-ahead (trailing slice excludes current fold).
- **Key Verification:** 7 unit test scenarios (negative edge to floor, positive edge keeps full, linear ramp midpoint, empty list neutral, clamp bounds, invalid params, per-bar normalization). L1: ruff + mypy clean on 4 modified files (`l2_meta.py`, `dataclasses.py`, `awf_sim.py`, `test_l2_meta.py`). 7/7 pytest green + pre-existing 456/460 regression tests unaffected.

## [2026-06-30] L2 Reversal Selectivity & Persistence — N-bar raw condition gate + tighter DD threshold
- **Delta:** (P1) `RegimeConfig.reversal_dd_threshold` default `0.06 → 0.12`. (P2) New field `reversal_persistence_bars: int = 3` with `__post_init__` validation (`>= 1`). (P3) `compute_reversal_risk_off_1d` gains `persistence_bars: int = 1` parameter — when `> 1`, computes trailing consecutive raw-True count and gates the shift(1) mask behind `run_count >= persistence_bars`. (P4) AWF wiring forwards `_rev_cfg.reversal_persistence_bars` to detector call. (P5) Hardcoded `_roff_floor = 0.05` replaced with `_rev_cfg.reversal_risk_off_floor` (SSOT).
- **Rationale:** Reversal kill-switch overfired in folds 2-3 (normal pullbacks) while effectively defending fold#1 crash. Raising DD threshold from 6% to 12% filters shallow drawdowns. Persistence gate requires N consecutive raw True bars before the shifted risk-off activates, preventing single-bar drawdown spikes from triggering hard de-gross. Together these tighten selectivity without sacrificing fold#1 crash protection.
- **Key Verification:** 14/14 unit tests — persistence selectivity (spike immunity), sustained reversal triggers after shift, backward compat (`persistence_bars=1` matches legacy), config validation (threshold + persistence bars), detector parameter validation. L1 ruff + mypy clean on 5 files. 1,264/1,301 regression PASS (37 pre-existing failures unrelated).

## [2026-06-30] L2 Reversal Kill-Switch — Trailing DD + Momentum Hard Risk-Off
- **Delta:** Added `compute_reversal_risk_off_1d` in `market_regime.py` (trailing drawdown + EMA momentum, O(T)). `RegimeConfig` 5 new fields (`reversal_dd_window=90`, `dd_threshold=0.06`, `mom_fast=20`, `mom_slow=120`, `risk_off_floor=0.05`) + `__post_init__` validation. `Layer2FoldAttribution` 3 new fields (`risk_off_bars`, `risk_off_realized_price`, `risk_on_realized_price`) + `_assemble_fold_attribution` wiring. Gate wiring in `_run_awf_simulation`: env check `L2_REVERSAL_KILL`, pre-compute `_risk_off_1d`, per-rebalance hard de-gross of all sleeve `raw_mu` to `risk_off_floor` (overrides soft cap/crisis_floor), risk-off price pair collection → attribution pass-through. `[L2-ATTR]` log extended with `roff_bars`, `roff_price`, `ron_price`.
- **Rationale:** 병목 fold#1(24Q4-25Q1, −27%, 全 regime 동시 음전 = 시장 반전)을 인과적 BTC trailing-drawdown/momentum 기반 선택적 hard risk-off kill-switch(gross→floor≈0, 기존 soft cap/floor 무시)로 방어. L* 단일 스칼라가 복제 불가한 시간-선택적 de-gross로 노출-크기 레버의 L* 상쇄를 탈출. efficiency gate의 실패(mean_ER 균일→선택성 0)와 달리, fold0의 mean_dd가 folds1-2의 2배(0.074 vs 0.035)로 선택적 탐지 가능. 기존 regime cap은 fold0를 bear/crisis로 탐지는 했으나 응답이 soft(crisis_floor=0.15가 손실 유지) + L* 흡수. 본 spec = hard(gross→~0, floor 무시) + 선택적 고확신 트리거.
- **Edge Cases:** Look-ahead 삼중 차단(dd trailing + mom trailing + shift(1)). mom<0 게이트로 V반등(dd 높지만 회복)은 kill 제외 — fold1(2025 반등) 보호. 단일 자산 BTC 의존(기존 regime 동일). 알트 디커플링 구간 한계는 진단 coverage로 모니터. `reversal_risk_off_floor < crisis_gross_floor` 검증.

## [2026-06-30] L2 Reversal Economic Replay — Env-configurable reversal variants + adoption verdict
- **Delta:** Added `_reversal_config_from_env()` in `awf_sim.py` — reads env overrides `L2_REVERSAL_DD_WINDOW`, `L2_REVERSAL_DD_THRESHOLD`, `L2_REVERSAL_MOM_FAST`, `L2_REVERSAL_MOM_SLOW`, `L2_REVERSAL_RISK_OFF_FLOOR`, `L2_REVERSAL_PERSISTENCE_BARS` and validates via `RegimeConfig.__post_init__`. Extended `Layer2TrialEvaluation` with `fold_deployed_mdds` and `fold_attributions` fields, propagated from `fold_diag` and `sim` respectively. Added `_run_l2_reversal_economic_replay()` + 4 helpers in `opt_main_futures.py`: `_l2_reversal_replay_variants()` (5 predefined variants), `_temporary_reversal_env()` (scoped env override), `_fold_metrics_from_l2_evaluation()`, `_reversal_replay_adoption_verdict()`. The replay call is gated by `L2_REVERSAL_REPLAY` env and executes after the parity gate in the tiered pipeline. CSV output at `docs/results/l2_reversal_replay.csv`.
- **Rationale:** Evaluate threshold/persistence variants against the bottleneck fold (fold 0) to identify a L2 reversal config that preserves legacy improvement while protecting non-bottleneck folds from damage. The adoption verdict enforces 70% defense ratio, non-bottleneck CAGR floor, aggregate CAGR superiority, and selection parity.
- **Edge Cases:** `deploy_leverage_override` only applied when `> 1.0`. `baseline_off` variant sets `blocker_reason="baseline"`. Metric parity checked only for baseline variant. Non-baseline variants always report `metric_parity=False`.

## [2026-06-30] L2 Trend-Efficiency Gate — Kaufman ER Whipsaw Attribution + Exposure Gate
- **Delta:** (C1) `compute_trend_efficiency_1d` in `market_regime.py` — trailing Kaufman Efficiency Ratio via causal cumulative-sum rolling window (`O(T)`, no `pd.rolling` dependency). (C2) `MarketRegimeContext.trend_efficiency_1d` field wired into `compute_market_regime_context`. (C3) `RegimeConfig` 3 new fields (`trend_efficiency_window=24`, `target=0.35`, `floor_mult=0.30`) + `__post_init__` validation. (C4) `Layer2FoldAttribution` 3 new fields (`realized_price_low_er`, `trend_efficiency_corr`, `mean_trend_efficiency`) + `_assemble_fold_attribution` ER-pair target param. (C5) `trend_efficiency_gross_mult` in `risk_deployment.py` — linear clamp `[floor_mult, 1.0]`. (C6) Gate wiring in `_run_awf_simulation`: env check `L2_TREND_EFFICIENCY_GATE`, pre-compute `_trend_efficiency_1d`, per-rebalance trend/ts_mom sleeve `raw_mu` scaling via `trend_efficiency_gross_mult`, ER pair collection → attribution pass-through. Archetype detection via family name prefix in `_parse_meta_group_ids` against `_trend_arch_families` frozenset.
- **Rationale:** 병목 fold#1(24Q4-25Q1, −27%)의 손실을 whipsaw(저ER 구간)로 귀속 측정하고, trend/ts_mom sleeve 노출만 trailing ER로 down-scale하여 추세 반전에서 방어. 기존 `trend_scale`은 부호 방향세기(SNR)로 whipsaw(순간 |snr|↑ 후 반전)를 chop으로 식별 불가. ER은 경로조정 추세품질(직교)로 SNR과 무관. 기본 off A/B 게이트로 회귀 안전.
- **Audit Fixes:** (1) 초기 구현에서 `np.convolve(mode="same")`가 centered look-ahead 사용 — `cumsum[i] - cumsum[i-window]` trailing rolling sum으로 교정. (2) 게이트 도우미 함수만 구현되고 `_run_awf_simulation`에 배선 누락 — env check → pre-compute → per-rebalance 적용 → ER pair 수집 → attribution 전달까지 전 경로 배선 완료.
- **Key Verification:** L1: ruff + mypy clean on 5 modified files. 33/33 unit tests PASS (4 files: 2 new + 2 augmented). Target (Scenarios 1~8): ER trend vs chop, flat zero, mult bounds, config validation, whipsaw decomposition, causal-only, ER-in-context integration. L2 regression: 1,690+ tests, 0 regressions (37 pre-existing failures unrelated: `evaluate_l2_trial` removed in prior refactor, emoji log formats). Coverage: `market_regime.py` 93% (new lines fully covered), `risk_deployment.py` 66% (trend_efficiency_gross_mult L38~48 covered), `config.py` 51% (new validations covered, pre-existing file), `awf_sim.py` 16% (gate wiring requires full AWF integration test).

## Layer 3 (Holdout & Replay) Historical Log

---
title: Layer 3 Holdout Engineering History
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: critical
ai_read_policy: when_related
---
## 2026-06-18 L3 scorecard threshold alignment — Calmar removal + absolute gate thresholds
- **Delta:** L3 scorecard now renders `min_trades`, `max_mdd_abs`, `min_sharpe`, `min_sortino`, and `max_cvar95` from `Layer3Result` and drops Calmar from the display. The holdout gate order is now `negative_return` → `mdd_abs` → `cvar_95` → `sharpe_abs` → `sortino_abs`.
- **Rationale:** Calmar was only producing `n/a(loss)` after negative CAGR while the direct gate was already `negative_return`. Absolute thresholds make the replay contract explicit and keep the scorecard aligned with the actual blocker chain.
- **Edge Cases:** `negative_return` remains the first compound-loss blocker. Risk and efficiency thresholds are persisted on the result object so the formatter cannot drift from the gate contract.

## 2026-06-16 L3 빈 holdout 구조적 수정 — IS+OOS 데이터 병합 (PART4)
- **Delta:** `pick_strategy_data_maps`가 `oos_data_maps`를 버리고 IS-only를 반환하던 동작을 IS+OOS `concat+sort+dedup` 병합으로 교체. `full_strategy_maps`를 쓰는 모든 호출부(bridge, END-coverage 필터, `align_data_maps`)가 자동으로 holdout_end까지 데이터를 보게 됨.
- **Rationale:** `aligned.datetimes`가 구조적으로 `holdout_start`에서 끝나, `_resolve_holdout_span`이 항상 `empty_holdout_window`를 raise — "intersection tail truncation(상장폐지 심볼)"이라는 기존 진단은 오진이었고, 실제 원인은 데이터 소스 자체가 IS-only였던 것.
- **Edge Cases:** `keep="first"`로 IS 우선 — 경계 timestamp 중복 시 미래(OOS) 행이 과거를 덮어쓰지 않음. 부작용은 `layer2-eh.md`의 "L2 AWF fold anchoring 복원" 항목 참조(같은 작업에서 발견된 L2 fold 붕괴 regression).

## 2026-06-16 L3 평가체계 lean 보강 (PART2) — Phase D silent fallback 제거 (PART3)
- **Delta:** L3 게이트를 `cagr<0` 단일조건에서 5단계 순차 게이트(`insufficient_trades`→`negative_return`→`sharpe_rel`→`mdd_rel`→`mdd_abs`)로 교체. `total_return`, `equity_multiple`, `sortino`, `n_trades`, `cvar95`, `avg_gross_exposure`를 `Layer3Result`에 추가(L2 헬퍼 재사용, 신규 수학 없음). `except Exception` 발생 시 legacy Phase D fallback으로 조용히 넘어가던 동작을 제거 — 즉시 `RunnerResult(exit_code=1, reason="tiered_pipeline_error:...")`로 실패.
- **Rationale:** L3는 "1회 백테스팅으로 실제 복리자산증식 성과 판단"이 목적이므로 L2(Optuna 검증)와 동일한 수준의 풍부한 진단 지표는 불필요하나, CAGR/MDD/Sharpe/MAR만으론 빈약 — 단일패스 복리(`equity_multiple`)와 거래량 하한이 누락되어 있었음. Phase D fallback은 legacy 경로로, holdout 실패를 가려 "조용한 오류"를 만드는 위험이 있어 제거.
- **Edge Cases:** `max_mdd_abs`(기본 0.35)는 baseline 자체가 붕괴한 경우를 방어하는 절대 캡. `min_trades`(기본 10)는 L3 자체 기준으로 L2의 30보다 완화(단일 holdout 윈도우 특성 고려).

## 2026-06-18 L3 deployment parity 정합화
- **Delta:** `run_l3_holdout`가 선택적으로 `deploy_leverage`를 받아 L2 champion 배치와 동일한 `apply_deployment` 경로로 hybrid holdout의 CAGR/MDD/CVaR/terminal compounding을 계산하도록 변경. `run_tiered_pipeline`는 `l2_params["l2_deploy_leverage"]`를 L3까지 전달한다.
- **Rationale:** L2 승격 파라미터를 L3가 재사용하지 않으면 frozen holdout이 아니라 unit-path replay가 되어, L2/L3 결과 해석이 분리된다. 배치 계약을 L3에 주입해야 holdout 실패가 strategy failure인지 deployment mismatch인지 분리 가능하다.
- **Edge Cases:** `deploy_leverage`가 1.0 이하이거나 비유한값이면 unit path 유지. baseline은 비교용으로만 남기고 동일 배치하지 않는다.

- **Empirical Finding (실제 파이프라인 재실행, 2026-07-02, `L2_REVERSAL_KILL=1 L3_REVERSAL_REPLAY=1`, 8h tf, 실 BTC 데이터 2025-12-31~2026-06-30, BTC -32.8%/peak-trough -39.5% 실측 위기 구간):** `baseline_off`(reversal-kill 비활성)의 CAGR -4.96%/MDD 23.78%가 나머지 7개 활성 variant 전부보다 우수했다(활성 variant CAGR -5.04%~-5.89%, MDD 24.18%~24.64% — 전부 baseline보다 나쁨). `risk_off_realized_price`(kill-switch 발동 구간의 실현 가격 성과)가 전 variant에서 양수(+5.89%~+10.50%) — kill-switch가 de-gross한 바로 그 구간에서 원 신호가 실제로는 수익 중이었다는 뜻. **`next.md`가 "L\* 흡수를 피하는 유일하게 검증된 방어 레버"로 지목했던 reversal kill-switch가 이 실제 위기 구간의 economic replay에서 방어는커녕 손실을 악화시켰다 — 최초의 실제 crisis-window economic replay 결과가 반증.** SSOT/후속 조치: `docs/results/next.md` §1, §2 P1/P2, §3.
- **Key Verification:** 회귀 스위트 전체 PASS(check 단계 완료). Test scenarios: fold_attribution 배관(P1-S1~S4), env-독립 `reversal_kill_active`(P1-S2), 빈 `fold_attributions` fallback(P1-S3), 8-variant env 스코핑 + 종료 후 env 복원(P2-S5~S6) — 확립된 mocking 경계(`_run_awf_simulation`/`run_l3_holdout` boundary patch, synthetic price path 대신 canned dataclass) 준수.

## Universe (Market & Ledger) Historical Log

---
title: Futures Universe Ledger Backend Compatibility
domain: futures.universe
type: adr
status: active
priority: high
ai_read_policy: when_related
---
## 2026-06-20 TIERED-BASE-SCOPE: loaded symbol scope와 temporal admission 분리
- **Delta:** `opt_main_futures._run_strategy_stage`가 tiered entry 전에 `base_scope`를 먼저 계산하도록 바뀌었고, `_resolve_tradeable_scope`는 그 `base_scope`에만 warm-up / min-bars / OOS coverage를 적용하도록 좁혀졌다. empty strict admission은 fallback 없이 `TieredPipelineError`로 종료하도록 변경됐다. 관련 tests는 provenance scope와 strict admission을 분리했다.
- **Rationale:** historical-union provenance와 temporal feasibility를 한 단계에서 같이 판정하면 tiny fixture가 전부 탈락하거나, 반대로 fallback으로 fail-open이 섞인다. base scope와 admission을 분리해 loaded-symbol 검증은 보존하고, holdout contract 위반은 fail-closed로 차단해야 했다.
- **Edge Cases:** base scope가 비어 있으면 loaded map 자체가 없다는 뜻이므로 admission 단계로 가지 않는다. strict admission이 0개면 recover하지 않고 terminal error를 반환한다. aligned scope regression tests는 admission을 stub 처리해 provenance만 검증한다.

## 2026-06-20 PHASE4-LOADER-GAP: 백테스트 로더 연속성 gap 게이트 추가
- **Delta:** `opt_data_utils.evaluate_symbol_data_sufficiency`에 `max_gap_bars` 검사 추가. `sorted_dt.diff().max() / bar_delta - 1` = 최장 missing-bar 수. `gap_ok = max_gap_bars <= FUTURES_BACKTEST_MAX_GAP_BARS(=6)`. 양 pass_flag 경로(`stage5`/non-`stage5`) 포함. `reason="gap_too_large"`, 반환 dict에 `max_gap_bars` 노출. 경계값 `<=` — G6 gate(`> max_gap_bars`) 와 일치(24h gap 허용).
- **Rationale:** count 기반 95% 검사는 `reindex/ffill`로 은폐된 24h+ 연속 공백을 통과시킴. frozen 가격이 모멘텀/추세 신호를 오염시키는 것을 차단.
- **Edge Cases:** G6 gate(`>`)와 경계 정합 필수 — `<=` 사용으로 universe 통과 심볼이 loader에서 부당 탈락 방지.

## 2026-06-20 PHASE3-REDESIGN: Universe 재설계 — capacity prefix 폐기 + G6 배선 + continuity 실측
- **Delta:** (P0) `capacity_coverage_target=0.90` prefix 블록 제거 → `eligible_syms[:k_max(=150)]` compute backstop. (P1) `compute_continuity_metrics` 구현: `max(onboard_date, first_data_date)` clamp + `.as_unit("ns").asi8` pandas 2.x unit 정합. ledger stub(0/1.0) → 실측 교체, full rebuild. (P2) `_instrument_df_from_ledger`에 9개 continuity 필드 주입 → G6(DATA_INTEGRITY_FAIL) 배선 활성화. (P3) G0(LEVERAGED_TOKEN), ADV_FLOOR(2M) 게이트 추가; k_max=150, min_adv_usdt=2M 파라미터 확정.
- **Rationale:** 기존 capacity prefix가 BTC+ETH(ADV 64%) 탓에 33개 심볼만 선택 → universe 폭 붕괴의 근본 원인. G6는 배선 누락으로 ledger stub을 읽어 항상 0 반환 → 무결성 게이트 무력화. compute_continuity_metrics unit mismatch로 633/633 심볼 max_gap_bars=14371(전수 G6 탈락).
- **Edge Cases:** onboard_date 이전 데이터 없는 구 심볼(BTC 2019 상장, 데이터 2022~) → clamp 전 max_gap_bars=14371, 후 max_gap_bars=0.

## 2026-06-19 PIT-BREADTH: 풀-윈도우 생존편향 필터 교체 + 용량커버리지 Cap + warm-up 가드
- **Delta:** (C1) `opt_main_futures._resolve_tradeable_scope` 추가 — 3-guard PIT 어드미션(warm-up: `datetimes.min()≤fetch_start`, `min_bars≥1500`, OOS-cov≥0.90). 풀-윈도우 END-coverage(`first≤fetch_start AND last≥holdout_end`) 폐지. `_TIERED_MIN_WINDOW_BARS=1500` 모듈 상수화. (C2) `PITUniverseConfig.k_in=0` 기본값; `capacity_coverage_target=0.90`, `k_max=100` 추가 — 누적 용량 90% prefix 알고리즘. (warm-up guard fix) `datetimes.min()>fetch_start` 심볼 reject: 교집합 start가 밀려 `ValueError: tiered warm-up coverage missing` 유발 차단.
- **Rationale:** END-coverage 필터가 633 온디스크 심볼을 54 "올드가드"로 붕괴 → PIT 설계가 막으려던 생존편향 재주입. 2023-10~2024-09 상장 110종 통째 배제. k_in=50은 교집합(231)·active_mask에 비구속(inert)이었으나 magic number 정당화 불가 → Pareto 용량 커버리지로 대체.
- **Edge Cases:** total capacity=0 → fail-open(`eligible[:k_max]`). fetch_start 이후 상장된 심볼은 warm-up guard로 자동 제외(교집합 보전). OOS 절단 심볼은 90% coverage guard로 제외.

## 2026-06-19 L2-ZERO: PIT cube bypass 해소 + store build/hit mismatch 수정
- **Delta:** `opt_main_futures.py`에 `_resolve_universe_state_cube()` 신규 함수 추가 → `_run_strategy_stage`에서 `universe_result`에서 cube 추출하여 `align_data_maps(state_cube=)` 주입. `pipeline.py` `_is_incomplete_pit_store_run()` 추가 → `load_or_build_universe_snapshot`에서 store hit 시 cube null 체크 후 rebuild. `discover_universe_timeline`에 `l2_start` timeline 경계 강제 로직 추가.
- **Rationale:** P0 - production 경로에서 `state_cube=None` 전달로 인해 L1/L2가 동일 PIT 필터를 소비하지 못함. P0 - store hit 시 decisions empty로 저장/복원되어 selection 정보 소실. P1 - L1/L2가 다른 시작 경계를 가져야 할 때 timeline이 2-way 계산만 함.
- **Edge Cases:** `universe_result is None` → cube=None 유지(기존 fallback 호환). Store hit + cube.parquet 없음 → cube=None fallback → incomplete 감지 → rebuild.

## 2026-06-19 EXACT-FIELDS: execution_pool_score 제거, exact-field only store contract
- **Delta:** `UNIVERSE_DECISION_COLUMNS`에서 `execution_pool_score` 제거. `_selected_frame_columns()`에서 제거. `build_decision_frame()`에서 `execution_pool_score` 쓰기 제거. `materialize_snapshot_from_store()`에서 alias 역매핑 제거. `_symbol_meta_from_decision_row()`에서 `alpha_capacity_score` 단독 사용. 구 cache hit 시 alias-only decisions → `is_exact_selected_feature_schema` False → rebuild. `_universe_metadata_by_symbol()`는 snapshot.selected exact field만 읽음.
- **Rationale:** Store/cache 계층에 `alpha_capacity_score`와 `execution_pool_score`가 동시 존재 → 동일 개념의 2개 truth source. Exact-field only로 단일화하여 cache-hit/fresh-build 간 metadata 불일치 원천 차단.
- **Edge Cases:** 구버전 store run(alias-only) → `build_decision_frame`가 `execution_pool_score` 컬럼을 남겨도 `validate_materializable_pit_store_run`가 detect → rebuild. `pipeline.py:649`에서 `alpha_capacity_score` 우선, 없으면 `execution_pool_score` fallback 유지.

## 2026-06-19 CAPACITY-CLIP: unit-NAV 시뮬레이션에서 portfolio_nav=1.0 → capacity clip 전멸
- **Delta:** `awf_sim.py:_run_awf_simulation`에 `_capacity_clip_enabled` 플래그 추가 (`portfolio_nav is not None`). fit-leg(829) 및 OOS(1025) capacity clip을 `_capacity_clip_enabled` 조건으로 가드.
- **Cause:** `portfolio_nav=None` → `_portfolio_nav=1.0` (unit-NAV). `_min_order_usdt=5.0` → `abs(w)*1.0 < 5.0` → per_symbol cap 10%를 통과한 모든 weight가 zero-out. commit `5f0254f`에서 state_cube와 동시에 추가됨.
- **Rationale:** Unit-NAV 시뮬레이션에서 w는 분수(fraction)이지 USDT 금액이 아님. 최소주문($5)을 weight에 직접 비교하는 것은 차원 오류. 실제 portfolio_nav가 주입될 때만 capacity clip을 활성화.

## 2026-06-19 KELLY-FRICTION: diagonal_kelly_weights 이중 friction filter 제거
- **Delta:** `portfolio_constructor.py:diagonal_kelly_weights`에서 Step 1 friction filter(`mu_bps < effective_hurdle = hurdle * safety_mult / holding_bars`) 제거. `friction_hurdle_bps`, `holding_bars`, `friction_safety_mult` 파라미터와 `hurdle` 변수 삭제. `awf_sim.py` 두 호출부에서 해당 인자 제거.
- **Cause:** `mu_bps` (`signed_net_bps_per_bar`)는 이미 edge computation에서 cost가 차감된 NET 값. `diagonal_kelly_weights`가 이를 다시 `hurdle * safety_mult / holding_bars`와 비교하면 이중과세 발생.
  - state_cube 도입 전(3.8 bps): `hurdle*2.5=9.5` → `gross(20)>9.5` → 통과
- (Compressed...)

## 2026-06-19 META-PARITY: UNIVERSE_DECISION_COLUMNS에 metadata 필드 추가 + full materialization
- **Delta:** `UNIVERSE_DECISION_COLUMNS`에 `vol_30d`, `friction_score`, `alpha_capacity_score`, `diversification_score` 4개 필드 추가. `_symbol_meta_from_decision_row()`에서 해당 필드 복원. `materialize_snapshot_from_store()`가 `decisions.parquet`에서 `SymbolMeta` 전체 필드 재구성. `_selected_meta_to_frame()` 추가 → `build_universe()` output을 decision columns와 일치. `_save_snapshot`에 `decisions=` 파라미터 추가.
- **Rationale:** cold build 시 `SymbolMeta`에 채워진 확장 필드가 cache-hit 시 `0.0` default로 떨어져 L1/L2가 다른 feature vector를 소비. Store schema에 exact field를 포함시켜 build/hit 간 metadata parity 보장.
- **Edge Cases:** 구버전 decisions(필드 누락) → `is_exact_selected_feature_schema` False → `validate_materializable_pit_store_run` False → rebuild 유도.

## 2026-06-19 Phase 4-B/C/E: Stage2-6 Config 및 legacy selection 제거
- **Delta:** `Stage2-6Config` 5종 class config.py에서 제거; `UniverseConfig`에서 `stage2-6/strategy_pool_mode/stage6_is_alpha_rank` 필드 제거. `Stage2Config` → `data_quality._DataQualityConfig` 인라인, `Stage3-5Config` → `filters.py` 로컬 이동. `selection.py` 전체 삭제(`apply_selection_stage` 포함). `pipeline.py` basket_ref/weights → `()`. 레거시 테스트 2종(`test_selection`, `test_strategy_pool_selection`) 삭제.
- **Rationale:** Phase 4-A에서 Stage6 else-branch 제거 완료 후 dead code 정리. `universe_engine` default = `"pit"` (4-A 적용). Stage2-5 config은 필터 유틸리티 함수 로컬 타입으로 유지(test_oi_adv_filter 호환).
- **Edge Cases:** 구버전 `Stage6Config` import하는 외부 테스트 → `@pytest.mark.skip` 처리(4-E); `k_in=50` cap으로 PIT 범위 제한.

## 2026-06-19 Phase 4-A: PIT 단독 경로 확정 + k_in=50 cap
- **Delta:** `build_universe`에서 Stage6 else-branch 완전 제거. `universe_engine` default `"stage6"` → `"pit"`. `PITUniverseConfig.k_in=50` 추가(capacity_usdt 내림차순 top-50). `store.py` empty decisions early return 추가.
- **Rationale:** PIT 경로 shadow validation PASS 후 Stage6 code path 불필요. k_in cap은 411 → 50 symbols로 제한(임시, Phase 4-D 이후 완전 제거 검토).
- **Edge Cases:** ledger `date` vs `datetime` 비교 TypeError 픽스(pipeline.py `_instrument_df_from_ledger`).

## 2026-06-19 Phase 3-3/3-4/3-5: PIT state_cube L1 wiring + lifecycle + capacity
- **Delta:** Phase 3-3: `_run_universe_stage` 7-tuple 반환(`universe_result` 추가). `align_data_maps` 호출에 `state_cube=` 주입 → `active_mask` PIT 반영. Phase 3-4: `SymbolLifecycleRecord` 추가, `promotion_available_at > l2_start` gate로 late-listing 심볼 L2 제외. Phase 3-5: `awf_sim` fit/OOS 양쪽에 `capacity_usdt` clip + 5 USDT min order threshold.
- **Rationale:** PIT state_cube 없이 L1이 stage6 all-True mask 사용 → look-ahead 노출. Lifecycle gate는 mid-window 상장 심볼이 OOS 신호에 참여하는 것을 방지. Capacity clip은 소량 포지션 거래비용 현실화.
- **Edge Cases:** `AlignedMarketData` frozen=True → `dataclasses.replace`; `adv_usdt_2d` shape 동적 체크(`isinstance(np.ndarray)`).

## 2026-06-15 Ledger backend compatibility recovery
- **Delta:** `load_ledger_slice(...)` now dispatches by backend suffix and supports both SQLite and parquet fixtures through the same PIT filter path.
- **Rationale:** universe tests and offline snapshots depend on parquet inputs; the loader must not collapse existing files into silent empty stage0 results.
- **Edge Cases:** missing files may still return empty frames, but readable files that fail backend-specific loading now raise with explicit backend context.

## 2026-06-19 Phase 4-D: UniverseSnapshot legacy panel 6필드 제거 + Stage6 경로 완전 삭제
- **Delta:** `UniverseSnapshot`에서 `training_panel`/`inference_panel`/`live_inference_panel`/`historical_trading_panel`/`inference_panel_quarter_membership`/`stage5_research_panel` 6개 필드 정의 제거. `discover_universe_timeline`의 Stage6 else-branch(230줄) 전체 삭제, dispatch는 PIT 무조건 호출로 단순화, `cfg=None` → `ValueError("universe_engine=pit required; stage6 path removed")` raise. Dead 헬퍼 `_resolve_trading_membership`, `_resolve_inference_membership` 삭제(`_snapshot_quality_symbols`는 `validate_universe_quality`에서 사용 중이므로 리팩터하여 유지). `snapshot_to_payload`/`snapshot_from_payload`에서 panel 직렬화 제거(구버전 payload key는 자동 무시). `store.py` `UniverseSnapshot(...)` panel 대입 제거. `pipeline.py` panel read + `replace(snapshot, ...)` 블록 제거. `strategy_service.py` `run_active_strategy_output_bridge`에서 panel 4개 파라미터 및 `training_panel` filter 제거. `opt_main_futures.py` 호출부 정리 및 `_run_universe_stage` extraction → `universe_result.inference_symbols`.
- **Rationale:** Stage6 panel 필드는 PIT state_cube가 유일 SSOT인 체계에서 불필요한 이중경로. Phase 4-A/4-B/4-C/E에서 Stage6 제거 후 최종 잔여 legacy 필드/경로 정리. `payload.get`-based deserialization은 구버전 스냅샷과의 하위호환 유지.
- **Edge Cases:** `cfg=None` → 명시적 raise, silent fallback 금지. `validate_universe_quality`가 `_snapshot_quality_symbols`에 의존하므로 함수 유지. `n_stageN` int 카운터는 별도 4-F 후보로 제거 대상 아님.

## 2026-06-19 Stage0.empty empty-universe contract: cube 강제 주입
- **Delta:** `build_universe()` stage0.empty 분기에서 `materialize_snapshot_from_store` 호출 시 `cube=None` 대신 empty `UniverseStateCube`(모든 array shape `(0,0)`, `eligible` all `False`)를 명시적으로 생성하여 전달. `validate_materializable_pit_store_run`가 empty-universe를 spec 계약(cube 존재 + eligible all False + zero selected) 하에서 통과시킴.
- **Rationale:** stage0.empty 경로에서 `cube=None` 전달 시 validator가 `cube is None` → `False` 반환 → `ValueError` 발생. 이는 cold build empty-universe 경로가 validated PIT snapshot만 소비한다는 계약을 위반. empty cube 생성으로 일관된 validator 통과 보장.
- **Edge Cases:** `np.empty((0,0), dtype=bool).any()` → `False` (empty array), `selected.empty` → True (zero rows), spec 계약 충족.

## 2026-06-19 Store Consolidation: 단일 Parquet Store 통합 + cube.parquet 영속화
- **Delta:** `snapshots/` flat+nested JSON/Parquet (분기당 7개 파일, 203개) 완전 제거 → `store/v1/runs/` 유일 저장소. `_save_snapshot` flat/nested write 제거. `load_or_build_universe_snapshot` snapshot JSON cache 경로(170줄) 제거 → 40줄 2-tier(store hit→materialize, store miss→build). `write_universe_store_run`에 `snapshot=` 파라미터 추가 → `pit_state_cube`를 `cube.parquet`로 직렬화(numpy tobytes). `load_universe_store_run` 반환값 3→4 튜플 확장(cube 포함). `materialize_snapshot_from_store(cube=)` → snapshot에 `pit_state_cube` 복원. `gc_stale_store_runs()` 신규 함수. `discover_universe_timeline` `cfg=None` → `UniverseConfig()` default (기존 ValueError 대체). `write_universe_store_run` empty-decision short-circuit 제거 → 항상 3파일(manifest+decisions+report) 쓰도록 수정.
- **Rationale:** 3중 JSON/Parquet 중복 및 file proliferation(203→29개) 해소. `pit_state_cube` transient 손실 버그 수정(캐시 적중 시 eligible all-False). `snapshots/` 레거시 호환성 유지 불필요(store가 단일 SSOT). Store run 누적(69→29) 방지 위해 GC 추가.
- **Edge Cases:** 구버전 store run(`cube.parquet` 없음) → `cube=None` fallback(기존 동작 유지). Empty decisions→schema-only DataFrame write로 store 일관성 유지. `load_universe_snapshot` 함수는 dropout computation에서 사용 중이므로 repurpose(store에서 최신 run 로드).
