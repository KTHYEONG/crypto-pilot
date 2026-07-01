---
title: Futures Signal Architecture
domain: futures.signals
type: architecture
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/signals/rules.py
  - src/domain/futures/signals/diagnostics.py
  - src/domain/futures/signals/contracts.py
  - src/domain/futures/signals/workflow.py
  - src/domain/futures/signals/timeframes.py
  - src/domain/futures/signals/ensemble.py
  - src/domain/futures/allocation/pipeline.py
  - src/domain/futures/allocation/metrics.py
  - src/domain/futures/strategy_runtime/bridge.py
change_triggers:
  - src/domain/futures/signals/rules.py
  - src/domain/futures/signals/diagnostics.py
  - src/domain/futures/strategy_runtime/bridge.py
dependencies:
  documents:
    - docs/architecture/regime.md
last_verified: 2026-06-24
---

# 1. Purpose
Generates vectorized rule panels with archetype/regime contexts, filtered through L1 breakeven hard gates and multiplicity controls to produce sparse candidate events. Manages prequential evidence snapshots for walk-forward validation across 4 TFs (4h/6h/8h/12h).

# 2. Core Logic & Math

**Signal Generation & Gating Sequence**
1. **Vectorization**: $S_{t} = f(\text{Data}_{1..t})$. Sparse triggers: $E_{t} = 1 \text{ if } (S_{t} \neq 0 \land S_{t-1} == 0) \text{ else } 0$. Strictly causal.
2. **Regime Gating**: Hard side_hint masking by archetype. `mean_rev` signals allowed in `("bull_quiet","bear_quiet","transition")`, blocked elsewhere when `mean_rev_gating_enabled=True` (default). `beta_neut` (residual_reversion) allowed in `("bull_quiet",)` only, blocked in all other regimes when `beta_neut_gating_enabled=True` (opt-in, default False). Gate condition: `cfg.regime_signal_gating_enabled OR (archetype-specific flag AND archetype match)`.
3. **L1 Breakeven Hard Gate**: $\frac{1}{N} \sum (\text{Edge}_{i}) > 0 \land t_{\text{stat}} \geq \text{min\_rule\_ir\_t}$.
4. **Profit Floor**: $\mu_{\text{OOS}} \geq \text{min\_variant\_oos\_profit\_bps}$.
5. **Regime-Cell Admission (OR-path)**: Bayesian posterior $P(\mu > \delta | \text{data}) \ge p_{\text{admit\_min}}$. Newey-West variance + cross-cell $\tau^2$ shrinkage.
6. **Multiplicity Controls**:
   - **BH-FDR**: $q > l1\_pair\_fdr\_alpha$ ($0.10$) → hard reject. Adjusts $m$ to $m_{\text{eff}}$ via TF diversity correlations.
   - **SPA**: Hansen's Single Predictive Ability (fail-closed circular bootstrap).

**Ensemble Shrinkage**
- Empirical-Bayes James-Stein on archetype cell means + variant-level priors.
- Bayesian prior precedes JS: $\hat{x}_a = w_{prior} \cdot \bar{x}_a + (1 - w_{prior}) \cdot \mu_{prior}$, $w_{prior} = n_{eff} / (n_{eff} + n_{prior})$. Disabled by default ($n_{prior}=0$).
- Archetypes with events < `l1_ens_min_display_events` → `insuf`.

**Archetype Labels** (`TRD`/`TMO`/`MRV`/`CRY`/`FLO`/`UNW`/`BTN`/`XS`)

**TF-Specific Signal Pools (28 Families)**
- `build_rule_signal_panels(family_filter)` post-filters panels. `CandidateStrategyConfig.per_tf_candidate_families` assigns per-TF pools: 4h (17, balanced), 6h/8h (9, trend+mixed), 12h (11, trend+carry).
- 3 crypto-native additions: `macd_4h`, `supertrend`, `ichimoku_trend`. 7 low-outcome families removed (`rsi_reversion`, `bollinger_reversion`, `vol_regime_reversion`, `funding_zscore_carry`, `gap_fade_1h`, `vwap_reversion_1h`, `volume_climax_1h`).

**Cross-Sectional Alpha Families (xs_alpha archetype)**
- 4 families: `xs_momentum`, `xs_carry`, `xs_flow`, `xs_oi_skew` — per-bar rank across symbols, beta-neutral by construction. Regime-agnostic (market-neutral, exempt from mean-reversion gating).
- Helpers: `_cross_sectional_rank_signed_2d` (per-timestamp rank → signed score + tercile side), `_beta_residual_return_2d` (rolling beta residual sum).
- Raw factors: `xs_momentum` = beta-residual return (L=12/48); `xs_carry` = `-funding_z_96/168` (short expensive funding); `xs_flow` = `flow_z_24`; `xs_oi_skew` = `-(oi_build_z_42 * sign(lsr_log_z_42))` (short crowded longs).

**Flow-Aware Panels**
- Shared cache: `flow_imbalance`, `flow_mean_6`, `flow_z_24`, `funding_z_96/168`, `funding_ts_slope`, `ret_1/12`, `ret_z_48`.
- Routes: `funding_flow_carry`/`funding_term_structure_carry`→CRY; `funding_flow_unwind`/`positioning_unwind`→UNW; `flow_exhaustion_reversal`→FLO; `flow_trend_continuation`→TMO; `lsr_oi_regime_filter`→BTN.
- `positioning_unwind`: 168-bar warm-up barrier. `lsr_oi_regime_filter`: side_hint = `-np.sign(lsr_log_z_42)`, stop_atr_mult=1.5, take_profit_atr_mult=2.0.

# 3. Architecture Flow

```mermaid
graph TD
    A[Market Data] --> B[Vectorized Indicators]
    B -->     C[CandidateSignalPanel 28 Families]
    C --> C1[Multi-TF Panel Injection]
    C1 --> D[Archetype & Regime Context]
    D --> E[L1 Breakeven & Profit Floor Gate]
    E --> F[Regime-Cell OR-path Admission]
    F --> G[Multiplicity: FDR & SPA]
    G --> H[Promoted Candidate Events]
    H --> I[Per-TF L1 Pipeline 4h/6h/8h/12h]
    I --> J[L2 on Master TF]
```

# 4. SWF & System Integrity

**Per-TF L1 Pipeline**
- `run_tiered_pipeline` executes L1 on `l1_tfs` (default 4h/6h/8h/12h). HTF panels via `build_multi_tf_panels`: native-resolution panels projected to base grid via `_project_panel_to_base_grid`, tagged with `native_tf`. `run_per_tf_l1` filters `native_tf == tf`, applies TF-specific gate overrides. `_aggregate_per_tf_l1` merges `oos_stacked` with `tf::` prefix, `gate_passed=any`. `_resolve_l2_master_tf`: `cfg.l2_master_tf` → TF with most L1 winning signals → `"8h"`.

**Prequential Evidence**
- Decoupled grid multiplier (outer_n×multiplier ≤ max_folds). Adaptive evidence gates: early snapshots relax `min_effective_obs`/`min_folds`.

**Readiness Gate (5-condition)**
- Fold Coverage ≥ 0.80, Match Ratio ≥ 0.90, $N_{eff}$ ≥ 3.0, Fold Ratio ≥ 0.50, Pooled LCB > 0.
- Per-TF overrides: 1h relaxed ($N_{eff}$ ≥ 3, sym ≥ 4, fold_ratio ≥ 0.40), 12h tightened ($N_{eff}$ ≥ 6, fold_ratio ≥ 0.55).
- **Probe Metric**: Defaults to `"breadth"` — per-decision probe uses the cross-sectional mean of all symbols' realized returns instead of top-k selection. Configurable via `l1_probe_metric`.
- **Probe Breadth Diagnostics (env-gated DEBUG, zero side-effect on gate)**: `L1_PROBE_DIAG` env var activates per-fold decomposition. Measurements: (a) **Breadth-decay** — `probe_gross_by_k` for k=3/10/20/-1 quantifies selection inflation; (b) **Net decomposition** — `gross − rt_cost` per k; (c) **Rank-IC** — Spearman ρ + Fisher-z tstat over predicted↔realized pairs; (d) **IC Monitor** — pooled IC tstat and sign-consistency ratio computed at gate evaluation time and logged at DEBUG level only (not a hard gate, spec §Track A). (e) **Realized distribution** — mean/median/positive-fraction. (f) **Regime side-split** — `regime_side_split` per-regime dict with `(long_fraction, long_real_mean_bps, short_real_mean_bps, n_long, n_short)`; exposes net-long bias in bear regime.
- **XS Factor Spread Diagnostics (env-gated DEBUG, zero side-effect on gate)**: `L1_XS_SPREAD_DIAG` env var activates gate-independent per-XS-factor OOS long-short spread measurement. Measurements: (a) **Per-bar tercile spread** — `realized_side_adjusted_gross_bps` mean per `decision_idx` (side-adjusted → mean = long-short P&L); (b) **Spread Sharpe** — `mean / max(std, 1e-9)`; (c) **Block-bootstrap LCB** — `moving_block_bootstrap_mean` P5 quantile; (d) **Rank-IC** — per-bar Spearman ρ(`score_z`, realized) with Fisher-z tstat; (e) **Long fraction** — proportion of long-side events per factor. Source: `realized_event_results` (pre-promotion superset, XS included) — bypasses promotion gate to detect whether per-pair gating masks portfolio-level XS alpha.

**Promotion Gate (4-condition)**
1. `hard_eligible`: structural gates all pass
2. `lcb_net_bps > l1_breakeven_floor_bps` (~7.5 bps): block-bootstrap P5 of incremental gross > RT cost
3. `q_value ≤ l1_pair_fdr_alpha` ($0.10$): binding BH-FDR
4. `quality_weight > 0`: `max(l1_qw_floor, max(0, 2P-1) · positive_fold_ratio · sample_scale)`. FDR hard reject overrides qw_floor. Probe winning cells get `l1_qw_probe_boost` (0.3) via `probe_prior_map`.

**PIT Universe Integration**
- `state_cube` → `AlignedMarketData.active_mask [T, N]` → `promotion_available_at`. Symbols with `promotion_available_at > l2_start` excluded from L2.

**Capacity Clip**
- `intended_notional < 5 USDT → w = 0`; `> capacity → proportional clip`. Active only with `portfolio_nav`.

# 5. Performance Optimizations

- **Adaptive Worker Cap**: Stage-specific caps: `evidence (4 if compact+≥8GB else 3)`, `outer (3)`, `l2_optuna (4)`. `pinned` parameter acts as safety-clamped upper bound (not hard override). `l1_nested_result_soft_cap_mb` enforces aggregate result OOM guard (`predicted_result_mb = 100 if compact else 400`). Oversubscription guard: `folds // workers < 2` → reduce. Memory formula: `estimated_proc_gb = max(0.8, 0.5 + frame_gb*0.5 + predicted_result_gb)`.
- **Prefit Overlap (OPT-4)**: `prefit_layer1_model` submitted to executor right after evidence fold submit. Runs in background during evidence IPC + snapshot + outer folds. On gate pass: `assemble_layer1_artifact` (1ms) attaches registry. `l1_fit_inference_artifact` 20.49s→0.00s (100% eliminated). Falls back to serial fit on prefit failure.
- **Prefork Cache Prime**: `prime_aligned_feature_cache` called once pre-fork → fork CoW shared across child processes.
- **Prequential Snapshot ThreadPool (OPT-3)**: `build_l1_prequential_evidence_snapshots` wraps per-snapshot work in `_build_snapshot` inner function. `n≤1` sequential, `n≥2` → `ThreadPoolExecutor(max_workers=n//2)`. Results sorted by `as_of_idx` for determinism. Numba bootstrap+GIL release makes threads effective.
- **Numba JIT**: `_rolling_robust_z_1d/2d`, `_cross_sectional_robust_zscore` (`_robust_zscore_numba` njit inner) via `@njit(cache=True)`. Membership warm/ready via `_calculate_warm_ready_numba`.
- **searchsorted Indexing (OPT-1)**: `load_futures_data_maps_for_symbols` Pass-2 replaces O(T) datetime mask + sum + argmax with O(log T) `np.searchsorted`. Applied to `is_end_idx`, `is_start_idx`, `oos_start_idx`.
- **t.ppf LRU Cache (OPT-3)**: `_t_ppf_cached(q_thousandths, df_int)` with `@lru_cache(maxsize=512)`. Each pair's `stats.t.ppf(1-alpha, df)` + `stats.t.ppf(power, df)` reduced to dict lookup. q=2 values (950, 800), df≈10-30 → ~100% hit rate.
- **Evidence IPC as_completed (OPT-2)**: `run_l1_nested_swf` collects evidence futures via `as_completed` instead of submit-order. Results sorted by `fold_id` post-collection for snapshot consistency.
- **Loop Invariant Hoisting**: `np.searchsorted` on state/readiness cube outside symbol loop ($O(N T \log T) \to O(T \log T + N V)$). Membership timeline normalized once (104→1).
- **Vectorized Volatility**: `pd.DataFrame.rolling().std(ddof=1)` over full matrix replaces column loop.
- **signal_batch_convert**: `iterrows` → `np.flatnonzero` vectorization (loop 2.64s→0.04s/fold, -98%).
- **Aligned Cache**: Numba JIT accelerated rolling/cross-sectional robust z-scores; pandas rolling apply replaced.
- **Multi-TF Bridge ThreadPool**: `build_multi_tf_panels` wraps each non-base TF (6h/8h/12h) processing into `_process_single_tf` inner function. `len(eligible_tfs) <= 1` → sequential; `>= 2` → `ThreadPoolExecutor(max_workers=2)`. Each TF allocates independent `list[CandidateSignalPanel]` — no shared mutation, GIL-released NUMBA works effectively. Per-TF exception isolated: caught inside `_process_single_tf`, returns `([] , None)`; other TFs unaffected.

- **Vectorized Top-k Selection**: `_vectorized_topk_per_bar` replaces per-bar Python loop in `select_candidate_events_for_portfolio`. Sort → `drop_duplicates` per-(bar,symbol) → `cumcount` rank → `ceil(bar_size×quantile)` keep → variant-cap backfill (per-(bar,family,variant) filter + re-rank). O(E log E) vectorized, 0 Python loops. Per-bar result identical via sorted cumcount tie-break.
- **Diagnostics Gating**: `enable_diagnostics` param on `select_candidate_events_for_portfolio`. Evidence folds (12/16 per TF) gate `selection_sensitivity`, `shadow_profiles`, `waterfall` to False. Deployed/outer fold path retains `enable_diagnostics=True` for diagnostic SSOT.
- **Prepare-Once Event Table**: `bridge.py` calls `prepare_labeled_events` once before WF loop → `PreparedLabeledEvents` passed through `run_candidate_walk_forward` and wired via `_GLOBAL_LABELED_EVENTS` (ProcessPool fork CoW). `build_candidate_dataset` fast path (numpy boolean mask + `frame.iloc`) eliminates per-window pandas to_numeric/isin/mask/copy.
- **Enrich Cache Precompute**: `PreparedLabeledEvents.enrich_cache` stores window-invariant fields: `arm` (vectorized affinity matrix), `entry_regime` (name-by-code), `overlay_mult`, `crisis_active`, `entry_regime_code`. `_precompute_enrich` called once per worker inside `build_candidate_dataset` (lazy init, ProcessPool-safe). Populates on first `sig_feat_names` or `skip_features` call; subsequent calls slice via `row_ids`. `score_pct` and `n_same` excluded (window-dependent — per-window `_compute_score_pct_variant_hist` on slide subset retained).
- **Schema-Once**: `frozen_identity_names` extracted from prepared events once; reused across all folds in `fit_candidate_feature_schema`. Identity feature schema computed only for the final anchored fit window.
- **Parquet I/O**: Baggage columns dropped (`close_time`, `no_trades`, `ignore`), numeric early-exit guard, redundant `.copy()` removed.
- **Conditional Copy/Merge**: `raw_df.copy()` gated by `needs_merge`. `_to_unix_ms` once per TF. Merged audit lightweight `{rows, cols}`. `_COL_GROUP_CACHE` for column-group mapping.
- **Ingestion ThreadPool**: `ThreadPoolExecutor` replaces `ProcessPoolExecutor` for DataFrame I/O (pickle overhead elimination).
- **PERF Coverage**: `[PERF] worker_calc` (memory estimation + worker decision), `[PERF] l1_nested_ipc_collect` (IPC vs pool_setup split). Per-TF unaccounted < 3%. Bridge stage: `_get_rss_mb()` via `/proc/self/status`, `_sample_rss()` per-stage delta, `wf_fold_times` per-fold timing, `[PROFILE][MERGE][SUMMARY]` merge statistics.
- **L1 WF Wall-Time Summary**: `[PERF] l1_wf_summary` logs avg selection/ds_fit/schema/edge_fit/inference across all folds + evidence/outer wall times all-TF aggregate. Bridge profile walk_forward always shown (`0s` → `"(skipped)"`).
- **Thread Control**: `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` set to `"1"` before fork. Prevents Numba `prange` + BLAS thread cascade (48→6 threads total).
- **Fork CoW Isolation**: Two-phase evidence/outer executor rolled back to single combined executor. Evidence sequences freed after snapshot build. Per-period `gc.collect()` with `del` hints. TF 간 `time.sleep(0.5)` for OS page recovery.
- **GC Control**: `gc.collect()` pre-fork, `gc.disable()` in child processes (CoW replication prevention), `gc.enable()` in `finally`. Additional: `gc.collect()` after `compute_rule_diagnostics()` to free ~5GB intermediate memory, `gc.collect()` after bridge return before tiered re-alignment. `del full_strategy_maps` after alignment in tiered path (dictionary wrapper reclaimed).
- **Parallel Signal & Event Generation**: `build_rule_signal_panels` runs independent signal families concurrently using `ThreadPoolExecutor` (max_workers=4) over nested closure states. `candidate_panels_to_events` partitions panel-to-event conversions across thread tasks to bypass sequential execution blocks.
  - **Regime×Policy Loop Pre-extraction**: `_convert_single_panel` extracts base arrays per-regime once (15× array indexing O(R×P)→O(R)) instead of re-extracting per-policy.
  - **No sort_values**: `candidate_panels_to_events` omits `sort_values("datetime")` — downstream consumers use entry_idx-based ordering, making the full-table sort (O(N log N)) unnecessary.
  - **Numba Per-Group Robust Zscore**: `_cross_sectional_robust_zscore` delegates to `_robust_zscore_numba` (`@njit`) — single argsort pass per group, O(U×E) Python loop replaced with O(E log E) Numba-compiled walk.
- **Parallel Diagnostics**: `compute_rule_diagnostics` parallelizes Pandas groupby aggregations and Newey-West/Bayesian statistical evaluations by executing independent `by_family`, `by_variant`, and `by_family_side` operations in parallel threads.
- **pandas 3.0**: `calendar.as_unit("ns").asi8` for nanosecond epoch. `tz_localize(None)` guard for `_to_unix_ms`.
