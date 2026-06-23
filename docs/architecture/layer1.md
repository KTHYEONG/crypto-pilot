---
title: Futures Signal Architecture
domain: futures.strategy
type: architecture
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/rule_signals.py
  - src/domain/futures/strategy/rule_diagnostics.py
  - src/domain/futures/strategy/exit_policies.py
  - src/domain/futures/strategy/candidate_contracts.py
  - src/domain/futures/strategy/tiered_workflow/pipeline.py
  - src/domain/futures/strategy/tiered_workflow/signal_selection.py
  - src/domain/futures/strategy/tiered_workflow/metrics.py
change_triggers:
  - src/domain/futures/strategy/rule_signals.py
  - src/domain/futures/strategy/rule_diagnostics.py
  - src/domain/futures/strategy_runtime/bridge.py
dependencies:
  documents:
    - docs/architecture/regime.md
    - docs/architecture/allocation.md
last_verified: 2026-06-23
---

# 1. Purpose
Generates vectorized rule panels with archetype/regime contexts, filtered through L1 breakeven hard gates and multiplicity controls to produce sparse candidate events. Manages prequential evidence snapshots for walk-forward validation across 4 TFs (4h/6h/8h/12h).

# 2. Core Logic & Math

**Signal Generation & Gating Sequence**
1. **Vectorization**: $S_{t} = f(\text{Data}_{1..t})$. Sparse triggers: $E_{t} = 1 \text{ if } (S_{t} \neq 0 \land S_{t-1} == 0) \text{ else } 0$. Strictly causal.
2. **Regime Gating**: Reversion signals blocked in specified high-risk regimes.
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

**Archetype Labels** (`TRD`/`TMO`/`MRV`/`CRY`/`FLO`/`UNW`/`BTN`)

**TF-Specific Signal Pools (31 Families)**
- `build_rule_signal_panels(family_filter)` post-filters panels. `CandidateStrategyConfig.per_tf_candidate_families` assigns per-TF pools: 1h (9, mean_rev), 2h (8, mixed), 4h (17, balanced), 6h/8h (7, trend), 12h (9, trend).
- 6 crypto-native additions: `gap_fade_1h`, `vwap_reversion_1h`, `volume_climax_1h`, `macd_4h`, `supertrend`, `ichimoku_trend`.

**Flow-Aware Panels**
- Shared cache: `flow_imbalance`, `flow_mean_6`, `flow_z_24`, `funding_z_96/168`, `funding_ts_slope`, `ret_1/12`, `ret_z_48`.
- Routes: `funding_flow_carry`/`funding_term_structure_carry`→CRY; `funding_flow_unwind`/`positioning_unwind`→UNW; `flow_exhaustion_reversal`→FLO; `flow_trend_continuation`→TMO; `lsr_oi_regime_filter`→BTN.
- `positioning_unwind`: 168-bar warm-up barrier. `lsr_oi_regime_filter`: side_hint = `-np.sign(lsr_log_z_42)`, stop_atr_mult=1.5, take_profit_atr_mult=2.0.

# 3. Architecture Flow

```mermaid
graph TD
    A[Market Data] --> B[Vectorized Indicators]
    B --> C[CandidateSignalPanel 31 Families]
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

**Readiness Gate**
- Fold Coverage ≥ 0.80, Match Ratio ≥ 0.90, $N_{eff}$ ≥ 3.0, Fold Ratio ≥ 0.50, Pooled LCB > 0.
- Per-TF overrides: 1h relaxed ($N_{eff}$ ≥ 3, sym ≥ 4, fold_ratio ≥ 0.40), 12h tightened ($N_{eff}$ ≥ 6, fold_ratio ≥ 0.55).

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
- **Deferred Artifact**: First TF computes `fit_layer1_inference_artifact`; remaining TFs skip via `defer_artifact=True`.
- **Prefork Cache Prime**: `prime_aligned_feature_cache` called once pre-fork → fork CoW shared across child processes.
- **Sequential Prequential Snapshots**: `ThreadPoolExecutor` rollback (GIL+cache thrashing on CPU-bound numpy/scipy → sequential recovers ~3.5s/TF).
- **Numba JIT**: `_rolling_robust_z_1d/2d`, `_cross_sectional_robust_z_2d` via `@njit(cache=True)`. Membership warm/ready via `_calculate_warm_ready_numba`.
- **Loop Invariant Hoisting**: `np.searchsorted` on state/readiness cube outside symbol loop ($O(N T \log T) \to O(T \log T + N V)$). Membership timeline normalized once (104→1).
- **Vectorized Volatility**: `pd.DataFrame.rolling().std(ddof=1)` over full matrix replaces column loop.
- **signal_batch_convert**: `iterrows` → `np.flatnonzero` vectorization (loop 2.64s→0.04s/fold, -98%).
- **Aligned Cache**: Numba JIT accelerated rolling/cross-sectional robust z-scores; pandas rolling apply replaced.
- **Parquet I/O**: Baggage columns dropped (`close_time`, `no_trades`, `ignore`), numeric early-exit guard, redundant `.copy()` removed.
- **Conditional Copy/Merge**: `raw_df.copy()` gated by `needs_merge`. `_to_unix_ms` once per TF. Merged audit lightweight `{rows, cols}`. `_COL_GROUP_CACHE` for column-group mapping.
- **Ingestion ThreadPool**: `ThreadPoolExecutor` replaces `ProcessPoolExecutor` for DataFrame I/O (pickle overhead elimination).
- **PERF Coverage**: `[PERF] worker_calc` (memory estimation + worker decision), `[PERF] l1_nested_ipc_collect` (IPC vs pool_setup split). Per-TF unaccounted < 3%. Bridge stage: `_get_rss_mb()` via `/proc/self/status`, `_sample_rss()` per-stage delta, `wf_fold_times` per-fold timing, `[PROFILE][MERGE][SUMMARY]` merge statistics.
- **Thread Control**: `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` set to `"1"` before fork. Prevents Numba `prange` + BLAS thread cascade (48→6 threads total).
- **Fork CoW Isolation**: Two-phase evidence/outer executor rolled back to single combined executor. Evidence sequences freed after snapshot build. Per-period `gc.collect()` with `del` hints. TF 간 `time.sleep(0.5)` for OS page recovery.
- **GC Control**: `gc.collect()` pre-fork, `gc.disable()` in child processes (CoW replication prevention), `gc.enable()` in `finally`. Additional: `gc.collect()` after `compute_rule_diagnostics()` to free ~5GB intermediate memory, `gc.collect()` after bridge return before tiered re-alignment. `del full_strategy_maps` after alignment in tiered path (dictionary wrapper reclaimed).
- **pandas 3.0**: `calendar.as_unit("ns").asi8` for nanosecond epoch. `tz_localize(None)` guard for `_to_unix_ms`.
