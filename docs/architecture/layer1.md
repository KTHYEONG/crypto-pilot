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
change_triggers:
  - src/domain/futures/strategy/rule_signals.py
  - src/domain/futures/strategy/rule_diagnostics.py
  - src/domain/futures/strategy/exit_policies.py
dependencies:
  documents:
    - docs/architecture/regime.md
    - docs/architecture/allocation.md
last_verified: 2026-06-14
---

# 1. Purpose
Generates vectorized rule panels with archetype/regime contexts, filtered through L1 breakeven hard gates and multiplicity controls to produce sparse candidate events. Manages prequential evidence snapshots for walk-forward validation.

# 2. Core Logic & Math

**Signal Generation & Gating Sequence**
1. **Vectorization**: $S_{t} = f(\text{Data}_{1..t})$. Sparse triggers: $E_{t} = 1 \text{ if } (S_{t} \neq 0 \land S_{t-1} == 0) \text{ else } 0$. Strictly causal.
2. **Regime Gating**: Reversion signals blocked in specified high-risk regimes.
3. **L1 Breakeven Hard Gate**: $\frac{1}{N} \sum (\text{Edge}_{i}) > 0 \land t_{\text{stat}} \geq \text{min\_rule\_ir\_t}$.
4. **Profit Floor**: Unconditional cost-based minimum: $\mu_{\text{OOS}} \geq \text{min\_variant\_oos\_profit\_bps}$.
5. **Regime-Cell Admission (OR-path)**: Rescues signals with strong orthogonal edge in specific regimes via Bayesian posterior: $P(\mu > \delta | \text{data}) \ge p_{\text{admit\_min}}$. Uses Newey-West variance and cross-cell $\tau^2$ shrinkage.
6. **Multiplicity Controls**:
   - **BH-FDR**: Limits false discoveries across pool expansion.
   - **SPA**: Hansen's Single Predictive Ability (fail-closed circular bootstrap).

**Ensemble Shrinkage**
- Empirical-Bayes James-Stein shrinkage applies to both archetype cell means ($\hat{\mu}_a \to \bar{\mu}$) and variant-level priors.
- A Bayesian prior layer precedes JS: $\hat{x}_a = w_{prior} \cdot \bar{x}_a + (1 - w_{prior}) \cdot \mu_{prior}$ with $w_{prior} = n_{eff} / (n_{eff} + n_{prior})$. When `l1_ens_prior_effective_n > 0`, archetypes with few events are shrunk toward $\mu_{prior}=0$ before JS, preventing small-sample negative edge artifacts. Disabled by default ($n_{prior}=0$).
- `predict_regime_conditional_ensemble` output: `validation_rank_ic` (diagnostic only, 0.0 default) in `validation_diagnostics`. IC is NOT a gate input; `mu_quality_shrinkage` feature is removed (was dead: validation_rank_ic=0.0 → lam=0 → mu collapse).

**Archetype Labels (`[ENS]` log)**
| Archetype key | Log label | Semantic |
|---|---|---|
| `trend` | `TRD` | Cross-sectional trend-following |
| `ts_mom` | `TMO` | Time-series momentum |
| `mean_rev` | `MRV` | Mean reversion |
| `carry_rev` | `CRY` | Carry / basis reversion |
| `flow_rev` | `FLO` | Order-flow reversion |
| `unwind` | `UNW` | Unwind / position exit |
| `beta_neut` | `BTN` | Beta-neutral / market-neutral |
- `[ENS]` numbers = archetype-pooled EB-shrunken mean edge (bps), NOT per-symbol averages.
- Archetypes with event count < `l1_ens_min_display_events` display `insuf` instead of numeric edge, preventing misleading small-sample signs.
- Unknown archetypes (not in the above 7) fall back to first-letter uppercase.

**Flow-Aware Panels & Conditioning Gates**
- `_safe_taker_imbalance_2d` converts taker buy volume into a cell-level imbalance cache and marks invalid cells as `False` without collapsing mixed-valid rows.
- `build_rule_signal_panels` reuses shared flow caches across `taker_imbalance_momentum`, `funding_flow_carry`, `funding_flow_unwind`, `flow_exhaustion_reversal`, `funding_term_structure_carry`, `flow_trend_continuation`, and `lsr_oi_regime_filter`.
- `funding_flow_carry` and `funding_term_structure_carry` route to `carry_rev`; `funding_flow_unwind` and `positioning_unwind` route to `unwind`; `flow_exhaustion_reversal` routes to `flow_rev`; `flow_trend_continuation` routes to `ts_mom`; `lsr_oi_regime_filter` routes to `beta_neut`.
- `funding_term_structure_carry` uses `funding_ts_slope = funding_z_96 - funding_z_168` to capture funding acceleration when short-term z exceeds long-term z in the same direction.
- `flow_trend_continuation` captures flow-supported trend continuation (flow_z_24 >= 1.0 + positive ret_12 + positive ret_1), long-only. Routes to `ts_mom` archetype (`flow_momentum_continuation` regime).
- `lsr_oi_regime_filter` emits a conditioning score when LSR z-score >= 1.0σ and OI build z-score >= 0.5σ, identifying positioning-dominated regimes. Emits directional side_hint (`-np.sign(lsr_log_z_42)`) to fade the crowded side, with stop_atr_mult=1.5 and take_profit_atr_mult=2.0. Routes to `beta_neut`.
- `positioning_unwind` enforces a 168-bar continuous valid data warm-up barrier before entry eligibility, preventing z-score noise in shallow data windows.
- Flow feature cache includes `flow_imbalance`, `flow_mean_6`, `flow_z_24`, `funding_z_96`, `funding_z_168`, `funding_ts_slope`, `ret_1`, `ret_12`, and `ret_z_48`.

# 3. Architecture Flow

```mermaid
graph TD
    A[Market Data] --> B[Vectorized Indicators]
    B --> C[CandidateSignalPanel]
    C --> D[Archetype & Regime Context Injection]
    D --> E[L1 Breakeven & Profit Floor Gate]
    E --> F[Regime-Cell OR-path Admission]
    F --> G[Multiplicity Gating: FDR & SPA]
    G --> H[Promoted Candidate Events]
    H --> I[L1 Nested SWF & Readiness Gate]
```

# 4. SWF & System Integrity

**Layer 1 Nested SWF**
- **Prequential Snapshots**: Evidence grids use decoupled multipliers and outer warm-up blocks to prevent early-fold starvation.
- **Adaptive Evidence Gates**: During the first `l1_evidence_early_snapshots` snapshots, `l1_pair_min_effective_obs` and `l1_pair_min_folds` thresholds are relaxed to `l1_pair_min_effective_obs_early` / `l1_pair_min_folds_early`, allowing sparse early folds to generate registry entries. Quality weight is the ultimate arbiter: pairs that pass relaxed structural gates but fail `probability_positive ≥ 0.5` still receive `quality_weight = 0`.
- **OOS Activation**: Enforces pooled Arch-Only mode during L1 to preserve statistical power ($N_{eff}$); regime is delegated to L2 risk overlays.
- **Readiness Gate**: Strict multi-condition screening:
  - Fold Coverage $\ge 0.80$, Match Ratio $\ge 0.90$, Effective Symbols ($N_{eff}$) $\ge 3.0$, Fold Ratio $\ge 0.50$.
  - **Pooled LCB**: Global profitability metric ($LCB > 0$) via stationary block bootstrap over all passed folds.
- **Right-Censoring Diagnostic**: `dropped_by_maturity_count` tracks events filtered by `exit_idx >= oos_end` per fold. Exposed in Outer Fold log as `[censored: N]` to distinguish genuine edge weakness from boundary truncation (especially last fold).

**Promotion Summary & L2 Gate**
- **Actual L2 gate**: `build_qualified_signal_registry` — admits evidence where `hard_eligible AND quality_weight > 0`, with `>= l1_min_signals_per_symbol` per symbol.
- **STATUS labels** (`[L2-PASS] Q:hi/mid/lo`) reflect q-value quality tier only; all admitted rows are L2-bound regardless of tier.
- **FAIL summary**: `[NOT PROMOTED] N pairs | top: <reason>xN` appears when `all_evidence` is provided, listing structural exclusion reasons for non-admitted pairs.
- **Promotion Filter**: Diagnostics-level filter (`apply_variant_promotions`) is advisory-only. When no variants are recommended by diagnostics, all events pass through unfiltered; the ultimate filtering authority is `compute_symbol_strategy_evidence` via structural gates and quality weight within the L1 SWF.

**PIT Universe Integration**
- `state_cube` (`UniverseStateCube [T, N]`) injected into `align_data_maps` → `AlignedMarketData.active_mask [T, N]`.
- Tiered entry scope is derived in two stages: `full_strategy_maps` is first reduced to a data-availability `base_scope`, then strict sub-window admission is applied before tiered execution begins. Empty strict admission is fail-closed.
- `active_mask` used as `SymbolLifecycleRecord` source: `first eligible bar per column = promotion_available_at`.
- **Promotion gate**: symbols with `promotion_available_at > l2_start` excluded from L2 `oos_stacked` before gate evaluation.
- `readiness_cube` (`StrategyReadinessCube`) computed after alignment via `evaluate_strategy_readiness`; injected via `dataclasses.replace(aligned, strategy_readiness_mask=...)`.

**Capacity Clip (awf_sim)**
- Per-bar capacity from `adv_usdt_2d [T, N]`: `intended_notional < 5 USDT → w = 0`; `> capacity → proportional clip`.
- Active only when `portfolio_nav` is provided (unit-NAV simulation skips the clip: weights are fractions, not USDT notional).

# 5. Data Integrity & Optimizations

- **Guards**: NaN/stuck-price blocks, length minimums, high-low violation checks.
- **Data Load Parallelization**: Utilizes `ThreadPoolExecutor` instead of multiprocessing to eliminate heavy pickle serialization and IPC overhead during parallel DataFrame loads.
- **Fast Datetime Bypass**: Skips redundant `pd.to_datetime` calls in `_resolve_tradeable_scope` if the input column is already in `datetime64` dtype, resolving datetime parsing bottlenecks to O(1).
- **NumPy-Backed Meta Alignment**: Meta columns are pre-converted to numeric at the ingestion stage. Within the `align_data_maps` loop, valid values are retrieved using fast NumPy masking on sliced views instead of Pandas Series creation, guaranteeing 100% data fidelity with zero look-ahead bias and optimized latency.
- **ALIGN-CUBE Loop-Invariant Hoisting**: `np.searchsorted` over `state_cube.calendar` (and `readiness_cube.calendar`) is computed once outside the symbol loop. `positions`/`t_valid`/`p_valid` are symbol-independent; complexity reduced from $O(N \cdot T \log T_{\text{cube}})$ to $O(T \log T_{\text{cube}} + N \cdot V)$ (V = valid bars). Pandas 3.0 nanosecond fix: `calendar.as_unit("ns").asi8` enforces `int64` nanosecond epoch instead of microsecond default.
- **Membership Timeline Hoisting**: `_normalize_timeline()` normalizes `timeline` / `inference_timeline` once before the symbol loop in `inject_membership_masks_into_maps`, eliminating 104× repeated quarter-start and `canonical_symbol` calls (52 syms × 2 maps).
- **Vectorized Volatility**: `volatility_2d [T, N]` computed via single `pd.DataFrame.rolling().std(ddof=1)` call over the full matrix, replacing a column-wise Python loop of N `pd.Series` allocations.
- **Conditional raw_df.copy()**: When `funding_df_prepared` and `metrics_df_prepared` are both `None`, the raw DataFrame reference is used directly without copying, eliminating redundant 30K×300 memory duplication. The copy path is preserved when any merge is required (`merge_asof` mutates in-place).
- **Single _to_unix_ms per TF**: `_to_unix_ms(raw_df["datetime"])` is computed once per timeframe inside `needs_merge` block. The resulting column is reused for both funding and metrics merge_asof calls, removing two unnecessary datetime→unix_ms conversions per TF.
- **Lightweight merged-stage audit**: `_append_stage_integrity("merged")` stores `{rows, cols}` only instead of calling `summarize_ohlcv_collection_integrity` (4 full-array scans: NaN, inf, gaps, OHLCV violations). The per-symbol `audit_df` groupby at the end of `load_futures_data_maps_for_symbols` conditionally selects only available integrity columns; missing columns default to zero.
- **Column group cache**: `_feature_group_coverage` uses a module-level `_COL_GROUP_CACHE` keyed by `(tf_label, frozenset(col_lower))` to pre-compute column→pattern-group mapping once per TF session. Subsequent calls for symbols with identical column sets perform O(C) lookups instead of O(C×P) string scans.
- **Parquet baggage column pruning**: `_load_cache` drops Binance API metadata columns (`close_time`, `no_trades`, `ignore`) immediately after parquet read — never used by any downstream domain code. Reduces per-file memory footprint by ~30% and eliminates string→numeric conversion overhead for these columns.
- **Numeric `_normalize_df` early exit**: When all non-datetime columns are already numeric (guaranteed after first `_save_cache`), the string→numeric loop is skipped via an `all(is_numeric_dtype)` guard. Cache-read path exits in O(N) dtype scan instead of O(C×N) full column conversion.
- **Removed redundant `.copy()`**: `collect_and_save(fetch_network=False)` returns `cache_df.loc[mask]` directly instead of `.loc[mask].copy()`. Boolean indexing in pandas always returns a copy, making the extra `.copy()` allocation redundant.
- **Algorithmic Optimizations**: Numba JIT bootstrap, $O(N \log N)$ vectorized percentiles, parent-process feature priming, Numba-JIT accelerated rolling/cross-sectional robust z-score loops to bypass pandas rolling overhead, and unified OMP-clamped multiprocessing pools.
