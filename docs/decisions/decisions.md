# Active Decisions Log (Sliding Window)

## [2026-07-03] [TASK_L1_DIV] [ADR_20260703_L1]
- **Context/Why:** Extreme trend-beta bias in L1 strategy promotions caused high portfolio concentration risk during crashes.
- **Resolution/What:** Implemented family_admission.py and evaluated non-trend candidates via seed-matched economic replay.
- **Impact:** Replay results showed baseline outperforming treatment (CAGR collapse to -17%), leading to final rejection.

## [2026-07-03] [TASK_L2_DR] [ADR_20260703_L2_DR]
- **Context/Why:** Correlation-aware sizing was absorbed by the L* optimizer, failing to limit leverage during correlation spikes.
- **Resolution/What:** Built Choueifaty-Coignard diversification ratio (DR) haircut gate in leverage calibration step.
- **Impact:** Phase 0 test disconfirmed DR correlation during market crashes, so default was set to False.

## [2026-07-02] [TASK_L2_COV_RE] [ADR_20260702_L2_COV]
- **Context/Why:** Previous correlated covariance mode test was limited to a single reduced trial (n=1, trial=50) due to ledger crashes.
- **Resolution/What:** Re-run diagonal vs correlated covariance A/B testing on full 200-trial after repairing data pipeline bugs.
- **Impact:** Correlated mode underperformed diagonal (CAGR -5.6% vs -5.0%), confirming L* absorption effect.

## [2026-07-02] [TASK_L3_EP] [ADR_20260702_L3_EP]
- **Context/Why:** Whipsaws in post-crash trailing drawdown detection required episode-level timestamps to diagnose.
- **Resolution/What:** Implemented ReversalEpisode extraction logic and stress_gap diagnostics based on half-spread z-score.
- **Impact:** Enables empirical validation of liquidity stress discriminative power for new crash indicators.

## [2026-07-02] [TASK_L3_REPLAY] [ADR_20260702_L3_REPLAY]
- **Context/Why:** Hard verification of crash defense logic was lacking actual historical economic replay in holdout windows.
- **Resolution/What:** Wired risk_off fold attributions to L3 and created run_l3_reversal_economic_replay harness for 8 variants.
- **Impact:** Replay showed baseline outperforming all variants (reversal-kill de-grossed profitable trades), disconfirming entry/exit tuning.

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
