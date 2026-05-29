# Latest Alpha Optimization & Performance Report

This file documents the execution details and performance metrics for the latest walk-forward futures alpha optimization run.

---

## 1. Execution Settings & Environment
- **Command:** `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode alpha --trials 1 --reference-date 2026-05-01`
- **Execution Date:** 2026-05-29
- **Timeframe:** `4h`
- **Reference Date:** `2026-05-01`
- **Strategy Anchor:** `lambdamart` (LightGBM rank-native algorithm, version-agnostic standard)

---

## 2. Dataset & Universe Coverage
- **Window Scope:** `2022-10-01` ~ `2026-03-31` (In-Sample: `2023-10-01` | Out-of-Sample: `2025-10-01`)
- **Universe Screening:** 214 target symbols analyzed, 201 loaded successfully (94% coverage).
- **Final Active Universe:** **96 assets passed** data readiness evaluation.
  - *Pruning Reasons:* 62 failed on `warmup_insufficient`, 34 failed on `panel_history_insufficient`, 9 failed on `is_coverage_short`.

---

## 3. LightGBM Model Training (Walk-Forward)
- **Feature Shape:** 59 features, 96 symbols, 6,462 timesteps ($T$).
- **Walk-Forward Splitting:** 2 cross-validation folds + 1 virtual live refit fold.
- **Parallel Optimization:** Loky backend parallelized across 6 CPU threads (LightGBM single-thread locked to maximize cache alignment).
- **Execution Time:** Completed 3 folds in **26,921.64 ms** (~26.9 seconds).
- **Expected Value pre-clipping ratios:**
  - **Fold 0:** negative = 53.2%, $P_{50}$ = -3.3 bps, $P_{90}$ = 75.0 bps, $P_{95}$ = 75.0 bps
  - **Fold 1:** negative = 20.8%, $P_{50}$ = 10.9 bps, $P_{90}$ = 22.7 bps, $P_{95}$ = 25.9 bps
  - **Virtual Refit:** negative = 64.2%, $P_{50}$ = -2.2 bps, $P_{90}$ = 3.6 bps, $P_{95}$ = 4.5 bps

---

## 4. Alpha Attributions & Scoreboard

### 1) Dense Ranker (In-Sample / OOS Fold)
- **Information Coefficient (Score IC):** **`ic=0.0731`**
- **T-Statistic:** **`t=9.71`**
- **Hit Ratio:** **`0.597`**
- **Breadth:** **`3.7`**

### 2) Final Alpha Scoreboard (Combined Multi-seed Metrics)
| Metric | Value | Threshold / Target | Status |
| :--- | :--- | :--- | :--- |
| **NET_IC** | `-0.0013` | Positive edge | ❌ Fail |
| **T-STAT** | `-0.36` | $\ge 2.0$ | ❌ Fail |
| **BREADTH** | `1.66` | $\ge 8.0$ | ❌ Fail |
| **DSR** | `0.0863` | High DSR | ❌ Fail |
| **BE_IC (12h)** | `0.0394` | Breakeven hurdle | ❌ Fail (gap = -406.8 bps) |

### 3) Regime-Specific Performance (Net IC)
- **Bull Regime:** `+0.006` (1.0x exposure)
- **Bear Regime:** `-0.005` (0.5x exposure)
- **Chop Regime:** `-0.007` (1.0x exposure)

---

## 5. Multi-Horizon Risk Sweep

Risk and breakeven evaluation sweeps performed at multiple future prediction horizons:

- **6h Horizon:** `ic = -0.002` (breakeven = 3.77 bps, breadth = 1.7) ❌ Fail
- **12h Horizon:** `ic = -0.001` (breakeven = 2.69 bps, breadth = 1.7) ❌ Fail
- **18h Horizon:** `ic =  0.001` (breakeven = 2.20 bps, breadth = 1.7) ❌ Fail

### 🏁 Final Promotion Status: `ALPHA_PASS: FALSE`
*The alpha profile was rejected for champion registry promotion due to weak out-of-sample performance and negative net information coefficient (Net IC).*
