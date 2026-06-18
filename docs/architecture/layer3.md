---
title: Futures Holdout & Validation Architecture (Layer 3)
domain: futures.strategy
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/tiered_workflow/pipeline.py
  - src/domain/futures/strategy/tiered_workflow/awf_sim.py
  - src/domain/futures/strategy/tiered_workflow/dataclasses.py
  - src/domain/futures/optimization/candidate_selector.py
  - src/domain/futures/optimization/final_evaluator.py
change_triggers:
  - src/domain/futures/strategy/tiered_workflow/pipeline.py
  - src/domain/futures/optimization/final_evaluator.py
dependencies:
  documents:
    - docs/architecture/layer1.md
    - docs/architecture/layer2.md
last_verified: 2026-06-16
---

# 1. Purpose
Executes the final out-of-sample (OOS) validation (Layer 3) to ensure strategy robustness. It encompasses both the "Frozen Holdout" evaluation within the Tiered Hybrid Architecture and the multi-seed "Layer 3 Stability Check" during final Optuna optimization.

# 2. Core Logic & Math

## 2.1 Layer 3 — Frozen Holdout (Tiered Pipeline)
Operates as the final verification seam for the CS Rank + Diagonal Kelly (Layer 2) portfolio, using a completely unseen holdout window `[ho_start, ho_end)`.

**Frozen Parameters:**
- Uses the optimal `l2_params` identified during the Layer 2 AWF simulation.
- Hyperparameters are **frozen**; no refitting or parameter adjustment occurs in Layer 3.
- `l2_deploy_leverage`가 존재하면 L3 holdout도 동일 배치 레버리지로 평가한다.

**Performance Metrics (Lean 5-Gate, single-pass compounding focus):**
- **CAGR / Sharpe / Sortino:** Computed on actual hybrid and baseline (1/N) returns using `bars_per_year(tf)` annualization (no vol-proxy approximation).
- **MDD / CVaR95:** Maximum peak-to-trough drawdown and 95% tail loss of the hybrid portfolio.
- **Deployment Parity:** L3 hybrid metrics are computed on `apply_deployment(rets, L*)` output when `deploy_leverage > 1.0`; otherwise L3 stays on the unit path.
- **MAR:** $\text{MAR} = \frac{\text{CAGR}}{\text{MDD} + 10^{-9}}$
- **`total_return` / `equity_multiple`:** Single-pass terminal compounding result (`equity_multiple - 1`), reusing the L2 terminal-multiple helper — no new math, lean reuse per design decision (L3 favors realized compounding over Optuna-grade diagnostics).
- **`n_trades`:** Realized trade count over the holdout window — feeds the `insufficient_trades` gate.
- **`avg_gross_exposure`:** Mean gross exposure across the holdout AWF simulation, diagnostic only.

**Holdout Gate (L3 Gate, ordered short-circuit):**
1. `no_holdout_returns` — holdout span produced zero returns.
2. `non_finite` — any of CAGR/MDD/Sharpe/Sortino/total_return/equity_multiple is non-finite.
3. `insufficient_trades` — $n_{\text{trades}} < \text{min\_trades}$ (default 10).
4. `negative_return` — $\text{total\_return} \leq 0$.
5. `sharpe_rel` — $\text{Sharpe}_{\text{hybrid}} < \text{Sharpe}_{\text{baseline}}$.
6. `mdd_rel` — $\text{MDD}_{\text{hybrid}} > \text{MDD}_{\text{baseline}}$.
7. `mdd_abs` — $\text{MDD}_{\text{hybrid}} > \text{max\_mdd\_abs}$ (default 0.35), absolute capital-protection cap independent of baseline.
- Replaces the legacy single `cagr < 0.0` check — `negative_return`(on `total_return`) is the direct single-pass compounding check; absolute MDD cap defends against a baseline that itself crashed.

**Holdout Data Scope (Data Integrity Fix, 2026-06-16):**
- `aligned.datetimes` passed into `run_l3_holdout` MUST span `[fetch_start, holdout_end]` — i.e. the IS and OOS per-symbol frames merged via `pick_strategy_data_maps` (see `docs/architecture/layer2.md` §2 Data Scope). Previously `aligned` was IS-only (ending at `holdout_start`), making `_resolve_holdout_span` always raise `empty_holdout_window` — a structural bug, not a strategy/data quality finding.
- `_resolve_holdout_span` now logs `start_idx/end_idx/n_bars/last_dt` on the empty-window error path for fast diagnosis if the merge ever regresses.

## 2.2 Layer 3 — Multi-Seed Stability Check (Optimization Stage)
When full Optuna optimization is executed, `check_stability_layer3` enforces that the selected champion strategy demonstrates parameter stability across different random seeds.

**Stability Gate (`FUTURES_TMP_LAYER3_HARD_GATE`):**
- If enabled, every stability-seed AWF replay must independently pass Layer-1 validation checks.
- Prevents overfitting to a specific noise realization within the AWF grid, blocking promotion if the parameter set is fragile.

# 3. Architecture Flow

```mermaid
graph TD
    A[Layer 2 Result & Params] --> B[Define Dummy WFFold for Holdout]
    B --> C[Run AWF Simulation with Frozen Params]
    C --> D[Compute L3 Metrics: CAGR, MDD, MAR, Sharpe]
    D --> E{L3 Gate: Hybrid >= Baseline?}
    E -->|Pass| F[Emit Layer3Result]
    E -->|Fail| G[Gate Blocked / Revert to Baseline]
    
    H[Optuna Best Trial] --> I[Layer 3 Stability Check]
    I --> J[Re-run AWF across N target seeds]
    J --> K{Pass L1 Hard Gates?}
    K -->|Pass| L[Champion Promotion Evaluation]
    K -->|Fail| M[Block Promotion: TMP_LAYER3_STABILITY_LAYER1]
```

# 4. Key Components

| Module | Role |
|--------|------|
| `tiered_workflow/pipeline.py` | Implements `run_l3_holdout`, defining the dummy fold and calling the AWF sim. |
| `tiered_workflow/awf_sim.py` | Shared simulation loop (`_run_awf_simulation`) executed with frozen L2 params. |
| `tiered_workflow/dataclasses.py`| Defines `Layer3Result` (CAGR, MDD, Sharpe, Sortino, MAR, total_return, equity_multiple, n_trades, cvar95, avg_gross_exposure, deploy_leverage, gate_passed, blocker_reason). |
| `optimization/candidate_selector.py` | Implements `check_stability_layer3` for multi-seed validation. |
| `optimization/final_evaluator.py` | Orchestrates the final champion evaluation, invoking Layer 3 stability checks. |

# 5. Integration with opt_main_futures.py

- In `--phase alo`, `opt_main_futures.py` executes Layer 1, Layer 2, and Layer 3 via `run_tiered_pipeline`. `run_l3_holdout` performs the evaluation over the `ho_start` to `end_date` window, logging diagnostics.
- In `--phase full`, the pipeline proceeds to `_run_optimization_stage` after strategy evaluation. The selected champion candidate undergoes the `check_stability_layer3` validation to confirm robustness against seed variance before proceeding to the final Champion Swap logic.
