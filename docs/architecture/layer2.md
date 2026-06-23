---
title: Futures Allocation Architecture
domain: futures.strategy
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/tiered_workflow/pipeline.py
  - src/domain/futures/strategy/tiered_workflow/selection.py
  - src/domain/futures/strategy/tiered_workflow/l2_gate.py
  - src/domain/futures/portfolio/covariance.py
  - src/domain/futures/portfolio/portfolio_constructor.py
  - src/domain/futures/strategy/walk_forward.py
  - src/domain/futures/portfolio/signal_composer.py
  - src/domain/futures/strategy/cs_rank.py
  - src/domain/futures/strategy/risk_deployment.py
change_triggers:
  - src/domain/futures/strategy/tiered_workflow/pipeline.py
  - src/domain/futures/strategy/tiered_workflow/selection.py
  - src/domain/futures/strategy/tiered_workflow/l2_gate.py
  - src/domain/futures/strategy/risk_deployment.py
dependencies:
  documents:
    - docs/architecture/signal.md
    - docs/architecture/layer1.md
last_verified: 2026-06-23
---

# 1. Purpose
Transforms L1 candidate events into optimal portfolio weights via cross-sectional ranking, regime-conditional shrinkage, and edge-throttled Diagonal Kelly sizing (3-Layer Tiered Hybrid Architecture). L2 Optuna optimizes 9 allocation parameters against scale-invariant Sortino_HAC objective, with deterministic deployment leverage from fit-leg calibration.

# 2. Tiered Hybrid Architecture

**Data Scope**: `full_strategy_maps` spans `[fetch_start, holdout_end]`. Tiered entry scope: `base_scope` (non-empty frame) → strict sub-window admission (warm-up/density/OOS coverage).

**L2 Fold Anchoring**: `build_walk_forward_folds(n_bars=ho_start_idx_l2)` — must pass bars up to `holdout_start`, not `len(aligned)`. Passing full length collapses fold count (regression: 3→1 folds).

**L2 Purge/Embargo**: Config `purge_bars`/`embargo_bars` (auto-derived from `max_holding_bars × purge_safety_mult`) now propagate to folds, preventing fold-boundary label overlap.

**L2 Signal Provenance**: `_signal_batch_fingerprint()` SHA-256 hashes batch boundaries, symbols, registry versions, event fields. `_layer2_experiment_key()` binds study name to window + fingerprint.

**Layer 1: SWF Validation**
- Prequential evidence, readiness gate: fold_cov≥0.80, match_ratio≥0.90, $N_{eff}$≥3.0, fold_ratio≥0.50, LCB>0.

**Layer 2: CS Rank & Diagonal Kelly AWF**

- **Signal Processing**: Absolute CS Ranking → BTC-β Neutralization.
- **Multi-TF Signal Pooling**: L1 per-bar net edge $\mu_i$ (sleeve = (symbol, strategy_id)) → symbol-level pooled edge via **precision-weighted combination**: $\mu_s = \frac{\sum_i c_i \mu_i}{\sum_i c_i}$ where $c_i = \text{quality_weight}_i$. Conviction cap: $c_s = \min(\sum c_i, \kappa \cdot \max c_i)$, $\kappa=1.5$. Guarantees $\min_i \mu_i \le \mu_s \le \max_i \mu_i$ → no mu inflation from multi-TF consensus. Direction conflict ($+\mu$ vs $-\mu$) → auto-netting via signed convex combination.
- **Kelly Sizing**: $w_s \propto f_k \cdot \mu_s / \sigma_s^2$ (friction masked). $\sigma_R = (q_{90} - q_{10})/2.563$. `vol_target=1.0` always active (RC-1 cascade prevention).
- **Edge-Conditional Throttle**: $m_t = \text{clip}((s - \text{floor}) / (\text{ref} - \text{floor}), 0, 1)^\gamma$ applied post-sizing.
- **Active Deployment Controls**: `deploy_cost_safety_mult`, `edge_throttle_min_active_mult`, `risk_budget_floor_ratio` + `risk_budget_max_scale`.
- **Search Space V9 (9 dims)**: `K_RANK`, `REBALANCE_BARS`, `CS_Z_SCORE_THRESHOLD`, `deploy_cost_safety_mult`, `edge_throttle_min_active_mult`, `edge_ref_bps`, `edge_throttle_gamma`, `risk_budget_floor_ratio`, `risk_budget_max_scale`.
- **Objective — Sortino_HAC_unit (Scale-Invariant)**: $J = \text{Sortino\_HAC\_unit} - \lambda_w \cdot \max(0, \tau_{wf} - \text{worst\_fold\_Sortino})$. `growth_lcb` demoted to diagnostic.
- **Phase B — fit-leg Deployment Calibration** (`risk_deployment.py`):
  - C1: fit-leg uses same OOS chain (rank→kelly→throttle→cost→funding), not equal-weight market avg.
  - C2: `calibrate_deployment_leverage(fit_rets_hybrid, l_hard_cap=20.0)` → $L^* = \text{clip}(\min(L_{mdd}, L_{cvar}), 1.0, 20.0)$. `cagr/mdd/cvar` deployed; Sortino/Sharpe/PSR unit-vol.
  - C4: `run_l2_awf(deploy_leverage=L^*)` → `apply_deployment(rets, L^*)`. `exchange_leverage_cap` (default 10×) limits exchange feasibility. `l2_deploy_cvar_margin` knob.
  - Binding ∈ {mdd, cvar, hard_cap, exchange_cap, none}. $L^*$ flows as `l2_params["l2_deploy_leverage"]` SSOT.
- **Vol Scaling**: Bidirectional via `allow_vol_upscale=True`, downscale-only default.
- **Gate Contract**:
  - Optuna feasibility (9-vector): deployment, leak, mdd, cvar, fold_pass_ratio, **recent_fold**, active_blocks, friction, trades.
  - **Friction Gate** (per-bar dimension): $|\bar{g}_s^{pb}| \ge \bar{c}_s^{pb}$ where $\bar{g}_s^{pb} = \text{signed\_gross\_bps\_per\_bar}$ (precision-pooled), $\bar{c}_s^{pb} = \text{expected\_cost\_bps\_per\_bar}$ (including `fixed_cost_safety_mult`). Signals with gross edge less than round-trip cost per bar are unprofitable → excluded from friction pass ratio gate.
  - Promotion (3-stage): Sortino ≥ 1.5 → Sharpe ≥ 0.7 → Calmar ≥ 0.5 + CAGR/MAR/PSR/growth_lcb/uplift.
  - Recent fold gate: latest non-empty deployed fold CAGR > 0 + optional Sharpe floor.
  - `l2_max_exchange_leverage` default 10.0 (`None` = cap disabled).
- **DSR → PSR**: DSR blocker removed. PSR≥0.90 gate (N=1). DSR diagnostic only (L1 FDR + L3 multi-seed handle real multiplicity).
- **OOS Fraction**: `ml_fit_fraction=0.55` + `ml_calibration_fraction=0.15` → OOS 30%.
- **Replay Championing**: `select_layer2_champion` replays all frontier candidates → `argmax(sortino_hybrid, cagr)`.
- **Fold Diagnostics**: `compute_layer2_fold_diagnostics()` → per-fold deployed CAGR/MDD, unit Sharpe, compound pass, selected symbols. `Layer2TrialEvaluation` stores recent_fold metrics + `fold_deployed_cagrs` as Optuna attrs.

**Layer 3: Deployment Parity**
- `run_l3_holdout(deploy_leverage=L^*)` uses same `apply_deployment(rets, L^*)` as L2 scorecard. $L^* \leq 1.0$ → unit path fallback.
- Frozen holdout gate: Sharpe ≥ Sharpe_baseline ∧ MDD ≤ MDD_baseline.

# 3. Optimization Flow

- **Step A (L1 Valid)**: Pipeline targets `l1`. Early exit if blocked.
- **Step B (L2 Prep)**: Causal signal batch from L1.
- **Step C (L2 Study)**: TPESampler maximizes Sortino_HAC_unit (200 trials, V9 9-param). Hard gates only (no soft penalty). Deterministic batch parallel via `ProcessPoolExecutor` (ask→eval→tell sequential). Champion blocked if `blocker_reason != ""`.
- **Step C-Post (Phase B)**: `calibrate_deployment_leverage` post-champion — no extra Optuna trials.
- **Step D (L3 Eval)**: Holdout with deployed leverage.

```mermaid
graph TD
    A[L1 Validated Signals] --> B[CS Rank & β-Neutralize]
    B --> C[Net Edge / Variance Est]
    C --> D[Diagonal Kelly + Throttle]
    D --> E[Vol-Targeting & Regime Caps]
    E --> F[Deployment L* Scaling]
    F --> G[Final Portfolio Weights]
    H[Optuna Flow] -.->|V9 9-param| D
    H -.->|Sortino_HAC_unit| G
    I[fit-leg Calibration] -.->|L* = clip(min(L_mdd, L_cvar))| F
```

# 4. Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `L2_ALLOC_SPACE` | Search space version | V9 (9 dims) |
| `L2_OPTUNA_TRIALS` | Optimization budget | 200 |
| `deploy_cost_safety_mult` | Deployment friction multiplier | config |
| `edge_throttle_min_active_mult` | Min active mult for positive edge | config |
| `risk_budget_floor_ratio` | Min vol ratio for under-deployed books | config |
| `risk_budget_max_scale` | Upper bound for floor scaling | config |
| `l2_deploy_cvar_margin` | CVaR safety margin for calibration | 0.20 |
| `exchange_leverage_cap` | Exchange max leverage | 10.0 |
| `vol_target` | Unit-vol normalization | 1.0 (always on) |
| `l2_min_sortino` | Promotion Sortino gate | 1.5 |
| `l2_min_sharpe_abs` | Promotion Sharpe sanity floor | 0.7 |
| `l2_min_calmar` | Promotion Calmar anchor | 0.5 |

# 5. Edge Cases
- **Survival Censoring**: Negative OOS fold predictions zeroed.
- **Fail-SAFE**: Replay falls back to best diagnostic candidate, preserves blocker reason.
- **NaN Protection**: Tradeable masks sever allocations on corrupt data without crash.
- **Empty fit_rets_hybrid**: OOS proxy fallback + `mdd_margin=0.30` buffer.
- **Recent fold empty**: `recent_fold_passed=None`, constraint = -1.0 (non-blocking). `l2_require_recent_fold_pass=False` disables.
