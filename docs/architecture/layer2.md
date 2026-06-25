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
last_verified: 2026-06-25
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
- **Bucket Routing (regime×family×TF dynamic gating)**: `l2_routing_mode="bucket"` (default) 활성화 시 pooling 전 sleeve 필터링 단계 추가. fit-leg에서 `compute_market_regime_context(code_1d: 6-state int8, BTC price regime)` 기반 per-fold bucket edge 계산 → $(regime\_code, family, TF)$ triplet의 realized edge $= \overline{side_j \cdot fwd\_ret(sym_j) \cdot 10000 - cost\_bps}$. OOS bar $t$에서 regime_now = `code_1d[t]` → 각 sleeve의 bucket edge lookup → `edge > l2_bucket_edge_floor_bps` (default 100bps)인 sleeve만 pooling으로 통과. 미관측 bucket은 `edge=0` 처리 → 자동 제외. min_n 미달 bucket은 family prior로 shrinkage: $e = (1-\lambda) e_{raw} + \lambda e_{family}$, $\lambda=0.3$.
- **Kelly Sizing**: $w_s \propto f_k \cdot \mu_s / \sigma_s^2$ (friction masked). $\sigma_R = (q_{90} - q_{10})/2.563$. `vol_target=1.0` always active (RC-1 cascade prevention).
- **Edge-Conditional Throttle**: $m_t = \text{clip}((s - \text{floor}) / (\text{ref} - \text{floor}), 0, 1)^\gamma$ applied post-sizing.
- **Active Deployment Controls**: `deploy_cost_safety_mult`, `edge_throttle_min_active_mult`, `risk_budget_floor_ratio` + `risk_budget_max_scale`.
- **Search Space V9 (9 dims)**: `K_RANK` (low=4, churn 방지), `REBALANCE_BARS`, `CS_Z_SCORE_THRESHOLD`, `deploy_cost_safety_mult`, `edge_throttle_min_active_mult`, `edge_ref_bps`, `edge_throttle_gamma`, `risk_budget_floor_ratio`, `risk_budget_max_scale`.
- **Objective — Sortino_HAC_unit (Scale-Invariant)**: $J = \text{Sortino\_HAC\_unit} - \lambda_w \cdot \max(0, \tau_{wf} - \text{worst\_fold\_Sortino}) - \lambda_t \cdot \text{mean\_turnover}$. `growth_lcb` demoted to diagnostic. Turnover penalty $\lambda_t = 0$ default (off) — backtest-safe, enable via `l2_turnover_penalty_weight`.
- **Phase B — fit-leg Deployment Calibration** (`risk_deployment.py`):
  - C1: fit-leg uses same OOS chain (rank→kelly→throttle→cost→funding), not equal-weight market avg.
  - C2: `calibrate_deployment_leverage(fit_rets_hybrid, oos_rets, l_hard_cap=20.0)` → $(L^*, \text{binding}, \text{cross\_valid\_MDD})$. $L^* = \text{clip}(\min(L_{mdd}, L_{cvar}), 1.0, 20.0)$. `oos_rets` 옵션 제공 시 OOS MDD 크로스 검증을 세 번째 반환값으로 전달. `cagr/mdd/cvar` deployed; Sortino/Sharpe/PSR unit-vol.
  - C4: `run_l2_awf(deploy_leverage=L^*)` → `apply_deployment(rets, L^*)`. `exchange_leverage_cap` (default 10×) limits exchange feasibility. `l2_deploy_cvar_margin` knob.
  - Binding ∈ {mdd, cvar, hard_cap, exchange_cap, none}. $L^*$ flows as `l2_params["l2_deploy_leverage"]` SSOT.
- **Vol Scaling**: Bidirectional via `allow_vol_upscale=True`, downscale-only default.
- **Gate Contract**:
  - Optuna feasibility (9-vector): deployment, leak, mdd, cvar, fold_pass_ratio, **recent_fold**, active_blocks, friction, trades.
  - **Friction Gate** (per-bar dimension): $|\bar{g}_s^{pb}| \ge \bar{c}_s^{pb}$ where $\bar{g}_s^{pb} = \text{signed\_gross\_bps\_per\_bar}$ (precision-pooled), $\bar{c}_s^{pb} = \text{expected\_cost\_bps\_per\_bar}$ (including `fixed_cost_safety_mult`). Signals with gross edge less than round-trip cost per bar are unprofitable → excluded from friction pass ratio gate.
  - **Cost Drag Gate** (promotion 17th blocker): $\text{cost\_drag} = \min(\frac{\sum \text{realized\_cost}}{\max(\sum |\text{realized\_price}|, \varepsilon)}, 100.0) > \text{l2\_max\_cost\_drag\_ratio}$ → BLOCK. Denominator uses `abs(realized_price)` to prevent long/short cancellation that drives cost_drag to infinity. Upper cap at 100.0 prevents degenerate degenerate books from blocking all trials. `l2_max_cost_drag_ratio` default 0.60.
  - Promotion (3-stage): Sortino ≥ 1.5 → Sharpe ≥ 0.7 → Calmar ≥ 0.5 + CAGR/MAR/PSR/growth_lcb/uplift + cost_drag.
  - Recent fold gate: latest non-empty deployed fold CAGR > 0 + optional Sharpe floor.
  - `l2_max_exchange_leverage` default 10.0 (`None` = cap disabled).
- **DSR → PSR**: DSR blocker removed. PSR≥0.90 gate (N=1). DSR diagnostic only (L1 FDR + L3 multi-seed handle real multiplicity).
- **OOS Fraction**: `ml_fit_fraction=0.55` + `ml_calibration_fraction=0.15` → OOS 30%.
- **Replay Championing**: `select_layer2_champion` replays all frontier candidates → `argmax(sortino_hybrid, cagr)`.
- **Fold Diagnostics**: `compute_layer2_fold_diagnostics()` → per-fold deployed CAGR/MDD, unit Sharpe, compound pass, selected symbols. `Layer2TrialEvaluation` stores recent_fold metrics + `fold_deployed_cagrs` as Optuna attrs.
- **Attribution (Always-On)**: `fold_attributions: tuple[Layer2FoldAttribution, ...]` returned by every `_run_awf_simulation` call regardless of `l2_diag_attribution_enabled` flag. Per-fold `realized_price`/`realized_funding`/`realized_cost` are accumulated unconditionally (O(N) per-bar dot). Alpha gap, sleeve samples, netting stats are diag-gated (`_diag`). Cost drag gate consumes attribution output.
- **L* Inflation Diagnostics (Always-On, DEBUG)**: `[L2-CALIB-CV]` 로그는 `calibrate_deployment_leverage` 내에서 fit-leg과 OOS 간 MDD_vol1 비율(MDD_ratio)을 계산하여 L* inflation을 정량화. `[L2-TRIAL-DIAG]` / `[L2-FINAL-DIAG]` 로그는 각각 trial 평가 및 최종 scorecard 경로에서 fit_CAGR_vol1/OOS_CAGR_vol1/fit_MDD_vol1/OOS_MDD_vol1을 분리 출력하여 alpha decay 여부 진단. `[L2-FIT-DIAG]` 로그는 `_run_awf_simulation`의 per-fold fit-leg 수익률에서 fit_CAGR, fit_MDD, fit_ann_vol, fit_sharpe를 계산하여 vol-targeting(실현 연율 변동성) 무결성 확인. `[L2-OOS-CAP]` 로그는 `calibrate_deployment_leverage`가 반환한 `cross_valid_MDD`로 OOS RiskUtil을 계산하여 L*의 OOS 과배치 여부 진단. `[L2-REPLAY]` 및 `[L2-REPLAY-GATE]`는 champion selection replay 시점의 stored vs replay metric 차이를 기록. `[L2-GATE]` 로그는 promotion gate의 모든 constraint별 actual vs threshold 비교를 한 줄에 출력.

**Layer 3: Deployment Parity**
- `run_l3_holdout(deploy_leverage=L^*)` uses same `apply_deployment(rets, L^*)` as L2 scorecard. $L^* \leq 1.0$ → unit path fallback.
- Frozen holdout gate: Sharpe ≥ Sharpe_baseline ∧ MDD ≤ MDD_baseline.

# 3. Optimization Flow

- **Step A (L1 Valid)**: Pipeline targets `l1`. Early exit if blocked.
- **Step B (L2 Prep)**: Causal signal batch from L1.
- **Step C (L2 Study)**: TPESampler maximizes Sortino_HAC_unit (200 trials, V9 9-param). Hard gates only (no soft penalty). Deterministic batch parallel via `ProcessPoolExecutor` (ask→eval→tell sequential) uses fork-safe global context with `L2_OPTUNA_BATCH_SIZE=2` by default, which dynamically falls back to 1 (sequential) if available memory drops below 3.0 GB to prevent WSL OOM. Champion blocked if `blocker_reason != ""`.
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
| `l2_routing_mode` | Sleeve routing mode: pool (legacy) or bucket (regime×family×TF) | "bucket" |
| `l2_bucket_cost_bps` | Bucket edge cost deduction | 6.0 |
| `l2_bucket_min_n` | Min bucket events before shrinkage | 30 |
| `l2_bucket_shrinkage` | Raw→family prior shrinkage rate | 0.3 |
| `l2_bucket_edge_floor_bps` | Bucket edge pass threshold | 100.0 |
| `l2_min_sortino` | Promotion Sortino gate | 1.5 |
| `l2_min_sharpe_abs` | Promotion Sharpe sanity floor | 0.7 |
| `l2_min_calmar` | Promotion Calmar anchor | 0.5 |

# 5. Edge Cases
- **Bucket Routing Look-ahead**: `compute_bucket_realized_edges` uses `fit_end=oos_start`, forward return at `t=fit_end-1` reads `close[oos_start]` — allowed (fit-leg only, OOS close price available in full dataset).
- **Bucket min_n shrinkage**: Bucket with count < min_n → family prior shrinkage prevents degenerate edge from single-event bucket.
- **Bucket unknown key**: `bucket_edges.get(key, 0.0)` → edge=0 < floor → auto excluded.
- **Regime stale**: regime_code_1d covers entire bar range; OOS bar with no regime uses `regime=0` fallback.
- **Survival Censoring**: Negative OOS fold predictions zeroed.
- **Fail-SAFE**: Replay falls back to best diagnostic candidate, preserves blocker reason.
- **NaN Protection**: Tradeable masks sever allocations on corrupt data without crash.
- **Empty fit_rets_hybrid**: OOS proxy fallback + `mdd_margin=0.30` buffer.
- **Recent fold empty**: `recent_fold_passed=None`, constraint = -1.0 (non-blocking). `l2_require_recent_fold_pass=False` disables.
