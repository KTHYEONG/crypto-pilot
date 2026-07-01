---
title: Futures Allocation Architecture
domain: futures.allocation
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/allocation/pipeline.py
  - src/domain/futures/allocation/selection.py
  - src/domain/futures/allocation/gates.py
  - src/domain/futures/allocation/metrics.py
  - src/domain/futures/allocation/simulation.py
  - src/domain/futures/allocation/search_space.py
  - src/domain/futures/allocation/replay.py
  - src/domain/futures/allocation/regime_policy.py
  - src/domain/futures/allocation/deployment.py
  - src/domain/futures/allocation/diagnostics.py
  - src/domain/futures/allocation/signal_batch.py
  - src/domain/futures/allocation/parity.py
  - src/domain/futures/allocation/scoring.py
  - src/application/futures/runner/pipeline.py
  - src/application/futures/runner/config.py
  - src/execution/opt_main_futures.py
change_triggers:
  - src/domain/futures/allocation/pipeline.py
  - src/domain/futures/allocation/selection.py
  - src/domain/futures/allocation/gates.py
  - src/domain/futures/allocation/deployment.py
  - src/application/futures/runner/pipeline.py
dependencies:
  documents:
    - docs/architecture/layer1.md
last_verified: 2026-06-30
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
- **Bucket Routing (regime×family×TF dynamic gating)**: `l2_routing_mode="bucket"` (default) 활성화 시 pooling 전 sleeve 필터링 단계가 동작한다. fit-leg bucket edge는 `compute_market_regime_context()`의 raw 6-state code를 `compress_regime_codes()`로 3-state(`bull`, `bear`, `crisis`)로 축약한 `effective_regime_code_1d`를 기준으로 계산한다. per-fold bucket edge는 $(regime\_code, family, TF)$ triplet의 realized edge $= \overline{side_j \cdot fwd\_ret(sym_j) \cdot 10000 - cost\_bps}$ 이다. OOS bar $t$에서 `regime_now = effective_regime_code_1d[t]`를 조회하고, `edge > l2_bucket_edge_floor_bps`(default 100bps)인 sleeve만 pooling으로 통과한다. 미관측 bucket은 `edge=0` 처리로 자동 제외하며, proof 실패 시 `pooled_edges_by_fold`를 3-state로 복제한 pooled fallback을 사용한다. `l2_regime_policy_mode`는 `filter`(legacy), `observe`, `soft`, `hybrid`를 지원하며, bucket map 위에 causal policy map을 적용해 sleeve를 block/downweight/allow/pool 한다. `soft`는 저신뢰 cell을 유지하되 비중만 줄이고, `hybrid`는 confidence + sign-consistency 조건이 맞을 때만 hard block을 허용한다. min_n 미달 bucket은 family prior로 shrinkage: $e = (1-\lambda) e_{raw} + \lambda e_{family}$, $\lambda=0.3$. Causal bucket reliability layer는 `RegimeBucketReliability`를 통해 fit/cal edge의 sign consistency, cal 최소 관측 수(`l2_regime_cal_min_n=20`), cal lift 하한(`l2_regime_min_cal_lift_bps=8`)을 검증하여 `allow/downweight/pool`을 결정한다. OOS debug bucket metric은 routing, training, selection에 사용하지 않는다. `RegimePolicyEffectSummary`는 per-fold action_ratio, pooled_ratio, mu_abs_ratio를 집계하며 `_policy_effect_is_visible()`이 `l2_regime_max_pooled_ratio_for_effective(0.80)`, `l2_regime_min_action_ratio_for_effective(0.10)`, `l2_regime_min_mu_abs_change(0.03)` 임계로 효과를 진단한다. 장애는 diagnostic 우선이며 pooled fallback과의 차이가 입증된 후에만 hard gate가 된다. Fold-level regime policy override: `build_regime_policy_by_fold`에서 `mean_cal_lift<0 & sign_consistency_ratio<0.6`인 fold는 모든 cell을 `action="allow"`, `reason="pooled_passthrough"`로 강제 전환 (regime demoted to exposure governor, not alpha selector).
- **Kelly Sizing**: $w_s \propto f_k \cdot \mu_s / \sigma_s^2$ (friction masked). $\sigma_R = (q_{90} - q_{10})/2.563$. `vol_target=1.0` always active (RC-1 cascade prevention).
- **Edge-Conditional Throttle**: $m_t = \text{clip}((s - \text{floor}) / (\text{ref} - \text{floor}), 0, 1)^\gamma$ applied post-sizing.
- **Trend-Efficiency Exposure Gate** (env A/B, default off): Multiplies trend/ts_mom sleeve `raw_mu` by `trend_efficiency_gross_mult(ER[t], target=0.35, floor_mult=0.30)` when `L2_TREND_EFFICIENCY_GATE` env is set. ER = Kaufman Efficiency Ratio from `compute_trend_efficiency_1d`. Non-trend sleeves (carry, XS, etc.) unaffected. Per-rebalance `(ER, realized_price)` pairs collected for fold attribution decomposition.
- **Reversal Kill-Switch** (env A/B, default off): `L2_REVERSAL_KILL` env activates pre-loop computation of `compute_reversal_risk_off_1d` (BTC trailing drawdown + negative momentum). Raw risk-off condition requires `persistence_bars` consecutive bars of drawdown+negative-momentum before the causal shift(1) produces the active mask. **Recovery hysteresis** (btc mode, shared state machine with panel mode): after persistent fires, the state stays True until `reversal_recovery_cooldown_bars` consecutive raw-off bars — exit counting uses raw signal (not persistent). At `recovery_cooldown_bars=0` (default), state tracks persistent byte-identically. At each rebalance bar $t$, if $risk\_off\_1d[t-1]$ is `True`, all sleeve `raw_mu` values are multiplied by `reversal_risk_off_floor` (selective hard de-gross, overrides soft cap and crisis_floor). Applied to all archetypes equally (market-wide reversal). Per-bar `(risk_off_flag, realized_price)` pairs collected for fold attribution decomposition via `risk_off_bars`, `risk_off_realized_price`, `risk_on_realized_price` in `Layer2FoldAttribution`. Config defaults are sourced from `RegimeConfig()`; env overrides `L2_REVERSAL_DD_WINDOW`, `L2_REVERSAL_DD_THRESHOLD`, `L2_REVERSAL_MOM_FAST`, `L2_REVERSAL_MOM_SLOW`, `L2_REVERSAL_RISK_OFF_FLOOR`, `L2_REVERSAL_PERSISTENCE_BARS`, `L2_REVERSAL_RECOVERY_COOLDOWN` via `_reversal_config_from_env()`.
- **Active Deployment Controls**: `deploy_cost_safety_mult`, `edge_throttle_min_active_mult`, `risk_budget_floor_ratio` + `risk_budget_max_scale`.
- **Search Space (9 dims, versionless)**: `L2_SEARCH_SPACE` in `allocation/search_space.py`. Parameters: `K_RANK` (low=4, churn 방지), `REBALANCE_BARS`, `CS_Z_SCORE_THRESHOLD`, `deploy_cost_safety_mult`, `edge_throttle_min_active_mult`, `edge_ref_bps`, `edge_throttle_gamma`, `risk_budget_floor_ratio`, `risk_budget_max_scale`.
- **Objective — Sortino_HAC_unit (Scale-Invariant)**: $J = \text{Sortino\_HAC\_unit} - \lambda_w \cdot \max(0, \tau_{wf} - \text{worst\_fold\_Sortino}) - \lambda_t \cdot \text{mean\_turnover}$. `growth_lcb` demoted to diagnostic. Turnover penalty $\lambda_t = 0$ default (off) — backtest-safe, enable via `l2_turnover_penalty_weight`.
- **Phase B — fit-leg Deployment Calibration** (`risk_deployment.py`):
  - C1: fit-leg uses same OOS chain (rank→kelly→throttle→cost→funding), not equal-weight market avg.
  - C2: `calibrate_deployment_leverage(fit_rets_hybrid, oos_rets, l_hard_cap=20.0, oos_budget_blend=0.5, oos_floor_cap=4.0)` → $(L^*, \text{binding}, \text{cross\_valid\_MDD})$. $L^* = \text{clip}(\min(L_{mdd}, L_{cvar}), 1.0, 20.0)$ (binding≠oos_blend). `oos_budget_blend` blends OOS-based floor into L* via convex combination, capped at `oos_floor_cap`. Binding `"oos_blend"` replaces hardcoded `min(2.0,…)`. `oos_rets` 옵션 제공 시 OOS MDD 크로스 검증을 세 번째 반환값으로 전달. `cagr/mdd/cvar` deployed; Sortino/Sharpe/PSR unit-vol.
   - C4: `run_l2_awf(deploy_leverage=L^*)` → `evaluate_l2_trial(deploy_leverage_override=L^*)` — 단일 평가 SSOT. `apply_deployment(rets, L^*)` 지표는 `Layer2TrialEvaluation`에서 직접 사용, `Layer2Result`는 어댑터(`_layer2_result_from_trial_eval`)를 통해 1:1 복사. `run_l2_awf`는 배포 전용 필드(turnover/weights/gate)만 추가 계산. `exchange_leverage_cap` (default 10×) limits exchange feasibility. `l2_deploy_cvar_margin` knob.
  - Binding ∈ {mdd, cvar, oos_blend, hard_cap, exchange_cap, none}. $L^*$ flows as `l2_params["l2_deploy_leverage"]` SSOT and stored in `Layer2Result.deploy_leverage` for parity tracking.
- **Vol Scaling**: Bidirectional via `allow_vol_upscale=True`, downscale-only default.
- **Gate Contract**:
   - Optuna feasibility (9-vector): deployment, leak, mdd, cvar, fold_pass_ratio, **recent_fold**, active_blocks, friction, trades.
   - **Friction Gate** (per-bar dimension): $|\bar{g}_s^{pb}| \ge \bar{c}_s^{pb}$ where $\bar{g}_s^{pb} = \text{signed\_gross\_bps\_per\_bar}$ (precision-pooled), $\bar{c}_s^{pb} = \text{expected\_cost\_bps\_per\_bar}$ (including `fixed_cost_safety_mult`). Signals with gross edge less than round-trip cost per bar are unprofitable → excluded from friction pass ratio gate.
   - **Cost Drag Gate** (promotion 17th blocker): $\text{cost\_drag} = \min(\frac{\sum \text{realized\_cost}}{\max(\sum |\text{realized\_price}|, \varepsilon)}, 100.0) > \text{l2\_max\_cost\_drag\_ratio}$ → BLOCK. Denominator uses `abs(realized_price)` to prevent long/short cancellation that drives cost_drag to infinity. Upper cap at 100.0 prevents degenerate degenerate books from blocking all trials. `l2_max_cost_drag_ratio` default 0.60.
   - Promotion (3-stage): Sortino ≥ 1.5 → Sharpe ≥ 0.7 → Calmar ≥ 0.5 + CAGR/MAR/PSR/growth_lcb(vol-matched via `std_hybrid`/`std_baseline` args) + uplift + cost_drag + worst_fold_cagr(`l2_min_worst_fold_cagr=-0.05`). `block_delta` demoted to diagnostic-only (`_growth_lcb_vol_matched_baseline` helper, no promotion blocking).
   - Champion Store Synthetic Crash Gate: After study success (`blocker_reason==""` and `best_evaluation` exists), `synthetic_crash_defense_verdict()` replays a synthetic ATH→decline path through `compute_reversal_risk_off_1d` with the actual deployment config (sourced via `_reversal_config_from_env()`). If the detector does not fire (`crash_fires=False`), the champion store update is skipped with a WARNING (`crash_defense_not_firing`). This gate is independent of `L2_REVERSAL_KILL` env — it detects code regression in the reversal-kill detector itself, not runtime activation status.
   - Recent fold gate: latest non-empty deployed fold CAGR > 0 + optional Sharpe floor.
   - `l2_max_exchange_leverage` default 10.0 (`None` = cap disabled).
   - Entry cooldown: `_resolve_tradeable_mask` → `apply_entry_cooldown` (causal, backward-only)로 새로 활성화된 심볼을 `l2_entry_cooldown_bars=12` bar 동안 지연 진입. `entry_block_spike` 경고는 `Layer2DeployableScore.entry_spike_penalty`로 candidate 순위에 패널티 부과.
- **DSR → PSR**: DSR blocker removed. PSR≥0.90 gate (N=1). DSR diagnostic only (L1 FDR + L3 multi-seed handle real multiplicity).
- **OOS Fraction**: `ml_fit_fraction=0.55` + `ml_calibration_fraction=0.15` → OOS 30%.
- **Replay Championing**: `select_layer2_champion` replays frontier candidates (gate-passed top 8, or fallback up to `l2_replay_max_fallbacks` default 24) via `ThreadPoolExecutor(max_workers=4)` for parallel evaluation. Gate evaluation is single pass (no pre-gate/final-gate dedup). Gate-pass champion selection: `argmax(sortino_hybrid, cagr, -trial.number)` — trial number tiebreaker ensures ThreadPool non-determinism safety. No gate-passed candidate exists 시 `Layer2DeployableScore` 공식으로 blocked fallback ranking: `score = cagr + 0.10 * min(sortino, 3) + 0.05 * min(calmar, 3) - 0.50 * max(0, -worst_fold_cagr) - 0.25 * max(0, threshold - positive_block_delta_ratio) - 0.20 * cost_drag - entry_spike_penalty`. Final run replay parity는 `cagr/mdd/fold_pass/trade_count` tolerance 이내로 검증 (`_assert_selection_replay_parity`). Parity divergence 시 `gate_passed=False, blocker_reason="parity_divergence"` hard-gate. SSOT 단일 평가 경로(evaluate_l2_trial) 도입 후 selection↔deploy CAGR은 구조적 동일 — parity_divergence는 회귀 탐지기로 유지.
- **Annualization TF SSOT (B1/B2)**: L2 master tf is resolved once via `_resolve_l2_master_tf(cfg, {}, probe_manifest)` in the runner and passed consistently to the L2 study, final deploy pipeline, and reversal economic replay. `Layer2Result.master_tf` carries the annualization tf used for deployment metrics. Post-pipeline, a defensive SSOT assert compares the runner-resolved `l2_master_tf` against `l2_final.master_tf`; mismatch triggers `gate_passed=False, blocker_reason="annualization_tf_mismatch"`, preventing silent CAGR/Sharpe ×2/×√2 divergence between champion selection and deployed metrics. Enriched cache propagation: final `run_l2_awf` receives `l2_study_result.sim_cache` (regime routing plan, bucket cache), eliminating cache mismatch between selection and final deployment. **Evaluation memoization**: `evaluate_l2_trial_cached` wraps the SSOT evaluator with a key `(id(cache), cfg_ch, id(signal_batch), id(caps), tf, deploy_lev)` — Optuna study loop bypasses cache (unique config per trial → hit=0), but selection replay + deployment share identical inputs → 2→1 call dedup. `Layer2StudyResult.eval_memo` propagates memo dict from selection to `run_tiered_pipeline`. `[L2-MEMO-PARITY]` DEBUG log verifies hit/miss parity. **L2 Reversal Economic Replay**: `L2_REVERSAL_REPLAY` env triggers post-parity evaluation of 8 reversal variants (`baseline_off`, `legacy_006_p1`, `balanced_010_p2`, `balanced_010_p3`, `current_012_p3`, `legacy_006_p1_cd4`, `legacy_006_p1_cd8`, `current_012_p3_cd8`) via `_run_l2_reversal_economic_replay()`. Each variant calls `evaluate_l2_trial` under `_temporary_reversal_env()` which now also sets `L2_REVERSAL_RECOVERY_COOLDOWN`. `L2ReversalReplayVariant` field `recovery_cooldown_bars` (default 0). `L2ReversalReplayResult` fields `dd_threshold`, `persistence_bars`, `recovery_cooldown_bars` for CSV traceability. Adoption verdict uses **dynamic stress fold identification**: `stress_idx = argmax(baseline fold MDDs)`. If `stress_mdd >= 0.15`, the verdict requires legacy improvement > 0, candidate >= 70% of legacy, and non-stress fold damage ≤ 1pp on the stress fold; if no fold exceeds `stress_mdd_threshold`, stress checks are skipped (spurious `legacy_no_improvement` prevention). Aggregate CAGR must exceed both baseline and legacy. Selection parity required. CSV output at `docs/results/l2_reversal_replay.csv`.
- **Optuna Parallel Execution**: `ProcessPoolExecutor(fork)` with batch `future.result()` loop. `_GLOBAL_L2_CTX` module-level global for fork-safe ctx sharing (numpy array CoW). OOM-guarded worker count: `max_workers = max(1, min(batch_size, cpu_cores, avail_gb/1.2))`. Memory < 1.2GB forces sequential fallback.
- **AWF Fold Reuse**: `TieredContext.awf_folds` pre-computed once before study and reused by `_resolve_l2_signal_batch_and_folds` across all trial evaluations, eliminating redundant `build_walk_forward_folds` calls.
- **Fold Diagnostics**: `compute_layer2_fold_diagnostics()` → per-fold deployed CAGR/MDD, unit Sharpe, compound pass, selected symbols. `Layer2TrialEvaluation` stores recent_fold metrics + `fold_deployed_cagrs`, `fold_deployed_mdds`, `fold_attributions` as Optuna attrs. Scorecard (`format_layer2_table`) carries evaluation window bottleneck verdict via `evaluation_window_bottleneck_verdict()` — checks whether any fold has MDD ≥ stress threshold (default 15%) and non-positive CAGR (crisis-caliber drawdown). If no such fold exists, the scorecard appends a `[WINDOW] NO-CRISIS-WINDOW` banner warning the operator against using this PASS as promotion evidence.
- **Regime Reliability Multiplier (A/B=off)**: `compute_regime_reliability_multiplier` reads trailing fold performance (`_bear_edge_by_fold` window length via `l2_regime_reliability_window`) to scale `bear_gross_cap` down during sustained negative bear edge. The multiplier uses a sign-first piecewise-linear ramp: mean trailing per-bar edge ≥ `pos_edge_at_full_bps` → 1.0, ≤ `neg_edge_at_floor_bps` → `floor`, otherwise interpolated. The multiplier is folded into `apply_regime_risk_cap` at the `bear_gross_cap` argument before weight clipping. In-window data is sampled before the current fold starts (look-ahead free). `_bear_edge_by_fold` accumulates unconditionally but only affects caps when `l2_regime_reliability_enabled=True`. `[L2-REGIME-RELIABILITY]` DEBUG log reports per-fold bear edge and multiplier.
- **Regime Routing Diagnostics**: `RegimeRoutingDiagnostics` records `active_state_count`, `conditioning_path`, `mean_lift_bps`, `nw_tstat`, `fold_pass_ratio`, `bucket_hit_pct_by_fold`, `js_divergence_by_fold`, `policy_diagnostics`, and `debug_diagnostics`. `RegimeRoutingPlan` carries `policy_by_fold` for fold-local causal policy application. L2 logs `"[REGIME]"` as a compact 3-state table summary and `"[REGIME-L2]"` as the routing verdict. When DEBUG is enabled, `debug_diagnostics` exposes pooled, effective_3, and raw_6 granularity stats plus the worst regime×family×TF cells and realized selected-regime return tables, and `policy_diagnostics` exposes `policy_mode`, `global_reliable`, `n_unstable`, `n_hard_block_eligible`, `sign_consistency_ratio`, and action counts. `RegimePolicyEffectSummary`는 per-fold action_ratio, pooled_ratio, block_ratio, mu_abs_ratio, quality_weight_ratio, edge_abs_ratio를 집계하며 `[L2-REGIME-POLICY-SUMMARY]` DEBUG 로그에 출력된다. Production routing still follows `effective_regime_code_1d`; raw 6-state output remains diagnostics-only.
- **Attribution (Always-On)**: `fold_attributions: tuple[Layer2FoldAttribution, ...]` returned by every `_run_awf_simulation` call regardless of `l2_diag_attribution_enabled` flag. Per-fold `realized_price`/`realized_funding`/`realized_cost` are accumulated unconditionally (O(N) per-bar dot). Whipsaw decomposition: `realized_price_low_er` = Σ price delta where trailing ER < target, `trend_efficiency_corr` = corr(ER, price delta), `mean_trend_efficiency` = fold-mean ER. Alpha gap, sleeve samples, netting stats are diag-gated (`_diag`). Cost drag gate consumes attribution output.
- **Edge-Survival Waterfall (DEBUG, diag-gated)**: `Layer2EdgeWaterfall` decomposes L1 expected edge → realized PnL into 4 stages per fold: `admitted_contrib` (equal-weight), `weighted_contrib` (Kelly/throttle/pre-cap), `capped_contrib` (regime + capacity clip), `realized_contrib` (post-friction). Stage loss terms `loss_weighting`, `loss_capping`, `loss_friction` isolate the dominant edge-erosion stage. Accumulators (`_attr_weighted`, `_attr_admitted`, `_cap_binding_bars`, `_sleeves_admitted_sum`) are scalars per fold. `w_precap = w.copy()` is captured before `apply_regime_risk_cap`. All stages share the same cumulative unit (Σ holding·dot(w_stage, mu) × 1e4 bps). `[L2-EDGE-WATERFALL]` DEBUG log emitted at fold end.
- **L* Inflation Diagnostics (Always-On, DEBUG)**: `[L2-CALIB-CV]` 로그는 `calibrate_deployment_leverage` 내에서 fit-leg과 OOS 간 MDD_vol1 비율(MDD_ratio)을 계산하여 L* inflation을 정량화. `[L2-TRIAL-DIAG]` / `[L2-FINAL-DIAG]` 로그는 각각 trial 평가 및 최종 scorecard 경로에서 fit_CAGR_vol1/OOS_CAGR_vol1/fit_MDD_vol1/OOS_MDD_vol1을 분리 출력하여 alpha decay 여부 진단. `[L2-FIT-DIAG]` 로그는 `_run_awf_simulation`의 per-fold fit-leg 수익률에서 fit_CAGR, fit_MDD, fit_ann_vol, fit_sharpe를 계산하여 vol-targeting(실현 연율 변동성) 무결성 확인. `[L2-OOS-CAP]` 로그는 `calibrate_deployment_leverage`가 반환한 `cross_valid_MDD`로 OOS RiskUtil을 계산하여 L*의 OOS 과배치 여부 진단. `[L2-REPLAY]` 및 `[L2-REPLAY-GATE]`는 champion selection replay 시점의 stored vs replay metric 차이를 기록. `[L2-GATE]` 로그는 promotion gate의 모든 constraint별 actual vs threshold 비교를 한 줄에 출력. `[AWF-SIM-FP]` 로그는 `_run_awf_simulation` 반환 직전 rets fingerprint, fold OOS bars, config fingerprint, 객체 ID를 DEBUG 레벨로 출력하여 champion-eval vs final-deploy 경로 간 sim 분기점을 격리한다. `[AWF-SIM-FP2]` 로그는 cache 내용해시(cache_ch), 전체 config 해시(cfg_ch), caps 해시(caps_ch), per-fold rets fingerprint, deploy_lev를 추가로 출력하여 객체 identity/CUDA repr truncate 사각지대를 보완한다.

**Layer 3: Deployment Parity**
- `run_l3_holdout(deploy_leverage=L^*)` uses same `apply_deployment(rets, L^*)` as L2 scorecard. $L^* \leq 1.0$ → unit path fallback.
- Frozen holdout gate: Sharpe ≥ Sharpe_baseline ∧ MDD ≤ MDD_baseline.

# 3. Optimization Flow

- **Step A (L1 Valid)**: Pipeline targets `l1`. Early exit if blocked.
- **Step B (L2 Prep)**: Causal signal batch from L1.
- **Step C (L2 Study)**: TPESampler maximizes Sortino_HAC_unit (200 trials, V9 9-param). Hard gates only (no soft penalty). Deterministic batch parallel via `ProcessPoolExecutor` (ask→eval→tell sequential) uses fork-safe global context with `L2_OPTUNA_BATCH_SIZE=2` by default, which dynamically falls back to 1 (sequential) if available memory drops below 3.0 GB to prevent WSL OOM. Champion blocked if `blocker_reason != ""`. When study succeeds (blocker empty + best_evaluation exists), champion store update is additionally gated by synthetic crash defense verdict — the update proceeds only when `_champion_promotion_allowed()` returns `(True, "")`. The helper performs two checks: (1) study validity (`blocker_reason==""` and `has_evaluation=True`) and (2) synthetic crash defense firing (`crash_fires=True`).
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
| `L2_SEARCH_SPACE` | Search space (versionless) | 9 dims in `allocation/search_space.py` |
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
| `l2_bucket_edge_floor_bps` | Bucket edge pass threshold | 50.0 |
| `l2_regime_policy_mode` | Regime policy mode: filter / observe / soft / hybrid | "hybrid" |
| `l2_regime_min_policy_confidence` | Minimum confidence for actionable policy | 0.55 |
| `l2_regime_hard_block_enabled` | Allow hard block only in hybrid mode with sign-consistency gate | False |
| `l2_regime_block_min_confidence` | Minimum confidence for hard block eligibility | 0.80 |
| `l2_regime_require_sign_consistency` | Require fit/cal sign consistency for hard block | True |
| `l2_regime_risk_cap_enabled` | Apply regime-state gross caps after weight composition | True |
| `l2_regime_reliability_enabled` | Dynamic bear cap downweight via trailing bear edge reliability | False |
| `l2_regime_reliability_window` | Trailing fold count for bear edge reliability estimation | 2 |
| `l2_regime_reliability_floor` | Minimum reliability multiplier floor for bear cap | 0.2 |
| `l2_regime_pooled_is_passthrough` | Treat pooled action as allow (passthrough) | True |
| `l2_regime_min_fit_n_floor` | Min n_fit floor for B-2 insufficient_fit_but_good_cal | 5 |
| `l2_regime_require_fit_n_for_downweight` | Require n_fit >= min_fit_n_floor for B-3 downweight preserve | False |
| `l2_min_sortino` | Promotion Sortino gate | 1.5 |
| `l2_min_sharpe_abs` | Promotion Sharpe sanity floor | 0.7 |
| `l2_min_calmar` | Promotion Calmar anchor | 0.5 |
| `l2_min_worst_fold_cagr` | Worst-fold CAGR gate threshold | -0.05 |
| `l2_min_positive_block_delta_ratio` | Block delta diagnostic threshold | 0.45 |
| `l2_regime_fold_override_enabled` | Fold-level regime policy override | True |
| `l2_cs_amp_enabled` | CS score amplification | True |
| `l2_cs_amp_mode` | Amplification mode: median_excess / power / tanh | "power" |
| `l2_cs_amp_alpha` | Amplification strength | 2.0 |
| `l2_cs_amp_power` | Power mode exponent | 2.0 |

# 5. Edge Cases
- **Bucket Routing Look-ahead**: `compute_bucket_realized_edges` uses `fit_end=oos_start`, forward return at `t=fit_end-1` reads `close[oos_start]` — allowed (fit-leg only, OOS close price available in full dataset).
- **Bucket min_n shrinkage**: Bucket with count < min_n → family prior shrinkage prevents degenerate edge from single-event bucket.
- **Bucket unknown key**: `bucket_edges.get(key, 0.0)` → edge=0 < floor → auto excluded.
- **Regime stale**: regime_code_1d covers entire bar range; OOS bar with no regime uses `regime=0` fallback.
- **Low-confidence policy**: `soft` mode preserves routing continuity via downweighting instead of hard exclusion.
- **Hybrid hard-block gate**: `hard_block_enabled` requires confidence and sign consistency; otherwise the cell is only downweighted.
- **Survival Censoring**: Negative OOS fold predictions zeroed.
- **Fail-SAFE**: Replay falls back to best diagnostic candidate, preserves blocker reason.
- **NaN Protection**: Tradeable masks sever allocations on corrupt data without crash.
- **Empty fit_rets_hybrid**: OOS proxy fallback + `mdd_margin=0.30` buffer.
- **Recent fold empty**: `recent_fold_passed=None`, constraint = -1.0 (non-blocking). `l2_require_recent_fold_pass=False` disables.
