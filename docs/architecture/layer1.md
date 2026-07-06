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
  - src/domain/futures/strategy/tiered_workflow/pipeline.py
  - src/domain/futures/optimization/metrics.py
  - src/domain/futures/strategy_runtime/bridge.py
  - src/domain/futures/alpha_foundry/contracts.py
  - src/domain/futures/alpha_foundry/recipes.py
  - src/domain/futures/alpha_foundry/cheap_gate.py
  - src/domain/futures/alpha_foundry/diversity.py
  - src/domain/futures/alpha_foundry/budget.py
  - src/domain/futures/alpha_foundry/posterior.py
   - src/domain/futures/alpha_foundry/pipeline.py
   - src/domain/futures/alpha_foundry/bridge_helpers.py
change_triggers:
  - src/domain/futures/signals/rules.py
  - src/domain/futures/signals/diagnostics.py
  - src/domain/futures/strategy_runtime/bridge.py
  - src/domain/futures/alpha_foundry/contracts.py
  - src/domain/futures/alpha_foundry/recipes.py
  - src/domain/futures/alpha_foundry/cheap_gate.py
  - src/domain/futures/alpha_foundry/diversity.py
  - src/domain/futures/alpha_foundry/budget.py
  - src/domain/futures/alpha_foundry/posterior.py
   - src/domain/futures/alpha_foundry/pipeline.py
   - src/domain/futures/alpha_foundry/bridge_helpers.py
dependencies:
  documents:
    - docs/architecture/regime.md
last_verified: 2026-07-06
---

# 1. Purpose
벡터화된 Rule Panels에 시장 Regime Context를 결합하고, L1 Breakeven Hard Gate 및 Multiplicity Control을 적용하여 유효한 Candidate Events를 생성한다. 4개의 Timeframe(4h/6h/8h/12h)에 걸쳐 Walk-forward Validation을 지원하는 Prequential Evidence Snapshots를 관리한다.

# 2. Core Logic & Math

### Signal Gating Sequence
1. **Vectorization**:
   - $S_{t} = f(\text{Data}_{1..t})$
   - Sparse Trigger: $E_{t} = 1 \text{ if } (S_{t} \neq 0 \land S_{t-1} == 0) \text{ else } 0$ (Causal)
2. **Regime Gating**:
   - `mean_rev` Archetype: `("bull_quiet", "bear_quiet", "transition")` 허용
   - `beta_neut` Archetype: `("bull_quiet",)` 허용
3. **L1 Breakeven Hard Gate**:
   - $\frac{1}{N} \sum (\text{Edge}_{i}) > 0 \land t_{\text{stat}} \geq \text{min\_rule\_ir\_t}$
4. **Profit Floor**:
   - $\mu_{\text{OOS}} \geq \text{min\_variant\_oos\_profit\_bps}$
5. **Regime-Cell Admission**:
   - Bayesian Posterior: $P(\mu > \delta | \text{data}) \ge p_{\text{admit\_min}}$
   - Newey-West variance와 cross-cell $\tau^2$ shrinkage 적용.
6. **Multiplicity Controls**:
   - **BH-FDR**: $q \le \text{l1\_pair\_fdr\_alpha}~(0.10)$
   - **SPA (Single Predictive Ability)**: Fail-closed circular bootstrap 검정.

### Ensemble Shrinkage
- Archetype cell mean과 variant-level prior에 대해 Empirical-Bayes James-Stein shrinkage 적용.
- Bayesian Prior: $\hat{x}_a = w_{prior} \cdot \bar{x}_a + (1 - w_{prior}) \cdot \mu_{prior}$
- $w_{prior} = n_{eff} / (n_{eff} + n_{prior})$

### Cross-Sectional Alpha Families (xs_alpha archetype)
- 4대 Family: `xs_momentum`, `xs_carry`, `xs_flow`, `xs_oi_skew`
- Per-bar rank score 변환을 통한 beta-neutral 구성 (Regime Gating 면제).
- `xs_momentum`: Rolling beta residual return (L=12/48)
- `xs_carry`: Funding rate z-score 역수 (`-funding_z_96/168`)
- `xs_flow`: Order flow imbalance z-score (`flow_z_24`)
- `xs_oi_skew`: Open Interest 빌드업과 LSR 스큐 결합 (`-(oi_build_z_42 * sign(lsr_log_z_42))`)

### Alpha Foundry Core [ADR_20260706_ALPHA_FOUNDRY_SYNC]
- `AlphaRecipe`: `recipe_id`, `family`, `variant`, `timeframe`, `archetype`, `indicator_params`, `side_rule_id`, `exit_policy_id`, `required_fields`, `causal_lag_bars`, `max_turnover_per_year`.
- `CheapGateEvidence`: cost-adjusted event summary with `gate_passed`, `reject_reasons`, `block_lcb_bps`, `rank_ic`, `turnover_per_year`.
- `L1VerificationUnit`: fold-bounded verification unit with `prior_mu_bps`, `prior_sigma_bps`, `allocated_fold_budget`, `early_stop_state`.
- `L1PosteriorEvidence`: `posterior_mu_bps`, `posterior_sigma_bps`, `prob_mu_gt_cost`, `lcb_net_bps`, `quality_weight`, `activation_contract`.
- `evaluate_panel_cheap_gate()` and `evaluate_alpha_cheap_gate_batch()` consume aligned `[T, N]` arrays, causal lag, funding cost, and stressed round-trip cost.
- `shrink_l1_evidence_hierarchical()` applies family/timeframe shrinkage with `w = n_eff / (n_eff + prior_effective_n)`.

### Alpha Foundry Bridge Wiring [ADR_20260706_ALPHA_FOUNDRY_MAIN_WIRING]
- `PanelRecipeBinding`: binding record with `panel_tag`, `recipe_id`, `source` (alpha_foundry/main/panels), `timestamp`.
- `AlphaFoundryBridgeReport`: per-TF report with `mode` (off/audit/gate), `panels_in`, `bound`, `survivors`, `reject_breakdown`, `warnings`.
- `AlphaFoundryL0Result`: typed result from L0 gate with `evidence_rows`, `passed`, `rejected`, `report`, `binding`.
- `run_alpha_foundry_l0_gate(panel_dfs, recipes, config)` — panel→recipe matching, cheap gate evaluation, audit/gate filtering, report generation with JSON artifact.
- `bind_panels_to_alpha_recipes(panel_dfs, recipes, max_per_family)` — variant normalization (`_normalize_variant()`, strip suffix/prefix), family filter, max-per-family budget.
- `_write_alpha_foundry_report(mode, rows, passed, tf)` — writes `logs/futures/alpha_foundry/{tf}_{timestamp}_report.json`.
- Mode behavior: `"audit"` preserves all bound panels with `gate_passed` flag; `"gate"` only forwards survivors (zero-survivor closes tiered L1 entry).

# 3. Core I/O Interfaces

### Input Data
- `AlignedMarketData`: OHLCV, indicators, flow, funding rates
- `MarketRegimeContext`: Compressed 3-state (`bull`, `bear`, `crisis`) 및 6-state regime codes

### Principal Data Structures (src/domain/futures/signals/contracts.py)
- `CandidateSignalPanel`:
  - `signed_score_2d`: NDArray[np.float64]
  - `side_hint_2d`: NDArray[np.int8]
  - `allowed_regimes`: tuple[RegimeName, ...]
  - `exit_policies`: tuple[SignalExitPolicy, ...]
- `SymbolStrategyEvidence`:
  - `mean_gross_bps`: float
  - `mean_incremental_bps`: float
  - `block_tstat_incremental`: float
  - `q_value`: float
  - `quality_weight`: float
  - `hard_eligible`: bool
  - `lcb_net_bps`: float
  - `adverse_regime_lcb_bps`: float | None — bear/crisis 구간 LCB(진단 전용, `quality_weight` 산식 미반영)
  - `adverse_regime_n_obs`: int
  - `adverse_regime_defended`: bool — `compute_adverse_regime_evidence()` 산출 [ADR_20260705_L1L2_REGIME_CONDITIONAL_WEIGHT]
- `QualifiedSignalRegistry`:
  - `by_symbol`: dict[str, tuple[SymbolStrategyEvidence, ...]]
  - `ready_symbols`: tuple[str, ...]

# 4. Architecture Flow

```mermaid
graph TD
    A[Market Data] --> B[Vectorized Indicators]
    B --> C[CandidateSignalPanel 28 Families]
    C --> C1[Multi-TF Panel Injection]
    C1 --> D[Archetype & Regime Context]
    D --> E[L1 Breakeven & Profit Floor Gate]
    E --> F[Regime-Cell OR-path Admission]
    F --> G[Multiplicity: FDR & SPA]
    G --> H[Promoted Candidate Events]
    H --> I[Per-TF L1 Pipeline 4h/6h/8h/12h]
    I --> J[L2 on Master TF]
```

# 5. Readiness & Promotion Gates

### Readiness Gate (5 Conditions)
- **Fold Coverage**: $\ge 0.80$
- **Match Ratio**: $\ge 0.90$
- **Effective $N$ ($N_{eff}$)**: $\ge 3.0$
- **Fold Ratio**: $\ge 0.50$
- **Pooled LCB**: $> 0$

### Promotion Gate (4 Conditions)
- **Hard Eligible**: Structural gates pass 통과 여부
- **LCB Net**: `lcb_net_bps > l1_breakeven_floor_bps` (~7.5 bps)
- **BH-FDR**: $q\text{-value} \le 0.10$
- **Quality Weight**: `quality_weight > 0`

# 6. Performance Optimizations
- **Adaptive Worker Cap**: 메모리 계산 및 fold 분할수에 따른 병렬 worker 수 조절.
- **Prefit Overlap (OPT-4)**: Evidence fold 수집 단계에서 Layer 1 모델을 백그라운드로 prefit하여 대기 시간 제거.
- **Prefork Cache Prime**: Process fork 이전에 shared cache를 초기화하여 CoW 메모리 효율화.
- **Numba JIT Acceleration**: Rolling z-score, cross-sectional rank, warm/ready 검사에 `@njit(cache=True)` 적용.
- **t.ppf LRU Cache**: Student-t 분포의 Percent Point Function 연산을 `@lru_cache`화하여 중복 연산 제거.
- **searchsorted Indexing (OPT-1)**: 시계열 datetime 매칭 탐색을 $O(T)$에서 $O(\log T)$로 최적화.
