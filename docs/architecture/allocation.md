---
title: Futures Allocation Architecture
domain: futures.strategy
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/candidate_ensemble.py
  - src/domain/futures/strategy/candidate_workflow.py
  - src/domain/futures/strategy/candidate_portfolio.py
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/ablation.py
change_triggers:
  - src/domain/futures/strategy/candidate_ensemble.py
  - src/domain/futures/strategy/candidate_workflow.py
  - src/domain/futures/strategy/candidate_portfolio.py
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/ablation.py
dependencies:
  documents:
    - docs/architecture/signal.md
    - docs/architecture/regime.md
    - docs/architecture/ML.md
last_verified: 2026-06-10
---

# 1. Purpose
Transforms L1 candidate events into optimal portfolio weights via regime-conditional shrinkage ensemble and stop-risk/Kelly sizing.

# 2. Core Logic & Math

**Regime-Conditional Shrinkage (Ensemble B0)**
- $\mu_{\text{net}} (a, g) = \frac{n_{a,g} \cdot \bar{x}_{a,g} + k \cdot \bar{x}_{\text{global}}}{n_{a,g} + k}$
- where $a$ = archetype, $g$ = entry_regime_code, $k$ = `ensemble_shrinkage_k`

**Fallback Logic (Two-level)**
- Level 1 (Missing regime): $\mu(a, g) \rightarrow \mu(a)$
- Level 2 (Missing archetype): $\mu(a) \rightarrow \mu_{\text{global}}$

**Signal Evaluation Gates (OOS)**
- $IC_{\text{rank}} = \text{Spearman}(\text{score}, \text{target})$ over OOS window. Gate: $IC_{\text{rank}} \geq 0.01$
- $t_{\text{stat}} = IC_{\text{rank}} \times \sqrt{\frac{N_{\text{oos}} - 2}{1 - IC_{\text{rank}}^2}}$. Gate: $t_{\text{stat}} \geq 0.8$
- Q10 Tail Risk: $\text{Fail Rate} \leq 0.65$

**Walk-Forward Survival Censoring**
- If fold realized edge $< 8.0$ bps (expected vs breakeven floor $\approx 3.75$ bps), fold fails.
- Failed fold predictions: $\mu, q10, q90, p_{\text{pass}} \rightarrow 0$ to prevent anti-selection in the portfolio pool.

# 3. Architecture Flow

```mermaid
graph TD
    A[L1 Events] --> B[candidate_workflow]
    B --> C{Backend: ensemble_b0}
    C --> D[fit_regime_conditional_ensemble]
    C --> E[predict_regime_conditional_ensemble]
    D -.->|Train Fold| E
    E --> F[CandidateModelOutput]
    F --> G[select_candidate_events_for_portfolio]
    G --> H[build_candidate_target_weights]
```

# 4. Core Variables & I/O

| Type | Variable | Description |
|------|----------|-------------|
| **Input** | `events: pd.DataFrame` | Candidate events from L1 signal generation |
| **Input** | `entry_regime_code: int` | Regime code at the time of signal entry |
| **Param** | `ensemble_shrinkage_k` | Regularization strength toward global mean |
| **Param** | `min_oos_rank_ic` | Minimum OOS Spearman Rank IC (default: 0.01) |
| **Param** | `min_ic_tstat` | Minimum IC t-statistic for signal validity (default: 0.8) |
| **Param** | `max_variant_oos_q10_fail_rate` | Maximum allowed fraction of events failing q10 threshold (default: 0.65) |
| **Output**| `expected_net_bps` | Shrinkage-adjusted expected return per event |
| **Output**| `target_weights` | Final portfolio allocation weights per event |
