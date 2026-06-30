---
title: Market Regime Architecture
domain: futures.strategy
type: architecture
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/market_regime.py
  - src/domain/futures/strategy/regime_evaluation.py
  - src/domain/futures/strategy/config.py
  - src/execution/opt_main_futures.py
change_triggers:
  - src/domain/futures/strategy/market_regime.py
  - src/domain/futures/strategy/regime_evaluation.py
  - src/execution/opt_main_futures.py
last_verified: 2026-06-30
---

# 1. Purpose
Establishes a causal market state from BTC price action for two consumers: a continuous risk overlay for position sizing and a diagnostics-only discrete regime code. L2 routing consumes the compressed 3-state summary (`bull`, `bear`, `crisis`), while the raw 6-state code remains a shadow diagnostic.

# 2. Core Logic & Math

**Volatility Targeting**
- $\hat{\sigma}_{t} = \sqrt{\text{EWMA}[ (r - \bar{r})^2 ]_{t} \cdot \text{bars\_per\_year}}$
- $\text{vol\_scale}_{t} = \text{clip}\left(\frac{\sigma_{\text{target}}}{\hat{\sigma}_{t}}, \text{min\_vol\_scale}, \text{max\_vol\_scale}\right)$

**Trend SNR Gate**
- $s_{t} = \ln(P_{t}) - \text{EMA}(\ln P)_{t}$
- $\text{snr}_{t} = \frac{s_{t}}{\text{std}(s)_{t}}$
- $\text{trend\_scale}_{t} = \frac{1}{2} (1 + \tanh(\text{snr}_{t}))$

**Trend Efficiency (Kaufman ER)**
- $ER_{t} = |P_{t} - P_{t-\text{window}}| / \Sigma_{i=t-\text{window}+1}^{t} |P_{i} - P_{i-1}|$
- $1$ = perfect trend, $\approx 0$ = chop/whipsaw
- NaN for $t < \text{window}$; zero-path divisor $\rightarrow 0.0$
- Consumed by L2 whipsaw attribution (`Layer2FoldAttribution`) and trend-efficiency exposure gate (`trend_efficiency_gross_mult`)

**Page-CUSUM Crisis Detector**
- $z_{t} = \frac{r_{t} - \text{median}_{\leq t}}{1.4826 \cdot \text{MAD}_{\leq t}}$
- $S^{+}_{t} = \max(0, S^{+}_{t-1} + z_{t} - k)$
- $S^{-}_{t} = \max(0, S^{-}_{t-1} - z_{t} - k)$
- Crisis triggers if $S^{+}_{t} > h$ or $S^{-}_{t} > h$.

**Overlay Compositor**
- $\text{overlay\_mult}_{t} = \text{vol\_scale}_{t} \cdot \text{trend\_scale}_{t}$
- If crisis active: $\text{overlay\_mult}_{t} = \text{crisis\_gross\_floor}$

**Discrete Quantizer (6-State)**
- Base: 2x2 grid (`bull/bear` by $\text{trend\_snr} \geq 0$, `quiet/volatile` by $\text{vol\_scale} \geq 1.0$)
- Override: If crisis active $\rightarrow$ `crash` (5). Transition $\rightarrow$ (4).
- Use: raw 6-state output is diagnostics-only; routing uses the compressed 3-state summary.

**Routing Summary**
- L2 production logs and routing decisions consume the compressed 3-state summary (`bull`, `bear`, `crisis`).
- `l2_regime_policy_mode` controls causal sleeve policy on top of bucket routing: `filter`, `observe`, `soft`, `hybrid`.
- `soft` keeps the route available and only downweights low-confidence cells, while `hybrid` can hard-block only when sign consistency and confidence criteria are met.
- Regime policy acts on `signal x symbol x tf` sleeves before symbol pooling, so `raw_mu` and `quality_weight` share the same state-aware scaling path as `sleeve_edges`.
- `l2_regime_scale_signal_mu` and `l2_regime_scale_quality_weight` control whether regime confidence reaches the allocation inputs or remains edge-only.
- State-level gross caps limit deployment exposure by regime class after portfolio weights are composed.
- Policy observability tracks pre/post edge, mu, and quality-weight totals so routing impact reaches the final sizing path and not only the diagnostics path.

# 3. Architecture Flow

```mermaid
graph TD
    A[BTC Close] --> B[Log Returns]
    B --> C[EWMA Volatility]
    C --> D[vol_scale]
    A --> E[Trend SNR]
    E --> F[trend_scale]
    B --> G[Robust Z-Score]
    G --> H[Page-CUSUM]
    H --> I[crisis_active]
    D --> J[overlay_mult]
    F --> J
    I -->|Override| J
    D --> K[Discrete Quantizer]
    E --> K
    I --> K
    K --> L[6-State Regime Code]
    J --> M[Portfolio Sizing]
    L --> N[Evaluation / ML Target]
```

# 4. Core Variables & I/O

| Type | Variable | Description |
|------|----------|-------------|
| **Input** | `P_t` | BTC Close Price at time $t$ |
| **Param** | $\sigma_{\text{target}}$ | Target annualized volatility. Bounds: `(0.0, 1.0]` |
| **Param** | `crisis_gross_floor` | Risk override multiplier during crisis. Bounds: `[0.0, 1.0]` |
| **Param** | $k, h$ | CUSUM drift and threshold, derived from target ARL. Bounds: `ARL > 0` |
| **Output**| `overlay_mult_1d` | Continuous risk multiplier applied to final portfolio weights. Bounds: `[0.0, max_vol_scale]` |
| **Output**| `code_1d` | Discrete regime integer (0-5) used for signal gating and B0 ensemble |
| **Output**| `trend_efficiency_1d` | Per-bar Kaufman Efficiency Ratio. Bounds: `[0, 1]`, NaN for warm-up |
| **Eval**  | `RegimeScoreCard` | Metrics C2(Persistence), C3(Distinctness), C4(Stability), C5(Coverage) |
| **L2 Param** | `l2_regime_bull_gross_cap` | Bull regime gross exposure cap (default `1.0`, full deployment). Bounds: `(0.0, 1.0]` |
| **L2 Param** | `l2_regime_bear_gross_cap` | Bear regime gross exposure cap (default `0.35`, bull-primary prior). Bounds: `(0.0, 1.0]` |
| **L2 Param** | `l2_regime_crisis_gross_cap` | Crisis regime gross exposure cap (default `0.25`, bull-primary prior). Bounds: `(0.0, 1.0]` |

# 5. Edge Cases & Handling
- **Flash Crash / Liquidity Vacuum:** Rapid extreme price drops trigger the CUSUM crisis condition, immediately snapping the `overlay_mult_1d` to `crisis_gross_floor` and ignoring naive volatility targeting bounds until the cooldown period expires.
- **Zero Volatility (Stale Data):** If exchange feeds freeze (resulting in zero returns and zero MAD), statistical safeguards (small epsilon additions to MAD) prevent division-by-zero during robust Z-score calculation.
