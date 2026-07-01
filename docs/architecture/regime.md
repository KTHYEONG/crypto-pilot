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

**Reversal Risk-Off Detector (BTC-only legacy)**
- $HW_{t} = \max(P_{t-\text{window}+1 : t})$ (trailing window high)
- $DD_{t} = 1 - P_{t} / HW_{t}$ (trailing drawdown)
- $Mom_{t} = \text{EMA}(P, \text{fast})_{t} - \text{EMA}(P, \text{slow})_{t}$ (momentum)
- $risk\_off\_raw_{t} = (DD_{t} \ge \text{dd\_threshold}) \land (Mom_{t} < 0)$
- $persisted\_run_{t} = persisted\_run_{t-1} + 1 \text{ if } risk\_off\_raw_{t} \text{ else } 0$
- $risk\_off\_persisted_{t} = 1_{\{persisted\_run_{t} \ge \text{persistence\_bars}\}}$ (N-bar consecutive raw condition)
- $risk\_off\_state_{t} = \text{recovery\_hysteresis}(risk\_off\_persisted, risk\_off\_raw, \text{cooldown})$ — after persistent fires, state stays True until $\text{cooldown}$ consecutive raw-off bars. Exit counting uses raw signal (not persistent). At $\text{cooldown}=0$ (default), state tracks persistent byte-identically.
- $risk\_off\_1d_{t} = shift(risk\_off\_state, 1)$ — bar $t$ uses information up to $t-1$ only (causal, look-ahead 0)
- $risk\_off_{bar} = True$ → L2 override of all sleeve raw_mu to `reversal_risk_off_floor` (overrides soft cap/crisis_floor)
- Config param: `reversal_recovery_cooldown_bars` (default 0) — exit hysteresis cooldown, shared with panel mode
- Env override: `L2_REVERSAL_RECOVERY_COOLDOWN`, parsed by `_reversal_config_from_env()` for btc mode
- Env-gated (`L2_REVERSAL_KILL`, default off), computed once from BTC close per simulation run

**Market State Panel (Breadth-Augmented Risk-Off)**
- Mode-gated by `reversal_mode`: `"btc"` (legacy BTC-only) or `"panel"` (breadth-augmented).
- **Cross-sectional Downside Breadth:**
  - $r_{t,i} = \ln(P_{t,i} / P_{t-\text{breadth\_window},i})$ per symbol $i$
  - $valid_{t,i} = \text{isfinite}(P_{t,i}) \land (P_{t,i} > \epsilon) \land \text{isfinite}(P_{t-\text{breadth\_window},i}) \land (P_{t-\text{breadth\_window},i} > \epsilon)$
  - $neg_{t,i} = valid_{t,i} \land (r_{t,i} < 0)$
  - $\text{breadth}_{t} = \frac{\sum_i neg_{t,i}}{\max(\sum_i valid_{t,i}, 1)}$ — fraction of symbols with negative momentum, $[0, 1]$, NaN-safe
  - $\text{breadth}_{t} = 0$ when $t < \text{breadth\_window}$ or $\sum valid = 0$
- **Breadth Hysteresis (stateful Schmitt trigger):**
  - `b_on = False` initially
  - $enter \le \text{breadth}_{t} \land \lnot b\_on \rightarrow b\_on = True$
  - $\text{breadth}_{t} < exit \land b\_on \rightarrow b\_on = False$
  - $enter > exit$ enforced by config validation (asymmetric hysteresis)
- **AND Gate:**
  - $raw\_on_t = btc\_off_t \land breadth\_on_t$ — both BTC and breadth axes must confirm
- **Persistence:** $N$ consecutive $raw\_on$ bars before activation (same as BTC-only persistence)
- **Recovery Hysteresis (asymmetric exit):**
  - $\text{state}_t = True$ when $persist\_on_t$ fires or $\text{state}_{t-1}$ remains True
  - $\text{state}_t = False$ only after $recovery\_cooldown\_bars$ consecutive $raw\_off$ bars
  - Config parameter `reversal_recovery_cooldown_bars` (default 0 = immediate release after raw-off)
- $risk\_off\_1d_t = shift(\text{state}, 1)$ — 1-bar shift for causal consumption
- Config: `breadth_mom_window`, `breadth_neg_frac_enter`, `breadth_neg_frac_exit`, `reversal_recovery_cooldown_bars`
- All new `RegimeConfig` fields validated in `__post_init__`

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
    A --> O[Rolling Max + DD]
    A --> P[EMA Fast / Slow]
    O --> Q[risk_off_raw]
    P --> Q
    Q --> R[shift(1) → risk_off_1d]
    R -->|BTC-only legacy| S[Selective Hard De-Gross]
    S --> M
    subgraph Panel [Market State Panel - breadth_off]
        U[Universe Close 2D] --> V[Log Returns per sym]
        V --> W[Fraction negative by sym]
        W --> X[Breadth Hysteresis]
        X --> Y[breadth_on]
    end
    Q -->|panel mode| Z[AND Gate]
    Y --> Z
    Z --> AA[Persistence]
    AA --> AB[Recovery Hysteresis]
    AB --> AC[shift(1) → risk_off_1d]
    AC --> AD[Selective Hard De-Gross]
    AD --> M
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
| **Param** | `reversal_dd_window` | Trailing high lookback for drawdown (bars, default 90). Bounds: `>= 2` |
| **Param** | `reversal_dd_threshold` | Drawdown threshold to flag risk-off (default 0.12). Bounds: `(0, 1)` |
| **Param** | `reversal_mom_fast` | Fast EMA span for momentum (default 20). Must be `< reversal_mom_slow` |
| **Param** | `reversal_mom_slow` | Slow EMA span for momentum (default 120). Must be `> reversal_mom_fast` |
| **Param** | `reversal_risk_off_floor` | Hard gross floor during risk-off bars (default 0.05). Bounds: `[0, crisis_gross_floor)` |
| **Param** | `reversal_persistence_bars` | Consecutive raw risk-off bars required before shifted activation (default 3). Bounds: `>= 1` |
| **Output**| `risk_off_1d` | Boolean mask [T], `True` = risk-off active. Causal: shift(1) after persistence gate, row 0 = `False` |
| **Eval**  | `RegimeScoreCard` | Metrics C2(Persistence), C3(Distinctness), C4(Stability), C5(Coverage) |
| **L2 Param** | `l2_regime_bull_gross_cap` | Bull regime gross exposure cap (default `1.0`, full deployment). Bounds: `(0.0, 1.0]` |
| **L2 Param** | `l2_regime_bear_gross_cap` | Bear regime gross exposure cap (default `0.35`, bull-primary prior). Bounds: `(0.0, 1.0]` |
| **L2 Param** | `l2_regime_crisis_gross_cap` | Crisis regime gross exposure cap (default `0.25`, bull-primary prior). Bounds: `(0.0, 1.0]` |
| **Input** | `P_2d` | Universe close prices (T x N matrix) for cross-sectional breadth |
| **Param** | `breadth_mom_window` | Log-return window for downstream breadth (default 21 bars). Bounds: `>= 2` |
| **Param** | `breadth_neg_frac_enter` | Fraction of negative-momentum symbols to enter breadth-on (default 0.50). Bounds: `(0, 1]` |
| **Param** | `breadth_neg_frac_exit` | Fraction to exit breadth-off (default 0.30). Bounds: `[0, 1)`, must be `< enter` |
| **Param** | `reversal_recovery_cooldown` | Consecutive raw-off bars before state resets to False in panel mode (default 0). Bounds: `>= 0` |
| **Param** | `reversal_mode` | Risk-off detector mode: `"btc"` (legacy BTC-only) or `"panel"` (breadth-augmented). Default: `"btc"` |

# 5. Edge Cases & Handling
- **Flash Crash / Liquidity Vacuum:** Rapid extreme price drops trigger the CUSUM crisis condition, immediately snapping the `overlay_mult_1d` to `crisis_gross_floor` and ignoring naive volatility targeting bounds until the cooldown period expires.
- **Zero Volatility (Stale Data):** If exchange feeds freeze (resulting in zero returns and zero MAD), statistical safeguards (small epsilon additions to MAD) prevent division-by-zero during robust Z-score calculation.
