# Quant Multiscale Futures Engine Evaluation Report (v6 Pipeline)

**Date**: 2026-07-24  
**Evaluation Horizon**: 182 days (4,380 4-hour bars / 17,520 1-hour bars)  
**Target Universe**: 120 Binance Perpetual Futures Symbols (Point-in-Time Causal Data)  
**Execution Mode**: Full Multiscale Pipeline (`--phase full`)  
**Data Integrity**: PASS (`integrity_ok: true`)  

---

## 1. Executive Summary & Level-by-Level Breakdown

| Pipeline Layer | Evaluation Scope & Strategy Config | Core Metric / Finding | Verdict & Status |
| :--- | :--- | :--- | :--- |
| **Level 1 (L1 Alpha Bank)** | 5 Admitted Signals across 4 Families (`trend_ema`, `basis_gap`, `xs_reversal`, `xs_momentum`) | Sign Consistency 75~100%, **Low SNR (< 0.1)** | PASS (Admitted 5/30) |
| **Level 2 (L2 Allocation)** | Dynamic Kelly Scaling ($f \in [0.25, 0.60]$), Asymmetric Leverage (Gross 2.0x, Long 1.5x, Short 0.5x), Funding Carry | **Log Growth Rate ($g$): -3.0470**, Equity Multiple: 0.3176 (-68.24% Return) | Volatility Drag (-71.60% MDD) |
| **Level 3 (L3 Validation)** | Sealed Holdout Gate (90-day holdout evaluation) | **`max_drawdown_exceeded`** | **REJECT** (Deployment Blocked) |

---

## 2. Level 1 (L1) Signal Bank Admission & Diagnostics

### Admitted Signal Catalog (5 Elite Signals)
- `trend_ema:slow` (Lookback 216h, Target 216h, Sign Consistency 1.00)
- `trend_ema:very_slow` (Lookback 432h, Target 432h, Sign Consistency 1.00)
- `basis_gap:fast` (Lookback 24h, Target 24h, Sign Consistency 0.75)
- `xs_reversal:fast` (Lookback 8h, Target 8h, Sign Consistency 1.00)
- `xs_momentum_slow:slow` (Lookback 216h, Target 216h, Sign Consistency 0.75)

### L1 Findings
- **Directional Edge**: Alpha signals demonstrate statistical significance after Benjamini-Hochberg FDR control ($q=0.05$).
- **SNR Deficit**: Information Coefficients (IC) range from +0.012 to +0.035, indicating a low Signal-to-Noise Ratio (SNR < 0.1).

---

## 3. Level 2 (L2) Portfolio Performance Matrix

| Metric Parameter | L2-v5 (Baseline Quarter Kelly 1.0x) | **L2-v6 (Dynamic Kelly & Asymmetric 2.0x)** |
| :--- | :--- | :--- |
| **Net Log Growth Rate ($g$)** | 0.0000 ~ 0.3044 | **-3.0470** |
| **Equity Multiple** | 1.0000 ~ 1.3559 | **0.3176 (-68.24%)** |
| **Maximum Drawdown (MDD)** | -12.10% ~ -49.50% | **-71.60%** |
| **Annualized Volatility** | 14.5% ~ 24.0% | **89.80%** |
| **Daily CVaR (95%)** | -0.40% | **-1.75%** |
| **Annual Capital Turnover** | 41.2 ~ 52.7 turns/yr | **57.3 turns/yr** |
| **Friction Control** | Smoothing $\alpha=0.03$, Hysteresis $\theta=6\text{ bps}$ | Retained Low Friction |

### L2 Failure Mechanism Analysis
1. **Over-Allocation under Low SNR**: Dynamic Kelly scaling pushed fractional exposure up to $f=0.60$ during transient signal spikes, exposing capital to 2.0x gross leverage when alpha edge was insufficient to overcome noise.
2. **Volatility Drag ($g \approx \mu - \frac{1}{2}\sigma^2$)**: Annualized portfolio volatility exploded to **89.80%**, causing geometric compounding drag and driving the net log growth rate deep into negative territory (-3.0470).
3. **Carry vs. Price Dispersion**: Funding rate carry yields (3~5% APR) were insufficient to offset capital losses caused by directional price volatility during regime shifts.

---

## 4. Level 3 (L3) Gate Verdict & Next Action

- **L3 Deployment Verdict**: **`REJECT`**
- **Block Reason**: `max_drawdown_exceeded` (Real MDD 71.60% > Limit 20.00%)
- **Governance Outcome**: Sealed Holdout Gate successfully blocked live capital deployment of high-leverage allocation.
- **Architectural Next Steps**:
  1. Refine L1 Signal Admission: Enforce higher conviction/SNR thresholds before enabling leverage.
  2. Revert L2 Risk Parameters: Restrict Kelly fraction to $f \le 0.20$ and re-enforce Target Volatility Cap at 15%.
