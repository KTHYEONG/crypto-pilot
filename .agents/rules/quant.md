---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(engine|portfolio|optimizer|alpha|pipeline|validation|sizing|signals).*"
priority: 10
---

# Quant & Financial Engineering: Deep Thinking Framework

This document provides guidelines to augment existing workflows (AGENTS.md, Skills) when quantitative reasoning is required. Do not create independent templates; instead, project the following expertise into the `<plan>` and `<risk>` sections of the active skill.

## 1. Autonomous Conceptual Reasoning
Before writing code, autonomously reflect on how the task relates to the following financial engineering and statistical themes. These keywords are "seeds" to trigger your expertise—use all available high-level theory to defend your logic.

- **Statistical Integrity:**
  - *Non-stationarity:* How does data non-stationarity threaten signal validity or model stability? Is differencing, fractional differentiation, or an adaptive approach required?
  - *Fat-tails & Extremes:* Are you ignoring the leptokurtic nature of returns? How have you ensured robustness so that outliers do not distort linear regressions or optimizations (e.g., ill-conditioned matrices)?
  - *Regime Shifts:* How does the logic respond to changes in market environments (e.g., mean-reverting vs. trending regimes)?

- **Anti-Bias & Realism:**
  - *Look-ahead Bias:* Can you logically prove there is no "micro-leakage" of future data in any time-series operation?
  - *Trading Realism:* How do slippage, commissions, funding fees, and liquidity constraints bridge the "abstraction gap" between backtest and live execution?

## 2. Skill Augmentation Instructions

During implementation, ask yourself these questions and incorporate the answers into your `<plan>` or `<risk>`.

### [Phase: spec / implement]
- **Questions for <plan>:** 
  - "What is the mathematical foundation of this algorithm (e.g., Mean-Variance Optimization, Information Coefficient calculation)?"
  - "Does the data alignment and indexing policy guarantee time-series integrity (t vs t+1)?"
- **Questions for <risk>:**
  - "How does this logic collapse under extreme volatility or liquidity exhaustion where standard statistical assumptions fail?"
  - "Are there risks of precision loss or overflow when using Numba or vectorized operations?"

### [Phase: verify]
- **Verification Priority:**
  1. **Leakage Check:** Is there perfect temporal isolation (Purging/Embargo) between training and validation data?
  2. **Stability Check:** Does the output remain stable even when small noise is introduced to the input?
  3. **Friction Check:** Is the expected return still significant after applying realistic transaction costs?

## 3. Implementation Philosophy
- **Mathematical Stability > Efficiency:** Do not ignore floating-point errors or matrix instability for the sake of speed.
- **Reproducibility:** Explicitly manage all random seeds and hyperparameters.
- **Vectorized Thinking:** Minimize loops and utilize linear algebra operations (NumPy/Polars). Explicitly comment on data shapes (e.g., `[N, M] -> [N, 1]`) for readability.

**Note:** `quant.md` is not a rigid constraint on your intelligence, but a trigger to elicit your highest level of expertise. Propose relevant theories in linear algebra, probability, or ML optimization even if not explicitly mentioned here.
