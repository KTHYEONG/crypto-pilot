---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(engine|portfolio|optimizer|alpha|pipeline|validation|sizing|signals|universe|loader|metrics).*"
  - on_file_path_glob: ["src/**/signals/**/*.py", "src/**/optimization/**/*.py", "src/**/validation/**/*.py"]
priority: 10
---

# Quant & Financial Engineering: Deep Thinking Framework

This document provides guidelines to augment existing workflows (AGENTS.md, Skills) when quantitative reasoning is required. Do not create independent templates; instead, project the following expertise into the `<plan>` and `<risk>` sections of the active skill.

## 1. Autonomous Conceptual Reasoning
Before writing code, reflect on financial engineering and statistical themes. Use high-level theory to defend your logic.

- **Statistical Integrity:** Non-stationarity, Fat-tails, Regime Shifts, and robustness against outliers.
- **Anti-Bias & Realism:** Look-ahead bias prevention, Trading Realism (slippage, fees, liquidity).

## 2. Skill Augmentation Instructions

### [Phase: spec / implement]
- **Questions for <plan>:** 
  - "What is the mathematical foundation of this algorithm (e.g., MVO, IC calculation)?"
  - "Does the indexing policy guarantee time-series integrity (t vs t+1)?"
- **Questions for <risk>:**
  - "How does this logic collapse under extreme volatility?"
  - "Are there risks of precision loss in vectorized operations?"

### [Phase: verify]
1. **Leakage Check:** Perfect temporal isolation (Purging/Embargo).
2. **Stability Check:** Robustness to input noise.
3. **Friction Check:** Significance after realistic costs.

## 3. High-Performance Computing (HPC) & JIT
For core engines and optimizers, follow these efficiency principles:
- **Zero-Loop Policy:** Avoid Python-level `for` loops in time-series operations. Use NumPy/Polars/Pandas vectorization.
- **JIT Compilation (Numba):** Use `@njit` for performance-critical bottlenecks, but ensure the logic remains simple enough for the compiler.
- **Memory Efficiency:** Pre-allocate arrays; avoid repetitive large-object copies.
- **Vectorized Thinking:** Explicitly comment on data shapes (e.g., `[N, M] -> [N, 1]`).

## 4. Financial Validation Standards
- **Walk-forward Validation:** Ensure the backtest pipeline follows the sliding window approach without data leakage.
- **9 Pillars of Integrity:** Refer to `docs/architecture/backtest-logic.md` for Conservation of Money and Exposure Cap rules.
- **IC Calibration:** Use Spearman Rank IC for signal evaluation to mitigate outlier influence.

**Note:** `quant.md` is a trigger to elicit your highest level of expertise. Propose relevant theories in linear algebra or ML optimization as needed.
