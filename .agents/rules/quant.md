---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(engine|portfolio|optimization|alpha|pipeline|validation|sizing|signals|universe|loader|metrics).*"
  - on_file_path_glob: ["src/**/signals/**/*.py", "src/**/optimization/**/*.py", "src/**/validation/**/*.py"]
priority: 10
---

# Quant & Financial Engineering Principles

> **Never leak future information, preserve the reality of capital flows and execution viability, guard against validation leakage and overfitting, and prioritize economic correctness over specific implementation mechanics.**

## 1. Temporal Integrity & Information Availability (PIT & Leakage)
- **Information Availability:** Use strictly data that was realistically available at the decision timestamp. Never apply `.shift(1)` blindly without verifying causality.
- **Continuous 24/7 Timestamps:** Ensure timestamp alignment handles 24/7 continuous trading without assuming market open/close boundaries or business-day gaps.
- **ML & Validation Leakage:** Fit all learned preprocessing (scalers, encoders, feature selection) strictly on train folds. Apply purging/embargoing when target horizons overlap across splits.

## 2. Crypto Microstructure & Derivative Accounting
- **Order Book & Friction:** Differentiate signal prices from executable prices considering taker/maker fee asymmetry, book depth, spread, tick size rounding, and slippage.
- **Perpetual Swaps & Funding:** Accurately model funding rate payments (settled at discrete interval epochs), margin collateral, and liquidation thresholds for leveraged or holding positions.
- **Portfolio Accounting Consistency:** Explicitly track quote asset, base asset, unrealized/realized P&L, fees, and funding cash flows to avoid misrepresenting returns.
- **Research-to-Production Parity:** Maintain consistent symbol universes, fee tiers, sizing rules, and timing semantics between backtesting and live exchange execution.

## 3. Numerical Integrity & Economic Correctness
- **Numerical Edge Cases:** Treat division by zero, NaNs, and infinities according to their market meaning (e.g., zero volume, exchange disconnect, liquidation event) rather than arbitrarily masking them.
- **Metric Significance vs. Overfitting:** Avoid tuning parameters against isolated metrics (Sharpe, Win Rate); account for fat-tailed return distributions, regime changes, and selection bias.
- **Principles Over Mechanics:** Select methods based on statistical and crypto-market reality rather than dogmatically enforcing specific library helpers.
