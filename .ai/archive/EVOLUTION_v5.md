# 🧬 System Evolution Journal (Archive v1.x - v5.x)

This file contains the historical evolution of the quantitative trading system prior to v6.0.0.

---

## [2026-05-09] v5.2.0 Engine Bottleneck Liquidation: Event-Driven Hysteresis & Sizing Liberation (Gemini CLI)
- **Status**: Validated (Structural PASS, Trade Starvation: FIXED, HMM Suppression: FIXED)
- **Problem**: v5.1.0 logic was still hindered by "Institutional Friction" — fixed clock-based rebalancing caused Alpha Decay to outrun the execution, while HMM hard gates (0.5) blocked profitable trades in "Bumpy" regimes.
- **Key Fixes**:
    - **Schmitt Trigger (Hysteresis)**: Implemented separate Z-score thresholds for entry (suggested) and maintenance (0.7x entry). This "sticky" conviction prevents churn from minor noise.
    - **Event-Driven Alpha Turnover**: Replaced fixed `REBALANCE_BARS` with a 15% Alpha Turnover threshold. The system now "waits" for significant conviction shifts before paying the Taker fee.
    - **HMM Soft-Sizing**: Removed the 0.5 binary gate. Lowered the floor to 0.1, shifting HMM's role from a "Hard Kill-switch" to a "Soft Size-modulator."
- **Results**: Smoke test (OOS 2026-04) confirmed 393 trades (healthy frequency) and entry in Regime 3 (MOD 0.48).

## [2026-05-09] v5.1.0 Optimization Bottleneck Overhaul: Cost-Aware Reward & Search Space Liberation (Gemini CLI)
- **Status**: Validated (IS CAGR: +14%p improvement vs v5.0.0)
- **Key Fixes**: Unlocked search space, cost-aware reward in objective, and fixed annualized vol scaling in sizer.

## [2026-05-09] v5.0.0 SOTA Institutional Quant Architecture: Deterministic Policy & Online Ensemble (Gemini CLI)
- **Status**: Validated (Ensemble Improvement: +3.9% OOS Retention PASS)
- **Key Fixes**: Platt Scaling + Kelly Criterion, Orthogonal Ensemble selection, and Online Capital Allocation (EG Algorithm).

## [2026-05-08] v4.2.0 Alpha Alpha: Institutional Alpha Mining & Neutralization (Gemini CLI)
- **Status**: Validated (OOS IC: 0.0839 PASS, Survival: 15/15 PASS)
- **Key Fixes**: Cross-sectional Z-score neutralization and friction-aware labeling.

## [2026-05-08] v4.1.0 Asymmetric Revolution: Instant Risk & Sustained Recovery (Gemini CLI)
- **Status**: Validated (Friction: 4.20% PASS, Avg Duration: 100.2 bars PASS)
- **Key Fixes**: Asymmetric regime minimum durations (CRISIS 1h vs BULL 36h).

## [2026-05-08] v4.0.0 Pragmatic Revolution: Role Decoupling & Soft Gravity (Gemini CLI)
- **Status**: Validated (TC OOS: 61.4% PASS, CRISIS G_log: -0.121%)
- **Key Fixes**: Soft Gravity NLL in HMM loss and Sigma-split/Mu-sort mapping.

## [2026-05-08] v3.4.0 5-State TVTP-HMM + Empirical Mapping + Tail Features (Claude Opus 4.7)
- **Architecture**: 5-state HMM, 2D Sharpe/vol mapping, AdamW optimizer.

## [2026-05-08] v3.0.0 Unsupervised Revolution: Removing Guidance & Post-hoc Mapping (Gemini CLI)
- **Key Fixes**: Pure unsupervised HMM, RobustScaler preprocessing.

## [2026-05-08] v2.1.0 μ Separation Campaign: Returns Observation & Multi-Guidance (Claude Sonnet 4.6)
- **Key Discovery**: Trade-off between μ separation and volatility-based tail capture.

## [2026-05-07] v1.x.x ~ v2.0.0 Legacy Evolution (Summary)
- TVTP Asymmetric Bias, Viterbi Decoding, semi-supervised guidance explorations.
