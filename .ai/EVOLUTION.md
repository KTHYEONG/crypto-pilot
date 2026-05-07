# 🧬 System Evolution Journal

This file tracks the logical progression and experimental results of the quantitative trading system. It serves as the primary context for AI agents to understand "why" certain changes were made and "what" was learned.

---

## [2026-05-07] v1.1.0 HMM Regime Separation Optimization (Gemini-2.0-Flash)
- **Status**: Validated (Improved State Separation)
- **Problem**: HMM was collapsing all states into "CHOP" due to high-dimensional noise (11 features) and indexing bugs.
- **Key Improvements**:
    - **Feature Pruning**: Reduced systemic HMM features from 11 to 4 core factors (Trend, Vol, Dispersion, OI Delta) to improve signal-to-noise ratio.
    - **Semi-supervised Init**: Implemented manual `locs` (means) initialization for JAX Student-t HMM to force semantic separation of Bull/Bear/Chop/Crisis.
    - **Bug Fixes**: Corrected volatility index mapping (Liq -> Vol) and fixed scale mismatch in Log-Wealth (G_LOG) labeling logic.
- **Results**:
    - Clear separation of BULL_TREND (G_LOG: 0.035%) and CRISIS (SIG: 0.658%).
    - Left-Tail Capture increased to 63.5%.
    - Stable regimes with average duration of 43.7 bars.
- **Lessons**: High-dimensional unsupervised clustering in HMM requires careful feature selection and prior bias to maintain economic interpretability.

---

## [Baseline] 2026-05-07: Initial SOTA Architecture
- **Status**: Stable (Champion Deployed)
- **Key Logic**: 
    - HMM-based regime filtering to avoid crisis periods.
    - GP-Alpha for capturing cross-sectional momentum.
    - Walk-forward validation with CAWF-R for robustness.
- **Metrics**: 
    - Refer to `logs/champion.json` for detailed performance stats.
- **Lessons**: Initial integration of HMM and ML Pipeline proved successful in reducing MDD during high-volatility regimes.

---
<!-- APPEND_POINT: New experiments will be added above this line -->
