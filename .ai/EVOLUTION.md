# 🧬 System Evolution Journal

This file tracks the logical progression and experimental results of the quantitative trading system. It serves as the primary context for AI agents to understand "why" certain changes were made and "what" was learned.

---

## [2026-05-08] v3.0.0 Unsupervised Revolution: Removing Guidance & Post-hoc Mapping (Gemini CLI)
- **Status**: Validated (CRISIS G_log: -0.112%, BULL G_log: 0.017%)
- **Problem**: v2.1.0 "Monolith" HMM was overloaded with 10,000x penalties and conflicting guidance, hitting a performance ceiling.
- **Key Fixes**:
    - **Pure Unsupervised HMM**: Removed all guidance masks and return-space penalties.
    - **Robust Fat-tail Preprocessing**: Replaced `QuantileTransformer` with `RobustScaler` (-15 to 15 clip).
    - **Post-hoc Semantic Mapping**: Rank discovered states by empirical MU/SIG after training.
- **Results**: Deepest risk isolation to date (CRISIS G_log: -0.112%). BULL G_log positive (0.017%).
- **Lessons**: De-coupling clustering (HMM) from labeling (Mapping) is the key to unlocking regime-based alpha. See [.ai/experiments/2026-05-08_hmm_unsupervised_refactor.md](experiments/2026-05-08_hmm_unsupervised_refactor.md).

---

## [2026-05-08] v2.1.0 μ Separation Campaign: Returns Observation & Multi-Guidance (Claude Sonnet 4.6)
- **Status**: Partial — μ separation achieved, Tail Capture FAIL (55-58%).
- **Key Discovery**: Fundamental trade-off between return-based observation (μ separation) and volatility-based tail capture in a guided 4-state architecture.
- **Root Cause**: Excessive guidance penalties (1000x-10000x) suppressed Likelihood learning.

---

## [2026-05-07] v1.x.x ~ v2.0.0 Legacy Evolution (Summary)
*Older entries from the iterative tuning phase of the Guided HMM architecture.*

- **v2.0.0 (P6)**: Introduced TVTP Asymmetric Bias. Found that SGD often overrides initial biases.
- **v1.9.0 (P5.B)**: Tightened Vol/OI thresholds to 95th percentile. Improved precision but recall suffered.
- **v1.8.0 (P5)**: Orthogonal tuning (Penalty recalibration). Found 10,000x penalty blocks LL learning.
- **v1.7.0 (P4)**: Introduced Viterbi Decoding for hard state paths. Solved "Noise Locked" soft posterior issues.
- **v1.3.0 ~ v1.6.0**: Explorations in semi-supervised guidance, hierarchical stress filters, and feature pruning.
- **Baseline**: Initial SOTA combining HMM-regime filtering with GP-Alpha.

---
<!-- APPEND_POINT: New experiments will be added above this line -->
