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

## [2026-05-08] v3.4.0 5-State TVTP-HMM + Empirical Mapping + Tail Features (Claude Opus 4.7)
- **Status**: Partial — Stability best-ever, TC ceiling confirmed at ~35% for post-hoc mapping paradigm
- **Phases Applied**: Phase1+2 (empirical mapping + AdamW) → CRISIS μ constraint → Phase5 (5-state) → Phase3 (13 features) → BULL/BEAR μ constraints
- **Key Wins**:
    - **Stability**: Avg Duration 104.1 bars (+36% vs v3.0.0), Switches 210, Friction **5.25%** (first <6%)
    - **BULL_CALM WEALTH_EXP**: Achieved for first time in Phase5 (G_log +0.051%)
    - **LGB Alpha**: Best OOS IC 0.326 (Phase5), driven by richer HMM regime features
    - **CRISIS TAIL_DEFENSE**: Maintained via μ<0 hard constraint
- **Architecture Delta from v3.0.0**:
    - 4-state → **5-state** (added `bull_vol_up` to separate parabolic up-moves from calm bull)
    - μ-only mapping → **2D Sharpe/vol empirical mapping** (Viterbi hard labels + real returns)
    - `returns_ser` now consumed (was silently unused)
    - EMA blend: post-training removed → **pre-training warm-start** (0.8×old + 0.2×fresh)
    - adam → **adamw(wd=1e-4)**; warmup iters 1 → 10
    - 10 features → **13 features** (+`macro_ret_5d_z`, `macro_ret_skew_24h`, `macro_ret_kurt_24h`)
    - TVTP: 5 features → **6 features** (+`macro_trend_168h` idx 0)
    - Semantic constraints: CRISIS μ<0, BEAR μ<0, BULL_CALM μ>0 (via `_swap_latent`)
- **Confirmed Tradeoff (architectural limit)**:
    - Unconstrained mapping → TC 65.8% but CRISIS μ>0 (semantic failure)
    - μ constrained → TC 20-35%, CRISIS semantically valid
    - Root cause: BULL_CALM 57.5% mass absorbs ~54% of worst-5% events; post-hoc mapping cannot redistribute this
- **Current Best Metrics**: TC IS/OOS 35.2%/35.2%, CRISIS G_log -0.103%, Duration 104.1 bars, Friction 5.25%, Score **53/100**
- **Next**: Option A (light NLL return penalty 200-400×) or Option B (metric redefinition). See [experiments/2026-05-08_hmm_5state_phase_evolution.md](experiments/2026-05-08_hmm_5state_phase_evolution.md)

---

## [2026-05-08] v4.0.0 Pragmatic Revolution: Role Decoupling & Soft Gravity (Gemini CLI)
- **Status**: Validated (TC OOS: 61.4% PASS, CRISIS G_log: -0.121%, Score: 78/100)
- **Problem**: v3.4.0 and earlier hit a performance wall by trying to force unsupervised clustering to match rigid semantic rules (TC target 60% with mu<0 constraint).
- **Key Fixes**:
    - **Feature Diet**: Reduced HMM inputs from 13 to 10. Removed noisy higher-order distribution features (skew/kurt) to refocus on Return/Vol channels.
    - **Soft Gravity NLL**: Reintroduced a light penalty (weight 200.0) in the loss function to guide CRISIS towards negative-mu clusters without overriding data-driven likelihood.
    - **Pragmatic 2D Mapping**: Replaced complex `_swap_latent` logic with a robust (Sigma-split -> Mu-sort) strategy.
- **Results**: Deepest risk-adjusted regime purity. Tail Capture jumped from 35% to **61.4%** while maintaining valid semantic separation.
- **Lessons**: HMM is a Macro Weather Station, not a Micro Umbrella. Decoupling macro regime sensing from micro tail defense is the key to top-tier robustness.

---

## [2026-05-08] v4.1.0 Asymmetric Revolution: Instant Risk & Sustained Recovery (Gemini CLI)
- **Status**: Validated (Friction: 4.20% PASS, Avg Duration: 100.2 bars PASS, Score: 94/100)
- **Problem**: Symmetric duration constraints (e.g., 24h for all states) caused dangerous lags in CRISIS detection and unnecessary wipsaws in stable regimes.
- **Key Fixes**:
    - **Asymmetric Post-Suppression**: Refactored Viterbi output to support per-state minimum durations.
        - CRISIS (1h) / BEAR (2h): Instant escape from tail events.
        - BULL_CALM (36h): Sustained validation before re-allocating capital.
    - **Sticky Penalty Calibration**: Upped `hmm_sticky_penalty_weight` to 100.0 to favor structural clusters over transient noise.
- **Results**: Dramatic friction reduction (7.5% -> 4.2%) while maintaining high-fidelity risk sensing. Avg regime duration crossed the critical 100-bar threshold.
- **Lessons**: Matching the algorithm's temporal inertia to the market's physical reality (sharp drops, slow recoveries) is the key to institutional-grade operational efficiency.

---
<!-- APPEND_POINT: New experiments will be added above this line -->
