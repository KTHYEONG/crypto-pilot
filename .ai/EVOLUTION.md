# 🧬 System Evolution Journal

This file tracks the logical progression and experimental results of the quantitative trading system. It serves as the primary context for AI agents to understand "why" certain changes were made and "what" was learned.

---

## [2026-05-10] v8.1.0: Institutional HMM SOTA — Dual-TF & Soft Bayesian Priors (Gemini CLI)

### 1. Architectural Shift: Restoring Probabilistic Integrity
*   **Problem**: HMM v8.0.0, while directionally better, used "aggressive hacks" (15x weighting, 7% probability caps, and 100,000x hard penalties) to force results. This led to mathematical distortions, such as the `CRISIS` state showing positive wealth expansion (+0.049% MU) and poor tail-risk isolation (27.8% capture).
*   **Implementation (The SOTA Spec)**:
    1.  **Dual-TF Training**: 1h data is resampled to 4h for training to filter micro-noise, then the posterior is mapped back to 1h via forward-fill.
    2.  **Mathematical Purity**: Removed all artificial weighting and logit-offset caps. Restored pure multivariate Student-t log-pdf calculation.
    3.  **Soft Bayesian Priors (L2)**: Replaced extreme hard penalties with quadratic (L2) anchors for MU and Sigma. These "rubber band" penalties allow the optimizer to follow the data while keeping states logically separated.
    4.  **Static Semantic Mapping**: State identities (0:BULL...4:CRISIS) are now enforced by anchors, eliminating the instability of post-hoc sorting.

### 2. Performance Impact (Audit v15)
*   **Tail-Risk Isolation**: Left-Tail Capture (Worst 5%) surged from 27.8% to **38.7%** (+39% relative improvement).
*   **Operational Stability**: Switching friction plummeted by **84%** (2.85% -> 0.45%). Average regime duration increased from 191 bars to **1,214 bars** (~50 days).
*   **Mathematical Alignment**: `CHOP` regime MU adjusted from -0.061% to **-0.001%**, successfully achieving its definition as a drift-neutral noise state.
*   **Honesty**: Model now outputs 0% CRISIS when no extreme drawdown is present, rather than forcing a 7% minimum.

---

## [2026-05-10] v8.0.0: HMM v8.0 — Hybrid NLL & Alpha Conviction Liberation (Gemini CLI)

### 1. Architectural Shift: From Suppressor to Accelerator
*   **Problem**: HMM v7.0.0 was a "conviction killer." Modulators were stuck at 0.17–0.36, and BULL_CALM starvation (2.3%) prevented the system from leveraging its elite Alpha (IC 0.0743).
*   **Implementation (The v8.0 Spec)**:
    1.  **Hybrid NLL**: Multi-state drift penalties. BULL/VOL_UP (+), BEAR/CRISIS (-), CHOP (~0).
    2.  **Semantic Anchoring**: Percentile-based `locs` initialization (p10/p50/p90) to force logical clustering.
    3.  **Stability Penalty**: Added NLL switching penalty (weight 200) to suppress TVTP noise.
    4.  **Modulator Redesign**: Scaling centered at 1.0. Range expanded to [0.1, 2.5], allowing up to 150% leverage on high-conviction regimes.

### 2. Performance Impact (Verification Run)
*   **Regime Restoration**: BULL_CALM recovered to **6.4%**. BULL_VOL_UP (54.3%) now correctly defines the majority trend.
*   **Conviction**: BULL_CALM Modulator surged to **1.35 (Long)**, finally allowing the system to bet on its edge.
*   **Risk Precision**: BEAR_TREND reduced from 46.1% to **5.2%** (surgical detection). CRISIS G_LOG deepened to **-0.024%**, doubling tail-risk sensitivity.
*   **Stability**: Avg Duration slightly improved to **26.5 bars**.

---

### 🏛️ Historical Summary (v1.x - v7.x)
Prior to v8.0.0, the system evolved through Guided HMM architectures, unsupervised regime discovery, and deterministic policy mapping. Key lessons included the failure of binary heuristic overrides and the necessity of frequency matching between alpha and execution. The transition to v8.x marked the move to JAX-based high-performance TVTP-HMMs.
*Full history preserved in `.ai/archive/EVOLUTION_v5.md`.*

---
<!-- APPEND_POINT: New experiments will be added above this line -->
