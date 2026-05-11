# HMM Regime Classifier — Evaluation Criteria (v2026-05-12)

**Version**: 2026-05-12-preventive-pivot  
**Effective From**: v9.2.0+  
**Review Cycle**: Quarterly

---

## Overview

This document defines the **Institutional-Grade** evaluation framework for the HMM regime classifier. Following the v9.2.0 architectural shift, the framework has moved from "Reactive Labeling" to **"Preventive Risk-Off"** assessment.

### The v9.2.0 Philosophical Shift
- **CRISIS as Insurance**: The `CRISIS` state is now defined as a **Preventive Early Warning (Top-Heavy)** signal. It triggers during market overheating (high funding, low volatility), often resulting in a **positive MU**. This is an acceptable "insurance premium" to avoid the subsequent tail event.
- **BEAR_TREND as Realized Risk**: The actual structural isolation of downward price action is now exclusively the responsibility of the `BEAR_TREND` state.

---

## Primary Evaluation Metrics (Hard Pass/Fail)

### 1. Lead-Lag Tail Capture (Preventive Power)
**Definition**: Among all bars where `CRISIS` state is triggered, what % are followed by a Tail Event (Return < -2σ) within the next **8 bars (32 hours)**?

**Target**: `> 40%`

**Severity Levels**:
- `> 40%`: ✅ PASS (Institutional Grade)
- `25% ~ 40%`: 🟡 ACCEPTABLE (Strategic monitoring required)
- `< 25%`: ❌ FAIL (Signal provides no foresight)

**Rationale**: In crypto, waiting for a crash to trigger a signal is too late. The value of this model lies in its ability to warn *before* the liquidation cascade begins.

---

### 2. BEAR_TREND MU (Realized Risk Isolation)
**Definition**: Average 4h bar return when HMM is in `BEAR_TREND` state.

**Target**: `< -0.2%`

**Severity Levels**:
- `mu < -0.2%`: ✅ PASS
- `-0.2% ≤ mu < 0.0%`: 🟡 ACCEPTABLE
- `mu ≥ 0.0%`: ❌ FAIL

**Rationale**: Since `CRISIS` has moved to a preventive role, `BEAR_TREND` must demonstrate pure bearish structural isolation. If BEAR MU is positive, the downward filtering logic is broken.

---

### 3. CRISIS Share (% of Time in Shutdown)
**Definition**: Fraction of total time classified as `CRISIS`.

**Target**: `1% ~ 8%` (Bilateral Band)

**Severity Levels**:
- `1% ≤ share ≤ 8%`: ✅ PASS
- `0.5% ≤ share < 1%` or `8% < share ≤ 12%`: 🟡 ACCEPTABLE
- `share < 0.5%` or `share > 12%`: ❌ FAIL

**Rationale**: `CRISIS` triggers a portfolio shutdown (Exposure = 0). Excessive triggers lead to unacceptable opportunity costs (frictional drag).

---

## Secondary Evaluation Metrics (Diagnostic)

### 4. Regime Information Coefficient (Spearman IC)
**Definition**: Spearman rank correlation between `p_crisis` and **4-bar forward returns**.

**Target**: `IC < 0.0` (Directionally correct)

**Rationale**: While the target is still negative, the window is shortened to 4 bars to match crypto dynamics. Given the preventive nature and mean-reversion, this is used for diagnostic trend analysis, not hard rejection.

---

### 5. Concurrent Tail Capture (High-Vol Isolation)
**Definition**: Fraction of worst-5% return bars classified as `CRISIS` or `BEAR_TREND` concurrently.

**Target**: `> 40%`

**Rationale**: While preventive power is primary, the model should still recognize a crash while it is happening. However, because `CRISIS` now suppresses itself during washout/capitulation bottoms (to avoid V-bounce contamination), this target is lowered from previous versions.

---

### 6. CRISIS MU (Early Warning Purity)
**Definition**: Average 4h bar return when HMM is in `CRISIS` state.

**Expectation**: **POSITIVE (+0.5% ~ +2.0%)**

**Rationale**: A positive MU for `CRISIS` validates that the signal is catching the "Top-Heavy" market phase (overheating) before the crash. A negative MU here would suggest the model is still lagging.

---

## Composite Scoring Logic

| Metric | Weight | Pass Criteria |
| :--- | :---: | :--- |
| **Lead-Lag Tail Capture** | **40%** | > 40% |
| **BEAR_TREND MU** | **25%** | < -0.2% |
| **CRISIS Share** | **15%** | 1% - 8% |
| **Concurrent Tail Capture** | **10%** | > 40% |
| **Regime Stability** | **10%** | > 24 bars |

---

## Revision History
- **2026-05-12 (v2.1)**: Preventive Pivot. Promoted Lead-Lag to Primary. Redefined CRISIS MU as positive-expected. Window shortened to 4-bar.
- **2026-05-12 (v2.0)**: Initial institutional revision. Added Lead-Lag/IC.
