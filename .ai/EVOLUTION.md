# 🧬 System Evolution Journal

This file tracks the logical progression and experimental results of the quantitative trading system. It serves as the primary context for AI agents to understand "why" certain changes were made and "what" was learned.

---

## [2026-05-10] v6.9.0: Advanced Alpha Features (v22) — Idiosyncratic & Microstructure (Gemini CLI)

### 1. Architectural Shift: Unlocking Directional Conviction
*   **Problem**: Alpha v21 was elite but trade frequency was low. Long-side participation was weak (PF 0.65) compared to Short-side (PF 3.82). The system lacked features to distinguish idiosyncratic strength from market beta.
*   **Implementation (The Alpha Depth)**:
    1.  **Idiosyncratic Residuals**: Added `idiosyncratic_return_24h` ($R_{asset} - \beta \cdot R_{btc}$) to isolate asset-specific strength.
    2.  **Microstructure Signals**: Added `price_impact_asymmetry` (Liquidity Vacuum) and `exhaustion_cascade_score` (Triple-confluence bottom sensing).
    3.  **Temporal Embedding**: Added `session_seasonality_sin/cos` to allow AI to learn session-specific patterns.
    4.  **Schema v22**: Bumped `GP_FEATURE_SCHEMA_VERSION` to force global cache invalidation.

### 2. Performance Impact (Full-Scale 5,000 Trials)
*   **Alpha Consistency**: **OOS IC maintained at 0.0743 (T-Stat 33.27)**.
*   **Safety Audit**: **MDD remained ultra-low at 0.15%**, confirming HMM robustness.
*   **The Blocking Bottleneck**: Despite better features, **CAGR stayed negative (-0.6%)**. 
*   **Diagnosis**: **Execution Dissonance**. The unified `CS_Z_SCORE_THRESHOLD` is too high (~1.0) for the Long edge.

---

## [2026-05-10] v6.8.0: Magnitude Refactor & OI Data Restoration (Gemini CLI)

### 1. Architectural Shift: Targeting Explosive Alpha
*   **Implementation**:
    1.  **OI Restoration**: Fixed `BinanceVisionDownloader` mapping and normalized timestamps. Backfilled 60 days of metrics.
    2.  **High-Volatility Features**: Added `liq_intensity_proxy` and `capitulation_proxy` (candle tails).
    3.  **Dynamic Friction Hurdle**: AI now ignores "noise-sized" winners using ATR-based targets.

### 2. Performance Impact
*   **Efficiency**: **Profit Factor surged from 0.49 to 0.75 (+53%)**.
*   **Precision**: **Win Rate improved to 57.26%**.
*   **Safety**: **MDD reduced to 0.16%**.

---

## [2026-05-10] v6.7.0: P1+P2 Integration — Magnitude Model & Linear Modulator (Opus 4.7)

### 1. Architectural Changes
*   **Modulator**: Shifted from tanh saturation to `clip(target_var/(RA·var), 0.25, 1.75)`.
*   **Magnitude**: Added `LGBMRegressor` per slot for 24h magnitude prediction.
*   **Slot Expansion**: Increased from 15 to 18 slots to accommodate interaction themes.

### 2. Experimental Results
*   **IS-OOS Retention**: Improved from -331% → **+86.4%** (structural stability).
*   **HO CAGR**: **+0.28%** (out-of-holdout positive for the first time).

---

## [2026-05-10] v6.6.0: Natural Risk-Adjusted Scaling (Gemini CLI)
- **Status**: Validated (Net Alpha improved from -65.4% to -15.94% (+49.5%p)).
- **Logic**: Replaced heuristic overrides with $RA_{dyn} = 1.0 + 3.0 \cdot P_{crisis} + 1.5 \cdot P_{bear}$.

---

## [2026-05-10] v6.5.0-p2-kelly-diag: Kelly Sizing Root Cause (Haiku)
- **Discovery**: Kelly sizing was fundamentally underpowered due to Platt Scaling under-discrimination ($ml\_calib\_prob \approx 0.496$).

---

## [2026-05-09] v6.4.x: Directional Symmetry & Execution Liberation (Gemini CLI)
- **Status**: Record **OOS IC 0.1466**.
- **Fixes**: Symmetric hybrid labeling and search space expansion. Identified 87% OOS defensive mapping as the blocker.

---

## [2026-05-09] v6.3.x: PLGD Objective & Hysteresis Restoration (Gemini CLI)
- **Status**: Validated (MDD 0.18%, PF 1.74).
- **Logic**: Maximizing Probabilistic Log Growth Deflation (PLGD) and Schmitt Trigger filtering.

---

## [2026-05-09] v6.0.0: Horizon Pivot (4h) & Decoupled Architecture (Gemini CLI)
- **Status**: Validated (OOS CAGR: +24.8% PASS, PF: 1.14 PASS).
- **Logic**: Upgraded from 1h to 4h base timeframe. Decoupled HMM from Alpha Ranking.

---

### 🏛️ Historical Summary (v1.x - v5.x)
Prior to v6.0.0, the system evolved through Guided HMM architectures (v1-v2), unsupervised regime discovery (v3), and deterministic policy mapping using Kelly Criterion (v5). Key lessons included the failure of binary heuristic overrides and the necessity of frequency matching between alpha and execution.
*Full history preserved in `.ai/archive/EVOLUTION_v5.md`.*

---
<!-- APPEND_POINT: New experiments will be added above this line -->
