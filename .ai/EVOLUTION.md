# G-ALPHA Evolution Log (Newest First)

## [2026-05-17] v9.0.3 A3+B1+B2 Final Implementation (Claude Haiku 4.5)

**Implemented**: Short model elimination (A3) + Triple-barrier labels (B1) + Market-neutral demeaning (B2)

**Results**:
- Learning time: 172s → 34s (−80% via short CatBoost elimination)
- Survivors: 6 slots → **2 slots** (both IC-linear baseline, zero CatBoost)
- IC Retention: 72% → 83.8% (+11.8%p, quality improvement)
- OOS-CS-IC: 0.0460 → 0.0369 (−0.009, diversity reduction but higher quality)

**Critical Discovery**: CatBoost + Triple-barrier has compatibility failure. Discrete 3-value labels {0, 0.5, 1} create 50% "neutral" pairs that uninform YetiRank pairwise loss. Linear IC-weighted blend of raw features survives where GB trees fail. **Validates Tier 1 hypothesis: Linear models may be more robust for low-SNR crypto CS alpha than nonlinear GB**.

**Decision Point**: Continue with (1) barrier tuning to reduce neutral%, (2) RMSE loss instead of YetiRank, or (3) Linear-first architecture.

---

## [2026-05-17] v9.0.2 A1+A2 Implementation (Claude Sonnet 4.6)

**Implemented**: Eliminate fake breadth (A1) + Add linear baseline (A2)

**Results**:
- Staged predictions (8 checkpoints from 1 model) → 3 independent models with subsample/rsm/seed variation
- Effective breadth: 4-6 (honest) vs 24 (claimed)
- IC Retention: 55.4% → 72.0% (+16.6%p improvement)
- OOS-CS-IC: 0.0565 → 0.0499 (−0.0066, expected from stronger FDR)
- Linear baseline IC: theme0=0.050, theme1=0.026 (competitive with CatBoost 0.057-0.068)
- Training time: 63s → 172s (+109s for 3 independent models)

**Validation**: A1 removed structural lie in breadth accounting. A2 proved linear factor blending is viable baseline—CatBoost non-linearity gap was theme-dependent, not universally superior.

**Findings**: Theme1 (Vol/MR) linear IC (0.026) matched CatBoost IC (0.017), indicating CatBoost's advantage is marginal for this subspace. Half-life gate removal (ICIR+pos_ratio+sub_period majority vote) was effective gate replacement.

---

## [2026-05-16] v9.0.1 Architectural Review & Improvement Identification (Claude Opus 4.7)

**Analysis**: Identified 5 fundamental flaws in v8.0 (staged prediction fake breadth, short label mirroring, overlapping label stat inflation, no BTC neutralization, over-tuned gates).

**Tiered Improvement Plan**:
- **Tier A (immediate)**: A1 (breadth), A2 (baseline), A3 (short)
- **Tier B (medium)**: B1 (triple-barrier + Newey-West), B2 (BTC-neutralization)
- **Tier C (cleanup)**: Collapse gate zoo, Walk-forward CV

**Root Cause Analysis**: CatBoost YetiRank (pairwise ranker) may be overfit for cross-sectional alpha with N_symbols ≤ 100 and IC ≤ 0.08. Fundamental Law IR=IC√breadth assumes honest breadth; with staged checkpoints correlating 0.85+, effective breadth was ~1.5-2 per theme, not 8.

---

## [2026-05-14 to 2026-05-16] v8.0 Bug Fixes (5 items) (Claude Opus 4.7)

**Fixed**:
1. Half-life leak: per-component tracking dict (hl_by_col) prevents variable shadowing
2. OOS CS-IC: ic_oos_by_slot dict added to meta, CSIC-OOS column displays OOS IC (was showing IS IC duplicate)
3. IC Retention formula: uses IS+OOS paired IC values, not phantom ic_by_slot_is key
4. Short label: replaced 1-y_labels mirror with -fwd_ret_6 based downside objective (BROKEN, reverted in A3)
5. n_trials: 48 → 6 (3 themes × 2 directions; staged snapshots are not independent)

**Result**: [READY] verdict with 16→10 survivors, but theme2 LONG failed on half-life gate (AR1 ρ requirement was too strict for 2.8-2.9b horizon).

---

## [2026-05-12] v8.0 Audit & Problem Statement (Initial)

**Problem**: G-ALPHA v8.0 audit output suspicious: half-life all 1.5b (same value across 48 slots = variable leak), IC Retention 0.0%, OOS-CS-IC values inflated, [READY] verdict untrustworthy.

**Analysis**: Found 5 concrete bugs in component_filter.py, miner.py (labeled Fix 1–5). After fixes: IC Retention 55.4%, [READY]-16 slots, half-life per-slot 2.1-5.2b. But theme2 still failing half-life gate due to AR1 gate being structurally flawed (rewards non-stationary IC, penalizes stable-noisy IC).

---

## Remaining Tier B+C Improvements (Not Yet Implemented)

### B3: Walk-Forward Validation (3-4 folds, embargo)
- Current: single IS/OOS split, one high-variance draw
- Replace with time-series walk-forward (Purged + Embargoed CV per López de Prado)
- Impact: Prove temporal generalization, not just cross-sectional

### B2-Extended: BTC-Beta Feature Neutralization
- Current: Features may be BTC-beta proxies, inflating IC in bull markets
- Implement: Cross-sectional regression of alpha signals against rolling BTC-beta per symbol
- Or: Neutralize target against BTC return component before IC computation
- Impact: Remove market-factor noise, expose true idiosyncratic alpha

### C1: Gate Zoo Collapse
- Current: 9 gates with hand-tuned thresholds (FDR, DSR, ICIR, tail, OOS, short, LSO, regime, bal)
- Simplify: FDR + OOS-IC (single criterion) + walk-forward stability (one metric)
- Demote rest to diagnostics (log only, don't gate)
- Impact: Reduce researcher degrees of freedom, improve robustness

### C2: Model Class Diversification
- Current: CatBoost YetiRank is sole ranker (failing with triple-barrier)
- Propose: (Linear Ridge + CatBoost + LightGBM) ensemble, ensemble weight by OOS-IC
- Impact: If one class fails, others provide signal; empirical validation of "best model"

---

## Schema Notes

- **version**: Semantic versioning. Major = architecture shift (e.g., v8.0→v9.0 is staged→independent). Minor = feature/parameter. Patch = bug fix.
- **previous_stable_state**: Git commit hash for rollback if regression detected.
- **deployment_readiness**: Subjective gate. READY = all tests pass + metrics healthy. MARGINAL = functional but fragile. BLOCKED = known showstoppers.
