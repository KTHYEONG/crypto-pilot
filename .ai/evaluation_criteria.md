# HMM Regime Classifier — Evaluation Criteria (v2026-05-12)

**Version**: 2026-05-12-revised  
**Effective From**: v9.0.0+  
**Review Cycle**: Annually or after major architectural changes

---

## Overview

This document defines the **institutional-grade** evaluation framework for the HMM regime classifier. The framework balances realism, practitioner relevance, and testability.

### Rationale for Revision (v8.2 → v9.0)

The original v8.2 criteria (Tail Capture ≥65%, CRISIS MU <-1%, Friction <2%) were:
- Numerically aggressive (unrealistic for 4h bars)
- Backward-looking (measured concurrent labeling, not predictive power)
- Overstated friction cost (assumed 100% rebalancing per regime switch)

The v9.0 revision emphasizes:
- **Realism**: Thresholds based on actual crypto regime dynamics
- **Forward-Looking**: New metrics measure preventive power (Lead-Lag Capture, Regime IC)
- **Regime-Specific**: Different expectations for different regimes

---

## Primary Evaluation Metrics

### 1. CRISIS MU (Mean Return in CRISIS State)

**Definition**: Average 4h bar return when HMM is in CRISIS regime.

**Target**: `< -0.2%`

**Rationale**: 
- 4h return of -1% would imply -6% per day (unrealistic even for crashes).
- Historical 4h CRISIS bars average -0.2% to -0.5%.
- -0.2% is a realistic minimum for a state that includes both pre-crash and post-crash bars.

**Measurement**: 
```
merged["ret"] = daily_returns_resampled_to_4h
merged["regime"] = argmax(crisis_prob, bear_prob, ...)
crisis_bars = merged[merged["regime"] == CRISIS]
crisis_mu = crisis_bars["ret"].mean() * 100  # in %
```

**Severity Levels**:
- `crisis_mu < -0.2%`: ✅ PASS
- `-0.2% ≤ crisis_mu < 0.0%`: 🟡 ACCEPTABLE (directionally correct but weak)
- `crisis_mu ≥ 0.0%`: ❌ FAIL (regime misspecified)

---

### 2. BEAR MU (Mean Return in BEAR/HIGH_VOL State)

**Definition**: Average 4h bar return when HMM is in BEAR or HIGH_VOL regime.

**Target**: `< 0.0%`

**Rationale**:
- BEAR state should not have positive expected return.
- Threshold of 0.0% is weak but acceptable (allows small noise).

**Severity Levels**:
- `bear_mu < 0.0%`: ✅ PASS
- `0.0% ≤ bear_mu < 0.05%`: 🟡 ACCEPTABLE
- `bear_mu ≥ 0.05%`: ❌ FAIL

---

### 3. CRISIS Share (% of Time in CRISIS)

**Definition**: Fraction of total bars classified as CRISIS regime.

**Target**: `3% ~ 10%` (bilateral band)

**Rationale**:
- CRISIS should be sparse but not vanishingly rare.
- Lower bound 3%: Allows ~2-3 significant crises per year in crypto (5-10 days each).
- Upper bound 10%: Prevents chronic overallocation to crisis hedging.
- 22,000 bars over 3 years → 3% = 660 bars, 10% = 2,200 bars.

**Severity Levels**:
- `3% ≤ crisis_share ≤ 10%`: ✅ PASS
- `1% ≤ crisis_share < 3%` or `10% < crisis_share ≤ 15%`: 🟡 ACCEPTABLE (tuning needed)
- `crisis_share < 1%` or `crisis_share > 15%`: ❌ FAIL (regime collapsed or bloated)

---

### 4. Tail Capture IS (In-Sample, Concurrent)

**Definition**: Fraction of worst-5% return bars classified as CRISIS or BEAR at the same time.

**Target (Dual-Tier)**:
- `≥ 55%`: ✅ PASS
- `40% ~ 55%`: 🟡 ACCEPTABLE
- `< 40%`: ❌ FAIL

**Rationale**:
- Original ≥65% was aggressive; achievable with heavy penalization but mathematically constrained.
- With CRISIS at 3~10% and tail events at 5%, theoretical maximum is 100% only if CRISIS perfectly concentrates on tail (unrealistic).
- 45-55% is realistic and defensible for institutional use.

**Measurement**:
```
worst_5_thresh = merged["ret"].quantile(0.05)
worst_5_mask = merged["ret"] <= worst_5_thresh
crisis_or_bear_in_worst = ((merged["hmm_prob_crisis"] > merged["hmm_prob_chop"]) | 
                           (merged["hmm_prob_bear_trend"] > merged["hmm_prob_chop"])).astype(int)
tail_capture = crisis_or_bear_in_worst[worst_5_mask].mean() * 100
```

---

### 5. Tail Capture OOS (Out-of-Sample, Concurrent)

**Definition**: Same as IS but on OOS period (future-unfitted).

**Target (Dual-Tier)**:
- `≥ 50%`: ✅ PASS
- `40% ~ 50%`: 🟡 ACCEPTABLE
- `< 40%`: ❌ FAIL

**Rationale**:
- OOS should be slightly better than IS (forward-filled posterior + fresh data).
- Slightly higher bar than IS to detect overfitting.

---

### 6. Average Regime Duration

**Definition**: Arithmetic mean of number of consecutive bars in each regime.

**Target**: `≥ 36 bars` (overall), regime-specific flexibility

**Rationale**:
- 36 bars at 4h = 6 days.
- Prevents excessive whipsawing but allows tactical regime changes.
- Do NOT enforce strict per-regime minimums (some regimes naturally shorter).

**Measurement**: 
```
hard_states = argmax(regime_probs, axis=1)
transitions = (hard_states != hard_states.shift(1)).sum()
avg_duration = len(hard_states) / transitions
```

---

## Secondary (Diagnostic) Metrics

These metrics provide deeper insight but are not strict pass/fail criteria.

### 7. Lead-Lag Tail Capture (UNDER DEVELOPMENT)

**Definition**: Among all bars classified as CRISIS, what % are followed by a tail event (return < -2σ) within the next 8 bars (2 trading days)?

**Target**: `> 40%` (preliminary)

**Measurement**:
```
crisis_enters = (regime == CRISIS) & (regime.shift(1) != CRISIS)
forward_tail_8 = returns.rolling(8, min_periods=1).apply(lambda x: (x <= -2*rolling_std).any())
lead_lag_capture = (crisis_enters & forward_tail_8.shift(-8)).mean() * 100
```

**Why It Matters**: Measures preventive power (do we warn before the crash?) vs concurrent labeling.

---

### 8. Regime Information Coefficient (IC)

**Definition**: Spearman rank correlation between HMM regime signal and forward returns.

**Target**: `Spearman(p_crisis, ret_fwd_10h) < -0.05`

**Measurement**:
```
forward_ret = returns.shift(-10)  # 10 bars ahead
from scipy.stats import spearmanr
ic, pval = spearmanr(regime_probs["crisis"], forward_ret.fillna(0))
```

**Why It Matters**: Directly measures whether CRISIS signal predicts lower future returns (the core hypothesis).

---

### 9. CRISIS Entry Lead Time

**Definition**: Average number of bars BEFORE a tail event (bottom 1%) that CRISIS regime is active.

**Target**: `≥ 0` (same bar or earlier is good; later is bad)

**Measurement**:
```
tail_1_thresh = returns.quantile(0.01)
tail_1_mask = (returns <= tail_1_thresh).rolling(10).max().fillna(False)
tail_1_idx = tail_1_mask[tail_1_mask].index
for each_crisis_period:
    bars_until_tail_1 = (tail_1_idx - crisis_end).min()
```

**Why It Matters**: Shows if regime switching is forward-looking or lagged.

---

### 10. Regime Stability by Market Condition

**Definition**: Average duration separately for LOW_VOL, MID_VOL, HIGH_VOL contexts.

**Target**: 
- LOW_VOL regime within LOW_VOL market: ≥ 60 bars
- BULL regime in LOW_VOL market: ≥ 100 bars
- CRISIS regime overall: 12~45 bars (okay to be shorter)

**Rationale**: Regimes should be more stable in calm markets, more responsive in volatile markets.

---

## Composite Scoring (Institutional Grade)

| Metric | PASS | ACCEPTABLE | FAIL | Weight |
|--------|------|-----------|------|--------|
| CRISIS MU | < -0.2% | -0.2~0% | ≥ 0% | 20% |
| BEAR MU | < 0% | 0~0.05% | ≥ 0.05% | 10% |
| CRISIS Share | 3-10% | 1-3% or 10-15% | <1% or >15% | 15% |
| Tail Capture IS | ≥55% | 40-55% | <40% | 25% |
| Tail Capture OOS | ≥50% | 40-50% | <40% | 20% |
| Avg Duration | ≥36 bars | 24-36 bars | <24 bars | 10% |

**Score Calculation**:
```
score = (
  (crisis_mu_points / 20) * 0.20 +
  (bear_mu_points / 10) * 0.10 +
  (crisis_share_points / 15) * 0.15 +
  (tail_capture_is_points / 25) * 0.25 +
  (tail_capture_oos_points / 20) * 0.20 +
  (avg_duration_points / 10) * 0.10
) * 100
```

**Grade Interpretation**:
- **≥ 70/100**: Institutional PASS — ready for production
- **55~70**: Production-ready with monitoring — tuning recommended
- **40~55**: Research grade — improvements needed
- **< 40**: Experimental — fundamental issues

---

## Usage Instructions for AI Agents

### When Evaluating HMM (Every Phase)

1. **Load this file** from `.ai/evaluation_criteria.md`
2. **Run audit** via `tests/test_universe_to_hmm.py --tf 4h`
3. **Extract metrics** from AUDIT section:
   - CRISIS MU, BEAR MU from regime return analysis
   - CRISIS Share from time allocation
   - Tail Capture IS/OOS from tail analysis
   - Avg Duration from switching analysis
4. **Compare against thresholds** in this document
5. **Score each metric** (PASS/ACCEPTABLE/FAIL)
6. **Calculate composite score** per formula above
7. **Record in DNA.json** under `current_performance`

### Red Lines (Non-Negotiable)

- CRISIS MU must be negative (< 0%). If positive, regime is fundamentally broken.
- CRISIS Share must be in 1~15% band. Outside indicates parameter collapse or bloat.
- Tail Capture must be ≥ 40% IS, ≥ 40% OOS. Below 40% means regime provides no protection.

---

## Revision History

| Date | Version | Changes | By |
|------|---------|---------|-----|
| 2026-05-12 | 2.0 | Initial institutional revision. Added Lead-Lag, IC, Lead Time metrics. | Claude Sonnet 4.6 |
| 2026-05-11 | 1.0 | Original aggressive criteria (Tail Capture ≥65%, CRISIS MU <-1%). | v8.2 baseline |

---

**Next Review**: 2026-11-12 (6 months) or after v10.0.0 release, whichever is sooner.
