# Experiment: 5-State TVTP-HMM + Empirical Mapping Evolution (Phase 1~3+patch)

## Date: 2026-05-08
## Researcher: Claude Opus 4.7

---

### 1. Starting Point (v3.0.0 baseline — pure unsupervised 4-state)
- TC IS: 21.8% FAIL | TC OOS: 21.5% FAIL
- CRISIS G_log: -0.133% TAIL_DEFENSE ✓ | BULL G_log: +0.017%
- Avg Duration: 73.6 bars | Switches: 297 | Friction: 7.42%
- Score: 42/100

**Root cause identified**: μ-only post-hoc mapping (sort by locs[:, 9]) caused BULL to dominate
68.4% of time, absorbing 60%+ of worst-5% events. `returns_ser` arg was unused.

---

### 2. Phase 1+2 — Empirical 2D Mapping + AdamW + EMA Fix

**Changes**:
- `_map_semantic_states`: replaced μ-only locs ranking with Viterbi hard-label + empirical (μ_emp, σ_emp, Sharpe) per latent state.
- `_assign_semantic_four_state`: CRISIS = argmin(Sharpe) among high-σ states; BULL = argmax(Sharpe) among low-σ.
- `returns_ser` now consumed: aligned to features_df.index via `reindex`.
- EMA blend moved to **pre-training warm-start**: `curr_p = 0.8*old + 0.2*fresh` before optimizer.init(); post-training EMA removed (was degrading convergence).
- optimizer: `optax.adam` → `optax.adamw(lr=0.02, wd=1e-4)`.
- Warmup: iters 1 → 10, added `_viterbi_hard_onehot` to warmup.
- Mapping computed once per fit-step on training window; reused at inference.

**Results (unconstrained)**:
| Metric | Value |
|---|---|
| TC IS | **65.8% ACCEPTABLE** |
| TC OOS | **67.7% ACCEPTABLE** |
| CRISIS μ | **+0.017% ❌** (parabolic up-move absorbed) |
| CRISIS G_log | +0.010% (semantics broken) |
| IC CRISIS_P t+1 (sign-adj) | +0.018 (sign correct) |
| Score | ~59/100 |

**Lesson**: 2D mapping + empirical returns dramatically improves TC but parabolic up-move clusters (high-σ, positive-μ) get labeled CRISIS when μ constraint is absent.

---

### 3. CRISIS μ<0 Constraint Patch

**Change**: `_assign_semantic_four_state` — if `emp_mu[crisis_idx] >= 0`, override to `argmin(emp_mu)` and remap remaining 3 states.

**Results**:
| Metric | Value |
|---|---|
| TC IS | 20.5% FAIL (reverted to v3.0.0 level) |
| CRISIS G_log | **-0.168%** (best ever) |
| Score | ~44/100 |

**Lesson**: μ<0 constraint restores CRISIS semantics but TC collapses because the min-μ state (6.9% time) cannot outcompete BULL (68.4%) in tail absorption.

**Confirmed tradeoff**: Without state proliferation, Phase 1+2 unconstrained (TC 65.8%) and Phase 1+2+patch (TC 20.5%) represent the two poles. Post-hoc mapping alone cannot achieve both simultaneously.

---

### 4. Phase 5 — 5-State HMM (CALM_BULL + VOL_UP split)

**Changes**:
- `n_states`: 4 → **5**
- New state: `bull_vol_up` (high-σ positive-μ, for parabolic up-moves)
- `_assign_semantic_five_state`: 5-way semantic assignment (CRISIS, BEAR, CHOP, BULL_CALM, VOL_UP)
- `locs` init: 5×13 with explicit (trend, vol) seeding per state
- `_TVTP_FEATURE_INDICES`: added `macro_trend_168h` (idx 0) → 6 TVTP features
- `HMM_SEMANTIC_PROB_COLUMNS` updated in `feature_engineering.py` to 5 columns
- `hmm_prob_bull_trend` = `bull_calm + bull_vol_up` derived column for backward compat
- `_calibrate_crisis_logit_offset`: generalized with `crisis_col` param
- `_normalized_entropy_k4` → `_normalized_entropy_k(probs, k)` generalized

**Results**:
| Metric | Value |
|---|---|
| TC IS | **29.6%** |
| TC OOS | **29.1%** |
| BULL_CALM G_log | **+0.051% WEALTH_EXP** (first time!) |
| VOL_UP TIME% | 0.5% (state collapsed — init insufficient) |
| CRISIS G_log | -0.131% TAIL_DEFENSE |
| LGB IS IC | **0.370** (best ever) |
| LGB OOS IC | **0.326** |
| Score | 50/100 |

**Lesson**: 5-state correctly separates calm bull; BULL_CALM achieves WEALTH_EXP for first time. But VOL_UP collapsed to 0.5% — parabolic up-move cluster not separating due to insufficient training examples matching `(trend=+1.5, vol=+2.0)` init.

---

### 5. Phase 3 — Return Tail Features (13 observations)

**Changes**:
- Added 3 new features to `SYSTEMIC_HMM_FEATURE_COLUMNS`:
  - `macro_ret_5d_z`: 120h rolling z-score of returns (slow-moving drawdown signal)
  - `macro_ret_skew_24h`: 24h return skewness (left-tail asymmetry)
  - `macro_ret_kurt_24h`: 24h return kurtosis (fat-tail density)
- Total features: 10 → **13**
- TVTP indices unchanged (0,4,5,6,7,8) — new ret_* features excluded from transition logits (obs-only)

**Results**:
| Metric | Value |
|---|---|
| TC IS | **35.2%** (best with valid CRISIS) |
| TC OOS | **35.2%** |
| Avg Duration | **104.1 bars** (best ever, +36% vs Phase 5) |
| Total Switches | **210** (best, -26% vs Phase 5) |
| Friction | **5.25%** (best, first time <6%) |
| CRISIS G_log | -0.082% UNSTABLE |
| BULL_CALM G_log | -0.002% (regression: label swapped to sideways state) |
| Score | 54/100 |

**Lesson**: Tail features (skew, kurt, 5d_z) dramatically improve stability by acting as slow-moving anchors. TC improves to 35.2%. But `_assign_semantic_five_state` maps BULL_CALM to the low-σ sideways state (μ≈0) — stability features cause HMM to cluster on distribution shape rather than return direction.

---

### 6. BULL_CALM + BEAR μ Constraint Patch

**Changes**:
- `_assign_semantic_five_state`: added `_swap_latent` for BULL_CALM (μ≤0 → swap to max-μ) and BEAR (μ>0 → swap to min-μ among non-crisis).
- CRISIS constraint retained.

**Results**:
| Metric | Value |
|---|---|
| TC IS | 30.8% FAIL (regressed from 35.2%) |
| TC OOS | 32.0% FAIL |
| CRISIS G_log | -0.103% **TAIL_DEFENSE** ✓ (restored) |
| BULL_CALM μ | still -0.006% (swap triggered but CALM still assigned to low-μ state via size dominance) |
| Score | 53/100 |

**Lesson**: BEAR μ<0 constraint fires and reassigns BEAR to a lower-σ state, reducing tail absorption. Multiple μ constraints fight each other and produce net TC regression. Stacking post-hoc constraints has reached diminishing returns.

---

### 7. Confirmed Architectural Limits of Post-Hoc Mapping

```
BULL_CALM 57.5% time → absorbs ~54% of worst-5% events regardless of label
Max achievable TC (CRISIS+BEAR only) ≈ 35% with current 5-state unsupervised
TC 65.8% achieved only when BEAR=41.2% (CRISIS semantics broken)
```

**Next recommended directions**:
- **Option A (Hybrid NLL)**: Reintroduce light μ_return_penalty (weight 200-400, not 10,000x) in NLL to soft-guide state separation without overriding likelihood. Expected TC: 45-55%.
- **Option B (Redefine metric)**: Accept TC ~35%, shift success metric to Sharpe-ratio gap between BULL_CALM and CRISIS/BEAR regimes for portfolio signal quality.
- **Option C (Rollback)**: Return to Phase 1+2 unconstrained (TC 65.8%) for pure tail-capture use case, accepting that CRISIS label ≠ semantic crisis.
