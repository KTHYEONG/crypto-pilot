# Alpha Execution Results - 2026-05-30 (Post FAIL-1/2 Fix)

## Executive Summary
- **Status:** `ALPHA_PASS=FALSE` (정직한 평가)
- **판정 사유:** G2(경제거래성) FAIL — post-clip IC 손실, basket net bps 음수, SWEEP 0/3
- **기술 개선:** 게이트 정정 완료(N_eff floor, G2 병합, clip_preservation_ratio 방향), P1 rank-sizing 기본 동작화
- **테스트:** 574 passed, 0 failed, overflow 경고 제거 ✅

---

## Execution Log (Latest Run)

```
🔬 [BE-EFF] N_raw=17.0 N_eff=1.5 sigma_r=666.3bps be_raw=0.0116 be_eff=0.0174 gap_resid_eff=+0.0273
🏅 [RANK-QUALITY L1] ic=0.0062 t=3.01 hit=0.130 breadth=3.7
🏅 [RANK-GENERALIZE] oos_rank_ic=0.0454 is_rank_ic=0.0098 retention=4.65

📊 [ALPHA SCOREBOARD]
Metric  | RESID_IC | T-STAT  | N_EFF | DSR    | BE_EFF(12h) | BEAR_IC
Value   | 0.0141   | 3.62    | 1.7   | 1.0000 | 0.0386      | 0.0240
Result  | ❌       | ✅      | 1.7   | ✅     | FAIL        | ✅

🧺 [L3-BASKET] ew_bps=-12.94 net_bps=-36.94 ir_t=-2.61 hit=0.494
🌐 [REGIME IC] Bull: 0.011 | Bear: 0.024 | Chop: -0.006
📈 SWEEP: [6h: ❌] [12h: ❌] [18h: ❌] (0/3)

>> ALPHA_PASS: FALSE
   [G1: resid_ic=0.0141 be_eff=0.0386 gap=-0.0244 t=3.62 bear_ic=0.0240]
   [G2: gap_raw=-0.0416 net_bps=-36.9 ir_t=-2.61 presv=-0.21 sweep=0/3]
   fail=['resid_ic_below_breakeven_eff', 'g2a_gap_raw_non_positive', 
         'g2b_basket_non_positive_or_ir_low', 'g2c_clip_preservation_below_0.5', 
         'g2d_sweep_no_pass']
```

---

## Key Metrics (FAIL-1/2 Corrections Applied)

| 항목 | 변경 전 | 변경 후 | 상태 |
|----|----|----|----|
| **clip_preservation_ratio** | -4.69 (역전 버그) | -0.21 | ✅ 정정됨 |
| **N_eff floor** | 15.0 (미실현 크레딧) | 1.7 | ✅ 적용됨 |
| **be_eff** | 0.0131 | 0.0386 | ✅ 정직화 |
| **overflow RuntimeWarning** | 2건 | 0건 | ✅ 제거됨 |
| **soft-hurdle 단위** | rank(-1,+1)에 bps 적용 → 무력 | EV→rank 순차 | ✅ 정정됨 |

---

## Technical Corrections Applied

### [FAIL-1] `clip_preservation_ratio` 분자/분모 정정
**정정:**
```python
# Before: _gating_ic / net_ic_dict["mean_ic"]  (pre/post, 역전)
# After:  net_ic_dict["mean_ic"] / _gating_ic  (post/pre, 정정)
```
**검증:** `-0.0030 / 0.0141 = -0.213` (실행 로그 presv=-0.21과 일치) ✓

### [FAIL-2] `_soft_hurdle` 단위 불일치 정정
**정정:**
```python
# Before: EV 단위에 soft-hurdle 미적용, rank-weight에 bps 기준 hurdle
# After: EV(return-fraction) 단위 soft-hurdle → rank-sizing 순차 처리
```
**효과:** 
- overflow 제거 (np.exp exponent clip 추가)
- 비용 크기가 gate 강도에 실제 반영

---

## Acceptance Criteria

| 기준 | 현재값 | 임계 | 상태 |
|----|----|----|----|
| **G1a** (resid_ic > be_eff) | 0.0141 > 0.0386 | NO | ❌ |
| **G1b** (t_stat ≥ 3.0) | 3.62 | OK | ✅ |
| **G2a** (gap_raw > 0) | -0.0416 | NO | ❌ |
| **G2b** (basket_net > 0) | -36.94 | NO | ❌ |
| **G2c** (presv ≥ 0.5) | -0.21 | NO | ❌ |
| **G2d** (sweep ≥ 1) | 0/3 | NO | ❌ |
| **최종** | — | G0∧G1∧G2∧G3 | ❌ **REJECT** |

---

## Interpretation

**ALPHA_PASS=FALSE 사유:**
1. **G1a 실패:** N_eff=1.7(emit-floor)로 be_eff=0.0386 상향 → resid_ic=0.0141 < breakeven
2. **G2 전면 실패:** 클립이 ranking skill을 파괴 (pre-clip IC 0.0141 → post-clip IC -0.0030)

**핵심:** 경질 클립이 0.0141의 유의한 신호를 -0.0030으로 반전시킴. P1(rank-sizing) 구현으로 보존비 개선 필요.

---

## Next Steps

1. **P1 효과 측정:** Optuna trial 실행 (`--mode optimize`)
   - `clip_preservation_ratio ≥ 0.5` 달성 확인
   - `basket_net_bps > 0` 개선 정도 추적

2. **P2 진입:** P1 성공 후 beta-neutral 실행 오버레이

3. **P3-5:** 순차적 진입 (details: `docs/specs/alpha0.md`)

---

**Test Status:** 574 passed, 0 failed ✅  
**Document Version:** 2026-05-30 (FAIL-1/2 수정, 정직한 평가 적용)
