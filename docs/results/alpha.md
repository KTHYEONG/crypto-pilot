# Alpha Execution Results - 2026-05-31

## 현재 상태
- **ALPHA_PASS:** FALSE — 잔여 블로커: `portfolio_ic_above_breakeven`, `multi_horizon_sweep_passes` (24bps 고정 마찰비용 적용에 의한 정직한 판정)
- **테스트:** 67 passed ✅

---

## 최신 실행 로그 (Phase 2.5 완료 - 24bps 고정 및 Breadth 최적화)

```
[RANK-SCOREBOARD] net_signal applied: q=0.45 long_nz=0.467 short_nz=0.467
🔧 [1B-RESID] market_fwd_std=0.0501 beta_mean=1.000 real_std=0.0692 resid_std=0.0472
🏅 [RANK-QUALITY L1] ic=0.0062 t=3.01 hit=0.130 breadth=3.7 | rank_score_long-short vs forward_gross_ret (C1 dense, unclipped)

📊 [ALPHA SCOREBOARD]
Metric | RESID_IC |  T-STAT  |  N_EFF   |   DSR    | BE_EFF(12h) | BEAR_IC
Value  |  0.0141  |    3.62  |    15.0  |  1.0000  |   0.0131  |  0.0240
Result |    ✅    |    ✅    |  N_eff   |    ✅    |  (gap=+10.3bps)  |    ✅

📊 [PASS=✅] fail=[] | net_ic=0.0126 be_raw=0.0290 gap_raw=-163.8bps
🧺 [L3-BASKET] ew_bps=34.12 net_bps=10.12 ir_t=5.78 hit=0.563 n=1417 | zw_bps=33.26(confound) | RANK-IC C3=0.0141
📊 [C3-EXEC]  NET_IC= 0.0126  T-STAT=   3.32  BRDTH=   3.08  BE_IC(12h)= 0.0290 gap=-163.8bps
🌐 [REGIME IC] Bull: 0.011 | Bear: 0.024 | Chop: -0.006

📈 SWEEP: [6h: ic=0.009 ❌] [12h: ic=0.012 ❌] [18h: ic=0.016 ❌]
>> ALPHA_PASS: FALSE [signal_skill_passes=OK portfolio_ic_above_breakeven=FAIL basket_net_positive=OK signal_preserved_after_selection=OK multi_horizon_sweep_passes=FAIL bear_market_basket_safe=OK] 
>> EXEC_DIAG: FAIL [port_ic=0.0126 be_raw=0.0290 gap_raw=-0.0164 basket_net_bps=10.12 fail=['portfolio_ic_below_raw_breakeven']]
```

---

## 누적 개선 이력

| 지표 | Phase 1b (NET 신호) | Phase 2 (비용상각-철회) | Phase 2.5 (24bps 엄격 고정 + Breadth) |
|------|-------------------|------------------|-------------------|
| N_eff / be_raw | 2.2 / 0.0343 | 2.2 / 0.0343 | **3.08 / 0.0290** (Breadth 향상 ✅) |
| basket ew_bps | +32.43 | +32.43 | **+34.12** |
| basket net_bps | +8.43 | +8.43 | **+10.12** (PnL 향상 ✅) |
| `signal_preserved_after_selection` (presv) | +1.01 | +1.01 | **+0.89** |
| `basket_net_positive` | ✅ | ✅ | **✅ (ir_t: 5.78로 상향)** |
| `multi_horizon_sweep_passes` | 0/3 ❌ | 2/3 ✅ (끼워맞춤) | **0/3 ❌ (정직한 판정)** |
| **`portfolio_ic_above_breakeven`** | ❌ | ❌ | **❌ (Gap -163.8bps로 축소)** |
| `bear_market_basket_safe` | nan | nan | **✅ (실측 통과)** |

---

## 게이트 판정 (Phase 2.5 결과)

| 게이트 코드 | 현재값 | 상태 |
|------------|--------|------|
| `signal_skill_passes` (resid_ic > be_eff) | 0.0141 > 0.0131 | ✅ |
| `signal_t_stat_too_low` (t_stat ≥ 3.0) | 3.62 | ✅ |
| `bear_regime_ic_negative` (bear_ic ≥ 0) | 0.0240 | ✅ |
| **`portfolio_ic_above_breakeven`** (port_ic > be_raw) | 0.0126 < 0.0290 | ❌ |
| `basket_net_positive` (net_bps > 0 ∧ ir_t ≥ 2) | +10.12 / +5.78 | ✅ |
| `signal_preserved_after_selection` (presv ≥ 0.5) | +0.89 | ✅ |
| **`multi_horizon_sweep_passes`** (sweep ≥ 1) | 0/3 | ❌ |
| `bear_market_basket_safe` | +10.12 bps | ✅ (실측 적용) |

**다음 단계:** `docs/specs/alpha2.md` 에 기록된 정공법을 따라 **Phase 3 (학습 타깃 잔차화 및 Idiosyncratic Features 보강)**에 진입하여 시그널 엣지 `net_ic` 수준을 $0.030+$ 이상으로 강인화해야 합니다.

