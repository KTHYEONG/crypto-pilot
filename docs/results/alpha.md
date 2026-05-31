# Alpha Execution Results - 2026-05-31

## 현재 상태
- **ALPHA_PASS:** FALSE — 잔여 블로커: `portfolio_ic_above_breakeven`, `multi_horizon_sweep_passes` (24bps 고정 마찰비용 적용에 의한 정직한 판정)
- **테스트:** 68 passed ✅ (단위 테스트 22 pass 포함)

---

## 최신 실행 로그 (Phase 3 완료 - Idiosyncratic Redesign)

```
[RANK-SCOREBOARD] net_signal applied: q=0.45 long_nz=0.467 short_nz=0.467
🔧 [1B-RESID] market_fwd_std=0.0501 beta_mean=1.000 real_std=0.0692 resid_std=0.0472
🏅 [RANK-QUALITY L1] ic=0.0079 t=3.67 hit=0.134 breadth=3.7 | rank_score_long-short vs forward_gross_ret (C1 dense, unclipped)
🏅 [RANK-GENERALIZE] oos_rank_ic=0.0294 is_rank_ic=-0.0072 retention=4.07 decision=continue

📊 [ALPHA SCOREBOARD]
Metric | RESID_IC |  T-STAT  |  N_EFF   |   DSR    | BE_EFF(12h) | BEAR_IC
Value  |  0.0145  |    3.71  |    15.0  |  1.0000  |   0.0131  |  0.0209
Result |    ✅    |    ✅    |  N_eff   |    ✅    |  (gap=+14.2bps)  |    ✅

📊 [PASS=✅] fail=[] | net_ic=0.0130 be_raw=0.0290 gap_raw=-160.1bps
📊 [RANK-IC C3] ic= 0.0145  t=   3.71  breadth=   1.94 | rank_score_long-short vs beta-resid (dense, unclipped, C3)
🧺 [L3-BASKET] ew_bps=24.37 net_bps=0.37 ir_t=3.82 hit=0.544 n=1417 | zw_bps=36.77(confound) | RANK-IC C3=0.0145
📊 [C3-EXEC]  NET_IC= 0.0130  T-STAT=   3.43  BRDTH=   3.08  BE_IC(12h)= 0.0290 gap=-160.1bps
🌐 [REGIME IC] Bull: 0.012 | Bear: 0.021 | Chop: 0.003

📈 SWEEP: [6h: ic=0.010 ❌] [12h: ic=0.012 ❌] [18h: ic=0.016 ❌]
>> ALPHA_PASS: FALSE [signal_skill_passes=OK portfolio_ic_above_breakeven=FAIL basket_net_positive=OK signal_preserved_after_selection=OK multi_horizon_sweep_passes=FAIL bear_market_basket_safe=OK] [IC_SKILL: resid_ic=0.0145 be_eff=0.0131 gap=+0.0014 t=3.71 bear_ic=0.0209 dsr=1.000] [BASKET: gap_raw=-0.0160 net_bps=0.4 ir_t=3.82 presv=0.89 sweep=0/3] [fail=['portfolio_ic_below_raw_breakeven', 'no_profitable_horizon_found']]
>> EXEC_DIAG: FAIL [port_ic=0.0130 be_raw=0.0290 gap_raw=-0.0160 basket_net_bps=0.37 fail=['portfolio_ic_below_raw_breakeven']]
```

---

## 누적 개선 이력

| 지표 | Phase 1b (NET 신호) | Phase 2.5 (24bps 엄격 고정 + Breadth) | Phase 3 (Idiosyncratic Redesign) |
|------|-------------------|-------------------|-------------------|
| N_eff / be_raw | 2.2 / 0.0343 | **3.08 / 0.0290** (Breadth 향상 ✅) | **3.08 / 0.0290** (동일) |
| basket ew_bps | +32.43 | **+34.12** | **+24.37** (단위 수축) |
| basket net_bps | +8.43 | **+10.12** (PnL 향상 ✅) | **+0.37** (실행 보수화) |
| `signal_preserved_after_selection` (presv) | +1.01 | **+0.89** | **+0.89** |
| `basket_net_positive` | ✅ | **✅ (ir_t: 5.78로 상향)** | **✅ (ir_t: 3.82로 유지)** |
| `multi_horizon_sweep_passes` | 0/3 ❌ | **0/3 ❌ (정직한 판정)** | **0/3 ❌ (정직한 판정)** |
| **`portfolio_ic_above_breakeven`** | ❌ | **❌ (Gap -163.8bps)** | **❌ (Gap -160.1bps로 소폭 축소)** |
| `bear_market_basket_safe` | nan | **✅ (실측 통과)** | **✅ (실측 통과, bear_ic=0.0209)** |

---

## 게이트 판정 (Phase 3 결과)

| 게이트 코드 | 현재값 | 상태 |
|------------|--------|------|
| `signal_skill_passes` (resid_ic > be_eff) | 0.0145 > 0.0131 | ✅ |
| `signal_t_stat_too_low` (t_stat ≥ 3.0) | 3.71 | ✅ |
| `bear_regime_ic_negative` (bear_ic ≥ 0) | 0.0209 | ✅ |
| **`portfolio_ic_above_breakeven`** (port_ic > be_raw) | 0.0130 < 0.0290 | ❌ |
| `basket_net_positive` (net_bps > 0 ∧ ir_t ≥ 2) | +0.37 / +3.82 | ✅ |
| `signal_preserved_after_selection` (presv ≥ 0.5) | +0.89 | ✅ |
| **`multi_horizon_sweep_passes`** (sweep ≥ 1) | 0/3 | ❌ |
| `bear_market_basket_safe` | +0.37 bps | ✅ (실측 적용) |

**다음 단계:** `docs/specs/alpha4.md` 에 기록된 설계를 따라 **Idiosyncratic Features**(BTC lead-lag, funding-basis 이격, 테마 상대 모멘텀)를 데이터 수집 및 피처 생성 파이프라인에 실체화하여 포트폴리오 `net_ic` 수준을 $0.030+$ 이상으로 강인화해야 합니다.
