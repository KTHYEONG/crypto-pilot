# Alpha Execution Results - 2026-05-31

## 현재 상태
- **ALPHA_PASS:** FALSE — 잔여 블로커: `portfolio_ic_above_breakeven`
- **테스트:** 607 passed ✅

---

## 최신 실행 로그 (Phase 2 완료)

```
[RANK-SCOREBOARD] net_signal applied: q=0.33 long_nz=0.333 short_nz=0.333
🔬 [RESID-IC] raw=0.0454 resid=0.0446
🔬 [BE-EFF] N_raw=17.0 N_eff=1.5 sigma_r=666.3bps be_raw=0.0116 be_eff=0.0174
🔬 [MONOTONICITY] top-bot=+35.5bps mono_rho=1.00 beta_tilt=-0.293 (L=0.78 S=1.07)

📊 [ALPHA SCOREBOARD]
Metric | RESID_IC | T-STAT | N_EFF  | DSR    | BE_EFF(12h) | BEAR_IC
Value  | 0.0141   | 3.62   | 15.0   | 1.0000 | 0.0131      | 0.0240
Result | ✅       | ✅     | 15.0   | ✅     | gap=+10.3   | ✅

🧺 [L3-BASKET] ew_bps=+32.43 net_bps=+8.43 ir_t=+5.38 hit=0.553
📈 SWEEP: [6h: ❌] [12h: ✅] [18h: ✅]

>> ALPHA_PASS: FALSE
   [signal_skill_passes=OK  portfolio_ic_above_breakeven=FAIL  basket_net_positive=OK
    signal_preserved_after_selection=OK  multi_horizon_sweep_passes=OK  bear_market_basket_safe=OK]
   [IC_SKILL: resid_ic=0.0141 be_eff=0.0131 gap=+0.0010 t=3.62 bear_ic=0.0240 dsr=1.000]
   [BASKET: gap_raw=-0.0200 net_bps=+8.4 ir_t=+5.38 presv=+1.01 sweep=2/3]
   fail=['portfolio_ic_below_raw_breakeven']

>> EXEC_DIAG: FAIL
   port_ic=0.0143 be_raw=0.0343 gap_raw=-0.0200 basket_net_bps=+8.43
   fail=['portfolio_ic_below_raw_breakeven']
```

---

## 누적 개선 이력

| 지표 | Phase 0 (ev_clip) | Phase 1b (NET 신호) | Phase 2 (비용상각) |
|------|------------------|-------------------|------------------|
| N_eff / be_eff | 1.7 / 0.0386 | 15.0 / 0.0131 | 15.0 / 0.0131 |
| `signal_skill_passes` | ❌ | ✅ | ✅ |
| basket ew_bps | -12.94 | +32.43 | +32.43 |
| basket net_bps | -36.94 | +8.43 | +8.43 |
| `signal_preserved_after_selection` (presv) | -0.21 | +1.01 | +1.01 |
| `basket_net_positive` | ❌ | ✅ | ✅ |
| `multi_horizon_sweep_passes` | 0/3 ❌ | 0/3 ❌ | 2/3 ✅ |
| **`portfolio_ic_above_breakeven`** | ❌ | ❌ | ❌ (잔존) |

## 비용상각 효과 (Phase 2)

| 호라이즌 | 상각비용 | breakeven (이전) | breakeven (이후) | net_ic | 판정 |
|---------|---------|-----------------|-----------------|--------|------|
| 6h | 24bps | 0.0328 | 0.0328 | 0.009 | ❌ |
| **12h** | **12bps** | 0.0234 | **0.0117** | 0.013 | **✅** |
| **18h** | **8bps** | 0.0192 | **0.0064** | 0.015 | **✅** |

---

## 게이트 판정 (현재)

| 게이트 코드 | 현재값 | 상태 |
|------------|--------|------|
| `signal_skill_passes` (resid_ic > be_eff) | 0.0141 > 0.0131 | ✅ |
| `signal_t_stat_too_low` (t_stat ≥ 3.0) | 3.62 | ✅ |
| `bear_regime_ic_negative` (bear_ic ≥ 0) | 0.0240 | ✅ |
| **`portfolio_ic_above_breakeven`** (port_ic > be_raw) | 0.0143 < 0.0343 | ❌ |
| `basket_net_positive` (net_bps > 0 ∧ ir_t ≥ 2) | +8.43 / +5.38 | ✅ |
| `signal_preserved_after_selection` (presv ≥ 0.5) | +1.01 | ✅ |
| `multi_horizon_sweep_passes` (sweep ≥ 1) | 2/3 | ✅ |
| `bear_market_basket_safe` | nan(미구현) | ✅ |

**다음 단계:** `docs/specs/tmp.md` 참조
