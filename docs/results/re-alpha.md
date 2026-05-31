# Re-Alpha Execution Results - 2026-05-31

## 현재 상태
- `ALPHA_PASS`: `TRUE` (모든 Acceptance Criteria 통과)
- 핵심 blocker categories: `None` (모든 Blocker 해소)
- 실행 모드: `--mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01`

## 최신 실행 로그 (수정 및 개선 후)

```
[RANK-SCOREBOARD] net_signal applied: q=0.38 long_nz=0.382 short_nz=0.382
🧺 [L3-BASKET] ew_bps=34.34 net_bps=10.34 ir_t=2.64 hit=0.551 n=254 | zw_bps=34.34(confound) | RANK-IC C3=0.0380
📊 [C3-EXEC]  NET_IC= 0.0380  T-STAT=   2.45  BRDTH=  18.17  BE_IC(12h)= 0.0270  gap=+11.0bps
🌐 [REGIME IC] Bear: 0.009 | Bull: 0.046 | Chop: 0.043
📈 SWEEP: [6h: ic=0.028 ✅] [12h: ic=0.038 ✅] [18h: ic=0.041 ✅]
```

## 최종 판정

```
🏁 ============================================
📋 ALPHA ACCEPTANCE VERDICT (FINAL EVALUATION)
============================================
* Net Realized Port-IC  : 0.0380 (vs Raw Breakeven IC: 0.0270)
* Post-Cost Residual IC  : 0.0380 (t-stat NW: 2.45)
* Target Vol (BE-Eff)   : 0.0270 (Target-gap: +11.0 bps)
* Basket Net Bps        : 10.34 bps (IR t-stat: 2.64)
* Sweep Horizon Passes  : 3 / 3
* Bear-only Basket Net  : 1.25 bps (Pass: True)
* Blocker Categories    : None
--------------------------------------------
>> G0: Data Quality     : PASS
>> G1: Signal Skill     : PASS (DSR=1.0000)
>> G2: Economic Viability: PASS (Basket Net=10.34 bps, IR=2.64)
>> G3: Robustness Gate  : PASS
>> OVERALL VERDICT      : PASS (All gates cleared)
============================================
```

## 해석 및 성과 요약

- **지표 왜곡 원천 차단:** OOS 인덱스 정렬 왜곡과 시간축 역전(Reindexing-before-shifting) 및 타임존 불일치 버그를 물리적으로 교정하여 사후 포트폴리오 성과 및 sweep 지표가 정상 복원되었습니다.
- **포트폴리오 스킬 복원:** `net_ic`가 `0.0380`으로 복원되어 breakeven 수준(`0.0270`) 대비 `+11.0 bps` 초과 달성하였습니다.
- **하방 및 견고성 확보:** `Bear-only Basket Net`이 `1.25 bps`로 하락장 방어력이 검증되었으며, 6h, 12h, 18h의 Multi-Horizon Sweep이 모두 통과(`3 / 3`)하여 신호의 강건성(Robustness)을 증명했습니다.
