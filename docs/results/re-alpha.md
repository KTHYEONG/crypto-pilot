# Re-Alpha Execution Results - 2026-05-31

## 현재 상태
- `ALPHA_PASS`: `FALSE`
- 핵심 blocker categories: `rank_skill`, `breadth`, `cost_turnover`
- 실행 모드: `--mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01`

## 최신 실행 로그

```
🔬 [SCORE-IC] dense_ranker ic=0.0347 t=4.51 hit=0.570 breadth=3.7
🔬 [OOS-RANKIC] ic=0.0347 t=4.51 n_bars=1417 cofinite_p50=17.0 bars_ge5_ratio=1.000
🔬 [RESID-IC] raw=0.0347 resid=0.0361 resid_hit=0.564
🔬 [BE-EFF] N_raw=17.0 N_eff=1.5 sigma_r=666.3bps be_raw=0.0116 be_eff=0.0174 gap_resid_eff=+0.0187
📊 ML_EVAL: nz(L=0.014 S=0.014) ic=0.0347 t=5.07 hit=0.214 obs=3704
💰 ML_COST: gate=9051.5bps floor=24.0bps pass=true
[REGIME-GATE] applied: bull=2431 bear=2905 chop=1096 unlabeled=30
```

## 최종 판정

```
📈 SWEEP: [6h: ic=0.000 ❌] [12h: ic=0.000 ❌] [18h: ic=0.000 ❌]
>> ALPHA_PASS: FALSE [signal_skill_passes=FAIL portfolio_ic_above_breakeven=FAIL basket_net_positive=FAIL signal_preserved_after_selection=FAIL multi_horizon_sweep_passes=FAIL bear_market_basket_safe=OK]
[fail=['signal_below_effective_breakeven', 'signal_t_stat_too_low', 'basket_net_lcb_non_positive', 'portfolio_ic_below_raw_breakeven', 'basket_net_not_profitable', 'signal_lost_after_selection', 'no_profitable_horizon_found']]
[blockers={'rank_skill': ['signal_below_effective_breakeven', 'signal_t_stat_too_low', 'portfolio_ic_below_raw_breakeven'], 'breadth': ['signal_lost_after_selection', 'no_profitable_horizon_found'], 'cost_turnover': ['basket_net_lcb_non_positive', 'basket_net_not_profitable'], 'regime_stability': []}]
>> EXEC_DIAG: FAIL [port_ic=0.0000 be_raw=0.0508 gap_raw=-0.0508 basket_net_bps=nan fail=['portfolio_ic_not_positive', 'portfolio_ic_below_raw_breakeven']]
```

## 해석

- `SCORE-IC`와 `OOS-RANKIC`는 양호했지만, 최종 alpha acceptance gate에서는 `C3` execution과 basket/sweep 단계가 무너졌다.
- `rank_skill`은 일부 지표에서 유지됐으나, `portfolio_ic_above_breakeven`과 `basket_net_lcb_non_positive`가 동시에 실패해 최종 alpha는 통과하지 못했다.
- `multi_horizon_sweep_passes=0/3` 이라서 sweep 기반 robustness도 확보되지 않았다.
