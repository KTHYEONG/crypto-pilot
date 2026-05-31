# Re-Alpha Execution Results - 2026-05-31 (Phase 1-6)

## 현재 상태
- `ALPHA_PASS`: `FALSE`
- 실행 모드: `--mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01`
- 목적: OOS-DIAG 계측 + OOS 인덱스 추출 경로(H1) 수정 검증

## 최신 실행 로그 (Phase 1-6 적용 후)

```
🔬 [OOS-DIAG] cause=sufficient_cofinite_check_ic oos_bars=1417 ge5_bars=1417
[OOS-DIAG] rank_cols=96 finite_rows=1417 oos_idx=1417 common_idx=1417 | common_syms[:3]=['AAVEUSDT', 'ADAUSDT', 'AVAXUSDT'] rank_cols[:3]=['1000LUNCUSDT', '1000SHIBUSDT', '1000XECUSDT']
Metric | RESID_IC |  T-STAT  |  N_EFF   |   DSR    | BE_EFF(12h) | BEAR_IC
📊 [PASS=❌] fail=['signal_below_effective_breakeven', 'signal_t_stat_too_low', 'basket_net_lcb_non_positive'] | net_ic=0.0000 be_raw=0.0526 gap_raw=-525.9bps
📊 [C3-EXEC]  NET_IC= 0.0000  T-STAT=   0.00  BRDTH=   0.00  BE_IC(12h)= 0.0526  gap=-525.9bps
>> ALPHA_PASS: FALSE [signal_skill_passes=FAIL portfolio_ic_above_breakeven=FAIL basket_net_positive=FAIL signal_preserved_after_selection=FAIL multi_horizon_sweep_passes=FAIL bear_market_basket_safe=OK] [IC_SKILL: resid_ic=0.0000 be_eff=0.0136 gap=-0.0136 t=0.00 bear_ic=nan dsr=nan] [BASKET: gap_raw=-0.0136 net_bps=nan ir_t=nan presv=nan sweep=0/3]
```

## 최종 판정

```
핵심 결과:
- OOS 인덱스 추출 경로는 복구됨 (`finite_rows=1417`, `common_idx=1417`).
- 그러나 현재 스모크에서는 `NET_IC=0.0000`, `ALPHA_PASS=FALSE` 상태.
- 따라서 본 단계는 "회귀 원인 가시화 + 인덱스 경로 수정"까지 완료, 성능 복원은 미완료.
```

## 테스트

```
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest \
  tests/e2e/test_cli_modes.py \
  tests/integration/execution/test_opt_main_futures_bypass.py \
  --tb=short

21 passed in 1.06s
```
