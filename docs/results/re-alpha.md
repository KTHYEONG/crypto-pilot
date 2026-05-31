# Re-Alpha Execution Results - 2026-05-31 (Phase alpha3)

## 현재 상태
- `ALPHA_PASS`: `FALSE`
- 실행 모드: `--mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01`
- 목적: signed-rank contract 붕괴(0-collapse) 제거 + OOS 평가 경로 견고화 검증

## 최신 실행 로그 (alpha3 적용 후)

```text
🔬 [SCORE-IC] dense_ranker ic=0.0347 t=4.51 hit=0.570 breadth=3.7
🔬 [OOS-DIAG] cause=sufficient_cofinite_check_ic oos_bars=1417 ge5_bars=1417
[OOS-DIAG] rank_cols=96 finite_rows=1417 oos_idx=1417 common_idx=1417 | common_syms[:3]=['AAVEUSDT', 'ADAUSDT', 'AVAXUSDT'] rank_cols[:3]=['1000LUNCUSDT', '1000SHIBUSDT', '1000XECUSDT']
[RANK-CONTRACT] c3_signed_nz=0.586 c3_signed_std=0.003136 c3_long_short_absdiff_p50=0.000000
🏅 [RANK-QUALITY L1] ic=0.0095 t=1.45 hit=0.522 breadth=16.7
📊 [RANK-IC C3] ic= 0.0174  t=   1.04  breadth=   8.79
🧺 [L3-BASKET] ew_bps=-7.26 net_bps=-31.26 ir_t=-1.13 hit=0.490 n=1405
📊 [C3-EXEC]  NET_IC= 0.0024  T-STAT=   0.14  BRDTH=  12.00  BE_IC(12h)= 0.0152  gap=-127.9bps
📈 SWEEP: [6h: ic=0.011 ❌] [12h: ic=0.002 ❌] [18h: ic=0.009 ✅]
>> ALPHA_PASS: FALSE [...]
```

## 판정

```text
핵심 결과:
- alpha2 단계의 계약 붕괴 증상(`breadth=0`, `rank ic=0`, `basket n=0`)은 해소됨.
- alpha3 수용 기준 충족:
  - [RANK-CONTRACT] signed_nz>0
  - [RANK-QUALITY L1] breadth>0
  - [RANK-IC C3] breadth>0
  - [L3-BASKET] n>0
- 잔여 이슈는 계약/파이프라인 결함이 아니라 성능 이슈:
  - post-cost basket 적자 (`net_bps=-31.26`)
  - 통계 유의성 부족 (`t=0.14`)
  - raw breakeven 미달 (`NET_IC 0.0024 < 0.0152`)
```

## 테스트

```text
uv run pytest tests/unit/domain/futures/strategy/test_alpha_evaluation.py --tb=short
46 passed in 0.97s

uv run pytest tests/unit/execution/test_opt_main_futures_strategy_mode.py tests/e2e/test_cli_modes.py tests/integration/execution/test_opt_main_futures_bypass.py --tb=short
33 passed in 1.14s
```
