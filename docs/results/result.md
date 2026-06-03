# Mode Alpha — 최신 검증 결과

**실행 일시:** 2026-06-03  
**실행 명령어:** `UV_CACHE_DIR=/tmp/uv-cache FUTURES_STRATEGY_NAME=candidate_ml PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase alpha --sync skip --timeframe 4h --trials 1 --date 2026-05-01`  
**상태:** `Active Signals: 0`, `Status: BLOCKED`

---

## 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견 | Stage6: 20개 선택
[DATA] 91/94 로드 (96.8%) | 준비 완료: 63개
[BRIDGE] Active Signals: 0 | Status: BLOCKED | Execution Time: 28.27s
[STRATEGY] Total Stage Time: 75.02s
```

---

## 핵심 진단

이전 결과 문서의 해석이었던 `ML gate p_pass < 0.40`는 최신 로그 기준으로는 정확하지 않다.

현재 blocker는 `gate`가 아니라 `edge/selection`이다.

```text
[DIAG][PIPELINE_GATE]
  calibrated=False
  reason=calibration_probability_collapse
  gate_p50=0.4266
  gate_p90=0.5632
  pct_ge40=0.618

[DIAG][PIPELINE_EDGE]
  mu_p50=-18.3bps
  mu_p90=-12.8bps
  q10_p10=-841.1bps
  utility_p50=-562.134

[DIAG][PIPELINE_SELECT]
  policy=utility_topk
  zero_reason=no_eligible_after_breakeven_floor
  eligible=0
  selected=0
  breakeven_floor=12.0bps
```

정리하면:

- `gate`는 calibration collapse로 보정을 버렸지만, raw probability 자체가 전부 0으로 눌린 상태는 아니다.
- `edge` 예측값이 전 구간에서 음수여서 `mu >= 12bps`를 만족하는 이벤트가 없다.
- 따라서 `utility_topk` 이전에 `eligible=0`이 되어 `Active Signals: 0`으로 종료된다.

---

## Candidate Top Strategies

| Rank | Strategy Name | Sample (OOS) | Profit(bps) | Win Rate | P/L | Score | Action |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | btc_regime_pullback:btc_pullback_50 | 1962 (675) | 29.8 | 43.1% | 1.28 | -0.122 | KEEP |
| 2 | funding_zscore_carry:fzs_168 | 3701 (1319) | 28.3 | 44.0% | 1.34 | 0.075 | KEEP |
| 3 | rsi_reversion:rsi_14 | 4431 (1879) | 20.6 | 49.0% | 1.34 | -0.047 | KEEP |
| 4 | funding_carry:funding_24 | 3777 (1580) | 11.9 | 42.4% | 1.39 | -0.028 | KEEP |
| 5 | cross_sectional_momentum:cs_mom_10 | 10658 (4572) | 5.0 | 45.0% | 1.25 | -0.016 | KEEP |
| 6 | funding_zscore_carry:fzs_48 | 1940 (701) | 2.4 | 40.3% | 1.35 | 0.046 | KEEP |
| 7 | cross_sectional_momentum:cs_mom_5 | 10607 (4499) | -0.9 | 47.1% | 1.18 | -0.009 | DROP |
| 8 | cross_sectional_momentum:cs_mom_20 | 15536 (6594) | -1.9 | 43.3% | 1.21 | -0.027 | DROP |
| 9 | btc_corr_regime:bcr_96 | 24426 (10067) | -3.9 | 41.7% | 1.25 | 0.036 | DROP |
| 10 | trend_donchian:donchian_18 | 2733 (1176) | -4.1 | 40.9% | 1.33 | 0.057 | DROP |

---

## Ablation Study

| Model Alias | CAGR | MaxDD | MAR | Equity | Pass |
|---|---:|---:|---:|---:|---|
| Equal Size | -23.9% | 58.8% | -0.41 | 439,624 | N |
| Kelly (No ML) | -0.6% | 2.0% | -0.29 | 982,572 | N |
| ML Gate | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| ML Gate+Edge | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| ML Full (Capped) | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| Cand. ML | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| Promo Filter | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| Val. Selection | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| Identity Feat | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| Market Feat | 0.0% | 0.0% | 0.00 | 1,000,000 | N |

---

## 해석

- 이번 변경으로 `OOS leakage`, `gate label fallback`, `hard-coded cost literal`, `smoke command drift`는 정리되었다.
- `Active Signals: 0`는 여전히 남아 있지만, 이제 원인은 `p_pass` 막연한 붕괴가 아니라 `edge model`이 OOS에서 전부 음수 `mu_net_decision_bps`를 내는 점으로 좁혀졌다.
- 다음 작업 우선순위는 `candidate_edge.py`의 target scale, feature/label alignment, cost-hurdle 과잉 반영 여부를 점검하는 것이다.
