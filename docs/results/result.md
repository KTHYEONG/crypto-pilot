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
[BRIDGE] Active Signals: 0 | Status: BLOCKED | Execution Time: 23.89s
[STRATEGY] Total Stage Time: 79.52s
```

---

## 핵심 진단

이전 결과 문서의 해석이었던 `ML gate p_pass < 0.40`는 최신 로그 기준으로는 정확하지 않다.

현재 blocker는 `gate`가 아니라 `edge/selection`이다.

```text
[DIAG][PIPELINE_GATE]
  calibrated=False
  reason=calibration_probability_collapse
  gate_p50=0.4469
  gate_p90=0.5501
  pct_ge40=0.740

[DIAG][PIPELINE_EDGE]
  mu_p50=9.6bps
  mu_p90=25.1bps
  q10_p10=-844.3bps
  utility_p50=-557.400

[DIAG][PIPELINE_SELECT]
  policy=utility_topk
  zero_reason=selected_nonzero
  eligible=448
  selected=45
  breakeven_floor=12.0bps
```

정리하면:

- `gate`는 여전히 calibration collapse 상태라 보정은 쓰지 않지만, raw probability는 충분히 살아 있다.
- `edge`는 이제 전체 음수 붕괴가 아니라 `mu_p50=9.6bps`, `mu_p90=25.1bps` 수준으로 복구됐다.
- `utility_topk`에서 `eligible=448`, `selected=45`까지 올라왔지만, live panel로 전개된 뒤 최종 `Active Signals`는 여전히 `0`이다.
- 즉 현재 blocker는 `ML selection`이 아니라 `bridge/live 전개` 쪽이다.

---

## Candidate Top Strategies

| Rank | Strategy Name | Sample (OOS) | Profit(bps) | Win Rate | P/L | Score | Action |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | funding_zscore_carry:fzs_168 | 3692 (1306) | 34.0 | 44.3% | 1.37 | 0.080 | KEEP |
| 2 | rsi_reversion:rsi_14 | 4409 (1838) | 28.3 | 49.6% | 1.40 | -0.032 | KEEP |
| 3 | btc_regime_pullback:btc_pullback_50 | 1962 (667) | 20.0 | 42.4% | 1.23 | -0.112 | KEEP |
| 4 | funding_carry:funding_24 | 3763 (1560) | 15.9 | 42.5% | 1.40 | -0.020 | KEEP |
| 5 | funding_zscore_carry:fzs_48 | 1939 (691) | 11.3 | 40.8% | 1.40 | 0.051 | KEEP |
| 6 | cross_sectional_momentum:cs_mom_10 | 10612 (4488) | 3.6 | 44.9% | 1.24 | -0.005 | KEEP |
| 7 | cross_sectional_momentum:cs_mom_20 | 15486 (6480) | -1.8 | 43.3% | 1.20 | -0.021 | DROP |
| 8 | cross_sectional_momentum:cs_mom_5 | 10566 (4417) | -2.8 | 47.1% | 1.17 | -0.004 | DROP |
| 9 | btc_corr_regime:bcr_96 | 24336 (9896) | -4.9 | 41.6% | 1.25 | 0.049 | DROP |
| 10 | vol_regime_reversion:vrr_40 | 4297 (1556) | -6.2 | 41.8% | 1.18 | -0.055 | DROP |

---

## Ablation Study

| Model Alias | CAGR | MaxDD | MAR | Equity | Pass |
|---|---:|---:|---:|---:|---|
| Equal Size | -23.9% | 58.8% | -0.41 | 439,624 | N |
| Rule Promo NL | -32.6% | 23.4% | -1.39 | 791,504 | N |
| Rule Promo Oracle | -40.8% | 29.2% | -1.40 | 733,208 | N |
| Kelly (No ML) | -0.6% | 2.0% | -0.29 | 982,572 | N |
| ML Gate | 0.0% | 0.0% | 0.04 | 1,000,039 | N |
| ML Gate+Edge | -4.3% | 12.9% | -0.34 | 875,260 | N |
| ML Full (Capped) | 0.0% | 0.1% | 0.06 | 1,000,109 | N |
| Cand. ML | 0.1% | 0.0% | 1.47 | 1,000,331 | Y |
| Direct Edge | 0.0% | 0.1% | 0.05 | 1,000,019 | N |
| Variant Prior | 0.1% | 0.1% | 1.20 | 1,000,380 | Y |
| Promo Filter | 0.1% | 0.1% | 1.80 | 1,000,707 | Y |
| Val. Selection | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| Identity Feat | 0.0% | 0.0% | 0.97 | 1,000,222 | Y |
| Market Feat | -0.0% | 0.0% | -0.15 | 999,959 | N |

---

## 해석

- 이번 변경으로 `OOS leakage`, `gate label fallback`, `hard-coded cost literal`, `smoke command drift`는 정리되었다.
- `Rule Promo NL`과 `Rule Promo Oracle`는 OOS-only로 다시 계산되면서, 전체기간 희석이 사라졌고 두 row 모두 명확한 음수 성과로 드러났다.
- `Cand. ML`, `Variant Prior`, `Promo Filter`, `Identity Feat`는 OOS slice 기준으로는 통과하지만, 브리지 live 경로에서는 여전히 `Active Signals: 0`이다.
- 다음 작업 우선순위는 `bridge/live 전개`에서 OOS selection이 실제 live panel로 어떻게 사라지는지 분리 진단하는 것이다.
