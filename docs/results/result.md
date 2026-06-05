# Mode Signal — 최신 검증 결과

**실행 일시:** 2026-06-05  
**실행 명령어:** `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase signal --sync skip --timeframe 4h --trials 1 --date 2026-05-01`  
**상태:** `Active Signals: 0 (sel=0)`, `Status: BLOCKED` ❌

---

## 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견 | Stage6: 20개 선택
[DATA] 91/94 로드 (96.8%) | 준비 완료: 63개
[BRIDGE] Active Signals: 0 (sel=0) | Status: BLOCKED | Execution Time: 21.09s
[STRATEGY] Total Stage Time: 21.69s
[PROMO_FILTER] no variants recommended by diagnostics; blocking all candidates (fail-closed)
```

---

## Candidate Top Strategies

| Rank | Strategy | Sample (Total/OOS) | Profit(bps) | Win Rate | P/L | Score | Action |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | `trend_pullback_continuation:tpc_20_100` | 710 / 287 | **99.8** | 37.6% | 1.66 | 0.056 | KEEP |
| 2 | `bollinger_reversion:bollinger_20` | 2318 / 834 | 31.1 | 45.0% | 1.18 | -0.091 | DROP |
| 3 | `vol_regime_reversion:vrr_40` | 1866 / 594 | 28.0 | 45.5% | 1.41 | -0.175 | KEEP |
| 4 | `residual_reversion:rr_48` | 769 / 290 | 27.4 | 42.4% | 1.04 | -0.171 | DROP |
| 5 | `dual_momentum:dm_24_96` | 9196 / 3715 | 27.0 | 37.2% | 1.14 | 0.073 | DROP |
| 6 | `rsi_reversion:rsi_14` | 3127 / 1284 | 26.9 | 46.3% | 1.30 | -0.047 | KEEP |
| 7 | `residual_reversion:rr_24` | 821 / 320 | 26.2 | 47.8% | 1.04 | -0.047 | DROP |
| 8 | `dual_momentum:dm_12_48` | 9243 / 3671 | 17.9 | 40.9% | 1.13 | 0.045 | DROP |
| 9 | `funding_zscore_carry:fzs_48` | 1287 / 506 | 16.4 | 42.9% | 1.37 | -0.097 | KEEP |
| 10 | `funding_carry:funding_24` | 2686 / 1114 | 16.4 | 42.8% | 1.45 | -0.071 | KEEP |

---

## Promotion 실패 집계

```text
[DIAG][RULE_RECOMMEND_FAIL_COUNTS]
event_density:40,hit_or_payoff:60,mean_edge:34,median_edge:93,min_obs:22,p10_edge:103,regime_edge:33
```

핵심 병목:
- `p10_edge`: 103
- `median_edge`: 93
- `hit_or_payoff`: 60
- `event_density`: 40
- `regime_edge`: 33

의미:
- signal-cell 단위로 쪼개도 핵심 문제는 여전히 평균 수익 부족보다 좌측 tail과 중앙값 품질이다.
- 일부 cell은 mean edge가 매우 높아도 `p10_edge`와 `median_edge`를 동시에 넘지 못한다.
- `dual_momentum` 계열은 `event_density`까지 겹쳐 배치 가능한 signal로 승격되지 못한다.

---

## 대표 실패 사례

| Variant | Signal Cell | OOS n | Mean | Median | P10 | Density | Hit | Payoff | Failed Checks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tpc_20_100` | `trend_pullback_continuation:tpc_20_100:trend_grind:bull_volatile` | 139 | 70.9 | -20.3 | -449.6 | 0.026 | 0.489 | 1.61 | `median_edge`, `p10_edge` |
| `tpc_20_100` | `trend_pullback_continuation:tpc_20_100:trend_grind:bull_quiet` | 316 | 50.0 | -107.0 | -387.2 | 0.060 | 0.434 | 1.37 | `median_edge`, `p10_edge` |
| `dm_24_96` | `dual_momentum:dm_24_96:momentum_follow:bear_volatile` | 669 | 129.3 | -16.7 | -578.3 | 0.127 | 0.492 | 1.81 | `median_edge`, `p10_edge`, `event_density` |
| `ema_12_72` | `trend_ma:ema_12_72:snapback:transition` | 202 | 90.5 | 19.0 | -246.6 | 0.038 | 0.515 | 3.02 | `p10_edge` |
| `fzs_168` | `funding_zscore_carry:fzs_168:snapback:bear_quiet` | 439 | 43.4 | 18.0 | -257.3 | 0.083 | 0.524 | 1.31 | `p10_edge` |

---

## 최종 판정

- `signal_cell`, `entry_regime`, `exit_policy_id` 기반 진단은 실제 런타임 로그에 반영됐다.
- fail-closed 동작은 유지됐다.
- 그러나 최종 promoted signal은 여전히 `0`이다.
- 현재 병목은 regime 분리 자체보다 `median_edge`와 `p10_edge` 중심의 tail 품질이다.
