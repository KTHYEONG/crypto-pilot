## L1+L2 통합 walk-forward + signal 단위 스크리닝 실제 데이터 측정 결과 — 2026-07-29

### 1. 실행 식별자와 원자료

| 항목 | 값 |
|---|---|
| 기준 실행 | `logs/futures/compound/20260729_075352/` |
| 원인 분해 계측 실행 | `scratch/verify_router_admission_diagnosis.py`, `scratch/verify_router_admission_diagnosis_v2.py` (동일 데이터·설정에 `evaluate_ensemble_admission` 스파이만 부착, production route는 그대로 실행) |
| 기준 명령 | `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42` |
| reference date / seed | `2026-07-15` / `42` |
| base timeframe | `1h` (L1 내부 결정 grid `4h`) |
| 입력 규모 | `5,442` bars × `51` symbols |
| model version | `quarterly-v1` |
| data manifest hash | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` |
| process | `exit_code=0` |
| L2/L3 보호 | `L2_DRY_RUN=1`, sealed L3 holdout 미소비 |
| 적용 스펙 | `docs/specs/l1_cash_only_exit_redesign.md` (P0 L1+L2 통합 walk-forward, P1 signal 단위 screening + horizon ladder 확장, P2/P3 6~7단 게이트 캐스케이드 → 복합 사후분포 검정 2건으로 축소) |

기준 artifact hash (직전 실행과 동일 — 둘 다 all-zero cash-only 산출물이라 해시가 우연히 일치):

- `result.json`: `216e76fd9567318d5d4e5b4b6270f81082845d288b34a502953efe764419afde`
- `target_weights.npy`: `2a161f690b5593fc026fbf44c11205b0afd6e228295cb756bb4c779092e6c102`

### 2. P0 — L1+L2 통합 walk-forward span 실측

| 계측값 | 직전(3단 분리) | 이번(통합 span) |
|---|---:|---:|
| fold/step 수 | 5 | **18** |
| OOS evidence bars | 1,700 | **3,240 (1.9배)** |
| warmup(guard-blocked) step 수 | — | **5** (`min_evidence_bars=900 / step_bars=180`와 정확히 일치) |
| 게이트 평가 도달 step 수 | 2/5 (40%) | **13/18 (72%)** |

`build_expanding_walk_forward_steps`가 quarterly 경로에서 `l3_start`까지 확장된 것을 확인했다(스펙 `[RULE-P0-4]`). fold 수 3.6배, OOS 표본 1.9배 확대는 산술 예측과 일치한다.

### 3. P1 — signal 단위 스크리닝 실측

`n_ic_bars=3,240`, `sidak_alpha=0.0053`, ladder 확장(`reversal_st`/`xs_reversal` 8종 → 8·12·24·48·72·96h, `xs_momentum_slow` 3종 → 216·432·648h) 적용 결과, family가 아닌 **signal 단위**로 8개가 독립적으로 admit됐다(family pooling 시절의 `√n_sig` t-팽창 없이):

| signal | n_ic_bars | t_newey_west | 판정 |
|---|---:|---:|:---:|
| `reversal_st:fast` (8h) | 3,240 | `+6.861` | **admit** |
| `reversal_st:medium` (12h) | 3,240 | `+8.698` | **admit** |
| `reversal_st:moderate` (24h) | 3,240 | `+5.880` | **admit** |
| `reversal_st:slow` (48h) | 3,240 | `+3.653` | **admit** |
| `xs_reversal:fast` (8h) | 3,240 | `+7.315` | **admit** |
| `xs_reversal:medium` (12h) | 3,240 | `+8.150` | **admit** |
| `xs_reversal:moderate` (24h) | 3,240 | `+4.883` | **admit** |
| `xs_reversal:slow` (48h) | 3,240 | `+4.373` | **admit** |
| `reversal_st:very_slow`/`ultra_slow`, `xs_reversal:very_slow`/`ultra_slow` | 3,240 | `+1.52` ~ `-0.23` | reject (`not_significant_after_sidak`) |
| `xs_momentum_slow:*` (216/432/648h) | 3,240 | `+2.221` ~ `+0.324` | reject (`not_significant_after_sidak`) |
| `trend_ema:*`, `momentum_ts:*`, `breakout_donchian:*`, `basis_gap:*` | 3,240 | 부호모순 또는 not_significant | reject |
| `smart_money_divergence:*` | 0 | `0.000` | reject (`insufficient_ic_samples`) |

이전 실행에서 family pooling으로 인해 admit됐던 `xs_reversal`(family 단위 t=+2.942)이 실은 2개 signal 평균으로 부풀려진 값이었다는 의심은, 이번 signal 단위 재측정에서 `xs_reversal:fast`(t=+7.315)와 `xs_reversal:medium`(t=+8.150) 둘 다 개별적으로도 강하게 유의함이 확인되며 해소됐다 — 이번 8종은 통계적 허위 admit이 아니라 실재하는 개별 신호다.

### 4. P2/P3 — 복합 사후분포 게이트 실측

`evaluate_ensemble_admission`이 실제로 호출되고 `circular_stationary_bootstrap_growth`의 세 번째 반환값(`prob_positive`, 직전까지 버려지던 값)이 캡처됨을 확인했다:

| 계측값 | 값 |
|---|---:|
| 게이트 평가 호출 수 | 18 (=step 수) |
| guard 차단(`insufficient_evidence`) | 5 |
| 실제 평가(guard 통과) | 13 |
| `admitted` | **0 / 13** |
| `prob_positive` (전 구간) | `0.0000` (13/13 모두) |
| `growth_2x_cost` (2배 비용 스트레스) | `-1.311` ~ `-1.189` (연환산 log growth) |

### 5. 근본원인 분해 — 비용이 총알파를 압도

`SignalFoldRecord`의 gross/cost/funding 성분을 step별로 분해한 결과(`scratch/verify_router_admission_diagnosis_v2.py`):

| step 누적 evidence bars | gross 연환산 | cost 연환산 | net 연환산 |
|---:|---:|---:|---:|
| 180 | `-0.11` | `-0.68` | `-0.79`* |
| 360 | `+0.23` | `-0.69` | `-0.47` |
| 900 | `+0.09` | `-0.68` | `-0.61` |
| 1,800 | `+0.17` | `-0.69` | `-0.53` |
| 3,060 | `+0.11` | `-0.70` | `-0.60` |

(*초기 warmup 구간은 표본이 적어 부호가 불안정)

**총알파(gross)는 연 +9%~+31%로 실재하고 P1의 강한 t-stat과 방향이 일치한다. 그러나 비용(cost)이 연 -68%~-70%로 거의 상수이며 총알파의 3~7배를 압도해 net이 항상 음수다.** 이는 게이트 산술이나 posterior 폐기 버그가 아니라, **실측된 gross-vs-cost 워터폴**이다.

의심되는 원인(범위 외, 미수정): 이번에 새로 admit된 8h/12h lookback 반전 신호는 순위가 2~3봉마다 뒤집히는 초고빈도 신호다. `build_fold_expert_books`가 이를 다른 신호와 동일한 Kelly 사이징으로 처리해, `alpha_smooth=0.08`/`band_frac=0.60`의 완충으로는 회전율 폭증을 충분히 억제하지 못하고 있을 가능성이 높다.

### 6. L2 / L3 결과

| metric | value |
|---|---:|
| L2 verdict | `no_evidence` |
| annualized log growth | `0.0` |
| CAGR | `0.0` |
| Sharpe | `0.0` |
| max drawdown | `0.0` |
| integrity_ok | `true` |
| L2 reasons | `active_days_ratio=0.0000<0.1`; `rebalances=0<30` |
| L3 verdict | `reject` |
| L3 reasons | `low_growth_probability`; `l2_not_pass` |

`target_weights.npy`는 `float32`, shape `(5442, 51)`이며 `nonzero=0`, `max_abs=0.0`, `mean_abs=0.0`이다. 자본 배치 0%, 현금 비중 100%는 이번에도 유지된다.

### 7. Handoff 원자료

`logs/l1_admission.jsonl` 최종 `EVAL` record:

```json
{
  "admitted_sleeves": 568,
  "distinct_series": 1,
  "oos_bars": 3240,
  "ann_growth": 0.0,
  "ann_lcb90": 0.0,
  "pw_block": 5.0,
  "turnover": 0.0,
  "cost_drag": 0.0,
  "positive_folds": 0,
  "fold_growths": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "mean_abs_net": 0.0,
  "admitted": false
}
```

`fold_growths`가 18개 항목 전부 정확히 `0.0`인 이유는 라우터가 단 한 step도 admit하지 않아 `mu_2d`/`weights_2d`가 전 구간에서 0으로 유지됐기 때문이다(P0가 만든 18-step span 자체는 정상 작동, P2 게이트가 매 step 일관되게 거부).

### 8. 결론

1. **P0(L1+L2 통합 walk-forward)와 P1(signal 단위 스크리닝)은 실제 데이터에서 스펙이 예측한 대로 정확히 작동했다** — fold 5→18, OOS 표본 1,700→3,240bar, family pooling에 의한 허위 admit 없이 8개 signal이 개별적으로 강한 통계적 유의성(t up to +8.70)으로 admit됐다.
2. **P2/P3(복합 사후분포 게이트)도 정상 배선됐다** — `prob_positive`가 실제로 계산·캡처되고, guard(5/18)와 통계 판단(13/18)이 스펙이 정의한 대로 분리되어 동작한다.
3. **그럼에도 여전히 cash-only다.** 그러나 이번 `no_evidence`의 사유는 더 이상 "도달 불가능한 게이트 산술"이 아니라 **실측된 비용 우위**다: gross alpha는 실재하고 연 +9~31%로 상당하지만, 비용이 연 -68~70%로 이를 3~7배 압도한다. 이는 스펙의 완료 기준("사유가 실제로 계산된 통계량이어야 한다")을 충족하는 정직한 결과다.
4. **임계값을 낮추거나 cash-only를 해제할 근거는 없다.** 다음 측정 대상은 게이트가 아니라 **8h/12h 초단기 반전 signal의 회전율·비용 모델**이다 — `build_fold_expert_books`의 Kelly 사이징이 이 속도의 신호에 적합한지, `alpha_smooth`/`band_frac` 완충이 이 신호군 전용으로 강화되어야 하는지가 다음 스펙의 후보다.

원본 artifact:

- [result.json](../../logs/futures/compound/20260729_075352/result.json)
- [manifest.json](../../logs/futures/compound/20260729_075352/manifest.json)
- [target_weights.npy](../../logs/futures/compound/20260729_075352/target_weights.npy)
- [l1_admission.jsonl](../../logs/l1_admission.jsonl)
- [l1_cash_only_exit_redesign.md](../specs/l1_cash_only_exit_redesign.md)
