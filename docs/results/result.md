## L1 family-only routing 실제 데이터 측정 결과 — 2026-07-29

### 1. 실행 식별자와 원자료

| 항목 | 값 |
|---|---|
| 기준 실행 | `logs/futures/compound/20260729_053222/` |
| 내부 계측 실행 | `logs/futures/compound/20260729_053625/` |
| 기준 명령 | `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42` |
| 계측 명령 | 동일 조건으로 `scratch/verify_l1_family_sleeve_routing.py --variant family_only` 실행. 현재 production route를 감싸 evidence만 기록 |
| reference date / seed | `2026-07-15` / `42` |
| base timeframe | `1h` (L1 내부 결정 grid `4h`) |
| 입력 규모 | `5,442` bars × `51` symbols |
| model version | `quarterly-v1` |
| data manifest hash | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` |
| process | 두 실행 모두 `exit_code=0` |
| L2/L3 보호 | `L2_DRY_RUN=1`, sealed L3 holdout 미소비 |

기준 artifact hash:

- `result.json`: `216e76fd9567318d5d4e5b4b6270f81082845d288b34a502953efe764419afde`
- `target_weights.npy`: `2a161f690b5593fc026fbf44c11205b0afd6e228295cb756bb4c779092e6c102`

### 2. Family screen 원자료

`n_ic_bars=1700`, `sidak_alpha=0.0061`, `declared_orientation=+1` 기준이다.

| family | signals | mean IC | NW t | 판정 | reason |
|---|---:|---:|---:|:---:|---|
| trend_ema | 5 | `+0.0018` | `+0.134` | reject | `not_significant_after_sidak` |
| momentum_ts | 5 | `-0.0136` | `-1.260` | reject | `not_significant_after_sidak` |
| breakout_donchian | 5 | `-0.0218` | `-2.251` | reject | `declared_orientation_contradicted` |
| basis_gap | 5 | `-0.0230` | `-2.524` | reject | `declared_orientation_contradicted` |
| reversal_st | 1 | `+0.0199` | `+2.753` | **admit** | — |
| xs_reversal | 2 | `+0.0332` | `+2.942` | **admit** | — |
| xs_momentum_slow | 2 | `-0.0028` | `-0.141` | reject | `not_significant_after_sidak` |
| smart_money_divergence | 2 | `0.0000` | `0.000` | reject | `insufficient_ic_samples`, `n_ic_bars=0` |

Family screen이 admit한 signal ID는 `reversal_st:*`, `xs_reversal:fast`,
`xs_reversal:medium`이다. 따라서 이번 실행에서 family gate 자체는 공집합이
아니다.

### 3. 구조적 sleeve와 prequential route 계측

| 계측값 | 값 |
|---|---:|
| `admitted_sleeves` (family/cluster 구조) | `57` |
| `distinct_series` | `1` |
| OOS evidence bars | `1,700` |
| `tested_hypotheses` | `6` |
| evidence rows | `15` |
| admitted evidence rows | `0` |
| active route bars | `0` |
| max active experts | `0` |
| `is_cash_only` | `true` |

신호별 evidence:

| signal | rows | max prior evidence bars | max posterior positive probability | rejection |
|---|---:|---:|---:|---|
| `reversal_st:fast` | 5 | 1,360 | `0.001` | warmup 3, growth probability 2 |
| `xs_reversal:fast` | 5 | 1,360 | `0.000` | warmup 3, growth probability 2 |
| `xs_reversal:medium` | 5 | 1,360 | `0.000` | warmup 3, growth probability 2 |
| **합계** | **15** | — | — | **warmup 9, growth probability 6** |

현재 `RegimeRouterConfig` 불변 임계값은 다음과 같다.

| parameter | value |
|---|---:|
| `min_evidence_bars` | `900` |
| `min_posterior_probability` | `0.90` |
| `min_effective_blocks` | `20` |
| `min_positive_inner_folds` | `2` |
| `n_bootstrap` | `1,000` |

### 4. 수익률 도메인 검증

실제 allocator output을 expert weight로 채점한 bootstrap 입력 6회에서:

| metric | observed |
|---|---:|
| minimum simple net return | `-0.004612` |
| maximum simple net return | `+0.004675` |
| observations per call | `1,020` 또는 `1,360` |
| `net <= -1` count | **0** |
| non-finite return/statistic rejection | **0** |

이전 raw-z 오배선에서 관측된 `-12.1053..+17.7465` 및 `net<=-1`은 현재
경로에서 재현되지 않았다. 이번 `posterior=0`은 수치 도메인 파손이 아니라
실제 성장 posterior가 `0.90` 바닥을 통과하지 못한 결과다.

### 5. L2 / L3 결과

| metric | value |
|---|---:|
| L2 verdict | `no_evidence` |
| annualized log growth | `0.0` |
| CAGR | `0.0` |
| excess growth LCB90 | `0.0` |
| excess growth probability | `0.5` |
| Sharpe | `0.0` |
| Sharpe probability | `0.5` |
| deflated Sharpe probability | `0.5` |
| max drawdown | `0.0` |
| annual volatility | `0.0` |
| annual turnover | `0.0` |
| cost drag ratio | `0.0` |
| capacity utilisation p95 | `0.0` |
| integrity_ok | `true` |
| L2 reasons | `active_days_ratio=0.0000<0.1`; `rebalances=0<30` |
| L3 verdict | `reject` |
| L3 reasons | `low_growth_probability`; `l2_not_pass` |

`target_weights.npy`는 `float32`, shape `(5442, 51)`이며 `nonzero=0`,
`max_abs=0.0`, `mean_abs=0.0`이다. 따라서 실제 자본 배치는 0%, 현금 비중은
100%다.

### 6. Handoff 원자료

최종 `EVAL` record:

```json
{
  "admitted_sleeves": 57,
  "distinct_series": 1,
  "oos_bars": 1700,
  "ann_growth": 0.0,
  "ann_lcb90": 0.0,
  "pw_block": 5.0,
  "turnover": 0.0,
  "cost_drag": 0.0,
  "positive_folds": 0,
  "fold_growths": [0.0, 0.0, 0.0, 0.0, 0.0],
  "mean_abs_net": 0.0,
  "admitted": false
}
```

구조적 sleeve가 존재한다는 사실과 handoff 성장 gate를 통과했다는 사실을
분리했다. `admitted_sleeves=57`은 후보 구조가 생성됐다는 뜻이고,
`positive_folds=0/5`와 `admitted=false`가 실제 배포 거부를 결정한다.

### 7. 실행 경고와 판정

- `smart_money_divergence`에 필요한
  `top_trader_long_short_ratio/long_short_ratio`가 없어 해당 family는
  `n_ic_bars=0`으로 측정 제외됐다.
- NumPy correlation 단계에서 표준편차 0에 대한 `invalid value encountered
  in divide` 경고가 있었으나 프로세스는 종료 코드 0으로 완료됐다.
- `integrity_ok=true`, return-domain violation 0건, sealed L3 미소비이므로
  이번 결과의 cash-only는 보호 동작으로 분류한다.

### 8. 결론

1. family-only 배선은 실제 데이터에서 `reversal_st`와 `xs_reversal`을
   구조적 sleeve로 생성했고, prequential route에 6개 가설·15개 evidence
   row를 전달했다.
2. 그러나 최대 posterior `0.001`~`0.000`으로 기존 `0.90` threshold와
   handoff `positive_folds=4` 조건을 모두 충족하지 못했다.
3. 임계값을 낮추거나 cash-only를 해제할 근거는 없다. 다음 측정 대상은
   gate가 아니라 admitted family의 OOS 성장성, 특히 reversal signal의
   fold별 net return 안정성이다.

원본 artifact:

- [result.json](../../logs/futures/compound/20260729_053222/result.json)
- [diagnostic result.json](../../logs/futures/compound/20260729_053625/result.json)
- [manifest.json](../../logs/futures/compound/20260729_053222/manifest.json)
- [target_weights.npy](../../logs/futures/compound/20260729_053222/target_weights.npy)
- [l1_admission.jsonl](../../logs/l1_admission.jsonl)
