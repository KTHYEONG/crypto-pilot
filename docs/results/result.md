## L1 CCI member quarantine — production comparison (2026-07-30)

### 1. 실행 식별자와 재현 조건

| 항목 | 이전 실행 | 신규 실행 |
|---|---|---|
| 실행 디렉터리 | `logs/futures/compound/20260730_022430/` | `logs/futures/compound/20260730_025740/` |
| 변경 | 10-member `trend_momentum` | `cci` 1개 격리(9-member) |
| 명령 | `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42` | 동일 |
| reference date / seed | `2026-07-15` / `42` | `2026-07-15` / `42` |
| data manifest hash | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` | 동일 |
| 입력 | 5,442 4h bars × 51 symbols | 동일 |
| 실행 상태 | `integrity_ok=true`, `dry_run=true`, exit code 0 | `integrity_ok=true`, `dry_run=true`, exit code 0 |
| L2/L3 보호 | `L2_DRY_RUN=1`, 동일 sealed quarterly window | 동일 |

신규 실행은 코드 변경 후 실제 production CLI를 재실행한 결과다. 데이터·기간·seed를
고정했으므로 관측된 차이는 CCI member quarantine과 그에 따른 downstream weight 변화로
귀속한다. 신규 실행 로그에는 기존과 동일하게 `smart_money_divergence` optional-field
누락 경고와 clustering 상관행렬의 NumPy divide warning만 남았고, P2 pipeline error는
없었다.

### 2. 변경 범위

`src/domain/futures/compound/l1_concept_bank.py`의 frozen registry에서 다음 한 항목만
제거했다.

```text
trend_momentum:
  before = trend_ema, momentum_ts, breakout_donchian, rsi, cci, mfi,
           aroon_oscillator, adx_directional, obv_trend, keltner_breakout
  after  = trend_ema, momentum_ts, breakout_donchian, rsi, mfi,
           aroon_oscillator, adx_directional, obv_trend, keltner_breakout
vol_regime = volume_zscore, bollinger_bandwidth  # unchanged
```

Admission formula, `max_leg_weight=0.50`, per-name cap, risk overlay, universe, execution
cost model, L2/L3 gates, and raw 37-signal catalog were not changed. CCI remains available in
the raw panel for future shadow diagnostics but contributes to no deployed concept.

### 3. L1 prequential evidence — final fold (17 folds)

| Concept | Metric | Before | After | Delta |
|---|---:|---:|---:|---:|
| `trend_momentum` | alpha_ann | 0.3712 | **0.4011** | **+0.0299** |
| `trend_momentum` | t_alpha (NW) | 1.881 | **2.049** | **+0.168** |
| `trend_momentum` | breakeven cost (bps) | 19.1 | **25.7** | **+6.6** |
| `trend_momentum` | mean turnover / bar | 0.088826 | **0.071380** | **−0.017446** |
| `trend_momentum` | positive folds | 13/17 | **14/17** | **+1 fold** |
| `trend_momentum` | posterior_positive | 0.971 | **0.984** | **+0.013** |
| `trend_momentum` | evidence_weight | 0.5000 | 0.5000 | 0.0000 |
| `vol_regime` | alpha_ann | 1.1849 | 1.1849 | 0.0000 |
| `vol_regime` | t_alpha (NW) | 2.414 | 2.414 | 0.000 |
| `vol_regime` | breakeven cost (bps) | 44.5 | 44.5 | 0.0 |
| `vol_regime` | mean turnover / bar | 0.121672 | 0.121672 | 0.000000 |
| `vol_regime` | positive folds | 10/17 | 10/17 | 0 fold |
| `vol_regime` | evidence_weight | 0.5000 | 0.5000 | 0.0000 |

CCI 격리는 trend leg의 신호 자체를 강화한 것이 아니라, trend book의 회전과 희석을
줄여 비용 차감 후 edge를 보존한 변화다. 두 concept 모두 여전히 0.5 cap에 걸리므로
leg weight가 달라진 것이 아니라 trend book 구성과 downstream risk-overlay 반응이 달라졌다.

### 4. L2 gate 비교

| Metric | Before | After | Delta | 판정 영향 |
|---|---:|---:|---:|---|
| verdict | `fail` | `fail` | — | 유지 |
| annualized log growth | 0.189253 | **0.213076** | **+0.023823** | 개선 |
| CAGR | 0.208347 | **0.237478** | **+0.029132** | 개선 |
| excess growth LCB90 | −0.033164 | **+0.024099** | **+0.057262** | 양수 전환 |
| stressed excess growth LCB90 | −0.091242 | **−0.022283** | **+0.068959** | 여전히 음수 |
| excess growth probability | 0.2870 | 0.3097 | +0.0227 | 0.90 미달 |
| equity multiple | 1.213461 | **1.239512** | **+0.026051** | 개선 |
| Sharpe | 1.337650 | **1.591068** | **+0.253418** | 개선 |
| Sharpe probability | 0.8770 | **0.9250** | +0.0480 | 개선 |
| deflated Sharpe probability | 0.600006 | **0.701918** | +0.101913 | 0.90 미달 |
| max drawdown | 0.068349 | 0.079906 | +0.011557 | 악화 |
| daily CVaR95 | −0.012746 | **−0.012191** | +0.000555 | 소폭 개선 |
| annual volatility | 0.137718 | **0.129506** | **−0.008212** | 개선 |
| annual turnover | 79.005419 | **57.454793** | **−21.550625 (−27.3%)** | 개선 |
| cost drag ratio | 0.316578 | **0.232733** | **−0.083844** | 개선 |
| capacity utilisation p95 | 0.178977 | 0.190617 | +0.011640 | 악화 |
| integrity_ok | true | true | — | 유지 |

L2 reasons는 6개에서 5개로 줄었다. `excess_growth_lcb90`은 음수에서 양수로 전환되어
해소됐지만, `stressed_excess_growth_lcb90`, `excess_growth_probability`,
`positive_outer_folds`, `deflated_sharpe_probability`, `capacity_utilisation_p95`는
아직 gate를 통과하지 못했다. 특히 turnover와 cost drag는 크게 개선됐지만, 현재 symbol
capacity 측정은 `0.190617`로 오히려 악화되어 CCI 제거만으로 capacity 병목을 해결할 수
없음이 확인됐다.

### 5. L3 dry-run 비교

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| verdict | `reject` | `reject` | — |
| posterior growth probability | 0.209167 | 0.212500 | +0.003333 |
| holdout days | 90 | 90 | 0 |
| max drawdown | 0.005622 | 0.006249 | +0.000627 |
| daily CVaR95 | −0.000424 | −0.000437 | −0.000013 |
| reasons | `l2_not_pass` | `l2_not_pass` | unchanged |

L3는 두 실행 모두 `l2_not_pass` 하나로 reject됐다. CCI 제거가 L3 posterior를 소폭
올렸지만 holdout MDD와 CVaR는 소폭 악화했으므로, L3를 통과했다고 해석할 수 없다.

### 6. 실제 target weights 비교

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| shape / dtype | `(5442, 51)` / `float32` | 동일 | — |
| nonzero bars | 2,796 (51.378%) | 2,796 (51.378%) | 0 |
| first / last nonzero index | 2,142 / 5,160 | 동일 | 0 |
| mean gross (active bars) | 0.339141 | **0.322806** | −0.016335 |
| max gross | 0.982529 | **0.742257** | **−0.240272** |
| gross p95 | 0.607872 | **0.588285** | −0.019587 |
| changed elements | — | 124,783 / 277,542 | 45.0% of cells |
| max absolute cell delta | — | 0.095954 | — |
| mean absolute cell delta | — | 0.001575 | — |
| total absolute weight delta | — | 437.259368 | — |

활성 봉 수와 holdout 진입 시점은 변하지 않았고, 포지션 크기만 전반적으로 낮아졌다.
최대 gross가 0.9825에서 0.7423으로 내려가 집중 extreme은 완화됐지만, L2의 별도
capacity utilisation 계산은 악화됐으므로 두 지표를 동일한 병목으로 취급하면 안 된다.

### 7. 결론과 다음 검증 경계

1. **CCI quarantine은 유효한 단일 변경이다.** 동일 데이터에서 L1 trend alpha,
   t-stat, breakeven, positive fold, L2 CAGR/Sharpe가 개선되고 annual turnover와 cost
   drag가 감소했다.
2. **전략은 아직 승인 상태가 아니다.** L2는 `fail`, L3는 `reject`이며 자동 배포/승격은
   일어나지 않았다.
3. **남은 핵심 병목은 capacity와 확률적 지속성이다.** CCI 제거 후에도 capacity p95가
   0.1906이고 stressed growth LCB90은 −0.0223이다. 다음 변경에서 concept 확장,
   leg-weight 개편, capacity 정책을 동시에 건드리면 attribution이 사라진다.
4. **후속 변경 제한**: 다음 실험도 한 번에 하나의 member 또는 하나의 allocator 규칙만
   변경하고, 동일한 pre-L3 chronological validation과 production replay를 통과해야 한다.

### 8. 원본 artifact

- [신규 result.json](../../logs/futures/compound/20260730_025740/result.json)
- [신규 manifest.json](../../logs/futures/compound/20260730_025740/manifest.json)
- [신규 target_weights.npy](../../logs/futures/compound/20260730_025740/target_weights.npy)
- [이전 result.json](../../logs/futures/compound/20260730_022430/result.json)
- [이전 target_weights.npy](../../logs/futures/compound/20260730_022430/target_weights.npy)
- [production 실행 로그](/tmp/l1_cci_quarantine_run.log)
