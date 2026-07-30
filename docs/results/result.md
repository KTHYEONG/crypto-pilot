## L1 신호 스테이지 재설계(`l1_signal_stage_redesign`) 실전 CLI 검증 결과 — 2026-07-30

### 1. 실행 식별자

| 항목 | 값 |
|---|---|
| 기준 실행 (evidence_weight 로깅 버그 수정 후, 최종본) | `logs/futures/compound/20260730_011331/` |
| 대조 실행 (수정 전, 배포 수치 동일성 확인용) | `logs/futures/compound/20260730_010639/` |
| 기준 명령 | `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42` |
| reference date / seed | `2026-07-15` / `42` |
| 입력 규모 | `5,442` bars(4h) × `51` symbols |
| data manifest hash | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` (직전 세션과 동일 스냅샷) |
| L2/L3 보호 | `L2_DRY_RUN=1` (dry_run=true), sealed L3 holdout 미소비 |
| 적용 스펙 | `docs/specs/l1_signal_stage_redesign.md` — signal 단계를 2-concept leg 파이프라인(`trend_momentum`, `vol_regime`)으로 전면 재설계 |
| universe (51종) | 1000BONKUSDT, 1000PEPEUSDT, 1000SHIBUSDT, AAVEUSDT, ADAUSDT, ALGOUSDT, APTUSDT, ARBUSDT, ATOMUSDT, AVAXUSDT, BCHUSDT, BELUSDT, BNBUSDT, BTCUSDT, CHZUSDT, CRVUSDT, DASHUSDT, DOGEUSDT, DOTUSDT, DYDXUSDT, ETCUSDT, ETHUSDT, FETUSDT, FILUSDT, HBARUSDT, ICPUSDT, IDUSDT, INJUSDT, JTOUSDT, LDOUSDT, LINKUSDT, LTCUSDT, NEARUSDT, ONDOUSDT, OPUSDT, ORDIUSDT, PENDLEUSDT, RIFUSDT, SEIUSDT, SOLUSDT, STGUSDT, SUIUSDT, TIAUSDT, TRXUSDT, UNIUSDT, WIFUSDT, WLDUSDT, XLMUSDT, XMRUSDT, XRPUSDT, ZECUSDT |

이번 검증은 스펙의 "Completion criteria"(`/check` green **AND** 실전 CLI 재실행에서 `l1_admission.jsonl`이 per-leg `alpha_ann`/`breakeven_cost_bps`/`t_alpha`/`evidence_weight`를 실제 계산값으로 기록)를 코드 실행으로 직접 확인한 것이다.

### 2. 검증 중 발견·수정한 결함: `evidence_weight` 로깅이 항상 0으로 찍히던 버그

1차 실행(`20260730_010639`)에서 `l1_admission.jsonl`의 모든 `LEG` 레코드가 `evidence_weight=0.0000`으로 기록됨을 확인했다. 원인 추적 결과, `evaluate_leg_alpha`(`l1_leg_evaluation.py`)가 `LegEvidence`를 생성할 때 `evidence_weight=0.0`을 하드코딩한 채로 `record_leg`에 그대로 로깅하고, 실제 가중치는 `l1_leg_admission.py`의 `accumulate_prequential_leg_weights`가 `compute_evidence_weight`를 별도 재계산해 배포에만 사용하는 구조였다 — **배포 자체(target_weights)는 정상**이었으나 진단 로그가 스펙의 완료 기준에 미달했다.

`evaluate_leg_alpha`가 자기 자신에 대해 `compute_evidence_weight`를 계산해 `dataclasses.replace`로 스탬프를 찍도록 수정하고, `accumulate_prequential_leg_weights`의 중복 계산을 제거해 단일 소스로 통일했다. `/check` 재검증(Cov 86% PASS, mypy strict PASS) 후 2차 실행(`20260730_011331`)에서 `evidence_weight`가 실제 계산값(`0.5000` = `max_leg_weight` 상한)으로 기록됨을 확인했다. **두 실행의 L2/L3 최종 수치는 완전히 동일** — 순수 진단 완결성 수정이었음이 실측으로 확인됐다.

### 3. Concept registry (2 concepts, 13 formula members)

| Concept | Mode | Member formulas |
|---|---|---|
| `trend_momentum` | xs | `trend_ema`, `momentum_ts`, `breakout_donchian` (기존 커널 재사용) + `rsi`, `cci`, `mfi`, `aroon_oscillator`, `adx_directional`, `obv_trend`, `keltner_breakout` (신규) |
| `vol_regime` | ts | `volume_zscore`, `bollinger_bandwidth`, `volatility_squeeze_keltner` (부활) |

### 4. Per-leg 실측 (`logs/l1_admission.jsonl`, `LEG` 태그, prequential fold별 실제 계산값)

fold `i`의 증거는 fold `0..i-1`만 사용(causal, [RULE-07]). 아래는 실전 실행에서 생성된 전체 30개 레코드다.

**`trend_momentum` (xs, n=15 folds)**

| fold | alpha_ann | beta | aSR | t_alpha | be_bps | turnover/bar | pos_folds | posterior | weight |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 0 | 2.0051 | 0.050 | 0.171 | 3.975 | 100.5 | 0.09114 | 3/3 | 1.000 | 0.5000 |
| 1 | 1.7972 | 0.010 | 0.158 | 4.236 | 92.0 | 0.08918 | 4/4 | 1.000 | 0.5000 |
| 2 | 1.6865 | −0.029 | 0.143 | 4.304 | 86.9 | 0.08864 | 5/5 | 1.000 | 0.5000 |
| 3 | 1.4496 | −0.045 | 0.130 | 4.269 | 72.6 | 0.09118 | 6/6 | 1.000 | 0.5000 |
| 4 | 1.4693 | −0.038 | 0.137 | 4.861 | 74.0 | 0.09071 | 7/7 | 1.000 | 0.5000 |
| 5 | 1.3987 | −0.039 | 0.136 | 5.154 | 70.3 | 0.09083 | 8/8 | 1.000 | 0.5000 |
| 6 | 1.3455 | −0.040 | 0.135 | 5.440 | 68.8 | 0.08935 | 9/9 | 1.000 | 0.5000 |
| 7 | 1.3380 | −0.040 | 0.137 | 5.830 | 68.2 | 0.08962 | 10/10 | 1.000 | 0.5000 |
| 8 | 1.3103 | −0.040 | 0.138 | 6.134 | 67.4 | 0.08882 | 11/11 | 1.000 | 0.5000 |
| 9 | 1.2725 | −0.039 | 0.138 | 6.394 | 65.5 | 0.08871 | 12/12 | 1.000 | 0.5000 |
| 10 | 1.3991 | −0.048 | 0.126 | 6.115 | 72.3 | 0.08833 | 13/13 | 1.000 | 0.5000 |
| 11 | 1.5748 | −0.043 | 0.133 | 6.695 | 81.6 | 0.08818 | 14/14 | 1.000 | 0.5000 |
| 12 | 1.5376 | −0.050 | 0.133 | 6.909 | 79.4 | 0.08840 | 15/15 | 1.000 | 0.5000 |
| 13 | 1.5252 | −0.054 | 0.134 | 7.199 | 78.6 | 0.08860 | 16/16 | 1.000 | 0.5000 |
| 14 | 1.4930 | −0.056 | 0.131 | 7.240 | 76.2 | 0.08944 | 17/17 | 1.000 | 0.5000 |

**`vol_regime` (ts, n=15 folds)**

| fold | alpha_ann | beta | aSR | t_alpha | be_bps | turnover/bar | pos_folds | posterior | weight |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 0 | 2.0335 | 0.567 | 0.087 | 2.033 | 91.2 | 0.10185 | 2/3 | 0.969 | 0.5000 |
| 1 | 1.8804 | 0.441 | 0.069 | 1.853 | 74.5 | 0.11530 | 3/4 | 0.977 | 0.5000 |
| 2 | 2.2836 | 0.444 | 0.081 | 2.417 | 88.7 | 0.11754 | 4/5 | 1.000 | 0.5000 |
| 3 | 1.7593 | 0.422 | 0.063 | 2.060 | 67.5 | 0.11898 | 4/6 | 0.991 | 0.5000 |
| 4 | 1.9546 | 0.394 | 0.070 | 2.472 | 72.9 | 0.12239 | 5/7 | 0.997 | 0.5000 |
| 5 | 1.8031 | 0.389 | 0.066 | 2.490 | 67.3 | 0.12240 | 6/8 | 0.998 | 0.5000 |
| 6 | 1.7973 | 0.359 | 0.067 | 2.681 | 65.2 | 0.12591 | 7/9 | 0.999 | 0.5000 |
| 7 | 1.5455 | 0.357 | 0.058 | 2.467 | 56.4 | 0.12505 | 7/10 | 0.999 | 0.5000 |
| 8 | 1.3165 | 0.321 | 0.050 | 2.225 | 47.2 | 0.12730 | 7/11 | 0.993 | 0.5000 |
| 9 | 1.2090 | 0.254 | 0.046 | 2.121 | 43.9 | 0.12571 | 8/12 | 0.996 | 0.5000 |
| 10 | 1.3901 | 0.269 | 0.052 | 2.533 | 51.4 | 0.12348 | 9/13 | 0.997 | 0.5000 |
| 11 | 1.4775 | 0.307 | 0.056 | 2.813 | 55.3 | 0.12207 | 10/14 | 0.999 | 0.5000 |
| 12 | 1.6439 | 0.283 | 0.062 | 3.219 | 60.6 | 0.12384 | 11/15 | 1.000 | 0.5000 |
| 13 | 1.5773 | 0.257 | 0.060 | 3.197 | 58.3 | 0.12347 | 12/16 | 1.000 | 0.5000 |
| 14 | 1.4943 | 0.275 | 0.057 | 3.168 | 55.8 | 0.12218 | 13/17 | 1.000 | 0.5000 |

`reasons=[]`(거부 사유 없음) 전 레코드. 두 concept 모두 전 fold에서 `evidence_weight`가 `max_leg_weight`(0.50) 상한에 도달 — posterior가 0.97~1.00으로 거의 항상 최대치이기 때문(`[RULE-09]` 가중치 공식: `min(2·max(Φ(t_alpha)−0.5,0), max_leg_weight)`).

### 5. 배포 (`target_weights.npy`, 5442×51 float32)

| metric | value |
|---|---:|
| nonzero 봉 비율 | **50.7%** (2,760 / 5,442 bars) |
| mean gross exposure (전체 봉 평균) | 0.1629 |
| mean gross exposure (활성 봉만) | 0.3212 |
| max gross exposure | 0.7442 |

L1이 signal 재설계 이래 처음으로 **cash-only가 아닌 실제 비영(非零) 포지션을 배포**했다 — 이전까지의 모든 기록(`admitted=0/37`)과 구조적으로 다른 결과다.

### 6. L2 / L3 최종 판정 (두 실행 동일)

| metric | value |
|---|---:|
| L1 admitted concepts | 2 / 2 (`trend_momentum`, `vol_regime`) |
| L2 verdict | **fail** |
| L2 annualized_log_growth | 0.2102 |
| L2 cagr | 0.2339 |
| L2 sharpe | 1.5632 |
| L2 sharpe_probability | 0.9155 |
| L2 deflated_sharpe_probability | 0.7075 (< 0.90 기준) |
| L2 excess_growth_probability | 0.3023 (< 0.90 기준) |
| L2 excess_growth_lcb90 | 0.0085 |
| L2 stressed_excess_growth_lcb90 | −0.0340 (양수 아님) |
| L2 max_drawdown | 0.0770 |
| L2 daily_cvar95 | −0.0119 |
| L2 annual_volatility | 0.1306 |
| L2 annual_turnover | 62.30 |
| L2 cost_drag_ratio | 0.2502 |
| L2 capacity_utilisation_p95 | 0.1837 (> 0.10 기준) |
| L2 integrity_ok | true |
| L2 reasons | `stressed_excess_growth_lcb90=-0.034045 not strictly positive`; `excess_growth_probability=0.3023<0.9`; `positive_outer_folds=2<3`; `deflated_sharpe_probability=0.7075<0.9`; `spa_pvalue=0.1050>0.1`; `capacity_utilisation_p95=0.1837>0.1` |
| L3 verdict | **reject** |
| L3 posterior_growth_probability | 0.1727 |
| L3 holdout_days | 90 |
| L3 reasons | `low_growth_probability`; `l2_not_pass` |

`capacity_utilisation_p95` 초과는 스펙의 `[LIMIT-04]`(엣지가 51종 서바이버 유니버스 내 유동성 낮은 종목에 집중되어 있다는 사전 실측)와 동일한 방향의 신호 — 별개 우연이 아니라 스펙이 이미 예견한 구조적 위험이 실제 게이트에서도 재확인된 것이다.

### 7. 부수 관측: 워닝 로그

실행 중 다음 워닝이 반복 관측됨 (기능 저해 없이 `/check` 통과·배포 정상이었으나 추후 조용히 시키거나 원인 규명할 잔여 과제로 기록):

```
src/domain/futures/compound/l1_concept_bank.py:122: RuntimeWarning: Mean of empty slice
  grid_avg = np.nanmean(stacked, axis=-1)
numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
```

`(t, s)` 셀에서 concept의 모든 member 신호가 동시에 NaN인 구간(예: 워밍업 초반 또는 데이터 공백 종목)에서 `np.nanmean`이 빈 슬라이스를 평균 내려 하며 발생 — 다운스트림에서 `eligible_2d`/`valid_3d` 마스킹으로 흡수되는 것으로 보이나(배포 수치 정상, 재현 동일), 명시적 마스킹 처리로 워닝을 원천 차단하는 정리는 아직 안 됨.

### 8. 결론

1. **구조적 목표 달성 확인**: 스펙의 핵심 목표("평가 대상 = 배포 대상", "L1이 실제로 신호를 낼 수 있게")가 실전 CLI 실행으로 확인됐다. `l1_admission.jsonl`의 30개 `LEG` 레코드 전량이 진짜 계산값(`alpha_ann`, `t_alpha`, `breakeven_cost_bps`, `evidence_weight`)이며, `target_weights.npy`의 50.7% 봉에 실제 비영 포지션이 배포됐다 — 이 프로젝트 기록 전체를 통틀어 최초.
2. **검증 과정에서 진단 완결성 결함 1건 발견·수정**: `evidence_weight` 로깅이 항상 0으로 찍히던 버그. 배포 자체엔 영향 없었으나(수정 전후 L2/L3 수치 완전 동일), 스펙의 완료 기준을 문자 그대로 충족시키기 위해 수정. `/check` 재확인 PASS(Cov 86%).
3. **L2/L3는 여전히 탈락 — 이것도 정직한 결과**: 구조적 결함(예전의 "무조건 cash-only")이 아니라, 통계적 강건성 게이트(`excess_growth_probability`, `deflated_sharpe_probability`, `positive_outer_folds`, `spa_pvalue`, `capacity_utilisation_p95`)를 실측 수치가 못 넘은 것. 특히 `capacity_utilisation_p95` 초과는 스펙이 사전에 문서화한 `[LIMIT-04]`(유동성 낮은 종목에 엣지 집중) 위험이 실제로 발현된 것으로, 우연이 아니라 예견된 실패 모드다.
4. **다음 착수점**: (a) `capacity_utilisation_p95` 초과 원인이 되는 저유동성 종목 비중을 낮추는 방향의 유니버스/capacity 정책 재검토, (b) `positive_outer_folds=2/3` 미달의 원인이 되는 fold간 비정상성 재조사, (c) L1LegPanel 조립 시 발생하는 "Mean of empty slice" 워닝의 명시적 마스킹 처리.

원본 artifact:

- [result.json (최종)](../../logs/futures/compound/20260730_011331/result.json)
- [manifest.json (최종)](../../logs/futures/compound/20260730_011331/manifest.json)
- [target_weights.npy (최종)](../../logs/futures/compound/20260730_011331/target_weights.npy)
- [result.json (수정 전 대조군)](../../logs/futures/compound/20260730_010639/result.json)
- [l1_admission.jsonl](../../logs/l1_admission.jsonl)
- [l1_signal_stage_redesign.md](../specs/l1_signal_stage_redesign.md)
- [l1_signal_stage_redesign_contract.json](../specs/l1_signal_stage_redesign_contract.json)
