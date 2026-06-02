---
title: Candidate ML Strategy Diagnostic Report
domain: futures/strategy
type: domain-spec
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/candidate_labels.py
  - src/domain/futures/strategy/candidate_gate.py
  - src/domain/futures/strategy/candidate_edge.py
  - src/domain/futures/strategy/candidate_portfolio.py
  - src/domain/futures/strategy/ablation.py
last_verified: 2026-06-02
---

# Candidate ML Strategy Diagnostic Report

**실행 환경:** `--phase alpha --timeframe 4h --sync skip`  
**실행 일자:** 2026-06-02  
**데이터 기간:** IS 2023-10-01 ~ 2025-10-01 / OOS 2025-10-01 ~ 2026-03-31  
**유효 심볼:** 33개 (pass) / 69개 (로드)  
**반영 구현:** gate label 분리, dataset gate target 전환, ablation `entry_idx` 보정, uncapped Kelly per-bar 정합화, net target contract 정합화, selection sensitivity 로그, payoff-aware rule recommendation

## 1. 요약

```
rule_only_equal_size                         CAGR  -7.44%   DD 22.95%   거래 있음
rule_only_fractional_kelly                   CAGR  -0.20%   DD  1.20%   거래 있음
rule_plus_ml_gate                            CAGR   0.00%   DD  0.00%   거래 0건
rule_plus_ml_gate_plus_edge                  CAGR   0.00%   DD  0.00%   거래 0건
rule_plus_ml_gate_plus_edge_plus_portfolio_caps CAGR 0.00%  DD 0.00%   거래 0건
candidate_ml_full                            CAGR   0.00%   DD  0.00%   거래 0건
```

핵심 판정:

1. gate collapse는 해소됐다.
2. edge target contract의 중복 cost subtraction는 제거됐다.
3. production threshold에서는 여전히 거래가 0건이다.
4. zero-trade의 직접 병목은 `q10` shortfall filter와 gate support 부족이다.

## 2. 최신 진단 로그

### 2-1. 라벨 분포

```
[DIAG][LABEL] events=5696 barrier_label1_rate=0.269 gate_label1_rate=0.416 mean_edge=-4.0 median_edge=-162.8 pct_edge_pos=0.416 p10_edge=-671.3 p90_edge=860.1
[DIAG][LABEL] events=5696 barrier_label1_rate=0.204 gate_label1_rate=0.414 mean_edge=21.6 median_edge=-159.5 pct_edge_pos=0.414 p10_edge=-640.8 p90_edge=921.5
```

해석:

- gate target은 `pct_edge_pos`와 정렬된다.
- pruning된 Donchian pool에서도 양수 edge pocket은 남아 있다.
- 다만 full distribution은 여전히 tail이 거칠다.

### 2-2. Gate 예측

```
[DIAG][GATE] n=2584 mean_p=0.4073 median_p=0.4075 max_p=0.4195 pct_ge55=0.000 pct_ge50=0.000 pct_ge45=0.000 calibrated=True
[DIAG][GATE] n=5696 mean_p=0.4075 median_p=0.4078 max_p=0.4201 pct_ge55=0.000 pct_ge50=0.000 pct_ge45=0.000 calibrated=True
```

해석:

- gate는 calibration collapse가 아니다.
- 하지만 현재 pruning된 후보군에서는 `0.55`가 support 밖이다.
- gate만 낮추면 해결되지 않고, q10가 여전히 전부를 막는다.

### 2-3. Edge 예측

```
[DIAG][EDGE] n=2584 target_scale=net cost_bps=24.0 mu_model mean=-0.5 max=8.3 pct_ge25=0.000 | mu_decision mean=-0.5 max=8.3 pct_ge1=0.399 | q10_net mean=-554.9 min=-1724.3 | utility mean=-555.346 max=-243.380
[DIAG][EDGE] n=5696 target_scale=net cost_bps=24.0 mu_model mean=-0.6 max=8.3 pct_ge25=0.000 | mu_decision mean=-0.6 max=8.3 pct_ge1=0.373 | q10_net mean=-576.8 min=-2773.5 | utility mean=-577.348 max=-210.106
```

해석:

- `candidate_edge.py`는 이제 net target scale로 정합화됐다.
- `mu_decision`은 일부 support에서 양수가 나오지만 평균은 아직 약하다.
- `q10_net`은 여전히 크게 음수라서 hard shortfall filter를 통과하지 못한다.

### 2-4. Selection 탈락

```
[DIAG][SELECT] total=2584 gate_fail=2584 edge_fail=1552 q10_fail=2584 all_fail=2584 passed=0 | thresholds(gate>=0.55 edge_net>=1.0 q10>=-80.0)
[DIAG][SELECT] total=5696 gate_fail=5696 edge_fail=3570 q10_fail=5696 all_fail=5696 passed=0 | thresholds(gate>=0.55 edge_net>=1.0 q10>=-80.0)
```

해석:

- production 조건에서는 여전히 `passed=0`이다.
- sensitivity 로그에서는 `q10>=-400`에서는 pass가 생기지만, `q10>=-80`에서는 0건이다.
- 즉 zero-trade의 직접 병목은 `q10` shortfall filter다.

### 2-5. Selection Sensitivity

```
[DIAG][SELECT_SENS] gate>=0.40 edge>=0.0 q10>=-400.0 passed=441 pass_rate=0.0774 top_variant=trend_donchian:donchian_18 top_pass=244
[DIAG][SELECT_SENS] gate>=0.40 edge>=1.0 q10>=-400.0 passed=408 pass_rate=0.0716 top_variant=trend_donchian:donchian_18 top_pass=227
[DIAG][SELECT_SENS] gate>=0.40 edge>=5.0 q10>=-400.0 passed=81 pass_rate=0.0142 top_variant=trend_donchian:donchian_18 top_pass=45
[DIAG][SELECT_SENS] gate>=0.40 edge>=0.0 q10>=-250.0 passed=5 pass_rate=0.0009 top_variant=trend_donchian:donchian_18 top_pass=3
[DIAG][SELECT_SENS] gate>=0.40 edge>=1.0 q10>=-250.0 passed=5 pass_rate=0.0009 top_variant=trend_donchian:donchian_18 top_pass=3
[DIAG][SELECT_SENS] gate>=0.40 edge>=0.0 q10>=-80.0 passed=0 pass_rate=0.0000 top_variant= top_pass=0
```

해석:

- gate/edge/q10의 joint policy를 봐야 한다.
- gate만 완화하거나 edge만 완화해서는 충분하지 않다.
- `q10` threshold가 가장 강한 차단기다.

## 3. 이번 패치의 효과 검증

### Fix-1 [DONE]: Gate label 분리

- `barrier_first_label`
- `profitable_after_hurdle_label`
- `triple_barrier_label` 유지

판정: **성공**

### Fix-2 [DONE]: Dataset gate target 전환

`build_candidate_dataset()`는 `profitable_after_hurdle_label`을 `y_gate`로 사용한다.

판정: **성공**

### Fix-3 [DONE]: Ablation `entry_idx` 1-bar 오프셋 제거

판정: **성공**

### Fix-4 [DONE]: Uncapped Kelly per-bar 정합화

판정: **성공**

### Fix-5 [DONE]: Edge target scale 정합화

`candidate_dataset.py`와 `candidate_edge.py`에서 net scale contract를 맞췄다.

판정: **성공**

### Fix-6 [DONE]: Selection sensitivity 로그 추가

`[DIAG][SELECT_SENS]`로 gate/edge/q10 threshold 조합별 pass count를 확인할 수 있다.

판정: **성공**

### Fix-7 [DONE]: Rule recommendation payoff-aware 전환

`pct_edge_pos >= 0.50` 고정 기준을 제거하고 payoff ratio와 q10 fail-rate를 같이 보도록 바꿨다.

판정: **부분 성공**

## 4. 현재 근본 원인 재정리

### RC-1 [RESOLVED]: Gate calibration collapse

gate는 더 이상 collapse 상태가 아니다.

### RC-2 [REDUCED]: Edge target contract 오류

중복 cost subtraction는 제거됐다. 다만 평균 edge는 아직 약하다.

### RC-3 [ACTIVE/CRITICAL]: Rule signal 품질과 tail risk

rule-only 성과는 생산 기준을 충족하지 못한다. 후보군은 존재하지만 tail이 거칠다.

### RC-4 [ACTIVE/CRITICAL]: q10 shortfall filter

`q10>=-80`은 현재 pruning된 Donchian pool에서 지나치게 엄격하다.

## 5. 권장 조치 업데이트

### P1 [즉시] Threshold 완화는 보류

`min_expected_net_bps`를 음수로 낮춰 거래를 강제로 만드는 것은 권장하지 않는다.

### P2 [최우선] q10 threshold sensitivity 재평가

`q10>=-80` 대신 `q10>=-250` 또는 `q10>=-400` 수준에서 OOS compounding을 다시 확인해야 한다.

### P3 [중요] gate threshold는 joint policy로 다뤄야 함

gate만 낮추는 방식은 edge/q10에서 다시 막힌다.

### P4 [중요] Rule signal 품질 진단

Donchian 이외 전략군까지 포함한 재진단이 필요하다.

## 6. 검증 상태

- unit tests: `32 passed, 27 warnings`
- strategy regression: `tests/unit/domain/futures/strategy --tb=short` `32 passed`
- alpha smoke: exit code `0`

