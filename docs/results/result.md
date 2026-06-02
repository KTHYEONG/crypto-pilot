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
**반영 구현:** gate label 분리, dataset gate target 전환, ablation `entry_idx` 보정, uncapped Kelly per-bar 정합화

---

## 1. 요약: gate calibration은 개선됐지만 ML 거래는 여전히 0건

```
rule_only_equal_size                         CAGR -24.50%   DD 58.16%   거래 있음
rule_only_fractional_kelly                   CAGR  -0.38%   DD  1.36%   거래 있음
rule_plus_ml_gate                            CAGR   0.00%   DD  0.00%   거래 0건
rule_plus_ml_gate_plus_edge                  CAGR   0.00%   DD  0.00%   거래 0건
rule_plus_ml_gate_plus_edge_plus_portfolio_caps CAGR 0.00%  DD 0.00%   거래 0건
candidate_ml_full                            CAGR   0.00%   DD  0.00%   거래 0건
```

이번 변경으로 `candidate_labels.py`의 gate 학습용 base rate는 의도대로 상승했다. 그러나 `candidate_edge.py`의 예측값이 여전히 전구간 음수여서 `select_candidate_events_for_portfolio()` 통과 이벤트는 계속 0건이다.

핵심 결론은 다음과 같다.

1. **해결된 문제:** gate classifier의 calibration collapse
2. **미해결 핵심:** rule signal 자체의 음수 기대값
3. **현재 판정:** 거래 0건은 selector 고장이 아니라 edge 방어 동작

---

## 2. 최신 진단 로그

### 2-1. 라벨 분포 (`candidate_labels.py`)

```
[DIAG][LABEL] events=90257
  barrier_label1_rate = 0.187
  gate_label1_rate    = 0.415
  mean_edge           = -17.1 bps
  median_edge         = -133.0 bps
  pct_edge_pos        = 0.415
  p10_edge            = -650.7 bps
  p90_edge            = +783.7 bps
```

해석:

- `barrier_label1_rate=0.187`은 기존 triple-barrier 성공률이 여전히 낮음을 보여준다.
- `gate_label1_rate=0.415`는 새 gate target이 비용/허들 차감 후 양수 edge 비율과 정렬됐음을 보여준다.
- 즉, 이번 패치의 직접 목표였던 **gate 학습 분포 완화는 성공**했다.

### 2-2. 게이트 모델 예측 분포 (`candidate_gate.py`)

전략 경로 중간 로그:

```
[DIAG][GATE] n=39196
  mean_p    = 0.4066
  median_p  = 0.4133
  max_p     = 0.6573
  pct_ge55  = 0.005
  pct_ge50  = 0.051
  pct_ge45  = 0.251
  calibrated = True
```

ablation full-set 로그:

```
[DIAG][GATE] n=90226
  mean_p    = 0.4172
  median_p  = 0.4198
  max_p     = 0.7091
  pct_ge55  = 0.019
  pct_ge50  = 0.083
  pct_ge45  = 0.302
  calibrated = True
```

해석:

- 이전 진단의 `mean_p ≈ 0.198`에서 `mean_p ≈ 0.41`로 상승했다.
- `pct_ge55`도 `0.1%` 수준에서 `1.9%`까지 증가했다.
- 따라서 `min_gate_probability=0.55`가 완전히 비현실적인 threshold인 상태는 아니게 됐다.

### 2-3. 엣지 모델 예측 분포 (`candidate_edge.py`)

전략 경로 로그:

```
[DIAG][EDGE] n=39196
  mu_gross mean = -19.5 bps
  mu_gross max  = -10.1 bps
  pct_ge25      = 0.000
  mu_net mean   = -43.5 bps
  mu_net max    = -34.1 bps
  pct_ge1       = 0.000
  q10_net mean  = -634.7 bps
  utility mean  = -652.415
```

ablation full-set 로그:

```
[DIAG][EDGE] n=90226
  mu_gross mean = -19.3 bps
  mu_gross max  = -10.1 bps
  pct_ge25      = 0.000
  mu_net mean   = -43.3 bps
  mu_net max    = -34.1 bps
  pct_ge1       = 0.000
  q10_net mean  = -640.7 bps
  utility mean  = -658.828
```

해석:

- 최대 예측조차 `mu_net max = -34.1 bps`로 음수다.
- `min_expected_net_bps = 1.0`을 통과하는 이벤트가 0건인 것은 모델 고장이 아니라 타깃 분포의 결과다.
- 즉 **현재 병목은 gate가 아니라 edge와 원신호 품질**이다.

### 2-4. 필터 탈락 분석 (`candidate_portfolio.py`)

전략 경로 로그:

```
[DIAG][SELECT] total=39196
  gate_fail = 38981
  edge_fail = 39196
  q10_fail  = 39196
  passed    = 0
  thresholds: gate>=0.55 edge_net>=1.0 q10>=-80.0
```

ablation full-set 로그:

```
[DIAG][SELECT] total=90226
  gate_fail = 88469
  edge_fail = 90226
  q10_fail  = 90226
  passed    = 0
  thresholds: gate>=0.55 edge_net>=1.0 q10>=-80.0
```

해석:

- gate 실패 비율은 줄었지만, `edge_fail`과 `q10_fail`은 여전히 100%다.
- 따라서 **거래 0건의 직접 원인은 edge / shortfall filter**다.

---

## 3. 이번 패치의 효과 검증

### Fix-1 [DONE]: Gate label 분리

기존:

```python
triple_label = 1 if barrier_label == 1 and edge_after_hurdle_bps > 0.0 else 0
```

현재:

- `barrier_first_label`
- `profitable_after_hurdle_label`
- `triple_barrier_label`은 backward compatibility 유지

결과:

- `barrier_label1_rate`: `0.187`
- `gate_label1_rate`: `0.415`
- gate probability 중심: `0.198 -> 0.4172`

판정: **성공**

### Fix-2 [DONE]: Dataset gate target 전환

`build_candidate_dataset()`는 `profitable_after_hurdle_label`이 있으면 이를 `y_gate`로 사용하고, 없으면 `triple_barrier_label`로 fallback 한다.

결과:

- 새 gate 학습 분포가 실제 양수 edge 비율과 일치
- 기존 fixture/legacy path도 유지

판정: **성공**

### Fix-3 [DONE]: Ablation `entry_idx` 1-bar 오프셋 제거

기존 `entry_idx - 1`은 look-ahead bias였다. 현재 ablation helper는 `entry_idx` 실행 바에 직접 weight를 기록한다.

판정: **성공**

### Fix-4 [DONE]: Uncapped Kelly per-bar 정합화

Variant 4에서 horizon-level expected edge를 `expected_holding_bars`로 나눠 per-bar variance와 차원을 맞췄다.

판정: **성공**

---

## 4. 현재 근본 원인 재정리

### RC-1 [RESOLVED]: Gate calibration collapse

이전 root cause였던 low-base-rate gate collapse는 해소됐다. `mean_p`와 `median_p`가 `0.4x` 수준으로 회복됐고, `pct_ge55`도 유의미하게 증가했다.

### RC-2 [ACTIVE/FATAL]: Edge 모델 전구간 음수

여전히 모든 이벤트의 net 기대값이 음수다.

```text
mu_net mean = -43.3 bps
mu_net max  = -34.1 bps
pct_ge1     = 0.000
```

이 상태에서는 `min_expected_net_bps=1.0`을 유지하는 한 통과 이벤트 0건이 정상이다.

### RC-3 [ACTIVE/CRITICAL]: Rule 신호 기대값 자체가 음수

```
rule_only_equal_size        CAGR -24.50%  DD 58.16%
rule_only_fractional_kelly  CAGR  -0.38%  DD  1.36%
```

ML이 학습하는 원재료가 평균적으로 손실이면, gate는 “덜 나쁜 신호”를 구분할 수 있어도 edge filter는 결국 전부 탈락시킨다.

### RC-4 [RESOLVED]: Ablation indexing bias

`entry_idx` 오프셋 버그는 수정됐다. 이는 현재 zero-trade 상태의 원인이 아니다.

---

## 5. 권장 조치 업데이트

### P1 [즉시] Threshold 완화는 보류

`min_expected_net_bps`를 음수로 낮춰 거래를 강제로 발생시키는 것은 운영 전략 기준으로 부적절하다. 이는 비용 차감 후 음수 기대값 거래를 허용하는 것이므로 diagnostic spike 용도 외에는 권장하지 않는다.

### P2 [최우선] Rule signal 품질 진단

다음 항목을 바로 점검해야 한다.

1. 각 family별 Spearman IC
2. `side` 방향 반전 시 성능 변화
3. `stop_atr_mult` / `take_profit_atr_mult`의 4h 크립토 적합성
4. family별 평균 `edge_after_hurdle_bps`, hit-rate, shortfall

### P3 [중기] Edge target / feature 재검토

현재 edge target의 중심이 음수라서 회귀기가 음수 예측을 하는 것은 자연스럽다. 다음 중 하나가 필요하다.

1. 원신호를 개선해 target 분포 자체를 양수 영역으로 이동
2. family conditioning feature를 강화해 일부 subset에서만 양수 discrimination이 가능하도록 설계

### P4 [유지] Gate 구조는 현행 유지

gate label 분리와 calibration 동작은 현재 타당하다. 추가 변경보다 rule/edge 원인 분석이 우선이다.

---

## 6. 검증 결과

실행 명령:

```bash
uv run pytest tests/unit/domain/futures/strategy/test_candidate_labels.py \
  tests/unit/domain/futures/strategy/test_candidate_dataset.py \
  tests/unit/domain/futures/strategy/test_ablation.py \
  tests/unit/domain/futures/strategy/test_candidate_edge.py \
  tests/unit/domain/futures/strategy/test_candidate_portfolio.py --tb=short
```

결과:

```text
15 passed, 6 warnings in 1.90s
```

진단 재실행:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase alpha --timeframe 4h --sync skip
```

결과:

- 종료 코드 `0`
- 최신 `[DIAG][LABEL]`, `[DIAG][GATE]`, `[DIAG][EDGE]`, `[DIAG][SELECT]` 로그 반영 완료

---

## 7. 최종 결론

이번 수정은 **gate calibration 복구 작업으로는 성공**했다. 하지만 전략 전체 관점에서는 **ML 거래 0건 문제가 아직 해결되지 않았다**.

현재 상태를 정확히 표현하면 다음과 같다.

1. gate는 이제 현실적인 분포를 출력한다.
2. edge는 여전히 모든 신호를 음수 기대값으로 본다.
3. 따라서 selector가 전부 탈락시키는 것은 정상 방어 동작이다.

다음 작업은 threshold 완화가 아니라 **rule alpha 품질 감사**여야 한다.
