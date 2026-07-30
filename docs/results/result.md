## Terminal futures risk projection — production replay comparison

### 1. 실행 조건 및 검증

| 항목 | 기존 baseline | 신규 replay |
|---|---|---|
| 실행 디렉터리 | `logs/futures/compound/20260730_025740/` | `logs/futures/compound/20260730_041009/` |
| reference date / seed | `2026-07-15 / 42` | `2026-07-15 / 42` |
| data manifest | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` | 동일 |
| 입력 | 5,442 bars × 51 symbols | 동일 |
| 실행 | `L2_DRY_RUN=1`, local sync, full phase | 동일 |
| integrity / dry-run | `true / true` | `true / true` |
| L2 / L3 | `fail / reject` | `fail / reject` |

신규 결과는 terminal projection과 NAV wiring, `max_name_weight_p95` 명칭 교정이 반영된
동일 데이터 재실행이다. 신호 공식과 성장 게이트는 변경하지 않았다.

검증 결과:

- `lean_check.py --spec-only`: **PASS**
- 관련 allocator/engine/validation/runner 테스트: **전부 PASS**
- 실행 로그: optional `smart_money_divergence` 필드 경고와 NumPy 상관행렬 warning만 존재

### 2. L2 결과 비교

| Metric | 기존 | 신규 | 변화 | 해석 |
|---|---:|---:|---:|---|
| annualized log growth | 0.213076 | **0.241269** | +0.028193 | 개선 |
| CAGR | 23.75% | **27.29%** | +3.54%p | 점추정만 개선, 기대수익 아님 |
| stressed excess growth LCB90 | -0.022283 | **+0.029067** | +0.051349 | hard floor 통과 |
| excess growth probability | 0.3097 | **0.3290** | +0.0193 | **여전히 0.90 미달** |
| equity multiple | 1.239512 | **1.274635** | +0.035123 | 개선 |
| Sharpe | 1.5911 | **2.3990** | +0.8079 | 개선 |
| Sharpe probability | 0.9250 | **0.9885** | +0.0635 | 통과 |
| deflated Sharpe probability | 0.7019 | **0.9582** | +0.2563 | 통과 |
| maximum drawdown | 7.99% | **3.06%** | -4.93%p | 개선 |
| daily CVaR95 | -1.219% | **-0.734%** | +0.485%p | 개선 |
| annual volatility | 12.95% | **10.18%** | -2.77%p | 개선 |
| annual turnover | 57.45 | **54.87** | -2.59 | 개선 |
| cost drag ratio | 23.27% | **20.60%** | -2.67%p | 운영 목표 20%에 근접 |
| max-name p95 | 0.190617 | **0.096732** | -0.093885 | 10% cap 통과 |

기존에는 5개 L2 실패 사유가 있었지만 신규에는
`excess_growth_probability=0.3290<0.9` 하나만 남았다. 따라서 terminal projection은
위험 캡 우회와 집중도 병목을 실제로 제거했지만, 전략 edge의 확률적 지속성을 증명하지는
못했다.

### 3. L3 및 배포 판단

| Metric | 기존 | 신규 | 변화 |
|---|---:|---:|---:|
| posterior growth probability | 0.2125 | **0.2372** | +0.0247 |
| holdout days | 90 | 90 | 0 |
| maximum drawdown | 0.625% | **0.622%** | -0.003%p |
| daily CVaR95 | -0.04374% | **-0.04346%** | +0.00028%p |
| verdict | reject (`l2_not_pass`) | reject (`l2_not_pass`) | 유지 |

L3 자체 holdout은 소폭 개선됐으나 L2 미통과로 승격되지 않는다. 현재 자동 배포 판단은
**cash-only/shadow 유지**가 맞다.

### 4. 실제 weights 및 cap 검증

`target_weights.npy` 비교:

| Metric | 기존 | 신규 |
|---|---:|---:|
| shape / dtype | `(5442, 51)` / `float32` | 동일 |
| active bars | 2,796 (51.38%) | 2,796 (51.38%) |
| max-name p95 | 0.148043 | **0.096466** |
| name-cap violation rows | 7.387% | **0%** |
| net-cap violation rows | 11.669% | **0%** |
| gross mean | 0.165852 | **0.161476** |
| gross p95 | 0.588286 | **0.556632** |
| max gross | 0.742258 | **0.666667** |
| changed cells | — | 35.24% |
| mean absolute cell delta | — | 0.000611 |

신규 artifact를 `float64`로 읽어 재검산하면 name/net/long/gross cap은 모두 통과한다.
short cap은 저장된 `float32` 반올림으로 50개 행이 `0.30000000287`까지 올라가며,
엄격한 `1e-12` 기준에서는 약 `2.9e-9` 초과로 기록된다. 이는 projection 계산 자체의
실패가 아니라 artifact dtype 정밀도 문제지만, 계약의 불변식까지 완전히 만족하려면 저장
artifact도 `float64` 유지 또는 저장 직전 안전 여유(epsilon) 적용이 필요하다.

### 5. 결론

1. **구조적 효과는 확인됐다.** terminal projection은 순노출·종목·gross 위험 캡의
   우회를 제거했고, max-name p95를 19.06%에서 9.67%로 낮췄다.
2. **위험 대비 성과도 개선됐다.** CAGR 점추정은 27.29%, 변동성 10.18%, MDD 3.06%,
   deflated Sharpe probability 95.82%로 기존보다 좋아졌다.
3. **그러나 배포 승인은 아니다.** excess growth probability 32.90%가 90% 기준에 크게
   못 미쳐 edge 지속성은 검증되지 않았다. CAGR 점추정을 실현 기대치로 사용하지 않는다.
4. **다음 단일 개선 후보**는 신호/레버리지 변경이 아니라 `excess_growth_probability`
   원인 분해다. 동시에 float32 artifact 정밀도 문제를 별도 작은 수정으로 닫아야 한다.

### 6. Artifact

- [신규 result.json](../../logs/futures/compound/20260730_041009/result.json)
- [신규 manifest.json](../../logs/futures/compound/20260730_041009/manifest.json)
- [신규 target_weights.npy](../../logs/futures/compound/20260730_041009/target_weights.npy)
- [기존 result.json](../../logs/futures/compound/20260730_025740/result.json)
- [기존 target_weights.npy](../../logs/futures/compound/20260730_025740/target_weights.npy)
- [replay log](/tmp/terminal_futures_risk_projection_run.log)
