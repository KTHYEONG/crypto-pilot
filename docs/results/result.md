# 2026-07-22 — Causal L2·온라인 정책 혼합 연결 및 실측 결과

## 실행 조건

- 명령:

  ```bash
  UV_CACHE_DIR=/tmp/uv-cache NUMBA_LOG_LEVEL=WARNING LOG_LEVEL=DEBUG \
  PYTHONPATH=. timeout 1800 uv run python src/execution/opt_main_futures.py \
    --phase l2 --sync skip --timeframe 4h --trials 120
  ```

- 실행 로그: `/tmp/l2_latest_spec_execution_v2.log`
- 종료 상태: **FAIL (code 1)**
- 총 실행 시간: **437.32초**
- peak RSS: **13,363MB** (SPEC 예산 12,288MB 초과)

## 변경 내용

- L2 fold를 기존 fallback에서 causal 4-fold로 교체했다.
- L2 각 fold에 실제 policy warm-up 구간을 확보했다.
- OOS 직접 Kelly 비중 계산을 4개 정책 weight matrix와 online allocator로 교체했다.
- 정책별 shadow return을 누적하고, 실현 block growth LCB가 양수일 때만 risk-on을 허용했다.
- 성장 근거가 없거나 risk scale이 안전하지 않으면 exact cash allocation을 사용한다.
- 데이터 구간이 없는 isolated fixture는 fallback fold를 만들지 않고 `insufficient_causal_l2_span`으로 차단한다.

## 데이터·L1 결과

| 항목 | 결과 |
|---|---:|
| L1 대상 심볼 | 126 |
| L1 입력 패널 | 81 |
| L1 bound 패널 | 79 |
| L1 통과 패널 | 10 |
| L1 상태 | PASS |
| 국면 분포 | bull 41.7% / bear 17.1% / crisis 41.1% |

## Causal fold 결과

정상 L2 study 경로에서 기존 `fit_bars=0` 문제가 제거됐다.

| fold | policy fit bars | OOS bars |
|---:|---:|---:|
| 0 | 546 | 409 |
| 1 | 956 | 409 |
| 2 | 1,366 | 409 |
| 3 | 1,776 | 412 |

단, crisis constraint replay에는 아직 구형 1-fold 경로가 남아 있어 해당 보조 경로에서 `fit_bars=0`이 재현됐다. 이는 다음 수정 대상이다.

## 온라인 정책 allocator 결과

동일 L1 신호에 대해 다음 4개 정책을 shadow book으로 평가했다.

- `equal_weight`
- `inverse_vol`
- `kelly`
- `l1_confidence_shrinkage`

전체 120 trial의 반복 실행 로그에서 관측된 allocator 상태는 다음과 같다.

| 상태 | 횟수 |
|---|---:|
| `abstain_cash` | 5,619 |
| `risk_off_cash` | 569 |
| `risk_on` | 122 |

해석: 신호가 있어도 robust growth LCB와 risk constraint를 동시에 만족하지 못하면 투자하지 않았다. 현금 대기는 정상 보호 동작으로 확인됐다.

## L2 성과 및 판정

| 항목 | 결과 |
|---|---:|
| Optuna trial | 120 |
| 최고 trial CAGR | 약 1.55% |
| 일부 trial 최고 관측 CAGR | 약 3.2% |
| joint-feasible trial | 0 |
| 최종 champion | 없음 |
| 최종 blocker | `no_feasible_trials` |

최종 감사 로그:

```text
completed=120/120
joint_feasible=0
failures={
  'fold': 120,
  'absolute_growth': 118,
  'sharpe_abs': 112,
  'deployment': 69,
  'cvar_95': 69,
  'recency_holdout': 69,
  'recent_fold': 43
}
```

## 핵심 진단

### 해결된 문제

- 정상 L2 경로에서 signal window와 policy fit window가 겹친다.
- L2 OOS가 직접 Kelly-only 경로를 사용하지 않는다.
- 성장 근거가 없을 때 1배 risk-on을 강제하지 않고 현금으로 전환한다.
- 4개 정책의 shadow posterior 계산이 실제 실행 경로에서 호출된다.

### 아직 해결되지 않은 문제

1. Production L2가 여전히 120회 Optuna를 실행한다. SPEC 목표인 production trial 0개와 불일치한다.
2. Crisis replay가 구형 1-fold/fallback 경로를 사용한다.
3. 최종 promotion이 아직 legacy fold·Sharpe·trade count 등 hard gate의 영향을 받는다.
4. causal 4-fold와 6개 worker 조합으로 peak RSS가 13.36GB까지 증가했다.

따라서 이번 결과는 **온라인 allocator 연결은 성공했지만 production L2 전체가 최신 SPEC으로 완전히 전환된 결과는 아니다.** 현재 실패 원인은 L1 신호 생성 실패가 아니라, production 입구에 남아 있는 Optuna·crisis legacy replay·legacy promotion gate의 구조적 잔존이다.

## 검증

- `uv run ruff check --fix`: PASS
- targeted regression: PASS
- `lean_check.py --spec docs/specs/causal-alpha-online-growth-engine_contract.json --skip-lint`: **PASS**
- strict mypy: PASS
- spec compliance: PASS
- coverage: **43%**
- coverage missing: `causal_alpha_tape.py:76-78`

## 다음 조치

- production L2에서 Optuna/champion selection을 제거하고 단일 causal replay로 전환
- crisis replay도 동일한 causal fold contract로 통일
- `GrowthSafetyConstraintVector` 5개만 production hard blocker로 사용
- worker 수와 fold replay 메모리를 줄여 RSS 12GB 이하로 제한
