# L2 Phase 성과 개선 세션 결과 — 2026-07-19 (Optuna 탐색 재현성 확보)

## 세션 요약

동일 `seed=42`, 동일 코드로 L2 파이프라인을 반복 실행해도 결과가 극단적으로 갈리는(gate-pass 7개 성공 vs 0개 완전 실패) 비재현성 문제의 근본 원인을 규명하고 해결했다. 원인은 병렬 최적화 로직 자체가 아니라, **기동 시점의 살아있는 시스템 RAM 상태**가 Optuna의 ask/tell 구조 자체를 바꿔버리는 숨은 스위치였다.

## 근본 원인

- `active_pipeline.py`의 `_run_tiered_l2_study`가 `study.optimize()` 호출 직전 `psutil.virtual_memory().available`을 읽어, 3.0GB 미만이면 `batch_size`(Optuna ask/tell **청킹 크기**, 기본 6)를 1로 강제 변경했다.
- 이는 단순 성능 파라미터가 아니라 **구조적으로 다른 두 알고리즘**을 무작위(=기동 시점 RAM 상태)로 선택하는 스위치였다:
  - `batch_size=1` → `study.optimize(n_jobs=1)`(Optuna 내장 순차 루프) — 매 trial마다 직전 결과를 즉시 반영.
  - `batch_size=6`(기본) → 수동 배치 루프 — 배치 내 6개 trial을 **전부 `ask()`한 뒤에야** `tell()`을 수행, 배치 내 2~6번째 trial은 같은 배치 앞선 trial들의 실제 결과를 전혀 반영 못 하는 stale한 이력으로 제안됨.
- `TPESampler(multivariate=True, group=True)`는 이력 의존도가 매우 높아 이 staleness가 120 trial 전체로 누적 전파 — 동일 seed에서도 완전히 다른 탐색 궤적을 만들었다.
- `max_workers`(`ProcessPoolExecutor`의 실제 동시 실행 수)는 `future.result()`가 제출 순서로 블로킹 수집되므로 `tell()` 순서에 영향을 주지 않는다 — 즉 "몇 개가 동시에 도는가"(안전하게 RAM 적응 가능)와 "샘플러가 무엇을 언제 보는가"(반드시 고정돼야 함)는 애초에 분리 가능한 별개의 축이었다.

## 해결

"논리적 ask/tell 청킹 크기"(`batch_size`, 고정)와 "물리적 동시 실행 워커 수"(`max_workers`, RAM 적응)를 분리했다. RAM 기반 `batch_size` 강제변경 블록을 삭제하고 `batch_size`는 config 고정값만 사용, 기존 `max_workers`의 RAM 적응(OOM 안전장치)은 100% 보존 — 병렬 처리 속도와 메모리 안전성을 그대로 유지하면서 재현성만 복원하는 최소 변경이다. 조사 중 기존 테스트가 이 RAM 분기를 mock 우회 트릭으로 의존하던 것을 발견해 config 직접 patch 방식으로 마이그레이션했고, 저RAM/고RAM 양쪽에서 `study.tell()` 시퀀스가 완전히 동일함을 직접 검증하는 종단 재현성 테스트를 신규 추가했다.

## 프로덕션 실측 (2026-07-19 기준일, seed=42, 120 trials, 2회 연속 실행)

```
Run 1: [L2-AUDIT] completed=120/120 joint_feasible=0 crisis_measured=120
       failures={'crisis_cagr': 110, 'cagr': 106, 'crisis_mdd': 85, 'sharpe_uplift': 40, 'fold': 28, 'recent_fold': 13, 'mdd': 3}
Run 2: [L2-AUDIT] completed=120/120 joint_feasible=0 crisis_measured=120
       failures={'crisis_cagr': 110, 'cagr': 106, 'crisis_mdd': 85, 'sharpe_uplift': 40, 'fold': 28, 'recent_fold': 13, 'mdd': 3}
```

- 최종 집계 라인(`completed`/`joint_feasible`/`crisis_measured`/`failures` 7개 항목) **바이트 단위 완전 일치**.
- Best CAGR 궤적도 동일한 마일스톤(6.12%→11.20%→12.59%→13.81%→16.40%→19.85%→23.92%→23.94%→25.61%→27.12%)에서 동일하게 갱신됨.
- `Current:` 진행률 라인에 미세한 순서 차이가 있었으나 tqdm 진행바 postfix의 터미널 출력 버퍼링 타이밍 차이일 뿐, trial 결과값/최종 집계에는 영향 없음을 확인.

원본 로그: `/tmp/l2_determinism_run1.log`, `/tmp/l2_determinism_run2.log`.

## Verdict

- **재현성**: ✅ **확보** — 이전에는 동일 seed로 gate-pass 7개(성공) vs 0개(완전 실패)처럼 극단적으로 갈리던 것이, 이번엔 두 실행 모두 정확히 같은 결과로 수렴.
- **⚠️ 결과 자체는 이번 검증 시점 조건에서 `no_feasible_trials`**: 재현성은 확보됐으나, 이전 세션에서 관측된 "gate-pass 7건 성공"은 우연한(더 유리한) RAM 상태에서 batch_size=1(순차 경로)로 실행됐던 결과였을 가능성이 있다 — 즉 이전 "성공"이 재현 가능한 champion이 아니라 실행마다 달라지는 요행이었을 수 있다는 뜻이기도 하다.
- **`/check`**: PASS (Cov 24%, 종단 재현성 회귀테스트 포함).

## 잔여 이슈

1. **[신규] 재현 가능한 기반 위에서 탐색공간/제약 재검토 필요**: 이제 결과가 결정적이므로, 현재 seed=42 조건에서 `no_feasible_trials`가 나오는 것이 진짜 탐색공간의 한계인지(제약이 과도하게 타이트) 판단할 수 있는 신뢰 가능한 기준선이 생겼다 — 다음 세션 최우선 분석 대상.
2. **다중 seed 검증 부재**: 재현성 fix로 "같은 seed → 같은 결과"는 보장되나, "seed 간 결과 분산이 얼마나 큰가"는 여전히 미검증 — 여러 seed로 champion 안정성을 평가할 필요.
3. **joint_feasible 상시 0**: 관측된 모든 실행(재현성 fix 전/후 포함)에서 13개 제약을 동시 만족하는 trial이 한 번도 나온 적 없음 — 근본적으로 탐색공간이나 게이트 임계값 재설계가 필요할 가능성.
4. **`[CRISIS-WINDOW-DETAIL]` 개별 라벨 불일치(경미, 이전 세션부터 이월)**: 상위 집계와 개별 윈도우 라벨 표기가 다른 기존 관측 이슈, 미해결.
