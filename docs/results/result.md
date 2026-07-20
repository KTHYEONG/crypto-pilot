# L0→L2 파이프라인 상세 소요시간/RAM 실측 — 2026-07-20 (다음 세션 최적화 기준선)

## 세션 요약

`--phase l2` 프로덕션 실행(seed=42, 120 trials) 1회를 `LOG_LEVEL=DEBUG`로 전 구간 계측해 universe 로드부터 L2 champion 선정까지 스테이지별 소요시간과 RSS를 기록했다. 목적은 다음 최적화 세션에서 "어디부터 손댈지" 바로 판단할 수 있는 기준선을 남기는 것 — 이번 세션에 완료한 L1 워커 병렬화(`docs/decisions/decisions.md` ADR_20260720_L1_MEMORY_FLOOR_ADAPTIVE_CALIBRATION)는 이미 이 실측치에 반영돼 있다(즉 아래 수치는 "적용 후" 상태).

- 전체 wall time: **353.46s**
- Peak RSS: **12414MB** (성능 예산 `performance.md` RSS<12GB 근접, 여유 매우 적음)
- 원본 로그: 세션 스크래치패드(`new_run.log`, 653줄), 이 세션에서만 생성/휘발됨 — 재현하려면 `logs/futures/optimization/l1_result_cache/*.pkl` 삭제 후 `LOG_LEVEL=DEBUG uv run python src/execution/opt_main_futures.py --phase l2 --trials 2 --sync skip` 재실행.

## 스테이지별 소요시간 (Top-level)

| 스테이지 | 소요시간 | RSS(시작→종료) | 비고 |
|---|---|---|---|
| universe (유니버스 스캔) | 1.8s | 327→419MB | discover 1.58s + validate 0.19s |
| data (OHLCV/펀딩 로드) | ~21.5s | 419→3045MB (+2598MB) | `load_futures_data_maps_for_symbols` 16.2s가 대부분 |
| bridge_post_align (TF 정렬) | **35.25s** | 3120→6269MB (+3149MB) | 7개 TF 그리드 동시 정렬, **RAM 단일 최대 증가 구간** |
| L0 게이트(phase1+phase3+pruning) | ~24.4s | 6038→7274MB | cheap_evidence 7.68s + canonical_gate 4.09s + cross_tf_pruning 10.66s, `n_passed=10/81 n_rejected=69` |
| **L1 nested WF 전체(7개 TF)** | **145.36s** | 7174→7316MB (평탄) | 아래 TF별 breakdown 참조 |
| L2 signal_batch + sim_cache | ~3.5s | 7316→7484MB | |
| **L2 Optuna 배치 루프(120 trials)** | **~75s** | 7484MB (변화없음) | RAM 안정적, 워커 부족 징후 없음(기존 결론 재확인) |
| **L2 champion 선정(select_layer2_champion)** | **35.80s** | 7484→7507MB | replay 후보 재평가 — L1 한 TF 처리량과 맞먹는 단일 병목, **다음 세션 조사 후보** |
| L2 최종 파이프라인(champion 재평가+정리) | 6.39s | 7507→7497MB | |
| (정리 후) | | →6873MB | 파이프라인 종료 시 GC로 RSS 대폭 감소 |

## L1 nested WF: TF별 breakdown (`[LIMIT-07]` 세밀한 TF부터 처리 순서 적용됨)

| TF | n_bars | 전체 소요 | `feature_cache_prime` | evidence(workers) | outer(workers) | wall(evidence+outer) |
|---|---|---|---|---|---|---|
| 1h | 23472 | 37.46s | **23.40s (63%)** | 10.4s (1) | 3.0s (1) | 13.4s |
| 2h | 11736 | 28.20s | **10.70s (38%)** | 13.4s (1) | 3.6s (1) | 17.1s |
| 4h | 6949 | 15.31s | 0.001s | 10.6s (1) | 1.8s (2) | 12.4s |
| 6h | 3912 | 16.08s | 3.39s | 9.6s (2) | 1.5s (2) | 11.1s |
| 8h | 2934 | 15.56s | 2.50s | 9.9s (2) | 1.5s (2) | 11.4s |
| 12h | 1956 | 14.43s | 1.67s | 9.6s (2) | 1.5s (2) | 11.2s |
| 1d | 978 | 7.84s | 0.80s | 5.4s (2) | 1.1s (2) | 6.5s |

- **`feature_cache_prime`가 1h/2h TF에서만 압도적으로 크다**(1h 23.4s, 2h 10.7s = 합계 34.1s, L1 전체 145.36s의 **23.5%**) — bar 수가 가장 많은 두 TF에 집중, 4h 이하는 사실상 무시 가능(캐시 재사용 추정). **다음 세션 1순위 최적화 후보**: 이번 세션(워커 병렬화, ~17% 개선)보다 단일 항목 절대량이 더 크다.
- 워커 캘리브레이션은 설계대로 동작(`docs/specs/l1_adaptive_worker_calibration.md` 참조): TF#1(1h)·TF#2(2h)는 관측치 부족으로 cold-start(workers=1) 유지, TF#3(4h)부터 outer가 2워커로 전환, TF#4(6h)부터는 evidence도 2워커로 전환.

## Peak RSS 위치 특정

- 스테이지별 표시 RSS는 7100~7500MB 수준을 벗어나지 않는데 `peak=12414MB`가 L1→L2 전환 시점부터 계속 표시됨 — **12414MB는 L1 nested WF 처리 도중 순간적으로 스파이크된 값**(스테이지 경계 로그 사이 어딘가, 아마 evidence fold 워커가 fork 직후 피처 계산을 하는 짧은 구간)이며, 정상 스테이지 로그의 조밀도로는 정확한 스파이크 지점을 못 잡음.
- **다음 세션 조사 필요**: `tracemalloc`이나 더 촘촘한(초 단위) RSS 폴링으로 12414MB 스파이크의 정확한 발생 지점을 특정하면, peak 자체를 낮춰 `tree_pss_cap_bytes`(현재 10GiB 고정) 헤드룸을 추가로 확보할 여지가 있음 — `l1_adaptive_worker_calibration.md`의 워커 캘리브레이션과는 별개 레버.

## 다음 세션 최적화 우선순위 (실측 근거 기반)

1. **`l1_nested_feature_cache_prime`(1h/2h TF) — 34.1s, L1 전체의 23.5%.** 이번 세션 워커 병렬화 효과(L1 -16.8%)보다 절대량이 크다. 무엇을 계산하는지, 캐시가 왜 4h 이하에서만 사실상 free인지부터 확인 필요(진짜 캐시 히트인지, 4h 이하가 애초에 계산량이 적은 건지 미확인).
2. **`select_layer2_champion` — 35.80s.** L2 Optuna 배치 루프(120 trials, 75s) 다음으로 큰 단일 블록. Replay 후보 재평가 로직이 병목인지, 워커 부족인지 미측정 — `l1_adaptive_worker_calibration.md` Appendix에서 낮은 우선순위로 미룬 "L2 실측"이 이 지점부터 시작하면 적절.
3. **`bridge_post_align`(TF 정렬) — 35.25s, RAM +3149MB 단일 최대 증가.** 시간보다 RAM 스파이크 원인으로 더 흥미로움 — peak RSS 12414MB의 실제 발생 구간일 가능성.
4. **Peak RSS 정확한 위치 특정.** 현재 12GB 예산에 근접(12414MB)해 있어 위 최적화들을 병렬성 확대 방향으로 밀어붙이기 전에 먼저 확인해야 안전.

---

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
