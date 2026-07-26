## 배포 provenance·탐색 다중성 스펙 구현 및 파이프라인 사상 최초 완주 — 2026-07-26

- 실행일: `2026-07-26`
- 실행 명령: `L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 검증 창: 2024-01-03 ~ 2026-07-01 (워밍업 90일 / L1 365일 / L2 365일 / L3 봉인 홀드아웃 90일)
- 데이터 축: **51개 CORE 완전 이력 심볼 × 5,460개 1시간 봉**
- 데이터 무결성: `true`
- 드라이런: `true` — 봉인 홀드아웃은 소모하지 않음
- 스펙: `docs/specs/deployment-provenance-and-search-multiplicity.md`
- exit_code: **0** (이전 두 차례 실행 모두 크래시로 미완주였던 것과 대비되는 사상 최초 완주)

### 배경

직전 항목(`20260726_114624`)에서 L2 게이트가 실질 마진으로 최초 PASS했으나, 실제 CLI 실행은 두 가지 크래시(`strategy_spec_hash is required`, `fold_manifest_hash is required`)로 미완주였다. 원인 조사 결과 표면적 크래시 아래에 구조적 결함 3건이 추가로 발견됐다.

- **DEF-01 (무결성)**: `engine.py`가 봉인 홀드아웃 `consume()` 호출 시 저장된 manifest 값을 그대로 되먹여, 홀드아웃 재사용 방지 검사가 항상 참(동어반복)이었다.
- **DEF-02 (무결성)**: 설정 탐색 다중성을 회계하는 `CandidateTrialLedger`가 `src/` 어디서도 호출되지 않는 dead code였다. 이전 12변형 그리드서치의 탐색 자유도가 DSR(deflated Sharpe probability)에 전혀 반영되지 않았다.
- **BUG-03 (잠복 크래시)**: `compute_trial_multiplicity`가 trial 1개일 때 NumPy 2.4에서 `ValueError`(corrcoef의 0-d 배열 처리 실패)를 던짐. 재현 확인.

### 설계 결정 (실측 기반)

다중성 과금 방식을 3개 가설로 실험 비교했다(`scratch/verify_*.py`, 프로덕션 `deflated_sharpe_probability` 함수 그대로 호출):

| 가설 | 방식 | 결과 |
|---|---|---|
| B (곱셈형) | `k_signal × K_eff_config(M,ρ)` | ρ 가정 필요, ρ=0.99에서도 DSR 0.000004로 과잉 처벌 → 기각 |
| C (행 추가) | trial 행렬에 config 스트림 row-append | 중복 50개 추가 시 k_eff 9.23→**3.74로 감소** — 탐색할수록 게이트가 쉬워지는 역인센티브 → 기각 |
| **D (가산형, 채택)** | `k_total = k_signal + (k_config−1)`, 참여비 실측 | 단조성(중복 padding에도 k_eff 불변)·하위호환(M=0=현행과 exact match)·비-게임성(고유 5개 > 중복 50개) 4개 속성 전부 통과 |

봉인 홀드아웃 재봉인 정책: 미소진 `spec_hash=''` seal만 1회 자동 backfill 허용, 이미 소진된 seal은 영구 무효(Q3 결정).

### 구현 및 검증 (`/implement` → `/check`)

- 신규: `src/domain/futures/compound/provenance.py` (해시 유도 4함수)
- 수정: `multiplicity.py`(BUG-03 가드 + `charge_config_search_multiplicity`), `contracts.py`(`CandidateTrialLedger.register/load_trial_returns`), `holdout_store.py`(`ensure_sealed`, `consume` 강화), `l1_sleeves.py`(fold_hash 배선), `engine.py`(해시 유도·ledger 배선·과금 순서), `compound_main.py`(전체 배선)
- `/check` 1차 실행에서 스펙 준수 도구 자체의 한계 2건 발견(점 표기 `Class.method` 계약명 미매칭, `contract.json` 키 스키마 불일치로 wiring 검증이 조용히 스킵됨) → `lean_check.py` 보정 후 wiring 8개 수동 재검증
- `/check` 과정에서 **실질 결함 2건 추가 발견 및 수정**:
  1. `trial_ledger.register()`가 `engine.py`에 전혀 호출되지 않고 있었음(다중성 회계의 핵심 축이 죽어있었음) → `candidate_hash`/`risk_policy_hash` 유도 및 자기제외(`exclude_candidate_hash`) 로직과 함께 배선
  2. `universe_state_hash` 사전검증이 DEF-01 동어반복 제거 과정에서 실수로 함께 삭제되어 `test_holdout_rejects_changed_universe_hash` 회귀 → 복원
- 최종 판정: 🟢 PASS — Wiring ✅ | Non-dummy AST ✅ | Mypy Strict ✅ | Regression Test ✅ | Coverage 94%

### 실제 CLI 재실행 결과 (구현 후 최초 완주, `logs/futures/compound/20260726_142427/`)

| 지표 | 값 |
|---|---:|
| verdict | **PASS** |
| absolute CAGR | 31.05% |
| benchmark-relative CAGR | 30.11% |
| Sharpe / probability | 1.352 / 0.9405 |
| deflated Sharpe probability | 1.000000 |
| excess growth LCB90 | +0.0527 |
| stressed excess growth LCB90 | +0.1284 |
| max drawdown | 4.23% |
| annual volatility | 10.06% |
| annual turnover | 6.63x |
| cost drag ratio | 2.74% |
| capacity utilisation p95 | 5.62% |
| integrity | `true` |

`20260726_114624`(수정 전, 크래시로 미완주)와 **완전히 동일한 수치** — 이번 수정이 전략 로직·성과에 어떠한 영향도 주지 않고 배관(provenance·다중성 회계)만 고쳤음을 실측으로 확인했다.

L3 결과: `verdict=shadow`, `reason=dry_run_holdout_not_consumed`. 봉인 홀드아웃 미소진 보존.

### 신규 로직 실측 동작 확인

- **봉인 홀드아웃 1회 backfill**: 로그에 `holdout quarterly-2026-06-30-0048c160d459: backfilling empty spec_hash` 기록. DB 확인 결과 해당 seal은 `strategy_spec_hash`가 빈 값에서 채워졌고 미소진 상태 유지(L2_DRY_RUN=1이라 `consume()` 미호출). 과거 이미 소진된 3개 seal(`multiscale-live`, `-07-24`, `-07-23`)은 예정대로 backfill되지 않고 영구 무효 유지(`[LIMIT-04]`).
- **다중성 ledger 최초 가동**: `data/futures/lake/candidate_trials.db`에 최초 행 1건 등록 확인(`l2_daily_returns` 365일치). 다음 실행부터 이 행이 "이전 탐색"으로 인식되어 다중성 회계에 반영된다. 이번 실행 자체는 사상 최초 등록이라 소급 과금 대상이 없다(`[LIMIT-03]`, 12변형 그리드서치는 ledger 도입 이전이라 소급 불가).

### 최종 판정

- 파이프라인이 사상 최초로 크래시 없이 완주했다. L2 PASS·L3 SHADOW·무결성 정상.
- 봉인 홀드아웃 결속과 탐색 다중성 회계가 실측으로 정상 동작함을 확인했다(더 이상 동어반복 검증·dead ledger가 아님).
- 전략 성과 수치는 수정 전후 완전히 동일 — 이번 작업은 순수 배관 결함 수정이며 알파 로직 변경이 아니다.
- **여전히 실전 매매에 사용하지 않는다.** `L2_DRY_RUN=0`(봉인 홀드아웃 실소비)은 사용자가 별도로 결정할 사안이며, 이번 실행에서 임의로 전환하지 않았다.
- 다음 실행부터는 설정을 바꿔 재탐색할 때마다 다중성이 정직하게 누적 과금되므로, 이후 그리드서치는 이번처럼 무비용이 아니라는 점을 인지해야 한다.
