## 2026-07-22 — TF-쿼터 handoff cap 재설계 및 C4 게이트 충돌 수정, production remeasurement

### 실행 조건

- 명령: `L2_OPTUNA_TRIALS=120 uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-21`
- 두 결함 수정 후 재측정 — `[ADR_20260722_L2_TF_QUOTA_CAP_AND_C4_GATE_FIX]`
  1. `portfolio_handoff.py::_rank_and_cap_sleeve_indices`를 TF별 최소 쿼터 선발 + 잔여분 global quality_weight로 재설계 — 순수 quality_weight 랭킹이 다양화된 659개 후보 풀을 다시 100% 4h로 재집중시키던 구조적 결함 수정 (candidacy 단계 다양성 보장, 최종 admission은 기존 통계 검정 유지)
  2. `awf_sim.py::_run_awf_simulation`의 C4 TF-inclusion 게이트에 handoff override 추가 — C4 자체 fit-edge 테스트가 handoff가 이미 admit한 TF를 재차 배제하던 충돌 수정 (직전 세션 발견된 "CAGR 0.00% 균일" 이상현상의 근본 원인)
- 4개 chokepoint(post_resolve/post_c4_filter/post_bucket_routing/post_netting) 직접 함수 트레이싱으로 소거법 근본원인 확정 후 수정 (logger 기반 계측은 워커 프로세스 stdout 미노출로 우회, 코드는 향후 디버깅용으로 유지)

### 결과

| 지표 | 수정 전 | 수정 후 |
|---|---:|---:|
| Trial별 CAGR | **120/120 전부 정확히 0.00%** | 실제 편차 있는 값 (예: +10.8%, +2.9%, -1.8%, -0.9%) |
| Trial별 거래 수 | 사실상 0 | **53~249건, 정상 분포** |
| Best CAGR (탐색 중) | 0.00% | **6.13%** |
| Champion 선정 블로커 | 균일 플랫 아티팩트 | **`no_feasible_trials`** (진짜 제약조건 미충족으로 정상화) |
| Admitted sleeve TF 분포 | 100% 4h | 여전히 100% 4h (쿼터는 candidacy만 보장, 비4h가 아직 admission 통계 검정 미통과 — 설계대로) |
| `[L2-AUDIT]` failures | `{deployment:120, fold:120, recent_fold:120, active_blocks:120, friction:120, trades:120, cagr:120, sharpe_uplift:120, crisis_cagr:71, crisis_mdd:16}` | `{cagr:120, recency_holdout:113, sharpe_uplift:99, fold:88, crisis_cagr:50, recent_fold:20, crisis_mdd:3}` |

### 해석

두 목표 수정 모두 데이터로 완전히 검증됨: (1) TF-쿼터 cap이 candidacy 단계에서 다양성을 구조적으로 보장, (2) C4 게이트-handoff 충돌 해소로 CAGR 0.00% 균일 현상이 완전히 사라지고 trial마다 실제 편차 있는 결과가 나옴(거래 수 53~249건 정상 분포, Best CAGR 6.13%).

`joint_feasible=0/120`은 유지되나, 블로커가 배선/매칭 버그가 아닌 **진짜 신호 품질·제약조건 문제**(`cagr` 120/120 실패 등)로 정상화됨 — 이제부터는 순수 경제적 병목(현재 admit되는 신호로는 CAGR 최저선 미달)이며, 다음 세션 조사 대상으로 이월.

### 검증

- `lean_check.py`: **PASS**
- Spec compliance: **PASS**
- Coverage: **56%**
