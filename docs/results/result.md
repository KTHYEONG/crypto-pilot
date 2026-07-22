## 2026-07-22 — crisis_rets 배선 복구 및 TF suffix 매칭 정규화, production remeasurement

### 실행 조건

- 명령: `L2_OPTUNA_TRIALS=120 uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-21`
- 두 결함 수정 후 재측정 — `[ADR_20260722_L2_CRISIS_WIRING_AND_TF_SIGNAL_LOSS_FIX]`
  1. `active_pipeline.py::_run_robust_l2_l3_outcome`에 누락됐던 `crisis_rets` 파라미터 배선 복구 (계산은 됐으나 전달이 빠져 매번 `crisis_context_mismatch` 강제 발생하던 버그)
  2. `signal_selection.py::_candidate_output_to_signal_batch`의 registry 매칭에 TF suffix 정규화 추가 — HTF-injection 경로가 붙이는 `_{tf}` suffix와 해당 TF 자체 registry의 무suffix 명명이 불일치해 4h 외 6개 TF 신호가 전량 폐기되던 버그
- `evaluate_portfolio_handoff` 반환값 계측 스크립트로 pre-cap sleeve pool의 TF 분포 확인 (scratch, 저장소 미포함)

### 결과

| 지표 | 수정 전 | 수정 후 |
|---|---:|---:|
| Pre-cap sleeve pool 크기/TF 분포 | 58개 (100% 4h) | **659개** (1h:205 2h:141 4h:58 6h:16 8h:167 12h:69 1d:3) |
| Handoff 사유 | (passed, 별개 이슈였음) | passed, admitted 18/25/21/32 |
| Champion 선정 블로커 | `crisis_context_mismatch` (매번 강제 발생) | **`no_feasible_trials`로 정상화** (crisis pairing 오류 소멸) |
| Admitted sleeve TF 분포 | — | 여전히 100% 4h (quality_weight 상위 32-cap 결과) |
| Best CAGR (120 trial 중) | 11.27% | **0.00% 균일** (120/120 전부 정확히 0%, 신규 이상현상) |
| `[L2-AUDIT]` failures | `{cagr:117, crisis_cagr:115, recency_holdout:104, sharpe_uplift:63, fold:52, recent_fold:37, crisis_mdd:16}` | `{deployment:120, fold:120, recent_fold:120, active_blocks:120, friction:120, trades:120, cagr:120, sharpe_uplift:120, crisis_cagr:71, crisis_mdd:16}` (신규 카테고리 등장) |

### 해석

두 목표 수정 모두 데이터로 검증됨: (1) `crisis_context_mismatch`가 완전히 사라지고 정상적인 `no_feasible_trials` 사유로 전환, (2) pre-cap sleeve pool이 58→659개로 확대되며 6개 TF(1h/2h/6h/8h/12h/1d)의 L1 검증 신호가 처음으로 signal_batch에 도달함을 확인.

다만 admitted sleeve는 여전히 100% 4h(659개 중 quality_weight 상위 32만 살아남는 cap 효과)이고, **예상치 못한 신규 이상현상**이 발생: 120개 trial 전부가 정확히 CAGR 0.00%로 균일 — 직전 실측(Best CAGR 11.27%)과 달리 예외/크래시 로그 없이 조용히 발생. `[L2-AUDIT] failures`에도 이전엔 없던 `deployment`/`active_blocks`/`friction`/`trades` 카테고리가 120/120으로 새로 등장. 원인 미상, 다음 세션 최우선 조사 대상으로 이월.

### 검증

- `lean_check.py`: **PASS**
- Spec compliance: **PASS**
- Coverage: **35%**
