## 2026-07-21 — Causal portfolio handoff production remeasurement

### 실행 조건

- 명령: `L2_OPTUNA_TRIALS=120 uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-05-01`
- 단일 프로세스 실행; multi-seed production 호출 제거 후 deterministic causal-portfolio 경로 사용
- 로그: `/tmp/portfolio_causal_l2_120_fixed.log`

### 결과

| 단계 | 결과 |
|---|---:|
| L1 admitted symbols | 113 / 118 |
| L1 layer audit entries | 117,911 |
| Crisis aligned symbols / bars | 53 / 7,221 |
| Crisis matched pairs / events | 140 / 61,756 |
| Causal handoff | **blocked: `all_folds_blocked`** |
| L2 Optuna trials | 0 / 120 |
| L2 simulated trades | 0 |
| L2 champion / L3 | not reached |

### 해석

L1 신호가 0인 것이 아니다. L1은 통과했지만, L2 탐색 전에 실행되는 fit/cal causal handoff에서 모든 fold가 sleeve를 차단했다. 따라서 fail-closed로 Optuna와 L2 거래 시뮬레이션을 시작하지 않았다. `CRISIS-STRESS events=61,756`는 위기 입력 이벤트 수이며 L2 체결 거래 수가 아니다.

현재 로그는 aggregate blocker만 남기므로 sleeve별 탈락 사유(`low_marginal_growth_lcb`, `low_positive_window_ratio`, `redundant_high_correlation`, `insufficient_family_diversity`)의 분포는 미확정이다. 다음 측정에서는 fold별 admitted 수와 rejection reason counts를 기록해야 한다.

### 검증

- `lean_check.py`: **PASS**
- Spec compliance: **PASS**
- Coverage: **38%**
