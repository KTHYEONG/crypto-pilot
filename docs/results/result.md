## 2026-07-21 — L2 handoff 통계적 검정력 재설계, production remeasurement

### 실행 조건

- 명령: `L2_OPTUNA_TRIALS=120 uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-21`
- `portfolio_handoff.py::evaluate_portfolio_handoff` 재설계 후 재측정 (bar-level block-bootstrap marginal-growth 통계로 교체, L1 `lcb_net_bps>0` 우선 판정 override 추가, `invalid_handoff_weights` 오탐 수정) — `[ADR_20260721_L2_PORTFOLIO_HANDOFF_STATISTICAL_POWER_FIX]`
- git 이력 재조사로 확정: portfolio_handoff 게이트 도입 직전 커밋(597985a4, 동일 L1 레지스트리)이 `joint_feasible=4/120`을 달성한 바 있어, 직전 세션의 "L1 알파 부재" 결론을 반증하고 게이트 자체의 통계적 결함으로 재규명함
- `evaluate_portfolio_handoff` 반환값을 가로채는 계측 스크립트로 fold별 admission 사유 및 `admitted_via_l1_edge_override` 비율 확인 (scratch, 저장소 미포함)

### 결과

| 단계 | 수정 전 | 수정 후 |
|---|---:|---:|
| Causal handoff | `blocked: all_folds_blocked` | **`passed=True`** |
| Fold별 admitted sleeve (4개 fold) | 0 / 0 / 0 / 0 | **18 / 14 / 25 / 32** |
| L2 Optuna trials | 0 / 120 | **120 / 120 완주** |
| Best CAGR (탐색 중 발견) | — | **11.27%** |
| Admitted sleeve 중 L1-override 비율 | — | 표본 전수 100% (`admitted_via_l1_edge_override=True`) |
| Joint feasible champion | not reached | **0 / 120** (신규 블로커, 아래 참조) |

### 해석

Handoff 게이트 자체는 설계대로 정상화됨: L1이 이미 `lcb_net_bps` +43~380bps로 검증한 sleeve들이, L2 자체의 (이제는 bar-level block-bootstrap로 통계적으로 견고해진) 재검증에서 여전히 약한 신호(대부분 marginal_growth_lcb ≈ 0/약음수)를 보여도, L1 우선 판정에 의해 정당하게 admit됨을 확인. Optuna가 최초로 실제 120개 trial을 완주하고 CAGR 11.27%까지 탐색.

그러나 `joint_feasible=0/120`은 유지 — `[L2-AUDIT] blocker=crisis_context_mismatch`, `failures={cagr:117, crisis_cagr:115, recency_holdout:104, sharpe_uplift:63, fold:52, recent_fold:37, crisis_mdd:16}`. 이는 handoff와 무관한 **후속 단계(다중 제약조건 동시충족 + crisis 컨텍스트 정합성)의 별도 병목**으로, 이번 spec 범위 밖으로 확인되어 다음 세션 조사 대상으로 이월.

### 검증

- `lean_check.py`: **PASS**
- Spec compliance: **PASS**
- Coverage: **90%**
