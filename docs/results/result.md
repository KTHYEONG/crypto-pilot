## 2026-07-21 — L2 portfolio handoff gate fix, production remeasurement

### 실행 조건

- 명령: `L2_OPTUNA_TRIALS=120 uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-21`
- `portfolio_handoff.py::evaluate_portfolio_handoff` 구조적 결함 5건 수정 후 재측정 (top-32 sleeve cap 강제, Kelly-proportional admission weighting, bootstrap LCB 퇴화 제거, family-diversity blanket-kill soft화, dead config 제거) — `[ADR_20260721_L2_PORTFOLIO_HANDOFF_GATE_FIX]`
- `evaluate_portfolio_handoff` 반환값을 가로채는 계측 스크립트로 fold별 rejection reason / LCB 분포 확인 (scratch, 저장소 미포함)

### 결과

| 단계 | 결과 |
|---|---:|
| L1 admitted symbols | 126 / 137 |
| L1 layer audit entries | 92,419 |
| Crisis aligned symbols / bars | 53 / 903 |
| Crisis matched pairs / events | 134 / 88,218 |
| Causal handoff | **blocked: `all_folds_blocked`** (수정 전과 표면 결과 동일) |
| Candidate sleeve pool (per fold) | 58 |
| Capped by `max_candidate_sleeves` | 26 / 58 (cap 정상 작동 확인) |
| Admitted sleeves (per fold) | 0 / 4 folds |
| Fold LCB 평균 범위 | −0.0063 ~ −0.0019 (fold별) |
| Fold 내 LCB 양수 sleeve 수 | 2 ~ 13 / 32 (일관성 부족으로 `low_positive_window_ratio` 추가 탈락) |
| L2 Optuna trials | 0 / 120 |
| L2 champion / L3 | not reached |

### 해석

게이트 자체의 결함(5건)은 계측으로 정상 동작 확인됨: `max_candidate_sleeves` cap이 58개 후보 중 26개를 실제로 절삭하고(수정 전에는 dead였음), Kelly 재가중으로 음수-평균 sleeve는 marginal delta≈0으로 정당하게 배제되며, `insufficient_family_diversity` blanket-wipe는 한 번도 발동하지 않았다(admitted=0이라 도달 자체가 없었음).

그럼에도 `all_folds_blocked`가 유지되는 이유는 게이트가 아니라 데이터: cap 통과 32개 sleeve 중 일부는 LCB>0이지만(fold별 2~13개) 3-subwindow 전반의 방향 일관성(`positive_window_ratio ≥ 2/3`)을 충족하지 못해 노이즈성으로 판정된다. 즉 L1 신호가 fit/cal 구간에서 통계적으로 견고한 순양(+) marginal growth를 보이지 못하는 것이 근본 원인이며, 이는 기존 세션들의 결론(Rank IC=0.000, mu≪breakeven)과 정합한다. **L2를 통과시키는 유일한 경로는 게이트 추가 완화가 아니라 L1 알파 재설계.**

### 검증

- `lean_check.py`: **PASS**
- Spec compliance: **PASS**
- Coverage: **86%**
