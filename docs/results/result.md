## 2026-07-22 — Kelly↔Equal-Weight shrinkage 도입, production remeasurement

### 실행 조건

- 명령: `LOG_LEVEL=DEBUG L2_OPTUNA_TRIALS=120 uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --date 2026-07-22`
- 선행 조치: `l2_gate.py::_cagr_gate_constraint` / `portfolio_handoff.py::evaluate_portfolio_handoff`에 `[EVAL]` DEBUG 계측 추가(로거 `__name__` → `opt_main_futures` 컨벤션 통일 병행 수정, 기존 침묵 버그 해소) 후 재실측 → `cagr_hybrid`가 `cagr_baseline`(동일 종목·동일 방향의 risk-matched 균등가중)보다 64.2%(77/120) trial에서 낮음을 확정 진단.
- 조치: `diagonal_kelly_weights`에 `kelly_shrink_to_equal ∈ [0,1]` 탐색 파라미터 신설 — shape-space에서 Kelly 비례 벡터와 균등가중 벡터를 블렌딩(`shrink=0.0` 기본값은 기존과 byte-identical). `L2_SEARCH_SPACE`에 탐색 항목 추가.

### 결과

| 지표 | 이전 (Kelly 100%) | 이후 (shrink 탐색 도입) |
|---|---:|---:|
| hybrid < baseline 비율 | 64.2% (77/120) | 60.8% (73/120) |
| uplift(hybrid−baseline) 중앙값 | −1.38%p | −0.26%p |
| uplift(hybrid−baseline) 평균 | −1.13%p | −0.26%p |
| +5%p 요구치 통과 trial | 1/120 | 2/120 |
| `[L2-AUDIT]` cagr 실패 | 120/120 | 120/120 |
| Best CAGR (탐색 중) | 8.79% | 6.68% |
| fold 실패 | 91/120 | 96/120 |
| crisis_cagr 실패 | 62/120 | 49/120 |
| joint_feasible | 0/120 | 0/120 |

### 해석

가설(저-SNR mu에서 Kelly-비례 사이징이 노이즈를 좇아 균등가중보다 열위)의 방향성은 실측으로 재확인됨 — uplift 중앙값이 거의 0으로 이동, hybrid 우위 비율도 소폭 개선. 그러나 `cagr:120/120` 실패는 불변 — 120-trial 예산 내 신규 탐색 차원(`kelly_shrink_to_equal`)이 기존 8개 파라미터와 동시 탐색되어 충분히 발현되지 못한 것으로 판단. Best CAGR 하락(8.79%→6.68%)·fold 실패 소폭 악화(91→96)는 탐색 예산 부족에 따른 국소 최적 미도달 가능성. 효과 크기 격리를 위해 trial 수 확대(300~500) 또는 `kelly_shrink_to_equal` 고정 그리드 A/B 스윕이 다음 조사 후보.

### 검증

- mypy strict: **PASS**
- Spec compliance: **PASS**
- Pytest: **100 passed**
- Coverage (신규 diff 라인 기준): `dataclasses.py` 98%, `l2_gate.py` 85%, `portfolio_handoff.py` 84%
