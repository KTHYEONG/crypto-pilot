# [Universe-Logic] 잔여 기술 부채 및 구현 로드맵 (Refined Summary)

## 1. 개요 (Status & Goal)
- **현황**: OOS label leakage, WF/AWF 계약 불일치, Alpha 단계 Hard Gate 등 일부 치명적 이슈는 해결 완료. 아직 Final Evaluation 정합성, Risk/Cost forecast contract는 보완 필요.
- **목표**: 최종 평가(Final Evaluation) 신뢰성 확보, 모델 정교화(Calibrator/Risk/Cost), 파라미터 일관성 달성.

---

## 2. 우선순위 이슈 매트릭스
| ID | 우선순위 | 핵심 이슈 | 영향 및 위험 요소 |
|---|---|---|---|
| **I1** | **P0/P1** | Final Eval Alpha 불일치 | Champion 선택 및 리서치 게이트(IS/HO/OOS) 비교 신뢰도 저하. |
| **I2** | **P1** | Calibrator `sample_weight` 누락 | EV 추정 불안정, 저유동성 샘플의 과도 반영. |
| **I3** | **P1** | Risk Layer (PCA/Factor) 미완성 | BTC Beta exposure 제어 약화 및 요인 분리 부족. |
| **I4** | **P1/P2** | Cost Layer (NLS Forecast) 미진 | 변동성/유동성 급변 시 비용 과소평가, 실용량 왜곡. |
| **I5** | **P2** | `EV_HURDLE_BPS` Fallback 불일치 | Config 누락 시 경로별 Trial Score 재현성 저하. |
| **I6** | **P2** | DD Scaling Path-Inconsistency | Precompute 단계의 정적 DD(0.0)로 인한 리스크 제어 부재. |

---

## 3. 상세 기술 명세 및 액션 아이템

### I1. Final Evaluation Alpha Artifact 일관성 (Critical)
- [x] IS/HO/OOS 모두 `build_strategy_alpha` -> `merge_ml_output` 흐름을 사용하도록 split rebuild 경로를 정리했다.
- [x] split 메타데이터(`strategy_name`, `config_hash`, `selected_horizon`, `model_family`, `cost_source`) 비교 게이트를 추가했다.
- [ ] OOS ensemble 평가와 IS/HO champion 평가의 semantics를 완전히 동일한 기준으로 맞추는 작업은 아직 남아 있다.

### I2. Quantile Calibrator 가중치 반영
- [x] `labels.py`가 생성한 `sample_weight`를 Calibrator 학습 시 `model.fit(sample_weight=...)`로 전달한다.
- [x] validation set에도 `eval_sample_weight`를 전달한다.

### I3. Risk Layer 고도화 (PCA/Beta)
- [x] BTC Beta 컬럼 부재 시 trailing return 기반 causal beta fallback을 계산하도록 했다.
- [ ] `RiskForecast` 컨트랙트 도입은 아직 미적용이다.
- [ ] `factor_exposure_3d`까지 포함한 factor-level risk split은 아직 구현되지 않았다.

### I4. Cost Layer의 동적 예측화
- [ ] `CostForecast` 명시적 도입은 아직 미적용이다.
- [ ] `cost = fee + spread + vol_buffer + k * sigma * sqrt(order/ADV)` 형태의 forecast contract는 아직 분리되지 않았다.

### I5. `EV_HURDLE_BPS` Fallback 단일화
- [x] `default_ev_hurdle_bps()` 공통 헬퍼를 도입해 fallback을 단일화했다.
- [x] `objectives.py`, `ml_context.py`, `samplers.py`, `signal_composer.py`, `ml_builder.py`, `validation/gates.py`의 분산 fallback을 정리했다.

### I6. Path-Aware Drawdown Scaling
- [x] `portfolio_constructor.py`의 static `current_dd` scaling을 제거했다.
- [x] DD scaling은 engine path-dependent overlay만 source of truth로 남겼다.

---

## 4. 최종 권장 수정 순서 (Roadmap)
1. **[Gate]** Final Evaluator Alpha 생성/Merge 절차를 OOS/IS/HO 전체에서 동일 기준으로 완전 일치시키기 (I1).
2. **[Refine]** Calibrator `sample_weight` 주입은 완료됨.
3. **[Standardize]** `EV_HURDLE_BPS` 헬퍼 통일과 BTC Beta Fallback은 완료됨.
4. **[Contract]** `RiskForecast` 및 `CostForecast` 모듈 분리(I3, I4)는 보완 필요.
5. **[Architecture]** Drawdown Scaling의 precompute 경로 제거는 완료됨.

---

## 5. 보완 필요 항목
- [ ] Final Evaluation에서 ensemble OOS와 champion split 비교 기준의 완전한 통일
- [ ] `RiskForecast` / `CostForecast` 전용 contract 분리
- [ ] `factor_exposure_3d` 포함 risk layer 확장
- [ ] Final evaluator split-consistency에 대한 직접 regression test 추가
- [ ] `CostSnapshot`/`resolve_cost_snapshot()`과 동일한 기준을 split metadata 검증에 재사용
