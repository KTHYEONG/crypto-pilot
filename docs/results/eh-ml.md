# ML Evaluation Hardening — 논리 압축

> **목적**: ML 평가 기준 4-Layer 재설계 논리 기록.
> 다른 AI / 채팅 세션에서 논리구조 참조용.

---

## 핵심 아이디어 (Why)

"ML이 동적으로 좋은 event를 골라내는가"를 정직하게 측정하지 못하던 기존 기준 → 허위 양성(False Positive) PROMOTED 발생.
재설계 원칙: **각 평가 계층이 독립적인 정보를 측정해야 함. 중복/잉여 조건은 제거.**

---

## Walk-Forward 구조 (변경 없음)

```
IS (fit+cal) ──────────────────────── OOS
├── fit_fraction=0.60  (LightGBM 학습)
├── cal_fraction=0.20  (calibration_fit / calibration_eval 5:5 split)
└── oos              → fold survival 판정
```

- `wf_scheme=anchored`, `n_folds=4`, `purge_bars=18`, `embargo_bars=18`
- OOS 구간: 2025-10-01 ~ 2026-03-31 (4 fold × ~45일)

---

## 4-Layer 평가 프레임워크

### Layer 0: Signal Pre-Qualification
→ `eh-signal.md` 참조

### Layer 1: Gate — Catastrophic Veto Only

**기존 문제**:
- Base rate ≈ 43% (ATR SL < TP 구조적 비대칭)
- `p_pass` max ≈ 0.496 → gate가 사실상 discriminator value = 0
- `gate disabled` 시 `np.ones()` 반환 → 전원 통과, selection에 기여 없음

**변경**:
- Gate는 항상 활성 (`enabled=True`)
- `p_pass`는 sizing confidence discount에만 사용 (selection score에 곱하지 않음)
- **단 하나의 역할**: `q10_net_bps < -catastrophic_shortfall_bps(300bps)` 이벤트 VETO

**제거된 기준**: `brier_skill`, `decile_lift`, `roc_auc` gate 판정, `selection_min_gate_probability_floor`, `selection_gate_mode`

**이유**: p_pass 0.43~0.46 범위에서 이 지표들은 측정 불가능하거나 의미 없음

---

### Layer 2: Edge Rank IC Gate

**기존 문제**:
- `edge_residual_model_enabled=True` 이면 IC=-0.621 모델도 그대로 배포
- `min_edge_rank_ic=0.01`은 코드 어디서도 판정에 사용되지 않는 dead config

**변경**:
- `calibration_eval` 셋에서 `Spearman(mu_pred, realized_edge)` 실계산
- `rank_ic >= 0.02` → residual model 활성 (`target_mode = "prior_residual"`)
- `rank_ic < 0.02` → prior-only fallback (`target_mode = "direct"`, center_pred = 0)

**수용 기준**:
| 조건 | 임계값 |
|---|---|
| `rank_ic_cal_eval` | `>= 0.02` |
| 최소 cal_eval 관측 수 | `>= 20` |

**로그**: `[EDGE_GATE] rank_ic=X threshold=0.0200 decision=accepted|rejected n=N`

**왜 0.02인가**: IC=0.02는 cross-sectional 순위 예측이 "랜덤보다 낫다"는 최소 증거. IC=0.005~0.01은 noise floor 안에 있음.

---

### Layer 3: Fold Survival — ML Selection Lift

**기존 문제**:
- `log_growth_proxy = mean(log1p(edge * 1e-4)) ≈ mean(edge) / 10000` — mean test와 수치적으로 동일한 잉여 조건
- ML이 무작위 선택보다 나은지 전혀 측정하지 않음

**변경**:
- `log_growth_proxy >= 0.0` 제거
- `ml_selection_lift_bps > 0` 추가:
  ```
  lift = mean(selected_events_edge) - mean(all_fold_oos_events_edge)
  ```
  ML이 fold OOS 전체 이벤트 평균보다 더 좋은 이벤트를 골랐을 때만 통과

**최종 fold survival 조건**:
```python
pass_survival = (
    selected_count >= 20                            # (1) 최소 거래 수
    and realized_mean >= 15.0                       # (2) 경제적 최소선 (RT cost × 2)
    and ml_selection_lift_bps > 0.0                 # (3) ML이 baseline 대비 양의 lift
)
```

---

## 핵심 설계 원칙

| 원칙 | 내용 |
|---|---|
| Fail-Closed | 모델 확신 낮으면 자본 미투입. threshold 완화로 통과 금지 |
| 이중계상 금지 | `p_pass`를 `mu`에 곱하지 않음. gate와 sizing 역할 분리 |
| Single-Unit Contract | label, ML target, sizing 모두 동일 risk-unit(s_i) 기준 통일 |
| Incremental Uplift | 각 컴포넌트가 calibration eval에서 독립적으로 가치 증명해야 활성화 |

---

## Selection Score 계약

**`expected_edge_direct` 모드** (현재 default):
- `selection_score = mu_net_decision_bps` (ML edge 예측값 단독)
- `p_pass`는 sizing confidence discount에만 사용
- `q10_net_bps < -300bps` → catastrophic veto

**이전 `additive_drag` 모드의 결함** (참고용):
- EU = `p_pass * mu - (1-p_pass) * |q10| - turnover`
- 결함1 이중계상: `mu`는 이미 기댓값인데 `p_pass`로 재축소
- 결함2 스케일 불일치: 수십bps 기댓값 - 수백bps MAE proxy → EU 깊은 음수 → eligible=0 교착

---

## 현재 상태 (2026-06-06)

| 계층 | 구현 | 동작 | 결과 |
|---|---|---|---|
| Layer 0 Signal Pre-Qual | ✅ | 26~29% 제거 확인 | 정제 완료 |
| Layer 1 Gate Veto | ✅ | catastrophic veto only | 정상 동작 |
| Layer 2 Edge IC Gate | ✅ | fold1~4 모두 IC < 0.02 → rejected | prior-only 전환 |
| Layer 3 ML Lift | ✅ | fold2 lift=-30.9bps → fail | 정직한 실패 판정 |

**BLOCKED 원인**: Residual model이 IC ≥ 0.02 달성 불가 → prior-only → eu_p90=0.769bps → selection 붕괴

**다음 돌파구**: Feature Engineering
- 진입 시점 momentum strength (entry bar의 signal intensity)
- variant별 최근 rolling IC (최근 n bar에서의 예측력)
- regime-specific feature (BTC drawdown depth, funding extremity 등)
- 목표: calibration_eval에서 rank_ic ≥ 0.02 달성

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `src/domain/futures/strategy/candidate_dataset.py` | Layer 0: signal pre-qual |
| `src/domain/futures/strategy/candidate_gate.py` | Layer 1: catastrophic veto |
| `src/domain/futures/strategy/candidate_edge.py` | Layer 2: rank_ic gate |
| `src/domain/futures/strategy_runtime/bridge.py` | Layer 3: fold survival + ML lift |
| `src/domain/futures/strategy/config.py` | `min_edge_rank_ic=0.02`, `signal_prequalify_min_obs=30` |
| `docs/specs/eval_criteria_hardening.md` | 설계 원문 (ephemeral spec) |
