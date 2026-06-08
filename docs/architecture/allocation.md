---
title: Futures Allocation Architecture
domain: futures.strategy
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/candidate_ensemble.py
  - src/domain/futures/strategy/candidate_workflow.py
  - src/domain/futures/strategy/candidate_portfolio.py
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/ablation.py
change_triggers:
  - src/domain/futures/strategy/candidate_ensemble.py
  - src/domain/futures/strategy/candidate_workflow.py
  - src/domain/futures/strategy/candidate_portfolio.py
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/ablation.py
dependencies:
  documents:
    - docs/architecture/signal.md
    - docs/architecture/regime.md
    - docs/architecture/ML.md
last_verified: 2026-06-09
---

# 1. Overview

Allocation 레이어는 L1에서 살아남은 candidate event를 받아 event별 `mu_net_decision_bps`, `q10_net_bps`, `p_pass`를 만들고, 이를 selection 및 Kelly/stop-risk sizing으로 연결합니다. 현재 active backend는 **regime-conditional shrinkage ensemble (`ensemble_b0`)** 입니다.

# 2. Core Components

| Component | Responsibility | File |
|-----------|----------------|------|
| `CandidateStrategyConfig` | `allocation_backend`, `ensemble_shrinkage_k` 및 backend switch 제공 | `config.py` |
| `RegimeConditionalEnsemble` | archetype-regime cell별 shrunk expected edge 보관 | `candidate_ensemble.py` |
| `fit_regime_conditional_ensemble` | train-window event만으로 cell mean/q10 shrinkage 추정 | `candidate_ensemble.py` |
| `predict_regime_conditional_ensemble` | OOS event를 `(archetype, regime_code)` lookup으로 scoring | `candidate_ensemble.py` |
| `_fit_and_predict_single_fold` | fold별 `ensemble_b0` / `ml_edge` backend 분기 | `candidate_workflow.py` |
| `select_candidate_events_for_portfolio` | `CandidateModelOutput` 기준 filtering 및 event selection | `candidate_portfolio.py` |
| `build_candidate_target_weights` | selected events를 Kelly / stop-risk sizing으로 weight 변환 | `candidate_portfolio.py` |
| `run_candidate_ablation` | allocation backend와 reporting 경로의 fallback 일관성 유지 | `ablation.py` |

# 3. Data Flow

```mermaid
graph TD
    A[L1-kept labeled events] --> B[candidate_workflow]
    B --> C[build_candidate_dataset]
    C --> D{allocation_backend}
    D -->|ensemble_b0| E[fit_regime_conditional_ensemble]
    D -->|ml_edge| F[fit_candidate_gate / fit_candidate_edge_models]
    E --> G[predict_regime_conditional_ensemble]
    F --> H[predict_candidate_gate / predict_candidate_edges]
    G --> I[CandidateModelOutput]
    H --> I
    I --> J[select_candidate_events_for_portfolio]
    J --> K[build_candidate_target_weights]
```

# 4. Business Rules & Invariants

- **Active default:** `allocation_backend="ensemble_b0"`가 기본값입니다.
- **No LGBM on active path:** `ensemble_b0` 경로에서는 `fit_candidate_gate`, `fit_candidate_edge_models`, `predict_candidate_gate`, `predict_candidate_edges`를 호출하지 않습니다.
- **Train-window only:** ensemble fitting은 현재 fold의 fit dataset event만 사용합니다. OOS event나 calibration event로 cell mean을 보정하지 않습니다.
- **Regime-conditional shrinkage:** cell 추정치는 `k = ensemble_shrinkage_k`를 사용해 global prior로 수축합니다.
  - `mu_hat(a,g) = (n_ag * mean_ag + k * mean_global) / (n_ag + k)`
- **Conditioning axis (`ensemble_conditioning`):** 기본값 `archetype_regime` — cell key는 `(archetype, regime_code)`이며 **`mu_net_decision_bps`가 `entry_regime_code`로 조건화됩니다**. `regime.md`의 "discrete code는 배분을 직접 구동하지 않음" 불변식의 **명시적 예외**(2026-06-09 reconcile). 대안: `archetype_only`(regime 제거), `auto`(fold 내부 purged-validation Rank IC로 선택). 현 추세-단일 신호 풀에서는 `archetype_regime`가 우세함이 실측됨(2026-06-09 反證: `archetype_only`/`auto` 적용 시 Fold 2 Rank IC 0.036→-0.102 악화). **재활성화 조건:** 신호 풀에 레짐별 부호 역전(mean-reversion ↔ momentum) 신호가 추가되어 C3 flip=Y / C4 rho>0 이 되면 `auto` 재평가 권장.
- **mu-quality shrinkage (`mu_quality_shrinkage_enabled`, 기본 False):** 활성 시 `final_mu = λ·mu_pred + (1−λ)·CS_mean`, `λ = clip(val_rank_ic / mu_quality_ic_full_scale, 0, 1)`. 추세-단일 풀에서는 selection을 평탄화해 왜곡 → 비활성 유지.
- **Two-level sparse-cell fallback:** unseen `(archetype, regime_code)`는 먼저 `archetype_mu_bps[archetype]`로, archetype도 없으면 `global_mu_bps`로 fallback합니다(q10 동일). `archetype_only` 모드는 cell을 건너뛰고 archetype→global만 사용합니다.
- **Schema stability:** backend가 달라도 downstream은 동일한 `CandidateModelOutput`만 소비합니다.
- **Selection unchanged:** B0는 `mu_net_decision_bps`, `q10_net_bps`, `p_pass` 생산자만 교체합니다. sizing/objective/AWF logic은 바꾸지 않습니다. Portfolio Selection 시 `expected_utility` 산출은 설정된 `downside_penalty`를 꼬리 위험(`q10_net_bps`)에 차감 반영하여 작동합니다.
- **Reporting parity:** ablation/reporting 경로는 challenger model이 비활성(예: `ensemble_b0` 모드)일 때도 active allocation backend인 B0 앙상블 모델의 예측값(`predict_regime_conditional_ensemble`)으로 우회하여 백테스트를 진행합니다.
- **Walk-Forward Thresholds:** 캐리 및 평균회귀 전략의 빈번한 복리 엣지 기회를 정상 반영하기 위해 Walk-Forward Fold 생존을 위한 실질 실현 엣지 하한(`min_fold_realized_edge_bps`)은 8.0 bps(가중 평균 RT 비용 7.5 bps의 약 1.07배)로 운용됩니다.

# 5. Data Schemas

### `RegimeConditionalEnsemble`

- `cell_mu_bps: dict[tuple[str, int], float]`
- `cell_q10_bps: dict[tuple[str, int], float]`
- `global_mu_bps: float`
- `global_q10_bps: float`
- `conditioning: str` — 실제 적용된 축 (`archetype_regime` | `archetype_only`)
- `archetype_mu_bps: dict[str, float]`, `archetype_q10_bps: dict[str, float]` — fallback 및 archetype_only용
- `validation_rank_ic: float` — in-fold purged validation Rank IC (auto 선택 및 shrinkage λ 산출용)

### `CandidateModelOutput`

- `events: pd.DataFrame`
- `p_pass: NDArray[float64]`
- `expected_net_bps`
- `q10_net_bps`, `q90_net_bps`
- `selection_score`
- `validation_diagnostics["allocation_backend"]`

# 6. Testing Expectations

- small cell은 global prior 쪽으로 shrink 되어야 합니다.
- lookup predict는 seen cell과 unseen cell fallback을 모두 정확히 처리해야 합니다.
- workflow test는 `ensemble_b0`에서 LGBM 호출이 없음을 증명해야 합니다.
- `phase=ml` 실행은 B0 active path와 ablation/reporting fallback까지 포함해 종료되어야 합니다.
- `ensemble_b0` 배분 모델의 평가는 OOS 예측값(`expected_net_bps`)과 실제 OOS 실현값 간의 Spearman Rank Correlation(Rank IC)을 계산하여 검증합니다.
