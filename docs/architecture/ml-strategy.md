# Futures ML Strategy Architecture

> last_verified: 2026-06-07 (continuous market regime overlay provider: vol-target + CUSUM, HMM removed)

## 1. Overview
본 문서는 `my-coin-traider` 프로젝트의 선물(Futures) ML 전략 아키텍처를 기술합니다. 본 아키텍처는 단순 순위 기반(Rank-based) 모델에서 벗어나, 개별 **Candidate Event**를 추출하고 이를 ML로 필터링하여 최종 **Target Weight**를 생성하는 파이프라인을 핵심으로 합니다.

## 2. ML 파이프라인 (Lifecycle)
```text
[Universe Selection] -> [Vectorized Signals] -> [Sparse Event Extraction]
      -> [Triple-Barrier Labeling] -> [Feature Engineering: Identity/Mkt/Symbol]
      -> [Model Training: Shrunk Edge/Downside]
      -> [Inference & Market Regime Context] -> [Cross-Sectional Alpha Selection]
      -> [Top-K Sparsification] -> [Portfolio Sizing]
```

## 3. 핵심 모듈별 상세 역할

### 3.1 Vectorized Signals & Event Extraction (`rule_signals.py`)
- **Vectorized Indicators**: `numba`와 `numpy`를 활용한 고속 벡터화 연산으로 EMA, Rolling Mean/Std, Log Return 등을 심볼별/타임스탬프별로 동시 계산.
- **Rule Signal Panels**: 정의된 전략 로직(Trend, Reversion 등)을 적용하여 밀집(Dense) 시그널 생성.
- **Sparse Event Extraction**: 시그널 문턱값을 넘는 진입 시점만 `Candidate Event`로 추출. 이때 사전에 통계적 유의미성(`KEEP`)이 검증된 전략 변종(Variant)만 필터링하여 노이즈 최소화.

### 3.1a Market Regime Context (`market_regime.py`)
- **Continuous Overlay Provider**: 시장 regime은 고정 상태수 분류가 아니라 `vol_scale * trend_scale` 연속 overlay를 기본 계약으로 사용한다.
- **Volatility Targeting**: BTC log return EWMA로 실현 변동성을 추정하고 `target_vol / realized_vol` 비율을 clip하여 고변동 구간을 자동 de-risk 한다.
- **Trend Scaling**: BTC log price의 자기정규화 `trend_snr`를 `tanh` smooth step으로 변환해 추세 강도에 비례한 scaling을 만든다.
- **Crisis Detection**: standardized BTC return에 two-sided CUSUM을 적용하고, ARL 기반 threshold를 사용해 급격한 구조 변화 시 `crisis_gross_floor`로 즉시 축소한다.
- **No Discrete/HMM Path**: HMM 또는 necessity-gated discrete regime 경로는 현재 시스템 계약에 포함되지 않는다.
- **Quality Gate**: `evaluate_regime_quality()`는 `cal_eval`에서만 persistence, leakage, overlay lift, crisis precision을 평가한다.

### 3.2 Leak-free Triple-Barrier Labeling (`candidate_labels.py`)
- **Dynamic Barriers**: 진입 시점의 Yang-Zhang OHLC volatility proxy를 기준으로 익절(TP), 손절(SL), 시간 제한(Time-Exit) 장벽을 동적으로 설정.
- **Next-Open Exit**: time exit는 `entry_idx + expected_holding_bars`의 open 가격을 사용 (close가 아님). Engine 진입 시점과 동일한 계약을 유지한다.
- **Cost-Aware Net Edge**: `net_event_bps = gross_event_bps - execution_cost_bps - realized_funding_bps - hurdle_bps`. taker 비용 + realized funding 기준.
- **Risk-Unit Normalization**: `z_i = net_event_bps / s_i`, `s_i = max(sl_thr_bps, min_risk_unit_bps)`. center/q10/q90 모델이 동일 stationary target을 학습한다.
- **Barrier Logic**:
  - `net_event_bps`: taker 비용 + funding 차감 후 실현 수익 (회귀 target)
  - `mae_r`: path MAE / s_i (risk-unit 하방 변동성)
  - `mfe_r`: path MFE / s_i (risk-unit 상방 잠재력)
  - `edge_after_hurdle_bps`: `net_event_bps`의 backward-compat alias

### 3.3 Multi-Group Feature Engineering (`candidate_dataset.py`)
학습 데이터셋은 세 가지 핵심 피처 그룹으로 구성됩니다.
- **Signal Pre-Qualification (Layer 0)**: fit split(`is_fit_split=True`)에서만 variant proof를 적용한다. `signal_prequalify_method="mean"`은 legacy 경로로 `mean_edge > 0` 와 `signal_prequalify_min_obs`만 확인하고, `"block_bootstrap"`/`"concurrency_t"`는 overlap-aware uniqueness weight를 사용해 `t-stat >= signal_prequalify_min_tstat`를 추가로 요구한다. 불합격 variant는 `uniqueness_weight=0` 처리되며 OOS inference에서는 global prior fallback을 유지한다.
- **Identity Features**: 전략 패밀리 및 변종 ID를 원-핫 인코딩하여 개별 로직의 고유 특성 반영.
- **Market State**: BTC 수익률, 추세, 전체 시장 변동성 및 분산, 시장 폭(Breadth) 등 거시 국면 정보.
- **Symbol State**: 개별 코인의 변동성 Z-score, 펀딩비 상태, 수익률 랭크 등 자산 고유 상태 정보.
- **Overlay Context**: `entry_idx - 1` 시점의 `overlay_mult`, `crisis_active`, `entry_regime_code`, `entry_regime`를 event context에 주입해 이후 gate/sizing이 동일한 causal snapshot을 사용한다.

### 3.4 Model Training & Calibration (`*_gate.py`, `*_edge.py`)
- **No Gate Classifier**: 별도 LightGBM gate classifier는 제거되었다. catastrophic veto는 `q10_net_bps < -catastrophic_shortfall_bps` 직접 판정으로 처리한다.
- **Downside/Upstream Ratio Diagnostic (`p_pass`)**: `p_pass`는 `clip(q10_return_r / mu_return_r, 0, 1)` 기반 downside/upside ratio로 유지된다. selection hard gate에는 사용하지 않고, sizing 단계에서 soft-discount로 곱해져서 target weight를 조절하거나 diagnostic confidence proxy로 사용한다.
- **Risk-Unit Edge (Regressor)**:
  - **Prior Shrinkage**: calibration-set 가중 평균으로 variant prior `mu_prior_i = E[z_i]` 추정. global prior와 shrinkage 결합.
  - **Honest Edge Gate**: 기본 게이트는 `edge_gate_mode="overlay_lift"`이며 `calibration_eval`에서만 overlay-applied realized lift를 평가한다. `lift_bps > 0`, `overlay_lift_tstat >= edge_gate_min_lift_tstat`, `n_eff >= edge_gate_min_n_eff`를 동시에 만족할 때만 수용한다. legacy `edge_gate_mode="rank_ic"`는 backward-compat 경로로만 유지한다.
  - **Residual Champion Fallback**: `edge_residual_model_enabled=False`가 기본값이다. residual path를 다시 켜더라도 gate 미통과 시 prior-only 또는 disabled로 fail-closed 한다. `EdgeModelValidation`에 `rank_ic_cal_eval`, `overlay_lift_bps`, `overlay_lift_tstat`, `n_eff`, `accepted`, `reason`을 기록한다.
  - `mu_i = mu_prior_i + mu_residual_i` (residual champion pass 시). `expected_net_bps_i = mu_i * s_i`는 표시/검사 전용이며 raw model 출력이 아님.
  - **Multi-Objective**: risk-unit center(z), mae_r, mfe_r 별도 Quantile Regressor 학습.

### 3.5 ML Gate & Dynamic Selection (`candidate_portfolio.py`)
시그널이 포트폴리오에 최종 편입되기까지의 **4단계 동적 필터링** 과정입니다.

1.  **Stage 1: Pointwise filtering (Gate & Utility)**
    - `selection_scope=per_timestamp`: timestamp 내부에서만 후보를 정렬하고 선택.
    - `utility_score`는 production 기준에서 `expected_edge_direct`를 사용.
    - `cost_floor_bps` 및 `selection_max_events_per_bar`는 절대 제약으로만 작동.
    - Gate는 항상 `q10` catastrophic veto로만 작동한다. `p_pass`는 selection hard gate, selection tie-break, `mu` 곱셈에 모두 관여하지 않는다.
2.  **Stage 2: Market Context Input**
    - 시장 국면 공급자는 `market_regime.py`의 연속 overlay/CUSUM 계약이다.
    - discrete regime state 또는 HMM 기반 leverage는 현재 활성 계약이 아니다.
3.  **Stage 3: Cross-Sectional Relative Filter (Alpha Selection)**
    - `CS_Z_SCORE_THRESHOLD`로 시장 평균 대비 독보적 엣지 보유 자산 선별.
4.  **Stage 4: Sizing**
    - **Stop-Risk Budget (기본)**: `abs(w_i) = min(max_symbol_weight, event_risk_budget / (s_i / 10_000))`. 진입 bar 종가 미사용 (look-ahead 제거).
    - **Calibrated Event Kelly (선택)**: `f_bin = clip(kelly_fraction * E[r|score_bin] / max(E[r²|score_bin], floor), 0, max_symbol_weight)`. `E[r]`, `E[r²]`, score-bin 경계는 calibration-fit에서만 추정하며 OOS에서 recalculate하지 않는다.
    - **Continuous Overlay Sizing**: `overlay_sizing_enabled=True`일 때 `signed_w = raw_w * sign(side) * overlay_mult(entry_idx-1)` 를 사용한다. discrete regime multiplier 표는 legacy fallback으로만 남는다.
    - **`p_pass`는 sizing에서 soft-discount로 반영**: sizing 단계에서 `p_pass` 가중치를 직접 곱해 target weight를 조절한다.
    - **Variant Concentration Cap**: timestamp 내부에서만 적용.
    - **Shadow Profiles Are Diagnostic Only**: shadow selection profile은 prediction-side counts(`eligible`, `selected_total`)만 요약하며, OOS realized 결과로 `best_shadow`를 승격하지 않는다.

### 3.6 Universe-to-ML Coupling
- **Metadata Propagation**: 유니버스의 정적 메타데이터(`vol_30d`, `friction_score`)가 ML 학습 피처와 진단 지표로 전달됨.
- **Diagnostics**: Bridge 단계에서 변동성 데실(Decile)별 생존율 등을 로깅하여 모델의 편향성 및 유니버스 적합성 모니터링.

## 4. 설계 원칙 (Design Principles)
- **Point-in-Time Integrity**: entry bar 종가 미사용, next-open exit, calibration-only score-bin moment 추정.
- **Purged Validation Horizon**: `purge_bars`/`embargo_bars`는 `max_holding_bars * purge_safety_mult`에서 자동 유도되며, hold horizon이 길어진 variant가 생겨도 split leakage가 없도록 유지한다.
- **Fail-Closed Selection**: 모델의 확신이 낮거나 시장 리스크가 크면 자본을 투입하지 않음. threshold 완화로 통과 금지.
- **Single-Unit Contract**: label, ML target, sizing이 동일한 risk-unit(s_i) 기준으로 통일. bps/q10/mfe를 혼용하지 않음.
- **Empirical Acceptance Gate**: gate는 `calibration_eval` 전용 overlay lift 판정이 기본이고, Kelly는 calibration-fit moment 추정으로만 활성화한다. residual rank-IC 경로는 legacy fallback이다. 시장 regime 품질 평가는 별도 `cal_eval` 전용 scorecard로 분리한다.
- **Continuous Market Regime**: 시장 regime 로직은 self-calibrating overlay가 기본이며, 손으로 정한 상태수/임계값 또는 HMM 분류를 전제하지 않는다.
- **Fold Survival (ML Lift)**: `realized_selected_edge`(default) 기준 — `selected_count >= min_fold_selected_events(20)` AND `realized_mean >= min_fold_realized_edge_bps(15)` AND `ml_lift_bps > 0` (선택 이벤트 평균 - fold OOS 전체 이벤트 평균). `min_wf_fold_pass_ratio(0.60)` 미달 시 fail-closed. (`realized_log_growth`는 backward-compat non-default 옵션으로만 잔존.)
- **Workflow Status Contract**: bridge는 성공 시에도 `wf_eligible`까지만 출력하고, `deployment_promoted`는 별도 배포 게이트에서만 사용한다.

## 4.1 Causal Ablation Framework (v2)

`run_candidate_ablation()`은 6개의 causal variant로 ML 각 컴포넌트의 증분 가치를 인과적으로 분리한다.

| # | Variant | 설명 |
|---|---|---|
| 1 | `rule_stop_risk` | Rule signal만, stop-risk sizing |
| 2 | `prior_rank_stop_risk` | + variant prior 순위 필터 |
| 3 | `prior_residual_rank_stop_risk` | + ML residual edge 추가 |
| 4 | `edge_plus_validated_gate_stop_risk` | + validated gate veto |
| 5 | `edge_plus_gate_event_kelly` | + calibrated event Kelly (caps bypass) |
| 6 | `full_portfolio_caps` | + 최종 portfolio caps 적용 |

- 모든 variant는 동일 OOS bars, fold boundaries, candidate set, execution contract를 사용.
- `trade_count == len(engine_trades)` (delta weight proxy 미사용).
- `real_edge_bps_p50`: `(pnl - entry_fee) / (entry_price * amount) * 10_000` 의 중앙값.

## 5. 핵심 기술 스택
- **Engine**: `numpy`, `pandas`, `numba` (고속 벡터 연산)
- **ML**: `lightgbm` (Tabular 데이터 최적화), `scikit-learn` (검증 및 보정)
- **Optimization**: `optuna` (전략 파라미터 최적화)
- **Validation**: Purged Walk-forward, Bootstrap, DSR/PBO

---
*참고: 상세 구현 로직은 `src/domain/futures/strategy/` 및 `src/domain/futures/optimization/` 경로의 각 모듈을 참조하십시오.*
