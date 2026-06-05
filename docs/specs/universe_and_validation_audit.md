# Universe Filter & ML Validation Loop Audit Report

## 1. Universe Filter (Stage 6) Structural Bottlenecks

### A. The Volatility Paradox (변동성 모순)
* **현상**: Stage 6의 다목적 스코어링 시스템에서 `alpha_capacity_score`는 30일 변동성(`vol_30d`)이 큰 자산에 가중치를 부여하여, 변동성이 큰 알트코인을 우선적으로 포트폴리오 유니버스에 선발합니다.
* **수식적 원인**:
  $$\text{alpha\_capacity} = 0.40 \times \text{vol\_norm} + 0.30 \times \text{dispersion\_norm} + 0.30 \times \text{regime\_indep}$$
  $$\text{tradeable\_score} = 0.50 \times \text{friction\_score} + 0.30 \times \text{alpha\_capacity\_score} + 0.20 \times \text{diversification\_score}$$
* **모순점**: 유니버스에는 변동성이 높은 자산을 가득 채우면서, 리스크 차단 필터인 `max_expected_shortfall_bps = 80.0`은 **자산의 고유 변동성과 관계없는 고정 절대값(bps)**으로 둔 데서 병목이 발생합니다.
* **결과**: ATR 1.5~2.5배 Stop을 쓰는 알트코인의 MAE(최대 역행폭)는 구조적으로 -300bps ~ -500bps를 넘나듭니다. 따라서 이 유니버스 하에서 모든 알트코인의 진입 신호는 리스크 필터에 의해 100% 차단(Catastrophic Pessimism)되는 마비 현상이 생깁니다.

---

## 2. ML Training & Validation Loop Logic Faults

### A. 1-Bar Return Feature Leakage (데이터 누수)
* **현상**: `sym_ret_1` 및 `mkt_ret_1` 피처가 T 시점 진입 결정 시점의 가격 움직임을 왜곡 전달하여 Look-ahead Bias(선행 편향)를 유발합니다.
* **수식적 원인**:
  `sym_ret_1[event_t - 1]`은 `close[event_t] / close[event_t - 1] - 1.0` 입니다.
  만약 진입이 T 시점의 시가(`open_2d[entry_idx]`)에 이루어진다면, 종가 기준 1-bar 리턴 피처가 T 시점 시가 진입 직후의 가격 변화율을 일부 선행 학습하거나 즉각 피드백 편향을 일으킬 수 있습니다.
* **결과**: 모델이 학습 시(In-Sample)에는 이 누수 피처를 최대로 활용하여 성능을 부풀리지만(IS mu_mean = 23bps), OOS(Out-of-Sample) 평가 시에는 실제 성능이 극단적으로 붕괴(OOS real_p50 = -63bps ~ -85bps)하게 만듭니다.

### B. Probability Calibration Collapse (확률 보정 붕괴)
* **현상**: LightGBM Classifier of output probability에 Sigmoid (Platt Scaling) 보정을 적용할 때, 클래스 불균형(Pos:Neg ≃ 4:5) 환경에서 예측 확률의 분산이 완전히 사라지는 **확률 붕괴(Probability Collapse)**가 유발됩니다.
* **결과**: `min_gate_probability_std = 0.03` 미달로 인해 soft gate 판정이 완전히 무력화되거나 왜곡됩니다.

### C. Extreme Pessimism of q10 Target (최악 케이스 타겟 편향)
* **현상**: `y_q10` 타겟의 정의가 `y_edge`와 `mae_raw` 중 더 최악의 값만 모사하도록 설계되어 지나친 비관주의를 학습합니다.
  $$y\_q10 = \min(\text{mae\_raw} - \text{cost\_hurdle}, y\_edge)$$
* **결과**: 알트코인은 원래 MAE가 크므로, q10 분위수의 기댓값이 -280bps 이하로 치우쳐 hard shortfall 차단 설정과 맞물리면 포트폴리오의 생존율이 0%로 수렴합니다.

### D. Purging & Embargo 설정의 적정성
* 현재 `purge_bars = 18` 및 `embargo_bars = 18` (4h bars = 72시간) 설정은 평균 보유 기간이 18 bars 이하일 때는 견고합니다.
* 그러나 만약 보유 기간이 20-30 bars를 넘나드는 Variant가 유입된다면, fit fold와 validation fold 간의 시간적 중첩으로 인한 오염 가능성이 상존합니다.

---

## 3. Recommended Actions & Roadmap

### Step 1. Immediate Phase 2 Implementation (단기 조치 완료)
* `exclude_immediate_return_features` 활성화로 즉각적 리턴 피처 배제 (Leakage 방지).
* `selection_shortfall_mode`를 `"penalty_only"`로 전환하여 하드 필터를 패널티 감점 체계로 전환.
* `downside_penalty`를 `1.0`에서 `0.3`으로 현실화.
* `gate_calibration_method`를 `"isotonic"`으로 전환하여 확률 보정 붕괴 해결.

### Step 2. Risk Target Volatility Normalization (중장기 과제)
* 고변동성 알트코인을 포용하기 위해 `max_expected_shortfall_bps = 80.0`과 같은 절대값 기준을 **자산별 14d ATR (bps) 대비 N배** 또는 **변동성 표준화(Volatility Normalized Shortfall)**로 개편할 것을 권장합니다.
* 예: `max_expected_shortfall = 1.5 * ex_ante_vol_bps`
