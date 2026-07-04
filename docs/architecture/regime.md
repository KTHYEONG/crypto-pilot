---
title: Market Regime Architecture
domain: futures.strategy
type: architecture
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/market_regime.py
  - src/domain/futures/strategy/regime_evaluation.py
  - src/domain/futures/allocation/replay.py
  - src/execution/opt_main_futures.py
  - src/application/futures/runner/pipeline.py
change_triggers:
  - src/domain/futures/strategy/market_regime.py
  - src/domain/futures/strategy/regime_evaluation.py
  - src/execution/opt_main_futures.py
last_verified: 2026-06-30
---

# 1. Purpose
BTC의 가격 흐름을 기반으로 포트폴리오의 익스포저를 실시간으로 제어하기 위한 연속 리스크 조절값(Continuous Overlay)과 L2 라우팅용 압축 3-state regime(`bull`, `bear`, `crisis`)을 도출한다. 기존 6-state는 내부 진단 데이터로만 활용된다.

# 2. Core Logic & Math

### Volatility Targeting
- **실현 변동성 측정**: $\hat{\sigma}_{t} = \sqrt{\text{EWMA}[ (r - \bar{r})^2 ]_{t} \cdot \text{bars\_per\_year}}$
- **변동성 조절 비율**: $\text{vol\_scale}_{t} = \text{clip}\left(\frac{\sigma_{\text{target}}}{\hat{\sigma}_{t}}, \text{min\_vol\_scale}, \text{max\_vol\_scale}\right)$

### Trend SNR Gate
- **Trend 신호대잡음비**: $s_{t} = \ln(P_{t}) - \text{EMA}(\ln P)_{t} \quad \implies \quad \text{snr}_{t} = \frac{s_{t}}{\text{std}(s)_{t}}$
- **추세 익스포저 비율**: $\text{trend\_scale}_{t} = \frac{1}{2} (1 + \tanh(\text{snr}_{t}))$

### Trend Efficiency (Kaufman ER)
- **추세 효율성 지표**: $ER_{t} = \frac{|P_{t} - P_{t-\text{window}}|}{\Sigma_{i=t-\text{window}+1}^{t} |P_{i} - P_{i-1}|}$
- L2 단계의 whipsaw 진단 및 추세 신호 필터링으로 유입.

### Reversal Risk-Off Detector
- **Drawdown 및 Momentum 측정**:
  - $DD_{t} = 1 - \frac{P_{t}}{\max(P_{t-\text{window}+1 : t})}$
  - $Mom_{t} = \text{EMA}(P, \text{fast})_{t} - \text{EMA}(P, \text{slow})_{t}$
- **동적 상태 전이**:
  - $risk\_off\_raw_{t} = (DD_{t} \ge \text{dd\_threshold}) \land (Mom_{t} < 0)$
  - $persistence\_bars$ 만큼 연속 충족 시 Active 전이.
  - $recovery\_cooldown\_bars$ 만큼 연속 해제 시 Release 전이 (Exit Hysteresis).
  - $risk\_off\_1d_{t} = shift(\text{state}, 1)$로 1시점 밀어 인과성을 확보하여 비중 조절에 인입.

### Market State Panel (Breadth-Augmented)
- **교차 섹션 하락 확산세**:
  - $\text{breadth}_{t} = \frac{\sum_i (r_{t,i} < 0)}{\sum_i valid_{t,i}}$
- **Schmitt Trigger 적용**:
  - $\text{breadth}_{t} \ge enter \quad \implies \quad b\_on = True$
  - $\text{breadth}_{t} < exit \quad \implies \quad b\_on = False$
- **결합 로직**: $raw\_on_{t} = btc\_off_{t} \land breadth\_on_{t}$ (BTC 하락세와 심볼 하락 확산세의 논리곱 교차 검증)

### Page-CUSUM Crisis Detector
- **이상치 감지 알고리즘**:
  - $z_{t} = \frac{r_{t} - \text{median}_{\leq t}}{1.4826 \cdot \text{MAD}_{\leq t}}$
  - $S^{+}_{t} = \max(0, S^{+}_{t-1} + z_{t} - k), \quad S^{-}_{t} = \max(0, S^{-}_{t-1} - z_{t} - k)$
  - $S^{+}_{t} > h$ 혹은 $S^{-}_{t} > h$ 발생 시 위기(`crisis_active`) 돌입.

### Overlay Compositor
- **최종 익스포저 배수**: $\text{overlay\_mult}_{t} = \text{vol\_scale}_{t} \cdot \text{trend\_scale}_{t}$
- 위기 상태 돌입 시 $\text{overlay\_mult}_{t} = \text{crisis\_gross\_floor}$ 로 상한 강제 차단.

# 3. Architecture Flow

```mermaid
graph TD
    A[BTC Close] --> B[Log Returns]
    B --> C[EWMA Volatility]
    C --> D[vol_scale]
    A --> E[Trend SNR]
    E --> F[trend_scale]
    B --> G[Robust Z-Score]
    G --> H[Page-CUSUM]
    H --> I[crisis_active]
    D --> J[overlay_mult]
    F --> J
    I -->|Override| J
    D --> K[Discrete Quantizer]
    E --> K
    I --> K
    K --> L[6-State Regime Code]
    J --> M[Portfolio Sizing]
    L --> N[Evaluation / ML Target]
    A --> O[Rolling Max + DD]
    A --> P[EMA Fast / Slow]
    O --> Q[risk_off_raw]
    P --> Q
    Q --> R[shift(1) → risk_off_1d]
    R -->|BTC-only legacy| S[Selective Hard De-Gross]
    S --> M
    subgraph Panel [Market State Panel - breadth_off]
        U[Universe Close 2D] --> V[Log Returns per sym]
        V --> W[Fraction negative by sym]
        W --> X[Breadth Hysteresis]
        X --> Y[breadth_on]
    end
    Q -->|panel mode| Z[AND Gate]
    Y --> Z
    Z --> AA[Persistence]
    AA --> AB[Recovery Hysteresis]
    AB --> AC[shift(1) → risk_off_1d]
    AC --> AD[Selective Hard De-Gross]
    AD --> M
```

# 4. Core Variables & I/O

| Type | Variable | Description |
|---|---|---|
| **Input** | `P_t` | BTC 종가 시계열 |
| **Param** | $\sigma_{\text{target}}$ | 연간 목표 변동성 |
| **Param** | `crisis_gross_floor` | 위기 상황 발생 시 포트폴리오 허용 최소 익스포저 배수 |
| **Param** | $k, h$ | CUSUM 이상치 감지용 drift 및 threshold 파라미터 |
| **Output**| `overlay_mult_1d` | 최종 비중 조절용 연속 익스포저 배수 시계열 |
| **Output**| `code_1d` | 신호 필터링 및 B0 앙상블용 6-state 코드 (0~5) |
| **Output**| `risk_off_1d` | Reversal Kill-Switch 동작 유무 마스크 |
| **L2 Param**| `l2_regime_bull_gross_cap`| Bull 상태의 포트폴리오 최대 총 익스포저 상한 (기본 1.0) |
| **L2 Param**| `l2_regime_bear_gross_cap`| Bear 상태의 포트폴리오 최대 총 익스포저 상한 (기본 0.35) |
| **L2 Param**| `l2_regime_crisis_gross_cap`| Crisis 상태의 포트폴리오 최대 총 익스포저 상한 (기본 0.25) |

# 5. Edge Cases & Handling
- **유동성 고갈 및 플래시 크래시**: 급격한 가격 이탈 시 CUSUM 알고리즘이 순간적으로 반응하여 `vol_scale` 연산을 우회하고 즉시 `crisis_gross_floor` 수준으로 포트폴리오 비중을 차단함.
- **장기 보합에 따른 변동성 붕괴**: 자산 시세가 장기간 정체되어 MAD가 0에 가깝게 하락하는 경우, 분모에 미소값 $\epsilon$을 더하여 분모 영(0) 분기를 방지하고 연산 무결성을 유지함.
