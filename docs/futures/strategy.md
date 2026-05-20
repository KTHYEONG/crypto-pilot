# Binance Futures 전략 아키텍처 (v1.0 - Edge & Risk Budget)

**최종 업데이트**: 2026-05-20  
**핵심 설계 목적**: 백테스트 엔진과 유니버스 인프라 위에서 복리 자산 증식을 극대화하는 전략 연구 계층을 구축한다. 본 문서는 특정 Alpha mining 방식이나 HMM 구현에 종속되지 않고, 검증 가능한 edge 생성, regime-aware risk budget, sleeve별 성과 분해를 표준 계약으로 정의한다.

---

## 1. 핵심 아키텍처 및 데이터 흐름

전략 계층은 백테스트 엔진에 직접 주문을 내지 않는다. 전략은 오직 **expected edge**, **confidence**, **risk budget**, **diagnostics**를 생성하고, 실제 weight 산출과 execution 검증은 기존 portfolio/backtest layer가 담당한다.

```text
build_strategy_panel(
    market_panel: pd.DataFrame,
    universe_snapshot: UniverseSnapshot,
    as_of: datetime,
    cfg: StrategyConfig,
) -> StrategyPanel
```

* **전략 독립성**: trend, carry, flow, ML model, regime model은 모두 동일한 출력 계약을 따라야 한다.
* **룩어헤드 차단**: 모든 feature는 `t`까지 닫힌 데이터만 사용하고, 체결은 backtest engine의 `t+1` 규칙에 위임한다.
* **복리 최적화 기준**: 단일 CAGR이 아니라 log growth, drawdown, turnover, funding drag, capacity를 함께 본다.

### 4개 프로세스 분리 구조

```text
┌─────────────────────────────────────────────────────────────┐
│ [프로세스 A: Feature & Sleeve Edge]                         │
│ - 4h decision grid 기준 causal feature 생성                 │
│ - trend/reversal/carry/flow/defensive sleeve score 산출     │
│ - cost-adjusted long/short edge 생성                       │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ [프로세스 B: Regime & Risk Budget]                          │
│ - market-wide posterior 생성                                │
│ - gross cap, EV hurdle, sleeve blend, no-trade buffer 조절  │
│ - crisis/tail guard는 exposure 축소에만 우선 적용           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ [프로세스 C: Evaluation Harness]                            │
│ - sleeve별 IC/EV/Cost/turnover/OOS decay 측정               │
│ - regime별 성능 분해 및 capacity impact 측정               │
│ - 후보 edge의 승격/폐기 사유 기록                           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ [프로세스 D: Portfolio Interface]                           │
│ - edge_long/edge_short/confidence/risk_budget만 전달        │
│ - Kelly/cap/quantization/execution은 portfolio layer 담당   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 디렉토리 구조 및 모듈 매핑

초기 구현은 기존 `alpha_factory`, `ml_pipeline/regime`, `portfolio` 코드를 재사용하되, 외부 계약은 HMM이나 특정 mining 구현에 묶지 않는다.

| 파일/모듈 | 역할 | 상태 |
|---|---|---|
| `src/domain/futures/strategy/` | 전략 계층 신규 루트. edge/regime/evaluation 계약의 공개 API | 신규 권장 |
| `strategy/contracts.py` | `StrategyPanel`, `SleeveEdgeFrame`, `RegimePosteriorFrame`, `RiskBudgetFrame` 정의 | P0 |
| `strategy/features.py` | 4h causal feature 생성. 가격, funding, OI, flow, liquidity feature 통합 | P0 |
| `strategy/sleeves.py` | trend/reversal/carry/flow/defensive sleeve score 산출 | P0 |
| `strategy/regime.py` | rule-based 또는 score-based regime posterior provider | P0 |
| `strategy/risk_budget.py` | posterior를 gross/EV hurdle/sleeve blend/no-trade buffer로 변환 | P0 |
| `strategy/evaluator.py` | sleeve별 IC, EV/Cost, turnover, regime별 성능 진단 | P0 |
| `strategy/pipeline.py` | 전체 strategy panel 오케스트레이션 | P0 |
| `alpha_factory/*` | 기존 sleeve/feature 구현. 신규 `strategy/*`로 점진 이전 또는 adapter화 | 재사용 |
| `ml_pipeline/regime/*` | 고급 regime provider 후보. P0 계약 뒤 P1/P2에서 선택적으로 연결 | 선택 |
| `portfolio/signal_composer.py` | legacy composer. 신규 risk budget 계약으로 대체 대상 | 정리 필요 |
| `portfolio/portfolio_constructor.py` | edge를 weight로 변환하는 최종 소비자 | 유지 |

---

## 3. 데이터 계약 및 출력 스키마

### 3.1 StrategyPanel

전략 계층의 최종 산출물이다. backtest/optimizer는 이 객체를 통해서만 전략 출력을 소비한다.

| 필드 | shape/index | 설명 |
|---|---|---|
| `edge_long` | `[B4h, N]` | long 방향 expected edge, simple return per bar 단위 |
| `edge_short` | `[B4h, N]` | short 방향 expected edge, simple return per bar 단위 |
| `confidence` | `[B4h, N]` | edge 신뢰도 `[0, 1]` |
| `turnover_hint` | `[B4h, N]` | 잦은 교체 가능성 또는 signal instability |
| `risk_budget` | `[B4h]` 또는 `[B4h, N]` | gross multiplier, EV hurdle, sleeve blend |
| `regime_posterior` | `[B4h, R]` | market-wide regime probability |
| `diagnostics` | table | sleeve, regime, cost, leakage 검증 결과 |

### 3.2 SleeveEdgeFrame

각 sleeve는 같은 형식의 score를 내야 한다.

| 컬럼 | 의미 |
|---|---|
| `sleeve_name` | `trend`, `reversal`, `carry`, `flow`, `defensive` |
| `raw_score` | 정규화 전 score |
| `normalized_score` | cross-section/time-series 정규화 후 score |
| `cost_adjusted_edge` | 비용 차감 후 edge |
| `confidence` | feature coverage, IC stability, regime compatibility 기반 신뢰도 |
| `turnover_hint` | 예상 회전율 proxy |

### 3.3 RegimePosteriorFrame

Regime은 모델 이름이 아니라 상태 posterior 계약이다.

| 상태 | 의미 | 기본 대응 |
|---|---|---|
| `regime_prob_risk_on_calm` | 상승/저변동, trend sleeve 우호 | gross 유지 |
| `regime_prob_risk_on_volatile` | 상승/고변동, trend 가능하나 tail 위험 증가 | EV hurdle 상향 |
| `regime_prob_risk_off_trend` | 하락 추세, short/defensive 우호 | long suppress |
| `regime_prob_chop_liquidity_thin` | 횡보/얇은 유동성, turnover 비용 위험 | no-trade buffer 확대 |
| `regime_prob_crisis` | 급락/상관 붕괴/청산 위험 | gross 강제 축소 |

### 3.4 RiskBudgetFrame

Regime posterior는 직접 방향성 알파에 곱하지 않고, 우선 risk budget으로 변환한다.

| 필드 | 범위 | 설명 |
|---|---:|---|
| `gross_multiplier` | `[0, 1.2]` | portfolio gross cap 배율 |
| `long_multiplier` | `[0, 1.5]` | long edge 배율 |
| `short_multiplier` | `[0, 1.5]` | short edge 배율 |
| `ev_hurdle_bps` | `>= 0` | 진입 최소 edge |
| `no_trade_buffer_bps` | `>= 0` | 리밸런스 생략 임계값 |
| `sleeve_weights` | sum ~= 1 | regime별 sleeve blend |

---

## 4. 7단계 전략 생성 Funnel

### Stage 0: 입력 정합성 검증 (Readiness)

* `market_panel`은 `(datetime, symbol)` MultiIndex를 사용한다.
* datetime은 UTC, monotonic increasing이어야 한다.
* 모든 feature는 decision bar 기준 closed data에서만 생성한다.
* 결측 feature는 해당 feature의 명시적 neutral value로 처리하고, 무음 drop 금지.

---

### Stage 1: Causal Feature 생성

* **목적**: 모든 sleeve가 공유할 공통 feature panel 생성.
* **기본 feature 그룹**:
  * price: momentum, reversal, range position, realized volatility
  * carry: funding level, funding z-score, funding momentum
  * flow: OI momentum, OI/price divergence, taker imbalance, CVD proxy
  * liquidity: ADV, spread proxy, volume drought, Amihud
  * risk: tail risk, beta, correlation shock, gap proxy
* **금지**: label horizon 이후 수익률, 미래 universe membership, 미래 delisting 지식 사용.

---

### Stage 2: Sleeve Edge 산출

각 sleeve는 독립적으로 long/short edge를 산출한다.

| Sleeve | 목적 | 대표 feature | 기본 방향 |
|---|---|---|---|
| `trend` | 지속성 포착 | 24h~7d momentum, breakout persistence | 추세 추종 |
| `reversal` | 과열/과매도 되돌림 | range position, short horizon reversal | 평균회귀 |
| `carry` | funding/basis 수익 | funding z, funding momentum, basis proxy | carry harvest |
| `flow` | 포지셔닝/수급 불균형 | OI change, taker imbalance, CVD divergence | flow following |
| `defensive` | 위험 회피/품질 필터 | tail risk, liquidity shock, beta shock | exposure 감산 |

초기 P0는 deterministic score 기반으로 구현한다. ML은 P1에서 sleeve weight calibration 또는 score shrinkage에만 사용한다.

---

### Stage 3: Cost Adjustment

* raw score는 반드시 비용 차감 후 edge로 변환한다.
* 최소 비용 항목:
  * taker fee
  * slippage/spread proxy
  * expected turnover cost
  * funding drag 또는 funding benefit
  * liquidity/impact penalty
* `EV/Cost < 1`인 signal은 기본적으로 entry 불가로 본다.

---

### Stage 4: Regime Posterior 생성

P0에서는 rule-based posterior를 표준으로 한다. HMM은 필수가 아니라 provider 중 하나다.

| 버전 | 방법 | 목적 |
|---|---|---|
| P0 | rule/score 기반 posterior | 재현성, 해석 가능성, 빠른 검증 |
| P1 | GMM 또는 Bayesian smoothing | 상태 전환 안정화 |
| P2 | Student-t HMM / nonlinear classifier | tail-aware regime 고도화 |

P0 posterior 입력:
* market return trend
* realized volatility percentile
* cross-sectional dispersion
* BTC/ETH beta shock
* funding stress
* liquidity shock
* drawdown depth and speed

---

### Stage 5: Regime-Aware Risk Budget

Regime은 edge 방향을 직접 결정하기보다 risk budget을 조절한다.

| 상태 | gross | sleeve blend | EV hurdle |
|---|---:|---|---:|
| `risk_on_calm` | 유지/상향 | trend, carry 증가 | 기본 |
| `risk_on_volatile` | 소폭 축소 | trend 유지, defensive 증가 | 상향 |
| `risk_off_trend` | 축소 | short, defensive 증가 | 상향 |
| `chop_liquidity_thin` | 축소 | reversal 제한, no-trade 증가 | 크게 상향 |
| `crisis` | 강제 축소 | defensive 최우선 | 최고 |

설계 원칙:
* posterior entropy가 높으면 gross를 줄인다.
* crisis는 방향 예측이 아니라 exposure control로 처리한다.
* regime별 sleeve weight는 train fold 내부에서만 추정하고 OOS에 고정 적용한다.

---

### Stage 6: StrategyPanel 조립

최종 조립 수식:

```text
edge_long =
    cost_adjusted_sleeve_blend_long
    * confidence
    * long_multiplier

edge_short =
    cost_adjusted_sleeve_blend_short
    * confidence
    * short_multiplier
```

`edge_long`, `edge_short`는 `portfolio_constructor.precompute_rebalance_weights()`의 입력으로 전달된다. 이후 Kelly scaling, vol target, gross/net/beta/per-symbol caps, quantization은 portfolio layer가 처리한다.

---

## 5. 평가 Harness 및 승격 기준

전략 후보는 backtest result만으로 평가하지 않고 sleeve 진단을 먼저 통과해야 한다.

### 5.1 Sleeve 단위 평가

| 지표 | 목적 | 최소 기준 |
|---|---|---|
| `rank_ic_mean` | cross-sectional 예측력 | 양수 |
| `rank_ic_tstat` | 안정성 | `> 1.0` 관찰 기준 |
| `turnover_adjusted_ic` | 비용 반영 edge | 양수 |
| `ev_cost_ratio` | 비용 대비 기대값 | `> 1.0`, 승격 후보는 `> 3.0` |
| `coverage` | feature 가용성 | `> 0.95` |
| `oos_decay` | IS 대비 OOS 열화 | 과도한 붕괴 금지 |

### 5.2 Regime 단위 평가

* 각 regime에서 sleeve별 성능을 별도로 기록한다.
* 특정 regime에서만 수익이 나는 sleeve는 해당 regime weight만 허용한다.
* posterior confidence가 낮은 구간에서 손실이 커지는 전략은 gross multiplier를 낮춘다.

### 5.3 Portfolio 단위 평가

backtest layer의 기존 gate를 그대로 사용한다.

* positive leg ratio
* worst leg TW
* mean leg TW
* DSR
* funding drag
* atomic 6M pass ratio
* intrabar decay
* AUM ladder

---

## 6. 하드웨어 및 연산 설계

대상 개발 환경:

```text
CPU: Intel i5-13600K
GPU: RTX 4070 Ti
RAM: 16GB
```

### 6.1 권장 연산 경계

* 4h decision grid에서 feature/edge 산출.
* 1m 데이터는 execution 검증 전용으로 사용.
* 50~200개 심볼, 수년치 4h panel은 CPU/NumPy/Pandas로 처리 가능.
* Optuna trial은 병렬 CPU 중심으로 운영.
* GPU는 P1 이후 tabular deep model 또는 regime 실험에 제한적으로 사용.

### 6.2 피해야 할 설계

* 전체 1m multi-symbol feature를 GPU에 상시 적재.
* 대형 transformer를 초기 전략 계층에 도입.
* HMM posterior를 edge 방향성에 강하게 곱하는 구조.
* feature 수백 개를 무차별 mining한 뒤 backtest score만으로 선별.

---

## 7. 구현 우선순위

| 우선순위 | 작업 | 산출물 |
|---|---|---|
| P0 | `strategy/contracts.py` 작성 | `StrategyPanel`, `RegimePosteriorFrame`, `RiskBudgetFrame` |
| P0 | deterministic sleeve edge 구현 | trend/reversal/carry/flow/defensive |
| P0 | rule-based regime provider 구현 | canonical 5-state posterior |
| P0 | risk budget mapper 구현 | gross/EV/no-trade/sleeve weights |
| P0 | sleeve evaluation harness 구현 | IC, EV/Cost, turnover, OOS decay report |
| P0 | portfolio 입력 adapter 구현 | `edge_long/edge_short` -> existing optimizer |
| P1 | IC shrinkage 및 fold-aware sleeve weight calibration | 과적합 억제 blend |
| P1 | GMM/Bayesian regime smoothing | posterior 안정화 |
| P1 | feature store/cache | 반복 trial 비용 절감 |
| P2 | Student-t HMM 또는 nonlinear regime classifier | tail-aware regime |
| P2 | lightweight ML edge model | sleeve별 residual edge 보정 |

---

## 8. 품질 검증 가드레일

### 8.1 Leakage Guard

* feature timestamp와 label/return timestamp를 명시적으로 분리한다.
* scaler, rank normalizer, sleeve weight calibration은 train fold에서만 fit한다.
* OOS에서는 fit된 파라미터만 사용한다.

### 8.2 Stability Guard

* NaN/Inf는 neutral 처리 후 diagnostics에 카운트한다.
* posterior row sum은 항상 1.0으로 정규화한다.
* confidence가 낮거나 entropy가 높은 구간은 exposure를 줄인다.

### 8.3 Trading Realism Guard

* edge는 비용 차감 후 값이어야 한다.
* turnover가 높은 sleeve는 EV hurdle을 추가로 요구한다.
* liquidity/capacity penalty는 universe와 중복 계산하지 않고, strategy layer에서는 entry confidence와 turnover hint에 반영한다.

### 8.4 Reproducibility Guard

* 모든 config는 hash 가능해야 한다.
* 모든 stochastic model은 seed를 명시한다.
* strategy panel은 parquet + metadata json으로 저장 가능해야 한다.

---

## 9. 기존 코드 전환 계획

### 9.1 유지할 것

* `alpha_factory/features.py`의 안전한 feature extraction 패턴.
* `alpha_factory/sleeves.py`의 5-sleeve 기본 구조.
* `ml_pipeline/regime/regime_contracts.py`의 canonical regime probability 개념.
* `portfolio/portfolio_constructor.py`의 Kelly/cap/quantization 소비 구조.

### 9.2 정리할 것

* 외부 계약에서 `hmm_prob_*` 직접 의존 제거.
* `signal_composer.py`의 regime 직접 곱셈 로직을 risk budget mapper로 이전.
* alpha mining 결과를 바로 portfolio edge로 쓰는 경로를 sleeve diagnostics 통과 후 사용하도록 변경.

### 9.3 나중으로 미룰 것

* 대형 ML alpha mining.
* HMM/Student-t HMM 고도화.
* transformer 기반 regime classifier.
* live trading 정책 자동 승격.

---

## 10. 적용 대상

* `src/domain/futures/strategy/*` 신규
* `src/domain/futures/alpha_factory/*` adapter 또는 점진 이전
* `src/domain/futures/ml_pipeline/regime/*` optional provider
* `src/domain/futures/portfolio/signal_composer.py`
* `src/domain/futures/portfolio/portfolio_constructor.py`
* `src/domain/futures/optimization/optimizer.py`
* `src/execution/opt_main_futures.py`
