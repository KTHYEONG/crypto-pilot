# Binance Futures ML 전략 아키텍처 (v3.0 - LambdaMART EV Ranker)

**작성일**: 2026-05-23  
**상태**: 신규 전략 전환 설계 확정  
**핵심 설계 목적**: 기존 deterministic sleeve harness를 legacy로 격리하고, LightGBM LambdaMART Ranker + LightGBM Quantile EV Calibrator 기반의 비용 차감 expected edge 전략을 `alpha_long` / `alpha_short` 계약으로 공급한다.

---

## 0. 2026-05-23 결정 사항

- 신규 기본 전략은 `ml_lambdamart_v1`로 정의한다.
- 기존 `src/domain/futures/strategy/` deterministic sleeve 코드는 신규 전략의 본체가 아니다.
- 기존 폴더는 아래 위치로 이전하여 보존한다.

```text
src/domain/futures/strategy/
  -> src/domain/futures/legacy/strategy_sleev/
```

- 새 `src/domain/futures/strategy/` 폴더는 ML 전략 전용 패키지로 재생성한다.
- legacy로 이동된 코드는 직접 import하지 않는다. 재사용 가치가 있는 순수 유틸은 새 strategy 패키지로 승격 또는 복제한다.
- 신규 ML 전략도 기존 runtime/backtest 계약을 유지한다.

```text
strategy builder
  -> alpha_panel[(datetime, symbol), alpha_long, alpha_short]
  -> strategy_runtime.bridge.MLPipelineOutput(alpha_panel=...)
  -> signal_composer
  -> portfolio_constructor
  -> execution_sim
```

---

## 1. 전략 목표와 수학적 목적함수

### 1.1 목적

복리자산증식 극대화는 단순 방향 예측 문제가 아니다. 신규 전략의 최적화 대상은 다음과 같다.

```text
maximize E[log(W_t+1 / W_t)]
subject to:
  - execution cost
  - funding cost
  - turnover
  - drawdown
  - liquidity/capacity
  - universe membership
  - per-symbol and gross exposure cap
```

현재 포트폴리오 레이어는 expected return `mu`를 분산으로 나누는 fractional Kelly 구조를 사용한다. 따라서 ML 모델은 주문이나 weight를 직접 만들지 않고, **비용 차감 후 양의 기대값을 갖는 per-bar expected return alpha**를 공급해야 한다.

### 1.2 핵심 모델 선택

선정 모델:

```text
LightGBM LambdaMART Ranker
+ LightGBM Quantile EV Calibrator
```

역할 분리:

| 구성요소 | 역할 | 산출물 |
|---|---|---|
| LambdaMART Ranker | 같은 timestamp의 cross-section 안에서 long/short 후보의 상대 순위 학습 | `rank_score[t, i]` |
| Quantile EV Calibrator | rank score와 market feature를 비용 차감 return 단위로 보정 | `ev_adj[t, i]` |
| IC Shrinkage | 최근 OOS-like rolling IC로 과신 방지 | `ev_shrunk[t, i]` |
| Alpha Splitter | signed EV를 long/short canonical alpha로 분리 | `alpha_long`, `alpha_short` |

최종 변환:

```text
alpha_long[t, i]  = max(ev_shrunk[t, i], 0.0)
alpha_short[t, i] = max(-ev_shrunk[t, i], 0.0)
```

---

## 2. 전체 데이터 흐름

```text
┌──────────────────────────────────────────────────────────┐
│ [Data Maps]                                               │
│ - OHLCV 1h/4h/1d                                          │
│ - funding_rate / funding_rate_sum                         │
│ - optional: premiumIndex, bookDepth, metrics(OI/LSR)       │
│ - universe membership masks                               │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ [Feature Builder]                                         │
│ - vectorized T x N feature panels                         │
│ - PIT rolling transforms                                  │
│ - cross-sectional robust ranks/z-scores                   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ [Label Builder]                                           │
│ - t signal -> t+1 execution                               │
│ - net forward log return                                  │
│ - cost/funding adjusted rank relevance                    │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ [AWF Trainer]                                             │
│ - chronological train/valid/test fold                     │
│ - group by datetime for LambdaMART                        │
│ - purge by label horizon                                  │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ [Inference + Calibration]                                 │
│ - rank score                                              │
│ - q10/q50/q90 EV forecast                                 │
│ - IC shrinkage and cost-aware clipping                    │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ [alpha_panel]                                             │
│ MultiIndex(datetime, symbol)                              │
│ columns: alpha_long, alpha_short                          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Existing Backtest/Optimization Path                       │
│ signal_composer -> portfolio_constructor -> execution_sim │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 디렉토리 구조 및 모듈 매핑

### 3.1 리팩토링 후 구조

```text
src/domain/futures/
├── legacy/
│   └── strategy_sleev/
│       ├── __init__.py
│       ├── builder.py
│       ├── combine.py
│       ├── config.py
│       ├── diagnostics.py
│       ├── momentum.py
│       ├── normalize.py
│       └── sleeves/
│           ├── base.py
│           ├── xs_reversal.py
│           ├── ts_momentum.py
│           └── carry.py
└── strategy/
    ├── __init__.py
    ├── builder.py
    ├── config.py
    ├── contracts.py
    ├── features.py
    ├── labels.py
    ├── dataset.py
    ├── ranker.py
    ├── calibrator.py
    ├── inference.py
    ├── diagnostics.py
    ├── cache.py
    └── common/
        ├── normalization.py
        ├── alignment.py
        └── validation.py
```

### 3.2 신규 모듈 책임

| 파일 | 책임 | 주요 라이브러리 |
|---|---|---|
| `config.py` | ML 전략 설정, feature/label/model/cv/cache config dataclass | `dataclasses` |
| `contracts.py` | `FeaturePanel`, `LabelPanel`, `FoldSpec`, `StrategyMLArtifacts` 스키마 | `dataclasses`, `numpy`, `pandas` |
| `features.py` | data_maps -> PIT feature panel 생성 | `numpy`, `pandas`, optional `numba` |
| `labels.py` | t+1 체결 기준 net forward return 및 rank relevance 생성 | `numpy`, `pandas` |
| `dataset.py` | T x N x F panel -> LightGBM long matrix 변환, group 배열 생성 | `numpy`, `pandas`, `lightgbm` |
| `ranker.py` | `lightgbm.LGBMRanker(objective="lambdarank")` 학습/예측 | `lightgbm`, `numpy` |
| `calibrator.py` | q10/q50/q90 quantile EV model 학습/예측 | `lightgbm`, `numpy` |
| `inference.py` | fold별 예측을 alpha_panel로 조립 | `numpy`, `pandas` |
| `diagnostics.py` | IC, NDCG, turnover, EV/cost, alpha coverage 진단 | `numpy`, `scipy`, `pandas` |
| `cache.py` | feature/label/model artifact 캐시 및 manifest hash | `pyarrow`, `pandas`, `json`, `hashlib` |
| `common/normalization.py` | robust z-score, rolling rank, winsorization | `numpy` |
| `common/alignment.py` | multi-symbol datetime 정렬, membership mask alignment | `numpy`, `pandas` |
| `common/validation.py` | no-leakage, dtype, monotonic index, finite value 검증 | `numpy`, `pandas` |

### 3.3 재사용 정책

기존 deterministic sleeve 코드 중 다음은 신규 strategy로 승격해서 재사용한다.

| 기존 파일 | 재사용 방식 | 이유 |
|---|---|---|
| `normalize.py::winsorized_cs_zscore` | `strategy/common/normalization.py`로 이동 또는 복제 | cross-sectional robust scaling에 필요 |
| `diagnostics.py::rolling_ic` | `strategy/diagnostics.py`로 이동 또는 복제 | rank score 및 EV shrinkage 진단에 필요 |
| `diagnostics.py::ic_summary` | `strategy/diagnostics.py`로 이동 또는 복제 | rolling IC gate 및 report에 필요 |
| `sleeves/xs_reversal.py` | feature generator의 prior feature로 재구현 | ML feature로 사용 가능 |
| `sleeves/carry.py` | funding feature generator로 재구현 | carry edge feature로 사용 가능 |
| `sleeves/ts_momentum.py` | multi-horizon momentum feature로 일반화 | 단일 sleeve가 아니라 feature group으로 사용 |

중요 규칙:

```text
새 src/domain/futures/strategy/ 는 src/domain/futures/legacy/strategy_sleev/ 를 import하지 않는다.
필요 로직은 새 strategy/common 또는 strategy/features 내부로 복사/승격한다.
```

---

## 4. 데이터 계약

### 4.1 입력 데이터

`data_maps[symbol][tf]`는 최소 다음 컬럼을 가진다.

| 컬럼 | 필수 여부 | 용도 |
|---|---:|---|
| `datetime` | 필수 | PIT 정렬 및 MultiIndex |
| `open`, `high`, `low`, `close` | 필수 | return, volatility, label |
| `volume` | 필수 | liquidity/volume feature |
| `funding_rate_sum` 또는 `funding_rate` | 권장 | carry, funding cost adjustment |
| `universe_active_mask` | 권장 | PIT tradable mask |
| `universe_entry_warm_mask` | 권장 | 신규 편입 warmup gate |
| `entry_block_mask` | 권장 | 학습/추론 sample mask |
| `kill_signal` | 권장 | label 및 execution eligibility |

### 4.2 확장 가능한 Binance 데이터

`docs/futures/binance_data/binance_data.md` 기준 사용 가능한 데이터는 다음과 같이 feature로 연결한다.

| 데이터 | 가용성 | Feature 사용 방식 | P0/P1 |
|---|---|---|---|
| OHLCV | 2019-09 이후, 2020+ Vision 중심 | return, vol, drawdown, ATR, volume rank | P0 |
| fundingRate | 전범위 FAPI, 2020+ Vision monthly | carry, funding persistence, funding z-score | P0 |
| premiumIndex / markPrice | Vision 가용, 연동 일부 비활성 | basis, basis z-score, mark-index dislocation | P1 |
| bookDepth | Vision daily 가용 | half-spread, depth imbalance, execution friction feature | P1 |
| bookTicker | Vision 가용 | best bid/ask spread, microstructure proxy | P1 |
| metrics OI / LSR | 2020-09 이후 Vision daily metrics | crowding, leverage pressure, long/short imbalance | P1 |
| exchangeInfo | 현재 메타 | tick_size, step_size, status, onboardDate | P0/P1 |
| universe ledger meta | PIT snapshot | ADV, execution_cost_bps, beta, cluster_id, listing_age | P0 |

P0는 모든 심볼에 안정적으로 있는 OHLCV + funding + universe meta만 사용한다. P1은 bookDepth, premiumIndex, metrics가 ledger/data_maps에 안정적으로 병합된 뒤 feature gate를 켠다.

---

## 5. Feature Engineering 설계

### 5.1 기본 원칙

- 모든 feature는 시점 `t` bar close까지의 정보만 사용한다.
- 체결 label은 `t+1` open 이후 수익률을 기준으로 만든다.
- cross-sectional transform은 같은 timestamp 내부에서만 수행한다.
- rolling scaler는 train fold에서 fit하고 valid/test에는 transform만 수행한다.
- NaN/Inf는 feature별 명시 정책으로 처리한다. silent drop 금지.
- feature dtype은 학습 직전 `float32`로 변환한다.

### 5.2 Feature Group

| 그룹 | Feature 예시 | 기대 효과 | 과적합 억제 |
|---|---|---|---|
| Return | `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_36`, `ret_skip_6_1` | short-term reversal/momentum 포착 | horizon 수 제한 |
| Volatility | realized vol 6/18/36, downside vol, ATR ratio | Kelly sizing에 유리한 risk state 반영 | log transform, clipping |
| Cross-section | return rank, vol rank, funding rank, volume rank | LambdaMART 목적과 직접 정렬 | timestamp별 robust rank |
| Carry | funding mean 1/3/6, funding sign persistence, funding z-score | futures 특유 carry edge | MAD z-score clipping |
| Liquidity | volume z, dollar volume rank, ADV rank, cost rank | turnover/cost 방어 | universe meta 우선 |
| Market context | BTC/ETH ret, market breadth, median return, dispersion | systemic state 반영 | 전체 시장 summary만 사용 |
| Beta/correlation | BTC beta, rolling corr, cluster_id | crowding/hedging 구조 반영 | rolling window 최소 90 bars |
| Basis/Premium P1 | mark-index basis, basis z, premium momentum | perp dislocation edge | availability mask 필수 |
| OI/LSR P1 | OI change, OI/ADV, top trader LSR z | leverage/crowding squeeze 감지 | 2020-09 이후 fold만 활성 |
| Microstructure P1 | half-spread, depth imbalance, bookTicker spread | cost-aware selection | 결측 시 feature mask 추가 |
| Sleeve prior | xs_reversal score, carry score, ts momentum score | 기존 deterministic edge를 ML prior로 흡수 | raw signal이 아닌 clipped rank |

### 5.3 Feature 수 제한

P0 feature budget:

```text
max_features = 64
required_features ~= 35~45
optional_features <= 20
```

Feature 수를 제한하는 이유:

- crypto futures universe는 cross-section N이 작다.
- 복수 horizon label을 만들면 effective sample이 겹친다.
- feature 수가 과도하면 fold별 regime noise를 학습하기 쉽다.

### 5.4 권장 P0 Feature Set

```text
price_return:
  ret_1, ret_3, ret_6, ret_12, ret_18, ret_36
  rev_3, rev_6, rev_12
  mom_12_skip_1, mom_36_skip_3

volatility:
  rv_6, rv_18, rv_36
  downside_rv_18
  atr_pct_14
  vol_of_vol_36

carry:
  funding_1
  funding_mean_3, funding_mean_6, funding_mean_18
  funding_z_30d
  funding_sign_persistence_6

liquidity:
  volume_z_18
  dollar_volume_rank
  adv_rank
  execution_cost_rank

cross_section:
  cs_rank_ret_6
  cs_rank_ret_18
  cs_rank_rv_18
  cs_rank_funding_6
  cs_rank_volume_18

market:
  btc_ret_6
  btc_rv_18
  market_median_ret_6
  market_dispersion_6
  positive_breadth_6

prior:
  xs_reversal_prior_6
  carry_prior_6
```

---

## 6. Label Engineering 설계

### 6.1 체결 정렬

신호 시점:

```text
feature[t] uses data up to close[t]
entry at open[t+1]
exit at close/open horizon endpoint depending execution contract
```

P0 label은 단순성과 leakage 방지를 위해 `horizon_bars=1`을 기본으로 한다.

P1에서 multi-horizon을 활성화한다.

```text
horizons = [1, 3, 6]
```

multi-horizon 사용 시 purge bars는 최소 `max(horizons)`로 둔다.

### 6.2 Net Return Label

Long label:

```text
long_net_log_ret[t, i]
  = log(exit_price[t+h, i] / entry_price[t+1, i])
    - taker_fee
    - slippage
    - expected_funding_cost
```

Short label:

```text
short_net_log_ret[t, i]
  = log(entry_price[t+1, i] / exit_price[t+h, i])
    - taker_fee
    - slippage
    + expected_funding_receive
```

Signed label:

```text
signed_net_ret[t, i] = long_net_log_ret[t, i]
```

Short alpha는 signed forecast가 음수일 때 분리한다.

### 6.3 LambdaMART Relevance

같은 timestamp의 eligible symbols 안에서 `signed_net_ret`를 rank bucket으로 변환한다.

```text
top 15%      -> relevance 4
top 15~35%   -> relevance 3
middle       -> relevance 2
bottom 15~35%-> relevance 1
bottom 15%   -> relevance 0
```

유니버스 N이 작을 때는 quantile bucket이 불안정하므로 최소 symbol 수를 둔다.

```text
min_group_size = 8
```

`min_group_size` 미만 timestamp는 ranker 학습에서 제외하고, calibrator에는 sample weight를 낮춰 포함할 수 있다.

### 6.4 Sample Weight

sample weight는 복리 목적과 실행 가능성을 반영한다.

```text
weight = active_mask
       * warm_mask
       * liquidity_weight
       * cost_weight
       * volatility_sanity_weight
```

권장:

```text
liquidity_weight = clip(log1p(adv_usdt) / median_log_adv, 0.25, 2.0)
cost_weight = clip(cost_median / execution_cost_bps, 0.25, 2.0)
```

단, 과도한 weighting은 대형 코인 편향을 키우므로 최종 weight는 `[0.25, 2.0]` 범위로 제한한다.

---

## 7. 모델 학습 설정

### 7.1 LambdaMART Ranker

라이브러리:

```text
lightgbm>=4.5.0
```

모델:

```python
lightgbm.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_data_in_leaf=200,
    feature_fraction=0.80,
    bagging_fraction=0.80,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=5.0,
    random_state=seed,
    n_jobs=n_jobs,
)
```

학습 입력:

```text
X_train: float32 [num_samples, num_features]
y_rank: int32 [num_samples]
group: int32 [num_timestamps], sum(group) == num_samples
sample_weight: float32 [num_samples]
```

평가:

```text
valid_metric = NDCG@top_k
top_k = min(5, max(2, group_size // 4))
```

### 7.2 Quantile EV Calibrator

모델:

```python
lightgbm.LGBMRegressor(
    objective="quantile",
    alpha=q,
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=5,
    min_data_in_leaf=250,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=10.0,
    random_state=seed,
    n_jobs=n_jobs,
)
```

Quantiles:

```text
q10 = downside forecast
q50 = median EV forecast
q90 = upside forecast
```

Calibrator feature에는 rank score를 추가한다.

```text
X_calib = [original_features, rank_score, rank_zscore, rank_percentile]
```

보수적 EV:

```text
downside = max(q50 - q10, 0.0)
upside = max(q90 - q50, 0.0)
asymmetry = upside / max(downside, eps)

tail_penalty = clip(downside / vol_forecast, 0.0, 2.0)
ev_adj = q50 - lambda_tail * downside
ev_adj = ev_adj * clip(asymmetry, 0.25, 2.0)
```

초기값:

```text
lambda_tail = 0.25
ev_clip_bps = 75
```

EV clip은 per-bar simple return 기준이다.

```text
ev_adj = clip(ev_adj, -0.0075, 0.0075)
```

### 7.3 Rank Score Calibration

Ranker score는 return 단위가 아니므로 단독 alpha로 사용하지 않는다.

```text
rank_score -> timestamp z-score -> percentile -> quantile EV model -> return units
```

최종 alpha는 `signal_composer`에서 다시 아래 처리를 받는다.

```text
mu_long  = BETA_ALPHA * alpha_long  - friction
mu_short = BETA_ALPHA * alpha_short - friction
```

따라서 ML alpha의 단위는 **simple return per decision bar**여야 한다.

---

## 8. Walk-Forward 학습 구조

### 8.1 Fold 구조

기본 fold:

```text
train_window = 24 months
valid_window = 3 months
test_window = 3 months
step = 3 months
purge_bars = label_horizon_bars
embargo_bars = max(1, label_horizon_bars)
```

학습 순서:

```text
for fold in folds:
    fit feature normalizer on train only
    train LambdaMART on train
    early stop on valid
    train q10/q50/q90 calibrators on train
    calibrate on valid
    infer only on test
    append fold alpha_panel
```

### 8.2 No-Leakage 규칙

- `feature[t]`는 `close[t]`까지만 사용한다.
- `label[t]`는 `open[t+1]` 이후 수익률을 사용한다.
- `rank group`은 동일 timestamp 내부에서만 생성한다.
- feature scaler, imputer, winsor boundary는 train fold에서만 fit한다.
- validation/test timestamp는 학습 group에 포함하지 않는다.
- optional 데이터의 availability mask는 feature로 넣되, 미래 가용 여부를 추론하지 않는다.

---

## 9. 연산 최적화 구조

### 9.1 메모리 레이아웃

Feature 생성 중간 표현:

```text
feature_3d: float32 [T, N, F]
label_2d: float32 [T, N]
mask_2d: bool [T, N]
```

LightGBM 학습 직전 long matrix:

```text
X: float32 [M, F]
y: float32/int32 [M]
group: int32 [G]
index_map: int64 [M, 2]  # time_idx, symbol_idx
```

원칙:

- DataFrame row-wise append 금지.
- per-symbol model 금지.
- feature는 가능한 2D NumPy 벡터 연산으로 생성한다.
- rolling 계산은 pandas groupby보다 NumPy stride/rolling helper 우선 사용한다.
- 병목 rolling 함수는 필요할 때만 `numba.njit(cache=True)`를 적용한다.

### 9.2 Cache

캐시 경로:

```text
data/cache_futures/strategy_ml/
├── features/
│   └── {tf}_{window_hash}_{feature_hash}.parquet
├── labels/
│   └── {tf}_{window_hash}_h{horizon}_{label_hash}.parquet
└── manifest/
    └── {run_id}_strategy_ml_manifest.json

logs/futures/models/strategy_ml/{run_id}/
├── fold_00_ranker.txt
├── fold_00_q10.txt
├── fold_00_q50.txt
├── fold_00_q90.txt
└── fold_metrics.parquet
```

Hash 구성:

```text
window_hash = fetch_start + end_date + symbols + tf
feature_hash = feature_config + source column availability
label_hash = horizon + cost config + execution alignment config
```

### 9.3 병렬화

권장 정책:

```text
feature generation: process/thread pool by feature group or symbol block
LightGBM training: n_jobs = min(4, physical_cores // active_optuna_workers)
Optuna workers: avoid nested oversubscription
```

환경 변수:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

LightGBM 내부 parallelism만 명시적으로 허용한다.

---

## 10. 과적합 억제 장치

### 10.1 모델 수준

- `max_depth <= 6`
- `num_leaves <= 31`
- `min_data_in_leaf >= 200`
- `feature_fraction <= 0.85`
- `bagging_fraction <= 0.85`
- `lambda_l2 >= 5.0`
- early stopping patience 50~100
- feature count P0 64개 이하

### 10.2 금융 시계열 수준

- anchored walk-forward validation
- horizon overlap에 따른 purge
- timestamp group 기준 검증
- OOS IC decay report
- feature importance stability check
- fold별 top feature concentration cap

### 10.3 Alpha 출력 수준

- per-bar alpha clipping
- rolling IC shrinkage
- tail downside penalty
- minimum tradable group size
- membership/warm mask 적용
- cost-aware EV hurdle는 기존 `signal_composer`에서 재적용

---

## 11. 기존 최적화 파이프라인과의 연결

### 11.1 Strategy Runtime

`src/domain/futures/strategy_runtime/bridge.py`는 신규 strategy builder를 호출한다.

```text
run_ml_pipeline_for_universe(...)
  -> strategy.builder.build_strategy_alpha(...)
  -> MLPipelineOutput(alpha_panel=alpha_panel)
```

### 11.2 Optimization

기존 strategy mode trial-time composer는 유지한다.

```text
alpha_long/alpha_short
  -> BETA_ALPHA
  -> friction subtraction
  -> EV_HURDLE_BPS
  -> xs_score_long/xs_score_short
```

추천 search range:

| 파라미터 | P0 범위 | 이유 |
|---|---:|---|
| `BETA_ALPHA` | 1.0 ~ 5.0 | ML alpha는 이미 return 단위로 calibration됨 |
| `EV_HURDLE_BPS` | 2.0 ~ 30.0 | 비용 차감 후 과도한 무거래 방지 |
| `REBALANCE_BARS` | 1 ~ 8 | ranker signal decay 확인 |
| `PORTFOLIO_KAPPA` | 0.15 ~ 0.60 | Kelly aggressiveness 조절 |
| `PORTFOLIO_F_KELLY_MAX` | 0.10 ~ 0.75 | tail risk 방어 |
| `MAX_EXPOSURE_PER_COIN` | 0.10 ~ 0.35 | concentration 방지 |

### 11.3 Portfolio

ML은 target weight를 만들지 않는다. 기존 `portfolio_constructor`가 처리한다.

```text
mu = xs_score
sigma = rolling covariance / sigma estimate
weight = fractional_kelly(mu, sigma)
weight = cap_projection(weight)
```

이 구조를 유지해야 ML 과신이 직접 레버리지 폭주로 연결되지 않는다.

---

## 12. 진단 로깅 태그

신규 태그:

| 태그 | 단계 | 핵심 필드 |
|---|---|---|
| `[ML-FEATURE]` | feature 생성 | rows, symbols, features, nan_ratio, cache_hit |
| `[ML-LABEL]` | label 생성 | horizon, valid_ratio, long_p95_bps, short_p95_bps |
| `[ML-FOLD]` | fold 분할 | fold, train/valid/test bars, purge |
| `[ML-RANKER]` | ranker 학습 | ndcg, rank_ic, best_iter, top_features |
| `[ML-CALIB]` | quantile calibrator | q10/q50/q90 spread, pinball_loss |
| `[ML-ALPHA]` | alpha 생성 | long_nz, short_nz, alpha_p95_bps, ic_lag_mean |
| `[ML-OOS]` | fold OOS 진단 | ic, ev_cost_ratio, turnover_proxy |

기존 태그와 연결:

```text
[ML-ALPHA]
  -> [ALPHA-MERGE]
  -> [ALPHA-ALIGN]
  -> [COMPOSE-DIAG]
  -> [LEG]
  -> [RUN-SUMMARY]
```

---

## 13. 품질 게이트

### 13.1 학습 게이트

| 지표 | Pass 기준 | Hard Fail |
|---|---:|---:|
| train finite feature ratio | >= 0.995 | < 0.990 |
| valid group count | >= 100 | < 50 |
| ranker valid NDCG@K | baseline 이상 | baseline 미만 3 fold 연속 |
| valid Spearman IC | > 0.0 | <= 0.0 3 fold 연속 |
| q50 calibration sign hit | > 0.50 | <= 0.48 |

### 13.2 Backtest 게이트

| 지표 | Pass 기준 | 목적 |
|---|---:|---|
| `positive_leg_ratio` | >= 0.60 | fold 안정성 |
| `ev_cost_ratio` | >= 1.00 | 비용 차감 edge |
| `mean(log_TW)` | deterministic sleeve baseline 이상 | 복리 성장 |
| `worst_leg_log_TW` | >= -0.10 | tail 방어 |
| `turnover_cost_ratio` | <= 0.35 | 비용 폭주 방지 |
| `MDD` | <= 60% | 생존성 |

### 13.3 Baseline 비교

비교 대상:

```text
1. zero-alpha quick-backtest
2. legacy deterministic sleeve strategy
3. ML ranker without quantile calibrator
4. full ML ranker + quantile calibrator
```

Full ML이 2와 3을 동시에 이기지 못하면 배포 후보가 아니다.

---

## 14. 구현 단계

### P0: 최소 ML Alpha Engine

목표:

```text
OHLCV + funding + universe meta 기반 LambdaMART/Quantile strategy
```

작업:

1. 기존 `src/domain/futures/strategy/`를 `src/domain/futures/legacy/strategy_sleev/`로 이동.
2. 새 `src/domain/futures/strategy/` 생성.
3. `common/normalization.py`, `diagnostics.py`에 기존 robust z-score/rolling IC 유틸 승격.
4. `features.py`에 P0 feature set 구현.
5. `labels.py`에 t+1 execution aligned net return label 구현.
6. `ranker.py`에 LightGBM LambdaMART 학습 구현.
7. `calibrator.py`에 q10/q50/q90 EV model 구현.
8. `builder.py`에서 fold inference 후 alpha_panel 반환.
9. `bridge.py`가 신규 builder를 호출하도록 확인.

### P1: Binance 확장 데이터 결합

목표:

```text
premiumIndex, bookDepth, metrics(OI/LSR) feature 활성화
```

작업:

1. data_maps 또는 ledger에 optional source columns 공급.
2. availability mask feature 추가.
3. basis/crowding/microstructure feature group 활성화.
4. optional feature가 없는 구간에서도 동일 schema 유지.

### P2: Ensemble 및 Stability

목표:

```text
horizon ensemble + seed ensemble + model decay monitor
```

작업:

1. horizon `[1, 3, 6]` ensemble.
2. seed ensemble 3개 이하.
3. fold별 feature importance stability report.
4. IC decay 기반 자동 shrinkage 강화.

---

## 15. 기본 설정 제안

```python
StrategyMLConfig(
    name="ml_lambdamart_v1",
    timeframe="4h",
    min_group_size=8,
    label_horizon_bars=1,
    train_months=24,
    valid_months=3,
    test_months=3,
    purge_bars=1,
    max_features=64,
    alpha_clip_bps=75.0,
    lambda_tail=0.25,
    ranker_n_estimators=800,
    calibrator_n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_data_in_leaf=200,
    feature_fraction=0.80,
    bagging_fraction=0.80,
    lambda_l2=5.0,
    early_stopping_rounds=75,
)
```

---

## 16. 금지 사항

- legacy `strategy_sleev` 직접 import 금지.
- legacy HMM, CatBoost, 기존 alpha_factory 의존 금지.
- per-symbol model 학습 금지.
- random split 금지.
- scaler/imputer full-fit 금지.
- target weight를 ML이 직접 생성 금지.
- `alpha_long_00` 등 legacy alias 재도입 금지.
- label horizon과 fold purge 불일치 금지.
- feature 결측을 무조건 0으로 채우는 처리 금지. 결측 사유별 mask 또는 train-only imputer를 사용한다.

---

## 17. 최종 판단

`LightGBM LambdaMART Ranker + LightGBM Quantile EV Calibrator`는 현재 futures strategy architecture에 가장 잘 맞는 ML 전환 방식이다.

이유:

- cross-sectional futures selection 목적과 LambdaMART ranking objective가 직접 정렬된다.
- quantile EV calibration이 Kelly sizing에 필요한 return-unit alpha를 제공한다.
- 기존 `signal_composer`, `portfolio_constructor`, `execution_sim`을 변경하지 않고 활용한다.
- deterministic sleeve는 폐기하지 않고 feature prior로 흡수할 수 있다.
- OOS fold, cost-aware label, tail penalty, IC shrinkage를 통해 과적합과 복리 손실 위험을 동시에 낮출 수 있다.

신규 전략의 핵심 산출물은 모델 자체가 아니라 다음 계약이다.

```text
PIT-safe, cost-aware, calibrated alpha_panel
```

이 계약만 지키면 기존 최적화/백테스트 시스템은 ML 전략을 기존 deterministic sleeve 전략과 동일한 방식으로 평가할 수 있다.
