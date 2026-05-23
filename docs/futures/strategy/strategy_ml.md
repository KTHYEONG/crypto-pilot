# Binance Futures ML 전략 아키텍처 (v3.1 - Runtime Contract)

**최종 검증/확정**: 2026-05-23  
**핵심 설계 목적**: `opt_main_futures.py`의 strategy stage에 비용 차감 expected return alpha를 공급한다. ML은 주문, weight, leverage를 직접 만들지 않고 `alpha_panel[(datetime, symbol), alpha_long, alpha_short]`만 생성한다.

## 0. 2026-05-23 반영 사항 (코드 사실)

- `src/execution/opt_main_futures.py`의 기본 strategy는 `ml_lambdamart_v1`이다.
- 실행 모드:
  - `quick-backtest`: `MLPipelineOutput()` 빈 출력 허용.
  - `strategy`: preloaded OOS/IS data maps를 사용해 strategy alpha를 생성하고, alpha merge 후 non-zero long/short를 hard validate한다.
  - `strategy-smoke`: strategy stage까지만 실행 후 종료한다.
- 연결 경로:

```text
opt_main_futures._run_strategy_stage()
  -> pick_strategy_data_maps()
  -> run_active_strategy_output_bridge()
  -> strategy_runtime.bridge.run_ml_pipeline_for_universe()
  -> strategy.builder.build_strategy_alpha()
  -> MLPipelineOutput(alpha_panel=...)
  -> merge_ml_output_into_is_and_oos()
  -> assert_strategy_alpha_ready()
```

- `assert_strategy_alpha_ready()`의 hard contract:
  - `alpha_panel`은 비어 있으면 안 된다.
  - 필수 컬럼은 `alpha_long`, `alpha_short`이다.
  - merge 후 각 symbol frame에 두 컬럼이 존재해야 한다.
  - strategy mode에서는 long/short 양쪽 모두 non-zero alpha가 최소 1개 이상 필요하다.
- 기존 deterministic strategy name(`momentum_v0`, `eh_st_v1`)은 `StrategyConfig` 호환을 위해 남아 있지만, 신규 ML 전략의 본체는 `ml_lambdamart_v1`이다.

---

## 1. 핵심 아키텍처 및 데이터 흐름

ML 전략은 **runtime bridge 뒤의 alpha supplier**로 동작한다.

```text
build_strategy_alpha(
    data_maps: dict[str, dict[str, DataFrame]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
) -> alpha_panel
```

* **PIT 보장**: feature는 `t` bar close까지의 정보만 사용하고 label은 `t+1` 이후 체결 수익률로 만든다.
* **단위 보장**: `alpha_long` / `alpha_short`는 decision bar당 simple return 단위다.
* **책임 경계**: ML은 alpha만 공급한다. `signal_composer`, `portfolio_constructor`, `execution_sim`이 비용 재차감, Kelly sizing, cap projection, 체결 시뮬레이션을 담당한다.

### 2개 프로세스 분리 구조

```text
┌─────────────────────────────────────────────────────┐
│ [프로세스 A: Data/Universe 준비]                     │
│ - sync, universe build, readiness gate               │
│ - data_maps / oos_data_maps 생성                     │
│ - selected symbols와 tf frame 정렬                   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌──────────────────┴──────────────────────────────────┐
│ [프로세스 B: Strategy Alpha 생성]                    │
│ - preloaded data_maps만 사용                         │
│ - feature/label/fold/inference                       │
│ - alpha_panel 반환                                   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌──────────────────┴──────────────────────────────────┐
│ [기존 Optimization/Backtest]                         │
│ - alpha merge                                        │
│ - signal compose                                     │
│ - portfolio construction                             │
│ - execution simulation                               │
└─────────────────────────────────────────────────────┘
```

---

## 2. 디렉토리 구조 및 모듈 매핑

`src/domain/futures/strategy/`는 ML alpha 생성 패키지다.

| 파일명 | 주된 역할 및 책임 | 주요 외부 라이브러리 |
|---|---|---|
| `__init__.py` | strategy 공개 API 노출 | - |
| `config.py` | `StrategyConfig`, `StrategyMLConfig` 및 파라미터 검증 | `dataclasses` |
| `contracts.py` | feature/label/fold/artifact 스키마 | `dataclasses`, `numpy`, `pandas` |
| `builder.py` | strategy name routing 및 alpha build entrypoint | `pandas` |
| `ml_builder.py` | ML feature-label-train-infer 오케스트레이션 | `numpy`, `pandas` |
| `features.py` | PIT feature panel 생성 | `numpy`, `pandas` |
| `labels.py` | t+1 체결 기준 net return label 생성 | `numpy`, `pandas` |
| `dataset.py` | panel을 LightGBM long matrix와 group 배열로 변환 | `numpy`, `pandas` |
| `ranker.py` | LambdaMART ranker 학습/예측 | `lightgbm`, `numpy` |
| `calibrator.py` | quantile EV calibrator 학습/예측 | `lightgbm`, `numpy` |
| `inference.py` | fold output을 canonical alpha_panel로 조립 | `numpy`, `pandas` |
| `diagnostics.py` | IC, NDCG, EV/cost, alpha coverage 진단 | `numpy`, `pandas` |
| `cache.py` | feature/label/model manifest 캐시 | `json`, `hashlib`, `pandas` |
| `common/alignment.py` | symbol별 datetime 정렬 및 mask alignment | `numpy`, `pandas` |
| `common/normalization.py` | robust rank/z-score/winsorization | `numpy`, `pandas` |
| `common/validation.py` | no-leakage, dtype, finite value 검증 | `numpy`, `pandas` |

---

## 3. 데이터 계약 및 영속화 스키마

### 3.1 입력 데이터 (`data_maps`)

`data_maps[symbol][tf]`는 최소 아래 컬럼을 가져야 한다.

| 컬럼 | 필수 여부 | 용도 |
|---|---:|---|
| `datetime` | 필수 | PIT 정렬 및 alpha merge key |
| `open`, `high`, `low`, `close` | 필수 | return, volatility, label |
| `volume` | 필수 | liquidity/volume feature |
| `funding_rate` 또는 `funding_rate_sum` | 권장 | carry feature 및 funding adjustment |
| universe meta columns | 권장 | ADV, execution cost, beta, cluster feature |
| eligibility masks | 권장 | active/warm/entry block/kill signal gating |

입력 frame은 `datetime` 기준 단조 증가해야 하며, timezone 혼합으로 merge key가 어긋나면 안 된다.

### 3.2 출력 데이터 (`alpha_panel`)

`MLPipelineOutput.alpha_panel`은 다음 계약을 따른다.

| 항목 | 계약 |
|---|---|
| Index | `MultiIndex(datetime, symbol)` |
| 필수 컬럼 | `alpha_long`, `alpha_short` |
| dtype | float 계열, finite value |
| 단위 | decision bar당 simple return |
| 부호 | 두 컬럼 모두 non-negative |
| 결측 | 금지. merge 불일치 구간은 data map merge 단계에서 0.0 fill 허용 |

변환 규칙:

```text
alpha_long[t, i]  = max(ev_shrunk[t, i], 0.0)
alpha_short[t, i] = max(-ev_shrunk[t, i], 0.0)
```

### 3.3 캐시 및 산출물

```text
data/cache_futures/strategy_ml/
├── features/{tf}_{window_hash}_{feature_hash}.parquet
├── labels/{tf}_{window_hash}_h{horizon}_{label_hash}.parquet
└── manifest/{run_id}_strategy_ml_manifest.json

logs/futures/models/strategy_ml/{run_id}/
├── fold_00_ranker.txt
├── fold_00_q10.txt
├── fold_00_q50.txt
├── fold_00_q90.txt
└── fold_metrics.parquet
```

캐시 hash는 window, symbols, tf, feature config, label horizon, cost config, source column availability를 포함한다.

---

## 4. ML Strategy Funnel 상세

### Stage 0: Runtime 입력 선택

* **목적**: `strategy` 모드에서 OOS frame이 존재하면 OOS data maps를 우선 사용하고, 없으면 IS data maps로 fallback한다.
* **구현 모듈**: `src/application/futures/optimization/strategy_service.py` -> `pick_strategy_data_maps()`
* **검증 규칙**:
  1. `strategy` 모드는 `preloaded_data_maps`가 필수다.
  2. `quick-backtest`는 빈 `MLPipelineOutput`을 허용한다.
  3. `strategy-smoke`는 strategy stage smoke 검증에 사용한다.

---

### Stage 1: Feature 생성

* **목적**: OHLCV, funding, universe meta를 PIT-safe feature panel로 변환한다.
* **구현 모듈**: `features.py`
* **P0 feature group**:
  - Return: short-term reversal/momentum, skip momentum
  - Volatility: realized vol, downside vol, ATR ratio
  - Cross-section: timestamp별 return/vol/funding/volume rank
  - Carry: funding mean, funding z-score, sign persistence
  - Liquidity: volume z-score, ADV rank, execution cost rank
  - Market context: BTC/ETH return, market breadth, dispersion
  - Prior: deterministic sleeve raw output이 아닌 clipped rank prior
* **기술 원칙**:
  - full-sample scaler fit 금지.
  - cross-sectional transform은 동일 timestamp 내부에서만 수행.
  - feature 수는 P0 기준 64개 이하.
  - 학습 직전 `float32` 변환.

---

### Stage 2: Label 생성

* **목적**: `feature[t]`와 `entry[t+1]`가 정렬된 비용 차감 forward return label을 만든다.
* **구현 모듈**: `labels.py`
* **기본 설정**:
  - `label_horizon_bars = 1`
  - `purge_bars >= label_horizon_bars`
  - `embargo_bars >= 1`
* **비용 모델** (정준 기준: `src/core/settings.py`):

```text
round_trip_cost = FILLS_PER_ROUND_TRIP × (fee_bps + slippage_bps) / 10000
               = 2 × (5 + 2) / 10000 = 0.0014  (14 bps)

long_net_ret  = forward_return - round_trip_cost
short_net_ret = -forward_return - round_trip_cost
```

Label이 차감하는 14 bps는 `execution_sim`이 실제로 부과하는 비용(진입 Taker + 청산 Taker + 양 side 슬리피지)과 정확히 일치합니다. 펀딩비는 보유 기간에 비례하므로 label 생성에서 별도 처리하지 않고 `execution_sim`이 누적합니다.

* **Signal Composer Gate**: `signal_composer.py`의 friction threshold도 동일하게 `round_trip_cost_bps() / 10000 = 14 bps`를 사용합니다. 추가로 `EV_HURDLE_BPS`(기본값 40 bps)만큼의 최소 edge를 요구합니다. 즉, 신호가 발화되려면 `alpha > 54 bps` 조건이 충족되어야 합니다.

* **계산 개념**:

```text
long_net_ret  = forward_return - round_trip_cost
short_net_ret = -forward_return - round_trip_cost
signed_net_ret = long_net_ret
```

Short alpha는 최종 signed EV가 음수일 때 `alpha_short`로 분리한다.

---

### Stage 3: Rank Dataset 구성

* **목적**: timestamp cross-section을 LambdaMART group으로 변환한다.
* **구현 모듈**: `dataset.py`
* **계약**:

```text
X: float32 [M, F]
y_rank: int32 [M]
group: int32 [G], sum(group) == M
sample_weight: float32 [M]
index_map: int64 [M, 2]  # time_idx, symbol_idx
```

* **relevance bucket**:
  - top 15% -> 4
  - top 15~35% -> 3
  - middle -> 2
  - bottom 15~35% -> 1
  - bottom 15% -> 0
* **소표본 방어**: `min_group_size = 8` 미만 timestamp는 ranker 학습에서 제외한다.

---

### Stage 4: LambdaMART Ranker

* **목적**: 같은 timestamp의 tradable symbols 안에서 상대 순위를 학습한다.
* **구현 모듈**: `ranker.py`
* **기본 모델**: `lightgbm.LGBMRanker(objective="lambdarank", metric="ndcg")`
* **핵심 제한**:
  - `max_depth <= 6`
  - `num_leaves <= 31`
  - `min_data_in_leaf >= 100`
  - per-symbol model 금지.
  - random split 금지.

Ranker score는 return 단위가 아니므로 단독 alpha로 사용하지 않는다.

---

### Stage 5: Quantile EV Calibration

* **목적**: rank score를 비용 차감 return 단위 EV로 보정한다.
* **구현 모듈**: `calibrator.py`
* **기본 모델**: `lightgbm.LGBMRegressor(objective="quantile")`
* **quantile**: q10, q50, q90
* **보수적 EV 변환**:

```text
downside = max(q50 - q10, 0.0)
ev_adj = q50 - lambda_tail * downside
ev_adj = clip(ev_adj, -alpha_clip, alpha_clip)
```

기본값:

| 파라미터 | 값 |
|---|---:|
| `lambda_tail` | `0.25` |
| `alpha_clip_bps` | `75.0` |
| `learning_rate` | `0.03` |
| `ranker_n_estimators` | `800` |
| `calibrator_n_estimators` | `600` |

---

### Stage 6: Walk-Forward Inference

* **목적**: train/valid/test fold를 시간순으로 분리하고, test 구간 예측만 alpha panel에 append한다.
* **구현 모듈**: `ml_builder.py`, `inference.py`
* **기본 fold**:

```text
train_window = 24 months
valid_window = 3 months
test_window = 3 months
step = 3 months
purge_bars = label_horizon_bars
embargo_bars = max(1, label_horizon_bars)
```

* **No-Leakage 규칙**:
  1. scaler, imputer, winsor boundary는 train fold에서만 fit한다.
  2. validation/test timestamp는 학습 group에 포함하지 않는다.
  3. optional source availability는 feature로 넣되 미래 가용 여부를 추론하지 않는다.
  4. label horizon과 purge bars가 불일치하면 실행하지 않는다.

---

## 5. 기존 최적화 파이프라인과의 연결

### 5.1 Runtime Bridge

`src/domain/futures/strategy_runtime/bridge.py`는 `build_strategy_alpha()`를 호출해 `MLPipelineOutput`으로 감싼다.

```text
run_ml_pipeline_for_universe(...)
  -> build_strategy_alpha(...)
  -> MLPipelineOutput(alpha_panel=alpha_panel)
```

`merge_ml_output_into_data_maps()`는 `alpha_panel`을 symbol별 frame의 `datetime`에 left merge하고, 미매칭 값은 `0.0`으로 채운다.

### 5.2 Signal/Portfolio 계약

기존 strategy mode composer는 유지한다.

```text
alpha_long / alpha_short
  -> BETA_ALPHA
  -> friction subtraction
  -> EV_HURDLE_BPS
  -> xs_score_long / xs_score_short
  -> portfolio_constructor
  -> execution_sim
```

추천 search range:

| 파라미터 | P0 범위 | 이유 |
|---|---:|---|
| `BETA_ALPHA` | `1.0 ~ 5.0` | ML alpha는 이미 return 단위로 calibration됨 |
| `EV_HURDLE_BPS` | `2.0 ~ 30.0` | 비용 차감 후 과도한 무거래 방지 |
| `REBALANCE_BARS` | `1 ~ 8` | ranker signal decay 확인 |
| `PORTFOLIO_KAPPA` | `0.15 ~ 0.60` | Kelly aggressiveness 제어 |
| `PORTFOLIO_F_KELLY_MAX` | `0.10 ~ 0.75` | tail risk 방어 |
| `MAX_EXPOSURE_PER_COIN` | `0.10 ~ 0.35` | concentration 방지 |

---

## 6. 전략 품질 검증 가드레일

`opt_main_futures.py`의 strategy stage와 optimization stage 사이에서 아래 조건을 만족해야 한다.

### 6.1 Runtime Contract Gate

| 지표 | Pass 기준 | Hard Fail |
|---|---:|---:|
| `alpha_panel.empty` | `False` | `True` |
| required columns | `alpha_long`, `alpha_short` 존재 | 누락 |
| merged symbol frames | `>= 1` | `0` |
| long non-zero count | `> 0` | `0` |
| short non-zero count | `> 0` | `0` |

### 6.2 학습/추론 Gate

| 지표 | Pass 기준 | Hard Fail |
|---|---:|---:|
| finite feature ratio | `>= 0.995` | `< 0.990` |
| valid group count | `>= 100` | `< 50` |
| valid Spearman IC | `> 0.0` | `<= 0.0` 3 fold 연속 |
| ranker NDCG@K | baseline 이상 | baseline 미만 3 fold 연속 |
| alpha p95 abs | `<= alpha_clip_bps` | clip 초과 |

### 6.3 Backtest Gate

| 지표 | Pass 기준 | 목적 |
|---|---:|---|
| `positive_leg_ratio` | `>= 0.60` | fold 안정성 |
| `ev_cost_ratio` | `>= 1.00` | 비용 차감 edge |
| `mean(log_TW)` | deterministic baseline 이상 | 복리 성장 |
| `worst_leg_log_TW` | `>= -0.10` | tail 방어 |
| `turnover_cost_ratio` | `<= 0.35` | 비용 폭주 방지 |
| `MDD` | `<= 60%` | 생존성 |

---

## 7. 진단 로깅 계약

신규 strategy 로그는 기존 실행 로그와 이어져야 한다.

| 태그 | 단계 | 핵심 필드 |
|---|---|---|
| `[ML-FEATURE]` | feature 생성 | rows, symbols, features, nan_ratio, cache_hit |
| `[ML-LABEL]` | label 생성 | horizon, valid_ratio, long_p95_bps, short_p95_bps |
| `[ML-FOLD]` | fold 분할 | fold, train/valid/test bars, purge |
| `[ML-RANKER]` | ranker 학습 | ndcg, rank_ic, best_iter, top_features |
| `[ML-CALIB]` | EV calibration | q10/q50/q90 spread, pinball_loss |
| `[ML-ALPHA]` | alpha 생성 | long_nz, short_nz, alpha_p95_bps, ic_lag_mean |
| `[ML-OOS]` | OOS 진단 | ic, ev_cost_ratio, turnover_proxy |
| `[ALPHA-MERGE]` | runtime merge | merged_syms, alpha_long_nz, alpha_short_nz |

권장 연결:

```text
[ML-ALPHA]
  -> [ALPHA-MERGE]
  -> [COMPOSE-DIAG]
  -> [LEG]
  -> [RUN-SUMMARY]
```

---

## 8. 구현 단계

### P0: 최소 ML Alpha Engine

목표:

```text
OHLCV + funding + universe meta 기반 CS-Demeaned GBT Ranker/Quantile alpha_panel
```

작업:

1. `features.py`에 P0 feature set 구현.
2. `labels.py`에 t+1 execution aligned net return label 구현.
3. `dataset.py`에 timestamp group 기반 LightGBM matrix 구현.
4. `ranker.py`에 CS-Demeaned GBT 학습/예측 구현.
5. `calibrator.py`에 q10/q50/q90 EV model 구현.
6. `inference.py`에서 fold별 `alpha_panel` 조립.
7. `builder.py`가 `ml_lambdamart_v1`일 때 ML path를 호출하도록 유지.
8. `strategy_service.assert_strategy_alpha_ready()` smoke를 통과한다.

### P1: Binance 확장 데이터 결합

목표:

```text
premiumIndex, bookDepth, metrics(OI/LSR) feature 활성화
```

작업:

1. data maps 또는 universe ledger에 optional source columns 공급.
2. availability mask feature 추가.
3. basis/crowding/microstructure feature group 활성화.
4. optional feature가 없는 구간에서도 동일 schema 유지.

### P2: Stability

목표:

```text
horizon ensemble + seed ensemble + model decay monitor
```

작업:

1. horizon `[1, 3, 6]` ensemble.
2. seed ensemble은 3개 이하로 제한.
3. fold별 feature importance stability report.
4. IC decay 기반 shrinkage 강화.

---

## 9. 금지 사항

- ML이 target weight, order, leverage를 직접 생성 금지.
- `alpha_long_00` 같은 legacy alias 재도입 금지.
- per-symbol model 학습 금지.
- random split 금지.
- scaler/imputer full-fit 금지.
- label horizon과 fold purge 불일치 금지.
- feature 결측을 무조건 0으로 채우는 처리 금지. 결측 사유별 mask 또는 train-only imputer를 사용한다.
- ranker score를 return-unit alpha로 직접 사용 금지.
- `strategy` mode에서 zero-only alpha를 성공으로 처리 금지.

---

## 10. 최종 계약

신규 ML 전략의 핵심 산출물은 모델 객체가 아니라 아래 runtime contract다.

```text
PIT-safe, cost-aware, calibrated alpha_panel
```

이 계약을 지키면 `opt_main_futures.py`는 ML 전략을 기존 백테스트/최적화 경로에 동일하게 연결할 수 있다.
aware, calibrated alpha_panel
```

이 계약을 지키면 `opt_main_futures.py`는 ML 전략을 기존 백테스트/최적화 경로에 동일하게 연결할 수 있다.
��.
