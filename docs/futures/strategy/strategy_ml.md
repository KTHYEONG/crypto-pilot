# Binance Futures ML 전략 아키텍처 (v3.3 - High Performance Optimized)

**최종 검증/확정**: 2026-05-23  
**핵심 설계 목적**: `opt_main_futures.py`의 strategy stage에 비용을 초과 극복하는 expected return alpha를 공급한다. ML은 주문, weight, leverage를 직접 제어하지 않고 오직 `alpha_panel[(datetime, symbol), alpha_long, alpha_short]`만을 순수하게 생성한다.

---

## 1. 핵심 아키텍처 및 데이터 흐름

ML 전략은 외부 거래소 API와 격리되어 preloaded 데이터만 소비하는 **순수 함수 형태의 Alpha Supplier**로 동작한다.

```text
build_strategy_alpha(
    data_maps: dict[str, dict[str, DataFrame]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
) -> alpha_panel
```

### 2개 프로세스 분리 구조

```text
┌─────────────────────────────────────────────────────┐
│ [프로세스 A: Data/Universe 준비 및 Gating]           │
│ - sync, universe build, 데이터 충족성 검사           │
│ - data_maps / oos_data_maps 일괄 로딩               │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼ [aligned data maps] (PIT 보장 패널 데이터)
                   │
┌──────────────────┴──────────────────────────────────┐
│ [프로세스 B: Strategy Alpha 생성 (ML Engine)]       │
│ - offline · preloaded data_maps만 소비              │
│ - 50개 고성능 피처(CS-Sharpe 포함) 및 Gross Label   │
│ - Double-Weighted Dataset ➡️ CS-Demeaned 학습       │
│ - L1 Hybrid Regularization ➡️ Cost Barrier Gate     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼ [alpha_panel] (MultiIndexed DataFrame)
                   │
┌──────────────────┴──────────────────────────────────┐
│ [기존 Optimization/Backtest 레이어]                  │
│ - Alpha Merge ➡️ Portfolio Construction ➡️ Exec Sim  │
└─────────────────────────────────────────────────────┘
```

---

## 2. 디렉토리 구조 및 모듈 매핑

`src/domain/futures/strategy/` 디렉토리 내 각 파일의 책임 범위 정의:

| 파일명 | 주된 역할 및 책임 | 주요 외부 라이브러리 |
|---|---|---|
| `__init__.py` | strategy 패키지 공개 API 통합 노출 및 Export 관리 | - |
| `config.py` | `StrategyConfig`, `StrategyMLConfig` 파라미터 검증 및 중앙 관리 | `dataclasses` |
| `contracts.py` | feature/label/fold/artifact의 엄격한 데이터 스펙(dataclass) 정의 | `numpy`, `pandas` |
| `builder.py` | strategy name 기반 라우팅 및 alpha build entrypoint | `pandas` |
| `ml_builder.py` | ML feature-label-train-infer 오케스트레이션 및 Cost Barrier Gate 적용 | `numpy`, `pandas` |
| `features.py` | PIT를 준수하는 robust feature panel(크로스섹션 Z-Sharpe 포함) 생성 | `numpy`, `pandas` |
| `labels.py` | t+1 체결 기준 마찰 비용이 제외된 순수 Gross Alpha 레이블 생성 | `numpy`, `pandas` |
| `dataset.py` | Double-Weighting이 연동된 LightGBM cross-section group 구조 변환 | `numpy`, `pandas` |
| `ranker.py` | L1 규제가 적용된 CS-demeaned GBT Regressor 학습 및 스코어 추론 | `lightgbm`, `numpy` |
| `calibrator.py` | 분위수 기반 동적 불확실성 조정 및 L1 규제 EV 보정 모형 적용 | `lightgbm`, `numpy` |
| `inference.py` | fold별 OOS 예측값을 MultiIndexed canonical `alpha_panel`로 결합 | `numpy`, `pandas` |
| `diagnostics.py` | Spearman IC, NDCG, alpha coverage 및 EV/cost 진단 보고서 작성 | `numpy`, `pandas` |
| `cache.py` | 연산 효율성 극대화를 위한 feature/label/model manifest 캐시 | `json`, `hashlib`, `pandas` |

---

## 3. 데이터 계약 스키마

### 3.1 입력 데이터 계약 (`data_maps`)
`data_maps[symbol][tf]` DataFrame은 아래의 필수 필드를 가지며, timezone이 없는 `np.datetime64`로 엄격히 정렬된다.
- `datetime` (Index / Key)
- `open`, `high`, `low`, `close` (수익률 및 변동성 연산)
- `volume` (유동성 피처용)
- `funding_rate` (Carry 피처용)
- `active_mask`, `warm_mask`, `entry_block_mask`, `kill_mask` (유니버스 가용성 필터)

### 3.2 출력 데이터 계약 (`alpha_panel`)
최종 출력되는 `alpha_panel`은 다음 계약 사양을 반드시 충족해야 한다.
- **Index**: `MultiIndex(datetime, symbol)`
- **필수 컬럼**: `alpha_long` (non-negative float), `alpha_short` (non-negative float)
- **단위**: decision bar당 simple return 스케일 (Bps)
- **결측치**: 금지 (미매칭 구간은 `0.0`으로 정적 치환)

---

## 4. 6단계 ML 전략 Funnel 상세

### Stage 1: Feature 생성 (`features.py`)
- **목적**: OHLCV, funding, universe meta를 PIT-safe 피처 텐서로 변환 (총 50개 피처 스키마 잠금).
- **주요 피처 셋**:
  - Reversal / Momentum: `ret_1` ~ `ret_36` 및 횡단면 랭크 팩터 (`cs_rank_ret_12`, `cs_rank_ret_36`).
  - Volatility: realized vol, downside vol, ATR ratio.
  - Carry / Liquidity: funding z-score, volume z-score, ADV rank.
  - **CS-Sharpe (퀀트 고성능 피처)**: 개별 변동성 대비 기대수익률 강도를 크로스섹션 랭크화 한 `cs_sharpe_6` 및 `cs_sharpe_18` 주입.
- **기술 스택**: `numpy`, `pandas` 기반 벡터화 롤링 연산.

### Stage 2: Label 생성 (`labels.py`)
- **목적**: `feature[t]` 시그널에 조응하는 `entry[t+1]` 진입 기반의 순수 마켓 초과수익률 타겟 생성.
- **핵심 사양**:
  - 학습 시점의 정보 소실 방지를 위해 거래 비용(`cost`)을 `0.0`으로 고정하여 **Gross Alpha**를 생성.
  - 롱/숏 대칭성을 확보하기 위해 `signed_net_ret`을 기준으로 분위 등급(`relevance` 0~4 등급)을 부여하여 랭킹 및 분위수 회귀의 기준으로 사용.

### Stage 3: Dataset 구성 (`dataset.py`)
- **목적**: 시계열 패널을 동일 시점의 횡단면 집합인 LambdaMART group 구조로 재정렬 및 가중치 고도화.
- **Double-Weighting System**:
  - 보합/횡보 장세의 무작위 노이즈 신호를 배제하기 위해, 실측 리턴의 절대값($|y_{ev}|$)에 비례하여 sample_weight를 동적으로 가중.
  - $$\text{sample\_weight} = \text{sample\_weight} \times (1.0 + 2.0 \times |y_{ev}|)$$
- **소표본 방어**: 유니버스 자산 수가 `min_group_size = 8` 미만인 타임스탬프는 과적합 방지를 위해 학습 집합에서 자동 제외.

### Stage 4: Relative Ranker (`ranker.py`)
- **목적**: 동일 크로스섹션 자산들 사이의 순수 상대 우위(Relative Rank) 학습.
- **로직**: 타겟에 크로스섹션 평균을 차감(`_cs_demean`)하여 마켓 베타를 완벽히 제거한 뒤, `metric="rmse"` 목적함수로 LightGBM Regressor 피팅.
- **최적화 설정**: 과적합 방지를 위해 L1 규제 `reg_alpha=1.5` 주입, 피처 무작위 분할 `feature_fraction=0.70`, 배깅 배율 `bagging_fraction=0.75` 및 학습률 `learning_rate=0.02` 고정 적용.

### Stage 5: Quantile EV Calibration (`calibrator.py`)
- **목적**: 랭킹 스코어를 기대수익률(Bps)의 절대 차원으로 복원하되 꼬리 위험을 정교하게 방어.
- **로직**: `q10`, `q50`, `q90` 분위수 예측기를 동시 학습 (L1 규제 `reg_alpha=1.5` 및 배깅/러닝레이트 최적화 동일 적용).
- **동적 불확실성 조정**: 
  - 예측 불확실성 폭($Uncertainty = q_{90} - q_{10}$)이 자산 전체 중앙값 대비 넓을수록 패널티 가중치 $\lambda_{dynamic}$을 증폭하여 알파를 강하게 삭감하고, 확실성이 높은 자산은 알파 강도(Bps)를 온전히 보존.
  - $$ev = \text{where}(q_{50} \ge 0, \ q_{50} - \lambda_{dynamic} \times downside, \ q_{50} + \lambda_{dynamic} \times upside)$$

### Stage 6: Inference & Gating (`ml_builder.py`)
- **목적**: Walk-forward 학습-검증-추론 연쇄 조율 및 최종 실질 신호 필터링.
- **Dynamic Cost Barrier Gate**:
  - 크로스섹션 데미닝이 완료된 최종 `ev_test` 상의 스코어에 대해, 실질 라운드 트립 거래 비용(수수료 + 슬리피지 장벽)보다 절대값이 작은 미약한 노이즈성 시그널들을 강력하게 `0.0`으로 소거하여 확실한 고Bps 알파만 추출.

### 런타임 최적화 가속화 기법 (Runtime Acceleration)
- **Zero-Copy 사전 캐싱 (Pre-computation)**: `prepare_backtest_inputs`와 `merge_membership_constraints_into_aligned` 등 하이퍼파라미터에 의존하지 않는 정적 데이터 슬라이싱/정렬 결과를 최초 1회만 연산하고, 런타임 메모리에 객체 레퍼런스(`_prepared_cache`) 형태로 캐싱하여 복사 연산을 원천 차단 (Prep 단계 소요시간을 0.03ms로 극소 소멸).
- **조기 가지치기의 역설 극복 (Pruning Bypass)**: JIT Numba 백테스팅 연산이 극도로 가속화(1.5ms 수준)됨에 따라, 조기 가지치기 판독을 위해 SQLite DB에 트랜잭션을 쏘고 락 경합을 대기하는 비용(24ms 수준)이 훨씬 더 크다는 성능 역설을 입증 및 해결. `FUTURES_PRUNING_ENABLED = False`를 적용하여 DB I/O를 100% 바이패스 처리해 연산 효율을 기계어 한계 속도(수 ms대)로 극대화.

---

## 5. 전략 품질 검증 가드레일

### 5.1 Runtime Contract Gate
추론 완료 시점의 최소 알파 무결성 보장선:
- `alpha_panel.empty == False`
- `alpha_long` 및 `alpha_short` 양방향 필수 존재
- 롱/숏 각각 최소 1개 이상의 non-zero alpha 값 가용할 것 (strategy mode 한정)

### 5.2 학습 및 추론 Quality Gate
- `spearman_ic > 0.0` (Spearman IC 3 fold 연속 음수 기록 시 하드 페일)
- `alpha_p95_abs <= alpha_clip_bps` (클리핑 범위 정합성 보장)
- `finite_feature_ratio >= 0.995` (데이터 오염 차단)

---

## 6. 금지 사항 (Anti-Patterns)

- **Portfolio Control 침범 금지**: ML이 target weight, order, leverage를 직접 계산하지 말 것. 오직 pure expected return alpha만 계산해야 함.
- **Look-Ahead Leakage 금지**: 미래 시점 데이터를 참조하여 Scaler를 피팅하거나 imputer를 계산하지 말 것.
- **Friction-less Trading 금지**: dynamic cost barrier gate를 우회하여 0~5 bps 수준의 미약한 시그널을 난사하여 수수료 폭주를 일으키지 말 것.
- **Deterministic Sleeve 복제 금지**: ML sleeve는 prior를 단순 복사하는 대리인이 아니며, 반드시 quantile calibration 경로를 경유해야 함.
