# 현재 Compound Futures 파이프라인 설명

이 문서는 `src/execution/opt_main_futures.py`를 실행했을 때 실제로 동작하는 로직을 기준으로 작성했다.

핵심 흐름은 다음과 같다.

```text
Binance 캐시 OHLCV
  ↓
날짜·시간축 정규화
  ↓
PIT UniverseStateCube + MarketFeatureCube
  ↓
18개 고정 recipe 계산
  ↓
L1: causal edge·불확실성 추정
  ↓
L2: robust forecast를 비중으로 변환
  ↓
수수료·슬리피지·impact·funding을 반영한 simulator
  ↓
L2 평가 + 최근 90일 L3 평가
```

## 1. 메인 실행 명령

```bash
UV_CACHE_DIR=/tmp/uv-cache \
PYTHONPATH=. \
uv run python src/execution/opt_main_futures.py \
  --sync skip \
  --date 2026-07-08 \
  --seed 42
```

`opt_main_futures.py`는 CLI를 호출하는 얇은 진입점이다. 실제 실행은
`src/application/futures/runner/cli.py`의 `run_compound_main()`에서 시작한다.

현재 허용되는 실행 옵션은 다음 네 가지다.

- `--date`: 기준일
- `--sync auto|skip`: 유니버스 ledger 동기화 방식
- `--refresh-universe`: 유니버스 갱신 요청
- `--seed`: 재현용 seed

`--phase`, `--trials`, `--timeframe`, `--mode`, `--symbols` 같은 기존 옵션은 제거됐다. L2에서 Optuna를 실행하지도 않는다.

## 2. 데이터 입력과 시간 정합성

### 2.1 현재 사용하는 데이터

메인 실행은 `data/futures/ohlcv/1h/*.parquet`의 시간별 OHLCV를 읽는다.

Binance 캐시가 다음처럼 저장되어 있어도 입력 어댑터가 표준 형태로 바꾼다.

```text
timestamp          → datetime (UTC)
quote_vol          → quote_volume
taker_buy_quote_volume → taker_buy_quote
```

특히 timestamp 정밀도를 내부 나노초 기준으로 통일한다. 이 처리가 없으면 원천 데이터의 밀리초/마이크로초와 PIT calendar의 정밀도가 달라져 모든 관측값이 잘못된 위치에 매칭될 수 있다.

### 2.2 기준일은 모든 단계에서 동일해야 한다

`reference_date`는 다음에 똑같이 사용된다.

1. 데이터 로더의 마지막 시점
2. PIT 유니버스 calendar의 마지막 시점
3. simulator의 평가 구간

데이터는 기준일과 다른 날짜로 읽고 유니버스만 기준일로 만들면 앞부분이 결측이 되어 simulator가 무결성 실패로 현금 대기한다.

### 2.3 PIT MarketFeatureCube

`build_compound_market_feature_cube()`가 각 symbol의 관측값을 state calendar에 backward as-of 방식으로 붙인다.

시점 `t`의 값은 `t`보다 늦게 발생한 관측값을 사용할 수 없다.

```text
사용 가능:
  observation_time <= t

사용 금지:
  observation_time > t
```

유니버스에 없거나 필수 OHLCV가 없는 symbol은 `entry_block=True`, `eligible=False`가 된다. 단, 이미 보유한 포지션을 줄이는 것은 허용하는 방향이 simulator의 책임이다.

## 3. 현재 유니버스의 주의점

현재 `resolve_universe_symbols()`는 아직 실제 historical ledger에서 symbol 목록을 반환하지 않는다. 그래서 캐시 symbol을 자동으로 찾지 못하면 메인 경로는 다음 fallback을 사용한다.

```text
BTCUSDT, ETHUSDT
```

따라서 이번 실측 결과의 “2종목”은 최종 유니버스 필터를 통과한 120개 후보가 아니다. 실제 PIT historical universe selector가 연결되기 전까지는 BTC·ETH 두 종목만으로 실행된다.

이 부분은 전략 성과보다 먼저 보완해야 할 운영상 병목이다. 18개 신호를 계산하더라도 2종목만 있으면 cross-sectional 신호의 통계적 의미와 분산 효과가 크게 제한된다.

## 4. 18개 신호의 의미

18개는 종목 수가 아니라 `signal family × horizon` 조합이다.

```text
6개 family × 3개 horizon = 18개 recipe
```

현재 family는 다음과 같다.

| Family | 의미 |
|---|---|
| `time_series_trend` | 한 종목의 추세 방향 |
| `breakout` | 최근 가격 범위 이탈 |
| `cross_sectional_momentum` | 종목 간 상대 강도 |
| `short_term_reversal` | 단기 급등락의 되돌림 |
| `carry_basis` | funding과 premium의 보유비용/수익 |
| `flow_positioning` | taker flow와 funding 기반 포지셔닝 |

각 family에 `4`, `12`, `24` bar horizon을 적용한다.

```text
time_series_trend_h4
time_series_trend_h12
time_series_trend_h24
...
flow_positioning_h24
```

L0는 최근 성과가 나쁘다는 이유로 recipe를 삭제하지 않는다. 필수 데이터가 없거나 계산 결과가 malformed인 경우만 유효하지 않게 표시한다.

## 5. L1: 신호를 평가하는 방식

L1은 raw score를 미래 수익률 예측값으로 바꾼다.

```text
raw score
  + 과거 label
  + purged causal folds
  + recipe 전체 평균 prior
  ↓
gross_mu
forecast_variance
reliability
```

### 5.1 Causal fold

현재 기본값은 4개 fold, purge 25 bar, embargo 1 bar다.

- 미래 label이 fit 구간에 섞이지 않는다.
- OOS 구간의 결과로 그 이전 forecast를 다시 만들지 않는다.
- 관측 수가 적은 symbol은 recipe 전체 prior 쪽으로 shrinkage한다.

### 5.2 L1 lifecycle

각 recipe는 다음 상태를 가질 수 있다.

- `ACTIVE`: 충분한 관측이 있어 L2에 사용 가능
- `SHADOW`: 관측 부족으로 참고만 함
- `RETIRED`: 데이터 무결성 또는 반복적인 음의 근거로 중단

다만 현재 구현은 lifecycle을 곧바로 전체 거래 중단 gate로 사용하지 않는다. 최종 영향은 L2 forecast의 크기·신뢰도·support로 전달된다.

## 6. L2: forecast를 포트폴리오 비중으로 변환

L2에는 정책 4개 경쟁이나 Optuna 탐색이 없다. 모든 recipe를 한 번에 결합하는 단일 allocator가 동작한다.

### 6.1 신호 결합

recipe별 forecast를 variance와 reliability로 가중 결합한다.

```text
precision = reliability / forecast_variance
mu_combined = precision-weighted mean
variance_combined = 1 / total precision
```

서로 반대 방향인 신호가 많거나 uncertainty가 크면 결합 결과가 작아진다.

### 6.2 Robust shrink

현재 핵심 식은 다음과 같다.

```text
robust_mu
  = sign(mu)
  × max(abs(mu) - 0.5 × sqrt(variance), 0)
```

이 식의 의미는 다음과 같다.

- 불확실성이 작으면 forecast를 사용한다.
- 불확실성이 forecast보다 크면 해당 symbol을 이번 bar에서 사용하지 않는다.
- 이는 hard rejection gate라기보다 보수적 position support 계산이다.

### 6.3 성장 최적화

allocator는 다음을 동시에 고려한다.

```text
기대수익
- covariance 기반 위험
- turnover 비용
- 이전 비중과의 변화량
```

비중 제한은 다음과 같다.

- 전체 gross exposure 최대 1.0
- 전체 net exposure 절댓값 최대 0.3
- 종목별 비중 절댓값 최대 0.1
- beta 노출 최대 0.25
- 유동성 capacity를 초과하는 진입 금지

### 6.4 현재 구현의 중요한 실제 동작

현재 데이터 실측에서는 L1의 개별 forecast가 존재했지만, L2 robust shrink 이후 두 종목 모두 `support=False`가 됐다.

즉 다음과 같은 결과다.

```text
raw signal 있음
  → L1 forecast 있음
  → uncertainty 차감 후 robust_mu = 0
  → L2 target weight = 0
```

이것은 simulator가 거래를 누락한 것이 아니라, 현재 L2가 불확실성을 너무 크게 판단해 거래하지 않은 것이다. 향후 개선 시에는 uncertainty calibration, forecast variance 단위, horizon 정규화, 2종목 fallback 문제를 함께 검토해야 한다.

## 7. Simulator의 실제 체결·복리 계산

simulator는 다음 시간 순서를 사용한다.

```text
bar t에서 결정
  ↓
bar t+1에서 새 비중 적용
  ↓
이후 다음 rebalance까지 비중 forward-hold
```

반영하는 항목은 다음과 같다.

- fee
- slippage
- market impact
- funding
- 매 bar equity 복리 누적
- stale data 2 bar 연속 시 무결성 실패 및 청산

따라서 결과 파일의 terminal equity가 실제 매 bar 수익률 누적 결과와 일치해야 한다.

## 8. L2와 L3 평가

### L2

L2는 전체 ledger를 그대로 평가하지 않고, sealed holdout 시작 전 구간만 평가한다.

주요 지표는 다음과 같다.

- annualized log growth
- growth confidence interval
- equity multiple
- maximum drawdown
- CVaR 95%
- annual volatility
- turnover
- execution integrity

### L3

L3는 최근 sealed holdout 구간을 별도로 평가한다. 파라미터를 다시 튜닝하지 않는다.

- `PROMOTE`: 양의 성장 posterior probability가 0.65 이상
- `SHADOW`: 안전하지만 판단 근거가 부족하거나 불확실
- `REJECT`: 무결성·위험 한계 위반 또는 성장 확률이 매우 낮음

## 9. 최근 실제 실행 결과

실행:

```text
기준일: 2026-07-08
주기: 1h
봉 수: 2,048
종목 수: 2
```

결과:

| 항목 | 결과 |
|---|---:|
| L2 annualized log growth | 0.0 |
| equity multiple | 1.0 |
| MDD | 0.0 |
| annual volatility | 0.0 |
| turnover | 0.0 |
| integrity | 정상 |
| target weight | 전부 0 |
| L3 | `SHADOW` |

결과 파일:

`logs/futures/compound/20260722_235413/result.json`

이 결과를 “전략이 0% 수익을 냈다”고 해석하면 안 된다. 실제로는 L2가 거래를 한 번도 허용하지 않아 손익 분포 자체가 생성되지 않았다.

## 10. 현재 우선순위

현재 구조에서 가장 중요한 개선 순서는 다음과 같다.

1. `resolve_universe_symbols()`를 실제 PIT historical ledger 기반으로 연결한다.
2. symbol 수가 충분한 데이터로 cross-sectional recipe를 재평가한다.
3. L1 forecast variance의 단위와 horizon별 calibration을 검증한다.
4. robust shrink 이후 `support=False`가 되는 비율을 모니터링한다.
5. 실제 포지션이 생긴 뒤에야 L2 성장률·비용·복리 성과를 해석한다.

현재 구조는 legacy pipeline과 Optuna를 거치지 않고 단일 compound 경로만 실행한다. 다만 “코드가 실행된다”와 “자산증식 edge가 검증됐다”는 서로 다른 상태이며, 최신 실측은 아직 후자가 충족되지 않았음을 보여준다.
