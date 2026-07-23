# 현재 Futures Compound 파이프라인

이 문서는 `src/execution/opt_main_futures.py` 실행 시의 최신 로직을 설명한다.

## 전체 흐름

```text
CLI 설정
  ↓
Data Lake runtime 생성(client + DuckDB catalog)
  ↓
로컬 snapshot 확인 ── 불완전 + 승인 없음 → 종료
  ↓ 승인된 경우에만 Binance sync
Parquet/DuckDB snapshot
  ↓
Daily PIT universe
  ↓
MarketFeatureCube(1m 체결시계 + 필요한 상위 timeframe 파생)
  ↓
L1: 12개 명시 recipe의 causal edge 검증
  ↓
Sparse AlphaEventTape
  ↓
L2: signed growth allocator + chronological simulator
  ↓
L2 walk-forward 평가 + 분리된 180일 L3 holdout
```

## 1. 실행과 데이터 수집

메인 진입점은 `opt_main_futures.py → cli() → run_multiscale_compound_main()` 하나다. 레거시 `run_compound_engine`, `run_compound_main`, Optuna production 경로는 사용하지 않는다.

주요 설정은 기준일, sync mode, universe/data-lake 설정, seed다. Binance 다운로드는 기본 실행하지 않는다. 로컬 cache가 완전하면 그대로 사용하고, 불완전할 때 사용자가 `--allow-network-sync`를 명시한 실행에서만 Vision/REST 수집과 Parquet 저장을 수행한다.

Data Lake는 다음만 canonical로 보존한다.

- broad: exchangeInfo, 모든 historical USDT perpetual 1h, funding event
- selected PIT union: 1m, premium 5m, mark/index 1m, metrics 5m
- 4h·12h·1d 등은 저장하지 않고 실행 시 UTC closed-bar 규칙으로 재생성

저장 규칙은 checksum 검증 → 임시 partition 작성 → atomic commit → DuckDB manifest 기록 순서다. checksum/schema 오류는 quarantine하고, hard cap 초과 시 canonical 데이터를 삭제하지 않고 실패한다. 실제 데이터 수집은 아직 수행하지 않았으며 사용자 승인 후 진행한다.

## 2. Daily PIT Universe

기준일 D의 membership은 D-1까지 공개된 정보만 사용한다. alpha 성과는 universe 순위 입력이 아니다.

기본 hard rule은 다음과 같다.

- USDT perpetual이며 당시 거래 가능
- listing age 90일 이상
- 60일 1h coverage 99% 이상
- 30일 median daily volume 20M USDT 이상
- round-trip cost 25bps 이하
- 최소 20종목, historical union 최대 240종목
- 신규 진입 rank 45 이내, 기존 보유 유지 rank 60 이내

각 날짜의 symbol union을 replay axis로 사용하므로 현재 생존 종목만으로 과거를 재구성하지 않는다. membership 탈락 포지션은 다음 executable 1m bar에서 청산한다.

## 3. MarketFeatureCube와 시간 정합성

`MarketFeatureCube`가 L1/L2의 유일한 typed market 입력이다. 내부 timestamp는 UTC int64 ns로 통일한다.

- observation time이 decision time보다 미래인 값은 사용하지 않는다.
- metrics/premium/funding은 `available_at <= decision_time`인 경우만 join한다.
- 결측 conditional field는 해당 recipe만 무효화한다.
- funding/premium/metrics를 forward-fill하거나 결측을 0으로 바꾸지 않는다.
- signal decision 시점과 체결 시점을 분리하고 첫 체결은 다음 1m bar다.

## 4. L1 신호 생성

현재 catalog는 `family × timeframe`을 기계적으로 곱하지 않고 12개 recipe를 명시한다.

| 계열 | recipe 예시 | 초기 상태 |
|---|---|---|
| time-series trend | 4h/12h/1d | CORE 후보 |
| residual momentum | 4h/12h | CORE 후보 |
| breakout | 4h/12h | 조건부 후보 |
| carry | funding event | 조건부 후보 |
| basis reversion | 1h | shadow |
| taker flow / OI | 15m/1h | shadow·조건부 |
| liquidity exhaustion | 15m | shadow |

각 recipe는 5-fold purged/embargoed causal OOS로 검증한다. 다음을 모두 만족해야 active event를 만들 수 있다.

- 5개 fold 중 최소 4개 net-positive
- net-growth 90% LCB 양수
- 2배 비용 stress에서도 median growth 양수
- `P(net growth > 0) ≥ 0.65`
- fold sign consistency `≥ 0.80`
- FDR `q ≤ 0.10`
- capacity와 5% minute participation 조건 충족

같은 family의 두 timeframe은 residual correlation `≤ 0.60`이고 incremental growth LCB가 양수일 때만 함께 사용한다. timeframe quota나 family quota로 신호를 억지로 채우지 않는다.

L1의 출력은 dense forecast가 아니라 sparse `AlphaEventTape`다. event에는 decision time, first executable time, expiry, hourly log-return rate, variance, frozen combination weight, model/data/fold hash가 들어간다. active event가 없으면 이는 정상적인 `NO_DEPLOYABLE_SIGNAL` 상태다.

## 5. L2 자산배분과 체결

L2에는 여러 정책 경쟁이나 Optuna가 없다. active event를 하나의 allocator가 직접 사용한다.

recipe forecast는 다음처럼 precision 결합한다.

```text
precision = reliability / forecast_variance
combined_mu = precision-weighted mean
combined_variance = 1 / total_precision
robust_mu = sign(mu) × max(|mu| - uncertainty_z × sqrt(variance), 0)
```

allocator 목적함수는 다음을 동시에 본다.

```text
growth = expected signed log-return
       - covariance risk
       - turnover cost
       - turnover regularization
```

long과 short 모두 허용한다. 단, capacity를 `capacity_usdt / NAV`로 환산한 종목별 cap, gross/net exposure, beta, entry block, uncertainty를 함께 적용한다. 후보 비중이 기존 비중보다 비용·불확실성을 충분히 개선하지 않으면 거래하지 않는다.

Simulator는 시간 순서대로 다음을 처리한다.

1. 이전 포지션의 bar 수익
2. 해당 시각의 funding event
3. membership exit 강제청산
4. 새 alpha state 계산
5. 다음 1m bar부터 partial fill·fee·slippage·impact 반영
6. NAV 복리 누적

## 6. L2/L3 평가

L2는 sealed holdout 시작 전 walk-forward ledger만 평가한다. growth, CI, equity multiple, MDD, CVaR95, volatility, turnover, integrity를 기록한다.

L3는 별도 180일 holdout이며 L2 fitting·selection에 사용하지 않는다. L3가 모델이나 threshold를 재학습하지 못하도록 hash와 구간을 고정한다.

## 7. 현재 상태와 다음 작업

코드 계약·strict mypy·pytest·coverage는 최신 check에서 `PASS`, coverage 93%다. 다만 이것은 로컬 구현 검증이며 실제 Binance 데이터 sync를 아직 실행했다는 뜻은 아니다.

다음 순서로 진행한다.

1. 사용자의 승인 후 `--allow-network-sync`로 소규모 dry-run 수집
2. checksum·row count·timestamp 범위·Parquet/DuckDB manifest 확인
3. broad 1h에서 실제 PIT daily universe를 생성하고 종목 수/coverage 보고
4. 해당 historical union에 selected 1m·premium·mark/index·metrics를 수집
5. 새 MarketFeatureCube로 L1 12개 recipe causal 결과를 산출
6. active event 수, L2 support 비율, 실제 거래 수와 비용 차감 성장률을 분석
7. parity와 main smoke가 통과한 뒤에만 기존 `enriched/`, 저장된 4h·8h·1d 파생본 등 중복 데이터를 삭제

현재는 1~7 중 실제 데이터 수집 전 단계다. 데이터 수집과 기존 파일 삭제는 별도 승인 없이는 실행하지 않는다.
