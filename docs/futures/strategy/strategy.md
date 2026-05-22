# Binance Futures 전략 아키텍처 (v2.0 - Minimal Viable Strategy)

**최종 업데이트**: 2026-05-22
**핵심 설계 목적**: legacy alpha/HMM 없이 순수 결정론적 신호로 백테스트 엔진·로직·유니버스 종합 검증.  
이 문서는 복잡한 알파 mining을 목적으로 하지 않는다. 기존 파이프라인이 end-to-end로 정상 동작하는지 확인하는 것이 1차 목표다.

---

## 1. 설계 원칙

- **legacy 의존 금지**: `src/domain/futures/legacy/*` 경로를 신규 strategy 코드에서 import하지 않는다.
- **HMM/alpha_factory 미사용**: regime posterior, ML alpha panel 없이 순수 가격·funding 데이터만 사용.
- **기존 파이프라인 재사용**: `signal_composer` → `portfolio_constructor` → `backtest_engine` 경로를 그대로 통과시킨다. 전략 계층은 `alpha_long/alpha_short` 배열만 생산한다.
- **P0 범위 최소화**: 단일 sleeve(XS momentum). P1에서 carry 추가. 다른 sleeve는 검증 후 결정.

---

## 2. 실제 실행 흐름 및 주입 지점

```text
[신규 strategy 모듈]
build_strategy_alpha(data_maps, symbols, tf)
  → alpha_panel: DataFrame[(datetime, symbol), alpha_long / alpha_short]
                  (simple return per bar 단위, strategy mode only)
         │
         ▼
[strategy_runtime/bridge.py]
run_ml_pipeline_for_universe(...)
  → MLPipelineOutput(alpha_panel=alpha_panel)
         │
         ▼
[optimization/optimizer.py]
raw_full["alpha_long"] / raw_full["alpha_short"]  (per-symbol)
  → strategy mode에서는 trial params(BETA_ALPHA, EV_HURDLE_BPS, REBALANCE_BARS)
    로 xs_score_long / xs_score_short를 trial-time 생성
         │
         ▼
[portfolio/signal_composer.py]
apply_linear_signal_composer_scores(df, alpha_long, alpha_short, params)
  → xs_score_long, xs_score_short  (friction 차감 + EV hurdle 적용 후)
         │
         ▼
[portfolio/portfolio_constructor.py]
precompute_rebalance_weights(close_2d, xs_long, xs_short, ...)
  → target_weights_2d [B4h, N]
         │
         ▼
[portfolio/execution_sim.py]  (Numba intrabar)
backtest_target_weights_intrabar_numba(...)
  → equity_curve, trades_df
```

**핵심 규칙**: 신규 전략은 `alpha_long`/`alpha_short` 생산에만 집중한다.  
friction 차감, EV hurdle, Kelly scaling, cap projection, 체결 시뮬레이션은 기존 모듈이 처리한다.
strategy mode에서는 `alpha_long_00` 같은 legacy alias를 사용하지 않는다.

---

## 3. alpha_panel 데이터 계약

### 3.1 인덱스 및 컬럼

| 항목 | 규격 |
|---|---|
| index | `(datetime UTC, symbol str)` MultiIndex, monotonic increasing |
| 해상도 | 4h decision bar 기준 closed bar |
| `alpha_long` | `float64`, simple return per bar 단위. strategy mode의 canonical long edge. NaN은 0.0 처리. |
| `alpha_short` | `float64`, 동일 단위. strategy mode의 canonical short edge. NaN은 0.0 처리. |

### 3.2 lookhead 금지 규칙

- 시점 `t`의 alpha는 `t` bar 종가까지만 사용한다.
- 체결은 `t+1` bar open에서 이루어진다(optimizer/backtest_engine이 보장).
- 스케일러, 랭크 정규화는 rolling window 내부에서만 fit한다.

### 3.3 quick-backtest와의 관계

`--quick-backtest` 플래그는 alpha_panel을 비워서(`MLPipelineOutput()`) 중립(zero) 신호로 동작한다.  
strategy mode는 `--strategy momentum_v0`처럼 명시적으로 진입하며, `bridge.py`가 alpha_panel을 주입한다.

---

## 4. P0 전략: XS Momentum Sleeve

### 4.1 설계 의도

가장 단순하고 검증된 신호: **cross-section 모멘텀**. 특정 period의 수익률 상위 심볼 long, 하위 심볼 short.

- legacy 의존 없음
- 파라미터 2개(lookback bars, top/bottom ratio)
- 신호 direction이 명확하여 결과 해석이 쉬움
- 비용 차감 후 EV가 양수인지 바로 확인 가능

### 4.2 신호 산출 수식

```text
모멘텀 score (심볼 i, 시점 t):
  mom_score[t, i] = simple_return(close[t, i], close[t - L, i])

strategy mode에서는 mom_score를 심볼별 edge로 유지하고, cross-sectional rank/threshold로
상위 tail은 `alpha_long`, 하위 tail은 `alpha_short`로 분리한다.
결측/NaN 심볼은 제외하며 해당 alpha는 0.0 처리한다.
```

- `L` = lookback_bars (권장 초기값: 6 bars = 24h, 18 bars = 3d 두 가지로 비교)
- `top_ratio` = 상위 비율 (권장: 0.3), `bottom_ratio` = 하위 비율 (권장: 0.3)
- 출력 단위: per-bar simple return edge. trial-time composer가 `BETA_ALPHA`와 `EV_HURDLE_BPS`를 적용해
  `xs_score_long/short`를 생성한다.

### 4.3 데이터 의존성

| 필요 데이터 | 경로 | 비고 |
|---|---|---|
| OHLCV 4h | `data_maps[symbol]["4h"]["close"]` | optimizer가 이미 적재 |
| funding_rate | `data_maps[symbol]["4h"]["funding_rate_sum"]` | carry sleeve 추가 시(P1) 사용 |

**신규 외부 데이터 불필요.** optimizer가 준비한 `data_maps`를 그대로 소비한다.

---

## 5. 모듈 구조 및 구현 범위

### 5.1 신규 파일 (P0)

```text
src/domain/futures/strategy/
├── __init__.py
├── momentum.py          # XS momentum alpha_long/alpha_short 산출
└── builder.py           # build_strategy_alpha() - data_maps → alpha_panel DataFrame
```

수정 대상 파일:
```text
src/domain/futures/strategy_runtime/bridge.py
  run_ml_pipeline_for_universe() 내부에서 builder.build_strategy_alpha() 호출
  → MLPipelineOutput(alpha_panel=alpha_panel) 반환
```

### 5.2 각 파일 책임

| 파일 | 역할 | 외부 의존 |
|---|---|---|
| `strategy/momentum.py` | closed bar에서 XS momentum score 산출 | numpy, pandas |
| `strategy/builder.py` | data_maps 소비 → alpha_panel DataFrame 조립 | momentum.py |
| `bridge.py` (수정) | strategy 주입 on/off 분기 | builder.py |

### 5.3 금지 사항

- `src/domain/futures/legacy/*` import 금지
- `src/domain/futures/alpha_factory/*` import 금지
- `src/domain/futures/ml_pipeline/*` import 금지
- HMM prob 컬럼(`hmm_prob_*`) 생성 금지 — signal_composer는 해당 컬럼 부재 시 자동으로 0처리함

---

## 6. signal_composer 동작 확인 (연결 계약)

signal_composer는 alpha_long/alpha_short 외 HMM prob 컬럼이 없으면 자동으로 `regime = 0` 으로 처리한다.  
즉 `REGIME_POLICY_ENABLED=False`(기본값) 상태에서:

```text
mu_long[t]  = beta_alpha * alpha_long[t]  - friction
mu_short[t] = beta_alpha * alpha_short[t] - friction

xs_long[t]  = mu_long[t]  if mu_long[t]  >= ev_hurdle else 0
xs_short[t] = mu_short[t] if mu_short[t] >= ev_hurdle else 0
```

strategy mode에서는 `optimizer.py`가 trial-time으로 `alpha_long/short -> xs_score_long/short`
를 생성하고, legacy `alpha_long_00` alias는 사용하지 않는다.

P0에서 조정 가능한 optimizer 파라미터:

| 파라미터 | 역할 | 권장 초기 범위 |
|---|---|---|
| `BETA_ALPHA` | alpha 신호 스케일 | 4.0 ~ 8.0 |
| `EV_HURDLE_BPS` | 진입 최소 edge (bps) | 1.0 ~ 3.0 |
| `REBALANCE_BARS` | 리밸런스 주기 | 4 ~ 8 |
| `ATR_MULT` / `TRAIL_MULT` | stop loss 범위 | 1.5 ~ 4.0 |
| `KELLY_SHRINKAGE` | Kelly 보수화 | 0.2 ~ 0.5 |

---

## 7. 평가 기준 (P0 최소 통과 기준)

신규 strategy 모듈의 첫 검증은 **backtest-engine.md 섹션 7의 기존 hard gate**를 그대로 통과하면 된다.  
sleeve 단위 IC/EV 분해, regime별 성능 분해는 P1에서 추가한다.

### 7.1 엔진 검증 목적 최소 기준

| 지표 | 조건 | 목적 |
|---|---|---|
| RuntimeError 없이 완주 | 필수 | 파이프라인 정상 동작 확인 |
| `positive_leg_ratio >= 0.5` | 권장 | 신호 방향성이 랜덤보다 나은지 |
| `ev_cost_ratio >= 1.0` | 권장 | 비용 차감 후 양의 edge |
| `funding_drag <= 0.30` | 권장 | funding cost 과부하 아님 |
| `MDD <= 60%` | 권장 | 레버리지/stop 세팅 적절 |

### 7.2 neutral baseline 비교

`--quick-backtest`(zero alpha) 결과를 baseline으로 먼저 실행한 뒤, momentum 전략 결과와 비교한다.  
momentum이 baseline보다 turnover_adjusted 성과가 개선되지 않으면 신호 품질을 재검토한다.

---

## 8. 구현 우선순위

| 우선순위 | 작업 | 산출물 |
|---|---|---|
| P0 | `strategy/momentum.py` — XS momentum 산출 | `alpha_long/alpha_short [B4h, N]` |
| P0 | `strategy/builder.py` — data_maps → alpha_panel 조립 | `alpha_panel DataFrame` |
| P0 | `bridge.py` 수정 — strategy 주입 분기 | `MLPipelineOutput(alpha_panel=...)` |
| P0 | `optimizer.py` 수정 — trial-time `alpha -> xs_score` 생성 | strategy mode 전용 composer path |
| P0 | `phase_runner.py` 수정 — strategy phase_ranges 상속 | A1/A2/B 동일 범위 보장 |
| P0 | `opt_main_futures.py` 수정 — phase budget/worker 고정 | `--trials` 일관성, phase B 단일 worker |
| P0 | `--quick-backtest` baseline vs momentum 비교 실행 | 검증 결과 (CAGR/MDD/EV·Cost) |
| P1 | funding carry sleeve 추가 (`alpha_short` 보완) | carry-adjusted alpha |
| P1 | 복수 lookback blending (6bar + 18bar) | IC 가중 blend |
| P1 | rolling IC 모니터링 harness | sleeve IC/OOS decay report |
| P2 | rule-based regime multiplier 추가 (portfolio 독립) | drawdown/vol 기반 gross scale |
| P2 | 5-sleeve 구조로 확장 | trend/reversal/carry/flow/defensive |

---

## 9. 기존 코드 전환 계획

### 9.1 유지 (현재 active, 신규 strategy가 소비)

- `portfolio/portfolio_constructor.py` — Kelly/cap/quantization 소비 구조
- `portfolio/signal_composer.py` — friction/EV hurdle 처리 (no HMM multiply 상태 유지)
- `portfolio/execution_sim.py` — Numba intrabar 체결 시뮬레이터
- `optimization/optimizer.py` — trial/awf 오케스트레이션
- `ml_pipeline/regime/regime_contracts.py` — canonical regime prob 스키마 (P2 이후 참조용)

### 9.2 P0에서 건드리지 않을 것

- `legacy/*` — 읽기·import 모두 금지. 참조 필요 시 코드를 독립 재구현.
- `alpha_factory/*` — 동일. shim이므로 사실상 legacy와 동일 취급.
- `ml_pipeline/regime/hmm_inferrer.py` 등 HMM 모듈 — P2까지 동결.
- `signal_composer.py` 로직 수정 — P0에서 파라미터 조정만 허용, 로직 변경 금지.

### 9.3 현재 운영 규칙

- strategy mode의 phase ranges는 `phase_runner`가 `base_ctx.phase_ranges`를 A1/A2/B에 상속한다.
- `--trials`는 A1/A2/B의 공통 phase budget으로 사용한다.
- phase B는 SQLite/Optuna 병렬 충돌 방지를 위해 단일 worker로 실행한다.
- `No elite components found`는 strategy mode에서 정상일 수 있으며, canonical signal은 `alpha_long/alpha_short`이다.

### 9.4 미래 결정 사항 (P2 이후)

- HMM/Student-t HMM provider 연결 여부: P0 momentum 결과가 안정적일 때만 평가.
- regime posterior 5-state 구조: P2 설계 시 `regime_contracts.py` 계약을 기준으로 재설계.
- live trading 승격: champion registry pass + AUM ladder 통과 후.

---

## 10. 적용 대상 파일

| 파일 | 변경 수준 |
|---|---|
| `src/domain/futures/strategy/momentum.py` | 신규 |
| `src/domain/futures/strategy/builder.py` | 신규 |
| `src/domain/futures/strategy/__init__.py` | 신규 |
| `src/domain/futures/strategy_runtime/bridge.py` | 수정 (분기 추가) |
| `src/execution/opt_main_futures.py` | 필요시 `--strategy` 플래그 추가 |
