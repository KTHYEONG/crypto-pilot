# Binance Futures 전략 아키텍처 (v2.2 - Multi-Sleeve Blending System)

**최종 업데이트**: 2026-05-22
**핵심 설계 목적**: legacy alpha/HMM 없이 순수 결정론적 신호로 백테스트 엔진·로직·유니버스 종합 검증 및 다중 슬리브(XS Reversal, TSMomentum, Carry) 동적 블렌딩 검증.  
이 문서는 복잡한 알파 mining을 목적으로 하지 않으며, 다중 슬리브 신호가 end-to-end로 정상 동작하고 최적화되는지 확인하는 것이 1차 목표다.

---

## 1. 설계 원칙

- **legacy 의존 금지**: `src/domain/futures/legacy/*` 경로를 신규 strategy 코드에서 import하지 않는다.
- **HMM/alpha_factory 미사용**: regime posterior, ML alpha panel 없이 순수 가격·funding 데이터만 사용.
- **기존 파이프라인 재사용**: `signal_composer` → `portfolio_constructor` → `backtest_engine` 경로를 그대로 통과시킨다. 전략 계층은 `alpha_long/alpha_short` 배열만 생산한다.
- **Multi-Sleeve 동적 블렌딩**: XS Reversal(기본 활성), Carry(기본 활성), TSMomentum(기본 비활성) 다중 슬리브를 구성하고, 롤링 Spearman IC 및 t-stat 기반으로 동적 가중치를 산출하여 결합한다.

---

## 2. 실제 실행 흐름 및 주입 지점

```text
[신규 strategy 모듈]
build_strategy_alpha(data_maps, symbols, tf, cfg)
  → 1. compute_multi_alignment_info 기반 가격 패널 정렬
  → 2. 활성 슬리브(XS Reversal, TSMomentum, Carry) raw 신호 산출
  → 3. Cross-sectional Winsorized z-score 및 Spearman IC 산출
  → 4. Dynamic Blending 및 t-stat Gate 평가를 통한 가중치 결합
  → 5. Grinold expected return calibration (to_return_units)
  → alpha_panel: DataFrame[(datetime, symbol), alpha_long / alpha_short]
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

## 4. 다중 슬리브(Multi-Sleeve) 아키텍처 및 세부 로직

### 4.1 XS Reversal Sleeve (기본 활성화)
가장 안정적이고 강건한 평균회귀 신호인 **Cross-Sectional Reversal**입니다.
* **설계 의도**: 최근 N개 바(Closed Bar) 동안의 수익률이 낮았던 하위 심볼을 매수(Long), 높았던 상위 심볼을 매도(Short)합니다.
* **신호 산출**:
  $$\text{rev\_score}[t, i] = -\ln\left(\frac{\text{close}[t, i]}{\text{close}[t - L, i]}\right)$$
  * `L` = lookback_bars (기본값: 6 bars = 24h)
  * `min_symbols_for_xs` = 최소 5개 심볼 이상 존재할 때만 스코어링 활성화.

### 4.2 TS Momentum Sleeve (기본 비활성화)
각 심볼별로 개별적인 시간축 모멘텀(Time-Series Momentum)을 평가합니다.
* **설계 의도**: 백테스트 검증 결과, 4h 해상도 및 1d 해상도 전체 구간에서 음의 IC(4h t=-6.8, 1d t=-2.3)를 기록하여 **기본값 비활성(`False`)**으로 제어합니다.
* **신호 산출**:
  $$\text{ts\_mom\_score}[t, i] = \ln\left(\frac{\text{close}[t - skip, i]}{\text{close}[t - L, i]}\right)$$
  * `L` = ts_momentum_lookback (기본값: 36)
  * `skip` = ts_momentum_skip (기본값: 1)

### 4.3 Carry Sleeve (기본 활성화)
Binance Futures의 8시간 원천 funding rate를 수집하여 Carry-adjusted edge를 포착합니다.
* **설계 의도**: 펀딩 비용 드래그를 상쇄하고 고펀딩 수혜 포지션을 취하기 위해 활성화됩니다.
* **신호 산출**:
  $$\text{carry\_score}[t, i] = \text{rolling\_mean}(\text{funding\_rate\_sum}, \text{smooth\_bars})$$
  * `smooth_bars` = carry_smooth (기본값: 6 bars = 24h)

### 4.4 동적 블렌딩 (Dynamic Blending) 및 Grinold expected return Calibration
* **Dynamic Weighting**: 각 슬리브의 롤링 Spearman IC 윈도우(`ic_window_bars = 180`) 통계를 바탕으로 t-stat가 `min_t_stat = 2.0`, `min_hit_ratio = 0.45` 등의 Hard Gate를 통과한 슬리브에 대해서만 `ic_shrinkage(0.5) * mean_ic` 만큼의 가중치를 배분하여 Dynamic blend를 수행합니다. 
* **Fallback**: 모든 슬리브가 통과하지 못하면 가장 절대값이 높은 `mean_ic`를 지닌 단일 슬리브를 강제 적용(Fallback)합니다.
* **Grinold Calibration**: 블렌딩된 z-score와 예상 선행 변동성(`sigma_lookback = 30`), 롤링 IC 강도를 곱하여 **to_return_units**로 expected return($\alpha_{hat}$) 단위를 캘리브레이션합니다:
  $$\alpha_{hat} = \text{score} \times \sigma_{fwd} \times \text{IC}_{lagged}$$

### 4.5 데이터 의존성

| 필요 데이터 | 경로 | 비고 |
|---|---|---|
| OHLCV 4h | `data_maps[symbol]["4h"]["close"]` | optimizer가 이미 적재 |
| funding_rate | `data_maps[symbol]["4h"]["funding_rate_sum"]` | Carry sleeve 연산용 |

---

## 5. 모듈 구조 및 구현 범위

### 5.1 파일 구조 (XS Reversal / TS Momentum / Carry 완비)

```text
src/domain/futures/strategy/
├── __init__.py
├── builder.py           # build_strategy_alpha() - data_maps -> alpha_panel DataFrame
├── config.py            # StrategyConfig, SleeveConfig, BlendConfig, RegimeConfig 정의
│                        # ⚠ RegimeConfig.enabled=False (regime 모듈 미구현, P2 이후 활성화)
├── combine.py           # blend_sleeves() - 가중치 기반 다중 슬리브 합성 및 Normalization
├── normalize.py         # Winsorized CS z-score, Grinold return calibration(to_return_units)
├── diagnostics.py       # rolling_ic, ic_summary, passes_ic_gate 검증 헬퍼
├── momentum.py          # XS Momentum 원형 (수식 연산 유틸)
└── sleeves/
    ├── base.py          # BaseSleeve 추상 클래스
    ├── xs_reversal.py   # XSReversalSleeve 구현
    ├── ts_momentum.py   # TSMomentumSleeve 구현
    └── carry.py         # CarrySleeve 구현

❌ regime/ 디렉토리: 미구현 (provider.py 없음). bridge.py가 strategy_cfg.regime.enabled를
   확인 후 분기하므로, enabled=False일 때 market_probs=pd.DataFrame()으로 안전하게 스킵됨.
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
| `sleeves/*.py` | 개별 sleeve 신호 산출 추상화 | numpy, pandas |
| `strategy/builder.py` | data_maps 소비 → Multi-sleeve 동적 결합 및 Grinold Calibration 진행 | sleeves, combine, normalize |
| `bridge.py` | strategy 주입 on/off 분기 | builder.py |
| `config.py` | `StrategyConfig` 및 `RegimeConfig` 등의 하이퍼파라미터 정의 | dataclasses |

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

| 파라미터 | 역할 | 권장 탐색 범위 | 검증 실측값 |
|---|---|---|---|
| `BETA_ALPHA` | alpha 신호 스케일 | 2.0 ~ 8.0 | 3.3 ~ 5.4 (momentum_v0) |
| `EV_HURDLE_BPS` | 진입 최소 edge (bps) | 1.0 ~ 20.0 | 12.6~28.5bps에서 xs_nz>0 확인 |
| `REBALANCE_BARS` | 리밸런스 주기 | 4 ~ 8 | - |
| `ATR_MULT` / `TRAIL_MULT` | stop loss 범위 | 1.5 ~ 4.0 | - |
| `KELLY_SHRINKAGE` | Kelly 보수화 | 0.2 ~ 0.5 | - |

> **단위 정합 주의**: Grinold calibrated alpha의 `long_p95 ≈ 11bps` (momentum_v0, 5-sym 기준).
> friction ≈ 12bps이므로 `BETA_ALPHA × alpha_p95 > friction + EV_HURDLE` 조건을 만족해야 xs_score > 0.
> `EV_HURDLE_BPS` 상한이 과도하면 무거래(xs_nz≈0) → zero_trades prune 발생. [COMPOSE-DIAG] 로그로 확인.

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
| P0 | `workflow.py` — strategy phase A1/A2/B 오케스트레이션 | A1/A2/B phase budget 관리 |
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

- strategy mode의 phase 오케스트레이션은 `workflow.py:run_phased_optimization_skeleton`이 담당한다 (`phase_runner.py` 파일명은 미사용 — 구 문서 drift).
- `--trials N`은 A1/A2/B 모두 동일하게 N trials로 배분된다 (`n_trials_a1 = n_trials_a2 = n_trials_b = N`).
  - 최솟값: A1 ≥ 20 (TPE startup), A2 ≥ 40 (BoTorch startup). `--trials 1`은 random 탐색 1회로 수렴 불가.
- phase B는 SQLite/Optuna 병렬 충돌 방지를 위해 단일 worker로 실행한다.
- `No elite components found`는 strategy mode에서 정상일 수 있으며, canonical signal은 `alpha_long/alpha_short`이다.
- `BETA_REGIME_BEAR`, `BETA_REGIME_CHOP` 파라미터는 탐색공간에 존재하나 strategy mode에서 hmm_probs=0이므로 실질적 영향 없음 (legacy 잔재, P2 이후 제거 예정).

### 9.4 미래 결정 사항 (P2 이후)

- HMM/Student-t HMM provider 연결 여부: P0 momentum 결과가 안정적일 때만 평가.
- regime posterior 5-state 구조: P2 설계 시 `regime_contracts.py` 계약을 기준으로 재설계.
- live trading 승격: champion registry pass + AUM ladder 통과 후.

---

## 10. 진단 로깅 태그 체계 (v2.2 추가)

`--mode strategy` 실행 시 아래 태그로 `grep`하여 각 단계의 정상 진행 여부를 확인한다.

| 태그 | 단계 | 파일 | 핵심 출력 항목 |
|---|---|---|---|
| `[STAGE]` | 파이프라인 단계 전이 | `opt_main_futures.py` | window/universe/data/strategy/optimize |
| `[ALPHA-BUILD]` | alpha_panel 산출 통계 | `strategy/builder.py` | sleeves, long_p95_bps, ic_lag_mean, fallback |
| `[ALPHA-MERGE]` | data_maps 병합 결과 | `strategy_runtime/bridge.py` | merged_syms, alpha_long_nz, regime_broadcast |
| `[ALPHA-ALIGN]` | AWF leg 정렬 후 alpha 잔존량 | `optimization/ml_context.py` | leg, bars, alpha_long_nz (첫 3 leg) |
| `[COMPOSE-DIAG]` | alpha→xs_score 단위 정합 | `optimization/objectives.py` | friction_bps, hurdle_bps, thr_bps, xs_long_nz (trial<3) |
| `[LEG]` | per-leg 백테스트 결과 | `optimization/objectives.py` | trades, log_ret, mdd, tw_nz (trial<5) |
| `[PRUNE]` | prune 사유 | `observability/run_tracker.py` | reason, params (trial<5: INFO, 이후 DEBUG) |
| `[RUN-SUMMARY]` | 종료 시 phase별 집계 | `application/.../optimization_service.py` | complete/pruned/failed, prune_reasons top-3 |

### 빠른 진단 명령

```bash
# 1) 단계 도달 확인
grep '\[STAGE\]' run.log

# 2) 신호 생존 확인 ★ 핵심
grep '\[ALPHA-BUILD\]\|\[COMPOSE-DIAG\]' run.log

# 3) leg별 무거래/손실 확인
grep '\[LEG\]' run.log | head -20

# 4) 최종 prune 원인 집계
grep '\[RUN-SUMMARY\]' run.log
```

### 의사결정 트리

```
[COMPOSE-DIAG] xs_long_nz ≈ 0 ?
├─ YES → alpha_l_p95 vs thr 비교
│        ├─ alpha < thr  → BETA_ALPHA↑ 또는 EV_HURDLE↓ 또는 Grinold calibration gain
│        └─ alpha ≥ thr  → signal_composer 합성 버그 점검
└─ NO  → [LEG] tw_nz ≈ 0 ?
         ├─ YES → Kelly/cap/min_notional 단계 문제
         └─ NO  → 거래 발생, log_ret<-0.1 → 신호 방향(IC 부호) 문제
```

**실측 참고값 (momentum_v0, 5-sym, 4h)**:
- `[ALPHA-BUILD]`: long_p95=11.3bps, ic_lag_mean=0.014, ic_neg_ratio=0.35, fallback=True
- `[COMPOSE-DIAG]`: friction=12.2bps, hurdle=12~28bps, xs_long_nz=0.03~0.19
- `[RUN-SUMMARY]`: phase_a1 pruned 12/30 (trial_should_prune:9, zero_trades_first_leg:3)

---

## 11. 적용 대상 파일

| 파일 | 변경 수준 |
|---|---|
| `src/domain/futures/strategy/momentum.py` | 신규 |
| `src/domain/futures/strategy/builder.py` | 신규 |
| `src/domain/futures/strategy/__init__.py` | 신규 |
| `src/domain/futures/strategy_runtime/bridge.py` | 수정 (분기 추가) |
| `src/execution/opt_main_futures.py` | `--strategy` 플래그, `[STAGE]` 로깅 |
