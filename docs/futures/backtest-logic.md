# Binance Futures 백테스팅 아키텍처 (v3.0 - Final, All-Decided)

**확정일**: 2026-05-20
**목적**: **과적합 억제 + 복리 자산 증식 극대화**. 모든 설계 결정은 단일 안으로 확정되었으며, 변경은 governance 결정 + 별도 OOS 재검증 후에만 허용한다.

---

## 0. 설계 원칙 (Locked)

* **우선순위**: Correctness > No-Leakage > Capacity > Compounding > Realism > Speed
* **탐색 score**와 **승격 gate**는 완전 분리한다.
* 중첩 OOS 성과는 리서치 관찰용. **승격 통계는 atomic non-overlap block만 사용**.
* 단일 자본 backtest로 배포 결정 금지. **AUM ladder 다구간 통과 필수**.
* 목적함수의 λ 가중치, hard gate 임계값, Inner AWF K, Atomic block 크기는 **모두 고정 상수**. 튜닝 금지.

---

## 1. 핵심 아키텍처

```text
[Layer A: PIT Data & Universe]
  1h/1m kline · 1m mark price · funding 8h · meta · delisting · snapshot
  knowledge_date 차단 / data_manifest coverage 검증
        │ UniverseSnapshot + Prepared Inputs
        ▼
[Layer B: Fold Scheduler]
  Inner AWF (K=8, IS=24M, leg=3M)         ← optimizer 목적함수
  Outer Rolling OOS (IS=24M/OOS=6M/3M)    ← 리서치 관찰 (승격 비사용)
  Atomic 6M Blocks (non-overlap)          ← 승격용 독립 단위
        │ target_weights / costs / diagnostics
        ▼
[Layer C: Portfolio & Execution]
  Signal seam → Fractional Kelly (×0.25) → gross/net/beta/per-symbol/vol caps
  Coarse 4h (friction pre-charge) → Intrabar 1m (mark_price 청산)
        │ leg metrics / trade ledger / equity
        ▼
[Layer D: Promotion & Registry]
  Hard gates → DSR → Intrabar dual-decay → AUM ladder → Champion 비교
```

### 1.1 모듈 매핑
| 파일 | 책임 |
|---|---|
| `backtest_preparation.py` | 1h/4h/1m/mark/funding/kill/universe 정렬 |
| `portfolio/portfolio_constructor.py` | seam 통합, 0.25x Kelly weight, 5 caps 투영 |
| `portfolio/execution_sim.py` | Numba intrabar/coarse 코어, mark-price 청산 |
| `optimization/optimizer.py` | Inner AWF, score/DSR/penalty |
| `optimization/evaluator.py` | 통계 평가 유틸 |
| `optimization/final_evaluator.py` | 최종 WF 재평가, champion 비교 |
| `validation/walk_forward.py` | atomic block 집계, hard gate 판정 |
| `universe/*` | snapshot, cost/capacity 기초 |
| `execution/opt_main_futures.py` | 전체 orchestration |

---

## 2. 데이터 계약

### 2.1 시간 해상도
* **Decision**: 4h (UTC closed bar) — 신호 산출 단위
* **Execution**: 1m — 체결/청산/펀딩 정밀 처리
* **Base grain**: 1h — feature 1차 가공

### 2.2 배열 계약
| 변수 | shape | 비고 |
|---|---|---|
| `close_2d` | `[B_4h, N]` | 4h 종가 |
| `target_weights_2d` | `[B_4h, N]` | 리밸런스 목표 비중 |
| `exec_o/h/l/c_1m` | `[B_1m, N]` | Vision `klines` |
| `mark_price_1m` | `[B_1m, N]` | **격리 청산 기준 (HARD)**. Vision `premiumIndexKlines`. P0-data |
| `funding_event_mask_1m` | `[B_1m, N]` | 8h 펀딩 이벤트 |
| `funding_rate_1m` | `[B_1m, N]` | 이벤트 시점 펀딩비 |
| `kill_signal_2d` | `[B_4h, N]` | 상폐/결측/강제 제외 |
| `sigma_3d` | `[B_4h, N, N]` | rolling covariance |

### 2.3 결측 바 정책
* 단일 결측: 직전 종가 forward-fill, **volume=0** (거래 불가)
* 연속 2봉 이상: `kill_signal=1` → 해당 leg에서 제외
* `data_manifest.parquet` coverage가 **단일 진실 소스**. 다른 fallback 금지.

### 2.4 Look-ahead 차단
* 신호 산출 `t`, 체결 `t+1` (4h 단위)
* `knowledge_date > as_of` 정보 전면 차단
* NaN/Inf 발생 심볼은 해당 바에서 neutral 또는 entry skip
* tick/step_size 는 현재 `exchangeInfo` 값 사용 (역사 이력 없음, 안정성 가정 명시)

---

## 3. Walk-Forward 계층 (확정값)

### 3.1 Inner AWF — 목적함수 평가
* **IS = 24M, K = 8 (leg = 3M)**
* 산출: `leg_log_TW[8]`, `worst_leg`, `mean_leg`, `MDD`, `turnover`, `funding_drag`, `EV/Cost`
* **이유**: K=6 은 binomial 잡음으로 false-negative 과다(P(X≥5|p=0.7)≈0.42), K=8 은 4-quarter 사이클을 leg 단위로 완전히 커버하면서 표본 분산 28% 감소.

### 3.2 Outer Rolling OOS — 리서치 전용
* `IS=24M, OOS=6M, step=3M` (50% overlap)
* **용도 한정**: drift 관찰, 파라미터 열화 진단, 리포트
* **승격 통계 직접 사용 금지** (double-counting)

### 3.3 Atomic OOS Blocks — 승격 단위
* **Non-overlapping 6M block (확정, 3M 폐기)**
* 2019-09 ~ 현재 기준 약 11 blocks 확보 가능
* 산출: `pass_ratio`, `median_log_growth`, `worst_block_mdd`, `capacity_pass_ratio`
* **이유**: 3M block 은 분기 단발 잡음에 knife-edge 판정 발생. 6M 은 운영 반기 cycle과 정합하며 표본 독립성·통계력 균형.

---

## 4. Boundary Contract (확정)

### 4.1 공식
```
boundary_purge_bars = max(
  label_horizon_bars,
  meta_label_horizon_bars,
  stateful_fit_leakage_bars,
  execution_delay_bars
)
```

### 4.2 Seam 등록 의무 (NEW)
* 모든 signal/feature 모듈은 `meta.purge_bars: int` 를 엔진에 **반드시 등록**한다.
* 미등록 모듈은 backtest 진입 거부 (fail-fast). `default=0` fallback 금지.

### 4.3 적용
* Inner AWF train pool 말단 ↔ test leg 시작
* Outer Rolling OOS IS→OOS 경계
* 단순 rolling indicator history 자체는 purge 대상 아님 (fit된 파라미터·미래 지식 전파만 대상)

---

## 5. 실행 정밀도 (확정)

### 5.1 단계 분리
| 단계 | 모드 | 목적 |
|---|---|---|
| Inner AWF 탐색 | Coarse 4h | 수백 trial 탐색, 속도 우선 |
| Candidate 재평가 | Coarse + friction pre-charge | 비용 과소추정 1차 차단 |
| Champion 게이트 | Intrabar 1m | mark-price 청산·갭·펀딩 현실 검증 |

### 5.2 Coarse pre-charge (bookDepth 연동)
* `taker_fee + virtual_spread + funding_drag_proxy + latency_buffer` 사전 차감
* `virtual_spread`: 2020+ Vision `bookDepth` median(ask−mid), 2019 Corwin-Schultz fallback

### 5.3 Intrabar 규약
* **격리 청산은 `mark_price_1m` 기준 — HARD**. `exec_low_1m` 대리 금지.
* 펀딩 정산은 8h 이벤트 시 **해당 1m 바 시작 시점에 보유 중인 포지션에만** 적용.
* 1m 경로 기반 stop/gap/liquidation/funding 처리 필수.
* `mark_price_1m` 미적재 시 champion 후보 자격 미부여.

### 5.4 Dual Decay Gate
* `coarse_CAGR > 0` 일 때: `percent_decay = (intrabar_CAGR − coarse_CAGR) / coarse_CAGR`
* 항상: `absolute_decay_bps_yr = (intrabar_CAGR − coarse_CAGR) × 10000`

**판정 (둘 다 통과)**:
| 조건 | 기준 |
|---|---|
| `percent_decay >= -15%` | coarse_CAGR > 0 시에만 적용 |
| `absolute_decay_bps_yr >= -500` | 모든 경우 적용 (연 5%p 손실 한계) |

---

## 6. 목적함수 및 Hard Gate (확정)

### 6.1 Score (λ 동결)
```
score = mean(log_TW_legs)
      - 0.50 * downside_semidev(log_TW_legs)
      - 1.00 * worst_MDD
      - 0.30 * CVaR_5
      - 0.20 * excess_turnover
      - 0.50 * funding_drag
      - 0.40 * AUM_impact_penalty
```
**6개 가중치는 고정 상수.** 변경 시 별도 거버넌스 + OOS 재검증 필요. **튜닝 목록에 포함 금지.** 이유: λ 튜닝은 meta-overfitting 최대 함정.

### 6.2 Hard Gate (compound-growth aware)
| 기준 | 확정값 | 변경 이유 |
|---|---|---|
| `min_positive_leg_ratio` | **0.55** | 0.70은 모멘텀류를 binomial 잡음으로 부당 탈락 |
| `worst_leg_tw_floor` | **0.85** | 0.95는 코인 변동성 본질과 충돌, 트렌드 전략 배제 |
| `mean_leg_tw_floor` (3M leg) | **1.015** | 분기당 +1.5% (연 6%) 최소 edge 요구 |
| `ergodicity_guideline_pct` | **15%** | ensemble↔time-avg 괴리 상한 유지 |
| `EV/Cost floor` | **3.0** | 비용 대비 edge 3배 |
| `DSR floor` | **0.60** | 다중검정 통과선 명시 |
| `funding_drag ceiling` | `drag / gross_return <= 0.30` | 펀딩이 수익 30% 이상 잠식 시 reject |
| `capacity_pass_ratio` | 50k/100k/250k **전부 pass** | deployment target zone |

### 6.3 Ergodicity 처리 (2단)
* 1차: normalized penalty (score 항이 아닌 별도 항)
* 2차: 15% hard gate (deviation = |log(ensemble_mean) − mean(log_TW_legs)|)

---

## 7. DSR 및 다중검정 제어

### 7.1 Trial Signature
```
[awf_leg_log_TW_1..8, turnover_cost_ratio, funding_drag_ratio, worst_leg_mdd]
```
(K=8 → 11차원 벡터)

### 7.2 n_trials_eff (entropy effective rank)
```
λ_i = eigenvalues(corr(signatures))
p_i = λ_i / Σλ_i
n_trials_eff = exp(- Σ p_i log p_i)
```

### 7.3 Pruned trial 처리 (확정)
* 완료된 leg 만 signature 에 사용, 미완료 leg 는 **cross-sectional median**으로 impute
* trial weight = `completed_legs / 8`
* pruning 자체로 가설 수 부풀리기 방지

---

## 8. Cost & Capacity (확정)

### 8.1 비용 모델
```
total_roundtrip_cost_bps =
  fee_bps              # 0.04% taker + maker_share × (maker_rebate - taker_fee)
  + spread_bps         # bookDepth (2020+) / Corwin-Schultz (2019)
  + impact_bps         # square-root, k=0.5
  + tick_cost_bps      # step_size 양자화 손실
  + latency_buffer_bps # 0.5 bps 고정
  + funding_proxy_bps  # 4h 단위 평균 펀딩 환산
```
* `maker_share = 0.5` (default, signal 카테고리별 override 허용)

### 8.2 Impact (k 고정)
```
impact_bps = 0.5 * sigma_1d * sqrt(order_notional / ADV_30d) * 10000
```
* k = 0.5 default. **월 1회 intrabar 실측 대비 calibration** 으로 갱신 (Execution Calibration Module).

### 8.3 AUM Ladder
```
AUM = [10k, 50k, 100k, 250k, 500k] USDT
```
* **승격 필수 통과**: 50k, 100k, 250k **3개 전부 pass**
* 10k, 500k 는 sanity (단일 실패 허용)
* `capacity_ceiling` = 첫 fail 직전 AUM
* 운영 자금 도달 시 `[1M, 2M]` 확장

### 8.4 minNotional / Step Quantization (NEW)
```
qty = floor(target_weight * equity / (price * step_size)) * step_size
notional = qty * price
if notional < minNotional_usdt:  # default 20 USDT
    qty = 0
```
* 잘린 잔여 비중은 다음 리밸런스에 흡수 (보유 현금 추적)
* 모든 AUM tier에서 동일 양자화 적용 → 10k tier 에서 small-cap 알트는 자동 배제

---

## 9. 포트폴리오 제약 및 리스크 오버레이

### 9.1 Caps (확정값)
| 제약 | 값 | 의미 |
|---|---|---|
| `gross_exposure_cap` | 3.0 | 총 절대 노출 |
| `per_symbol_cap` | 0.10 | 단일 자산 10% |
| `net_exposure_cap` | ±0.30 | 일방향 쏠림 |
| `beta_exposure_cap` | ±0.50 | BTC beta |
| `target_ann_vol` | 0.20 | portfolio vol target |

### 9.2 Fractional Kelly (확정)
* **Kelly weight × 0.25 (quarter Kelly)**
* 이유: full Kelly 는 코인 변동성 환경에서 path-dependent 파산확률 과다. 0.25x 가 ergodicity hard gate 와 정합.

### 9.3 Drawdown Overlay (2단)
| 트리거 | 액션 |
|---|---|
| rolling 30d loss > 10% | `gross_cap × 0.7` |
| rolling 30d loss > 15% | `gross_cap × 0.4` |
| recovery: rolling 30d loss < 5% | 단계적 복귀 (한 단계/주) |

---

## 10. Universe 연동

### 10.1 Snapshot 규칙
* Quarterly rebuild (UTC 분기 첫 거래일)
* 진입 필터: `listing_age_days >= 90`, `vol_30d <= 400%`, `adv_usdt_median >= 25M`, `funding_zscore` 이상치 제외
* `UniverseSnapshot(as_of)` 영속화 (사후 재구성 금지)

### 10.2 Mid-period 멤버십 변경
* 신규 진입: snapshot 교체 후 + 자체 `listing_age >= 90`
* 퇴출: 직전 4h close 에서 `target_weight=0`
* 상폐/kill: 다음 1m open 시장가 강제 청산
* **신규 universe 진입 시 첫 30 decision bars (≈5일) warm-up** — 신호 산출만, 진입 금지

### 10.3 OI/ADV Crowding 필터 (조건부 활성)
* `fetch_metrics_daily` 구현 완료 후 활성: `oi_usdt_median / adv <= 12.0`
* 미활성 기간: `vol_30d` + `funding_zscore` 간접 커버 (수용 결함, P1-data 우선순위)
* 2019 ~ 2020-08 구간: OI 필터 영구 비활성 (데이터 없음)

### 10.4 AUM-aware Universe (P2 deferred)
* **초기 운영: 단일 universe (500k tier 기준)**
* 운영 1년 후 tier별 분리 활성화 — 그 전에는 활성화 금지

---

## 11. Champion 승격 (확정 흐름)

### 11.1 Sequential Gate
```
1. Inner AWF 통과 (§6.2 hard gates ALL)
2. Outer Rolling OOS 명백한 붕괴 없음 (참고)
3. Atomic 6M blocks: pass_ratio >= 70% (8/11 이상)
4. Intrabar 1m: TW > 1.0, dual decay 통과, MDD < hard limit
5. AUM ladder: 50k/100k/250k 전부 pass
6. 기존 champion 비교 (§11.2 우선순위)
7. Registry 승격 또는 보류
```

### 11.2 Champion 비교 우선순위 (확정)
1. `atomic_oos_pass_ratio`
2. `capacity_ceiling`
3. `median_log_growth`
4. `worst_block_mdd`
5. `intrabar absolute_decay_bps_yr`

단일 CAGR 비교 금지. 동률 시 `DSR` tie-break.

---

## 12. 필수 리포트

### 12.1 성과
CAGR · log_growth · MDD · Calmar · Sortino · DSR

### 12.2 실행력
EV/Cost · turnover_cost_ratio · funding_drag_ratio · percent_decay · absolute_decay_bps_yr

### 12.3 구조 안정성
positive_leg_ratio · worst_leg_TW · ergodicity_deviation · atomic_oos_pass_ratio

### 12.4 Capacity
AUM ladder pass/fail · capacity_ceiling · marginal_impact_slope

### 12.5 Universe
snapshot_hash · median_ADV · median_execution_cost · forced_dropout_rate

---

## 13. Alpha / HMM Seam (요약)

* 모든 입력은 `precompute_rebalance_weights()` 이전에 정규화되어 들어온다.
* 엔진은 alpha/HMM 내부 학습을 모른다. 계약은 (시점 정합성, NaN 규약, 단위, gross/beta/capacity 연결, `purge_bars` 등록) 만.
* HMM 상태 전이 학습은 trial 탐색 공간에 포함 금지. regime-aware gross damp / crisis override 정책 입력으로 한정.

---

## 14. 추가 전문 모듈 (확정 우선순위)

| 모듈 | 우선순위 | 비고 |
|---|---|---|
| Execution Calibration Loop | P1 | k_impact, spread coeff 월간 재보정 |
| No-Trade Buffer / Turnover Optimizer | P1 | `delta_w < 2 * cost_bps` 시 거래 생략 |
| Funding / Basis Carry Sleeve | P2 | `premiumIndexKlines` 적재 후 |
| Open Interest / Crowding Risk | P1-data | `daily/metrics` 적재 후 |
| Tail Risk Overlay (CVaR, gap stress) | P2 | LUNA/FTX replay block |
| Regime-Conditional Capacity | P2 | BULL/CHOP/CRISIS 별 capacity |
| Capacity Frontier Engine | P1 | ladder 5포인트 + sqrt-fit |
| Beta / Market Exposure Controller | P0 | §9.1 5 caps 의 beta_cap 구현 |

---

## 15. 데이터 수집 전제조건

| 기능 | 데이터 | Vision 경로 | 상태 | 미구현 시 대안 |
|---|---|---|---|---|
| 격리 청산 mark price | `mark_price_1m` | `daily/premiumIndexKlines/` | **미구현 (P0)** | `exec_low_1m` 대리 — 정밀도 저하 인지 |
| basis_z_score | premiumIndex | `daily/premiumIndexKlines/` | **미구현 (P2)** | `vol_30d` 상한 부분 커버 |
| OI/ADV crowding | `sum_open_interest` | `daily/metrics/` (2020-09~) | **미구현 (P1)** | vol + funding_zscore |
| spread_bps (2020+) | bookDepth | `daily/bookDepth/` | ✅ 활성 | — |
| OHLCV (2020~) | klines | `daily/klines/` | ✅ 활성 | — |
| OHLCV (2019) | CCXT | API | ✅ 동결 parquet | — |
| funding rate | fundingRate | FAPI + Vision monthly | ✅ 활성 | — |
| 상폐/onboardDate | exchangeInfo | FAPI | ✅ (645/731) | Vision S3 첫 파일 보완 |
| tick/step size | exchangeInfo | FAPI | ⚠ 현재값만 | 안정성 가정, 2020 이전 불확실성 명시 |

### 15.1 데이터 기간별 백테스트 제약
| 구간 | 제약 |
|---|---|
| 2019-09 ~ 2019-12 | CCXT OHLCV, Roll spread, OI 없음, basis 없음 |
| 2020-01 ~ 2020-08 | bookDepth spread 가용, OI 없음 |
| 2020-09 ~ | OI/metrics 가용 (downloader 구현 후) |

### 15.2 실행 자격 규칙
* **champion 승격은 `mark_price_1m` 적재 완료 이후에만 가능**.
* 그 전까지는 후보를 "intrabar 참고치" 상태로 보관, mark_price 적재 후 재검증.

---

## 16. 구현 우선순위 (확정)

| P | 작업 |
|---|---|
| **P0-data** | `fetch_premiumindex_daily` → `mark_price_1m` 적재 |
| P0 | Atomic 6M blocks + champion 승격 로직 분리 |
| P0 | Boundary contract seam interface (`purge_bars` 등록 의무) |
| P0 | AUM ladder capacity gate (5 tiers, 3-tier 필수 pass) |
| P0 | minNotional/stepSize 양자화 (execution_sim) |
| P0 | Hard gate 재조정 (0.55 / 0.85 / 1.015 / 0.60 DSR) |
| P0 | beta/net exposure cap + Fractional Kelly 0.25x |
| P1 | Coarse friction pre-charge (bookDepth half-spread 연동) |
| P1 | `n_trials_eff` entropy effective rank |
| P1 | Dual decay gate (percent + absolute) |
| P1 | DD overlay 2-단계 (10% / 15%) |
| **P1-data** | `fetch_metrics_daily` → OI/ADV crowding 필터 복구 |
| P1 | Execution Calibration Loop (월간 k_impact 재보정) |
| P2 | AUM-aware universe snapshot |
| P2 | Regime-Conditional Capacity |
| P2 | Tail Risk Overlay (CVaR, gap stress, LUNA/FTX replay) |

---

## 17. 적용 대상

* `src/execution/opt_main_futures.py`
* `src/domain/futures/backtest_preparation.py`
* `src/domain/futures/portfolio/portfolio_constructor.py`
* `src/domain/futures/portfolio/execution_sim.py`
* `src/domain/futures/optimization/optimizer.py`
* `src/domain/futures/optimization/evaluator.py`
* `src/domain/futures/optimization/final_evaluator.py`
* `src/domain/futures/validation/walk_forward.py`
* `src/domain/futures/universe/*`
