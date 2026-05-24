# ML 전략(`ml_lambdamart_v1`) Alpha 품질 & Feature Engineering 개선안

> 작성일: 2026-05-23
> 대상: 횡단면 ML 전략 알파 파이프라인 (`src/domain/futures/strategy/`)
> 배경: Anchored per-leg refit 구조 버그 수정 후 `zero_trades_first_leg` 전량 prune 문제는 해소되었으나,
> IS holdout log return이 약 `-0.0002 ~ -0.004` 수준으로 음수에 머물러 `FUTURES_AWF_MU_LOG_MIN=0.0` 배포 게이트 미통과.
> 본 문서는 **방향 1(Alpha 품질)** 과 **방향 2(Feature Engineering)** 를 코드 레벨로 심층 검토한 개선안이다.

---

## 0. 현황 진단 (측정 기반)

### 0.1 파이프라인 구조 요약

| 단계 | 파일 | 핵심 동작 |
|---|---|---|
| Feature | `features.py:41` | 50개 feature 텐서 `[T, N, F]` 생성 |
| Label | `labels.py:35` | t+1 진입, `label_horizon_bars=6`(24h) forward log-ret, **GROSS(cost=0)**, funding 차감 |
| Long matrix | `dataset.py:63` | `[T,N,F]` → LightGBM 행렬, **double-weight** `w·(1+2|y_ev|)` |
| Ranker | `ranker.py:41` | **CS-demeaned LGBMRegressor (RMSE)** — 명칭과 달리 LambdaMART 아님 |
| Calibrator | `calibrator.py:79` | q10/q50/q90 Quantile regression (rank_score를 추가 feature로) |
| Conservative EV | `calibrator.py:146` | `q50 − λ·downside`(long) / `q50 + λ·upside`(short), λ 동적 |
| Cost barrier | `ml_builder.py:214` | `2×(5+2)=14bps` 미만 시그널 0으로 마스킹 |

### 0.2 측정된 증상 (run `ml_opt_fixed_20260523_234235.log`)

```
[ML-FEATURE] rows=6576 symbols=37 features=50
[ML-LABEL]   eligible=0.2870 sample_weight_mean=1.9723
[ML-ALPHA]   long_nz=0.0137 short_nz=0.0088 long_p95=0.00bps short_p95=0.00bps   ← 글로벌 alpha p95가 0bps
[ML-ANCHORED] anchor_end=2850 train=[0,2301) ... target_long_nz=0.2568           ← leg0 학습 12.6개월
[LEG] leg0 log_ret=+0.103 / leg1=+0.001 / leg2=−0.049 / leg3=+0.081 / leg4≈0.000  ← leg2 구조적 약점
```

핵심 신호:
- **글로벌 alpha의 `p95=0.00bps`**: cost barrier 14bps가 시그널 분포의 95%ile조차 0으로 만든다 → 신호 변별력이 비용 장벽 대비 너무 약함.
- **eligible=0.287**: 학습 가능 표본이 28.7%에 불과 (warmup·active·kill 마스킹 후).
- **leg별 수익 편차 극심**: leg0/3은 +8~10%, leg2는 −5%. 모델이 특정 레짐(횡보·변동성 정상화 구간)에서 역효과.

---

## 방향 1. Alpha 품질 (학습 데이터 정합성 & 모델 신뢰성)

### 1.1 [심각] Anchored 초기 leg 학습 데이터 부족 + 가장 중요한 leg가 가장 약함

**문제**
Anchored refit는 leg마다 `[0, anchor_i)` 까지만 학습한다 (`ml_builder.py:build_ml_strategy_alpha_anchored`).

| leg | anchor_end | 학습 범위 | 학습 개월 | 학습 표본(추정) |
|---|---|---|---|---|
| 0 | 2850 | [0,2301) | **12.6** | ~24k rows |
| 1 | 3148 | [0,2600) | 14.2 | ~27k |
| 2 | 3446 | [0,2898) | 15.8 | ~30k |
| 3 | 3744 | [0,3196) | 17.5 | ~33k |
| 4 | 4042 | [0,3494) | 19.1 | ~36k |

AWF `pos_frac` 게이트(≥3/5 leg 양수)는 **모든 leg를 동등하게** 요구하는데, 정작 leg0(12.6개월)이 가장 데이터가 적어 가장 불안정하다. 50-feature × 800-tree GBT에 12.6개월(eligible 28.7% 반영 시 ~24k rows)은 과적합 위험 구간이다.

**개선안**
- **(a) 학습 표본 하한 가드 + 동적 트리 수 축소**: `build_ml_strategy_alpha_anchored`에서 학습 행 수에 비례해 `n_estimators`/`num_leaves`를 줄여 thin-data 과적합 차단.
  ```python
  # ml_builder.py build_ml_strategy_alpha_anchored 내부, fit_ranker 직전
  n_train_rows = int(train.X.shape[0])
  if n_train_rows < 20_000:
      ml_cfg = replace(ml_cfg,
                       ranker_n_estimators=min(ml_cfg.ranker_n_estimators, 400),
                       num_leaves=min(ml_cfg.num_leaves, 15),
                       min_data_in_leaf=max(ml_cfg.min_data_in_leaf, 60))
  ```
- **(b) 표본 가중 시간 감쇠(time-decay)**: 최근 데이터에 더 큰 가중을 주어 적은 데이터에서도 최신 레짐 반영. `dataset.py:build_long_matrix`의 `sample_weight`에 `exp(−(anchor−t)/τ)` (τ≈3개월 bars) 곱.
- **(c) 검증**: leg0 모델의 valid IC(`[ML-RANKER] valid_mean`)를 leg4와 비교 — 격차가 0.001 미만으로 좁혀지면 성공.

---

### 1.2 [심각] 학습 horizon ↔ 실행 horizon 불일치

**문제**
- 라벨: `label_horizon_bars=6` → **6 bar(24h) forward return**으로 학습 (`labels.py:38, config.py:148`).
- 실행: 최적 trial의 `REBALANCE_BARS=21` → **약 84h(3.5일) 보유**.

모델은 24h 수익률을 예측하도록 학습했는데 백테스트는 3.5일 보유한다. 24h 예측 alpha가 84h 동안 평균회귀(reversal)로 소멸·역전될 수 있다. leg2의 −5%는 단기 모멘텀이 중기에 역전된 전형적 패턴으로 의심된다.

**개선안**
- **(a) 멀티-horizon 라벨 앙상블**: `label_horizon_bars`를 단일값이 아닌 `(6, 12, 24)` 다중으로 두고, 각 horizon별 EV를 학습한 뒤 실행 `REBALANCE_BARS`에 가장 가까운 horizon을 가중 선택. 최소 침습안으로는 `label_horizon_bars=12`(48h)로 상향해 실행 보유기간과의 간극 축소.
- **(b) REBALANCE_BARS 탐색 범위를 horizon 정합 구간으로 제약**: `opt_config.py`의 `REBALANCE_BARS` choices를 `(3,6,12)`로 좁혀 24h 라벨과 정합. (현재는 21까지 허용되어 불일치 유발)
- **(c) 검증**: horizon별 IC decay 곡선 로깅 — `ret_fwd_6`, `ret_fwd_12`, `ret_fwd_24`에 대한 score IC를 `build_quality_report`에 추가, decay가 완만한 horizon을 실행 기준으로 채택.

---

### 1.3 [중간] `relevance` 라벨 死재(dead weight) + 모델명 불일치

**문제**
- `labels.py:12 _build_relevance`가 5단계(0~4) 백분위 relevance를 계산하지만, 실제 `fit_ranker`는 **`y_ev`(CS-demeaned return)에 대한 RMSE 회귀** (`ranker.py:59 objective="regression"`). relevance는 `ndcg_proxy_at_k` 진단에만 쓰이고 학습에 미반영.
- 즉 "LambdaMART"라는 이름과 달리 LambdaRank 목적함수가 전혀 사용되지 않으며, relevance 계산은 매 fold마다 낭비된다.

**개선안 (택1)**
- **(a) 진짜 LambdaRank 채택**: cross-sectional ranking 본질에 맞게 `LGBMRanker(objective="lambdarank")` + `group=`(timestep) + `label=relevance`로 전환. 횡단면 순위 학습은 절대수익 노이즈에 강건하다.
- **(b) Regression 유지 + relevance 제거**: 현 CS-demeaned regression이 의도라면 `_build_relevance` 호출 제거(`labels.py`)하고 NDCG 진단을 `y_ev` 기반 Spearman으로 대체. 모델명을 `ml_csreg_v1` 등으로 정정.
- **권장**: (a). 횡단면 alpha의 핵심은 *순위*이며, RMSE 회귀는 outlier 수익(극단 펌핑)에 과민하다. LambdaRank는 NDCG@k 최적화로 상위/하위 종목 변별에 집중한다.

---

### 1.4 [심각] 품질 게이트가 음수 IC를 통과시킴

**문제**
`diagnostics.py:178 passes_quality_gate`:
```python
report.get("spearman_rank_ic", -1.0) >= -0.05   # ← 음수 IC 허용!
```
IC가 −0.05까지 통과한다. 즉 **역방향 예측 모델도 게이트를 통과**해 alpha로 병합된다. 이것이 leg별 수익이 음수로 나오는 직접 원인 중 하나다. `BlendConfig.min_mean_ic=0.02, min_t_stat=2.0`(config.py:62-63)라는 제대로 된 기준이 정의돼 있으나 ML 경로에서 미사용.

**개선안**
- **(a) IC 게이트 정상화**: anchored 함수 및 `build_ml_strategy_alpha`에 IS valid-IC 하한 적용.
  ```python
  # ml_builder.py, calibrate 직후 (anchored & 글로벌 공통)
  valid_ic = spearman(rank_valid, valid.y_ev)   # 횡단면 IC
  if valid_ic < 0.005:                            # 최소 edge 요구
      _logger.warning("[ML-GATE] valid_ic=%.4f below floor; alpha suppressed", valid_ic)
      # 해당 leg alpha를 0으로 (거래 안 함) → 음수 leg 손실 차단
  ```
- **(b) `passes_quality_gate` 하한 상향**: `spearman_rank_ic >= 0.0` 최소, 가능하면 `>= 0.01`.
- **(c) 검증**: 게이트 적용 후 leg2의 음수 수익이 0(거래 억제)으로 바뀌고 `pos_frac`이 개선되는지 확인.

---

### 1.5 [중간] Cost-blind 학습 + 사후 14bps 절벽 마스킹

**문제**
- 라벨은 `cost=0.0`(gross, `labels.py:50`)로 비용을 학습하지 않는다.
- 비용은 사후에 `ml_builder.py:214`에서 14bps **하드 절벽(step)** 으로 마스킹된다.
- 결과: 모델은 비용 인식이 없고, EV가 14bps 미만이면 일괄 0 → `[ML-ALPHA] p95=0.00bps`. 신호 분포가 비용 장벽 바로 아래에 몰려 대부분 소멸.

**개선안**
- **(a) Cost-aware 라벨**: net edge를 학습하도록 라벨에 round-trip 비용을 반영하되, **부드러운 hurdle**로. `signed_net_ret`에서 `2×(fee+slip)/10000` 차감 후 학습(현재 gross → net 전환). 이러면 모델이 비용 초과 종목을 직접 우선순위화.
- **(b) 절벽 → soft-thresholding**: `ml_builder.py:217`의 하드 마스킹을 soft-shrinkage로 교체.
  ```python
  # 기존: ev_test[sl] = where(|ev|>=barrier, ev, 0)
  # 개선: 비용만큼 축소(soft), 음수화 방지
  ev_test[sl] = np.sign(ev_test[sl]) * np.maximum(np.abs(ev_test[sl]) - barrier, 0.0)
  ```
  → 14bps 근방 신호가 일괄 소멸하지 않고 연속적으로 감쇠, 거래 빈도·변별력 동시 확보.
- **(c) 검증**: `[ML-ALPHA] p95`가 0bps에서 양수로 회복되는지, alpha_nz 분포가 매끄러워지는지 확인.

---

## 방향 2. Feature Engineering

### 2.1 [심각] 50개 중 ~40개가 RAW feature → market-beta 노이즈 주입

**문제**
횡단면 모델인데 `ret_1, ret_3, rv_6, funding_1, basis_1` 등 대부분이 **절대값(raw)** 으로 투입된다 (`features.py:203-254`). 크립토는 BTC와의 공통 베타가 0.6~0.9로 매우 높아, raw return의 분산 대부분이 *시장 공통 요인*이다. 이는 횡단면 *상대* 변별에 노이즈로 작용한다. 현재 CS-rank 처리된 것은 약 10개(`cs_rank_*`, `cs_sharpe_*`)뿐.

**개선안**
- **(a) Beta-residualized return**: BTC 회귀 잔차를 핵심 모멘텀 feature로.
  ```python
  # features.py: 종목별 168h 롤링 베타로 BTC 성분 제거
  beta_i = rolling_cov(ret_1_i, btc_ret_1) / rolling_var(btc_ret_1)   # window=168(28일)
  idio_ret_6 = ret_6 - beta_i * btc_ret_6        # 순수 idiosyncratic 모멘텀
  cs_rank_idio_ret_6 = cross_sectional_rank(idio_ret_6, mask, min_group_size)
  ```
- **(b) 주요 raw feature의 CS-rank 쌍 추가**: `ret_3, rv_6, downside_rv_18, oi_z_18`에 대해 `cs_rank_*` 버전 동반 투입(raw는 절대 수준 정보용으로 유지).
- **현 feature 64개 상한**(`config.py:154 max_features=64`) 내 여유 14개 → 우선순위 높은 CS-rank/idio feature부터 채움.

---

### 2.2 [중간] Interaction & Regime-conditional feature 부재

**문제**
모든 feature가 1차(선형 입력)이며, GBT가 상호작용을 학습하긴 하나 depth=6 제약(`config.py:161`)에서 고차 상호작용 표현력이 제한적이다. 또한 `RegimeConfig.enabled=False`(`config.py:87`)로 레짐 상태가 모델에 미투입.

**개선안**
- **(a) 명시적 interaction feature** (도메인 사전지식 주입):
  - `mom_vol_interaction = cs_rank_ret_6 × (1 − cs_rank_rv_18)` — 저변동성 모멘텀(질 높은 추세)
  - `carry_basis_div = funding_mean_6 − basis_mean_6` — funding/basis 괴리(차익 신호, 크립토 고유)
  - `breadth_mom = cs_rank_ret_6 × positive_breadth_6` — 시장 동조 추세
- **(b) 경량 레짐 state feature**: HMM 없이도 규칙 기반 3-state(trend/chop/crisis) soft posterior를 feature로. `RegimeConfig`의 vol/trend 임계값 재활용(이미 구현된 vol_window·trend_ma 로직). leg2(횡보 구간) 변별에 직접 기여.
- **검증**: SHAP 또는 LightGBM `feature_importances_`로 신규 feature 기여도 확인, 하위 기여 feature는 `max_features=64` 압박 시 제거.

---

### 2.3 [중간] 모멘텀 term-structure & 크립토 고유 feature 미흡

**문제**
모멘텀 lookback(`ret_6,12,18,36`)은 있으나 **가속도/곡률**(term structure)이 없다. 또한 크립토 강예측 인자들이 누락:
- funding-basis spread (현물-선물 괴리)
- Amihud illiquidity (`|ret|/dollar_volume`)
- OI-price divergence (가격↑ + OI↓ = 약한 추세)

**개선안 (신규 feature)**
```python
# features.py feats 리스트에 추가
("mom_accel_6_18",  cs_rank(ret_6 - ret_18, ...)),       # 모멘텀 가속/감속
("amihud_illiq_18", cs_rank(rolling_mean(|ret_1|/max(dollar_volume,ε),18), ...)),
("oi_price_div_6",  cs_rank(sign(ret_6) * sign(-oi_ret_6), ...)),  # 다이버전스
("funding_basis_spread", cs_rank(funding_mean_6 - basis_mean_6, ...)),
("downside_up_ratio_18", cs_rank(downside_rv_18 / max(rv_18,ε), ...)),  # 하방 편향
```
- **검증**: 각 신규 feature의 단독 univariate IC를 `[ML-FEATURE-IC]` 로그로 출력, |IC|>0.01 인 것만 잔류.

---

### 2.4 [낮음] Microstructure feature 과소 활용

**문제**
미시구조 feature가 2개(`micro_hl_spread_1, micro_close_to_hl_1`)뿐. 4h 바에는 일중 정보가 풍부하다.

**개선안**
- `overnight_gap = (open_t − close_{t-1}) / close_{t-1}` (갭 반전 신호)
- `intrabar_vol = (high−low)/close` 의 18-bar z-score (변동성 확장 감지)
- 우선순위는 낮음 — 2.1~2.3 적용 후 `max_features` 여유가 있을 때.

---

## 3. 구현 우선순위 & 검증 계획

### 3.1 우선순위 (영향도 × 난이도)

| 순위 | 항목 | 영향도 | 난이도 | 기대 효과 |
|---|---|---|---|---|
| **P0** | 1.4 IC 게이트 정상화 | ★★★ | 낮음 | 음수 leg 손실 즉시 차단 → pos_frac↑ |
| **P0** | 1.5 soft cost-shrinkage | ★★★ | 낮음 | alpha p95 회복, 거래 변별력↑ |
| **P1** | 2.1 beta-residual + CS-rank 확장 | ★★★ | 중간 | 횡단면 IC 근본 개선 |
| **P1** | 1.2 horizon 정합 | ★★ | 중간 | leg2 reversal 손실 완화 |
| **P2** | 1.3 LambdaRank 전환 | ★★ | 중간 | 순위 학습 강건성 |
| **P2** | 2.3 크립토 고유 feature | ★★ | 중간 | 신규 alpha 소스 |
| **P3** | 1.1 thin-data 가드 | ★ | 낮음 | 초기 leg 안정화 |
| **P3** | 2.2 interaction/regime | ★ | 중간 | leg2 레짐 변별 |

### 3.2 단계별 검증 (각 단계 후 측정)

1. **단위 IC 측정**: `[ML-RANKER] valid_mean` 및 신규 `[ML-FEATURE-IC]` 로그로 valid IC ≥ 0.01 확인.
2. **leg별 log_ret 분포**: `[LEG] leg_i log_ret` 5개 모두 또는 ≥4개 양수 목표 (`pos_frac ≥ 0.6` → 0.8).
3. **alpha 분포 건전성**: `[ML-ALPHA] p95 > 0bps`, `long_nz/short_nz`가 0.10~0.30 안정 구간.
4. **게이트 통과**: `[ENSEMBLE] No members` → 1개 이상 trial이 `FUTURES_AWF_MU_LOG_MIN=0.0` 통과.
5. **최종 OOS**: AWF 통과 후 OOS holdout CAGR/Sharpe/MDD 보고.

### 3.3 회귀 방지

- 각 변경은 **독립 PR**로 분리 (P0 → P1 → P2 순), 단계마다 `--trials 500 --mode strategy` 재실행으로 leg별 지표 회귀 모니터링.
- `cost=0` → `cost=net` 전환(1.5a)과 soft-shrinkage(1.5b)는 **동시 적용 금지** (이중 비용 차감 위험). 1.5a 우선, 효과 측정 후 1.5b 판단.
- Feature 추가는 `max_features=64` 상한 준수. 추가분만큼 저기여 raw feature 제거 (importance 하위부터).

---

## 4. 요약

근본 원인은 **두 축의 결합**이다:
1. **신호가 비용 대비 약함** (방향 2): raw feature의 beta 노이즈로 횡단면 IC가 낮아, gross alpha조차 14bps 비용 장벽을 못 넘는다 (`p95=0.00bps`가 직접 증거).
2. **약한 신호를 거르지 못함** (방향 1): IC 게이트가 음수까지 허용(`>=−0.05`)하고, 비용을 하드 절벽으로 처리해 음수 edge leg가 그대로 손실로 실현된다.

**최단 경로**: P0 두 항목(IC 게이트 정상화 + soft cost-shrinkage)만으로도 음수 leg 손실이 차단되어 `pos_frac` 게이트 통과 가능성이 높다. 이후 P1(beta-residual feature)로 IC 자체를 끌어올려 양(+)의 OOS edge를 확보하는 것이 정석 경로다.
