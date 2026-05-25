---
title: Binance Futures ML Strategy
domain: futures-strategy-ml
type: domain-spec
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/
  - src/domain/futures/optimization/objectives.py
change_triggers:
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/strategy/ranker.py
  - src/domain/futures/optimization/objectives.py
  - src/domain/futures/strategy/diagnostics.py
last_verified: 2026-05-25 (EV gate placement fix: gated copy for output, ungated for quality metrics)
---

# Binance Futures ML Strategy

## 1. Overview
`opt_main_futures.py`의 strategy stage에 비용을 초과 극복하는 expected return alpha를 공급하는 ML 기반 전략입니다. 주문, 비중, 레버리지를 직접 제어하지 않고 오직 `alpha_panel`만을 생성합니다.

---

## 2. Core Components

| Component | Responsibility |
|---|---|
| `ml_builder.py` | ML feature-label-train-infer 오케스트레이션 |
| `features.py` | PIT-safe 피처 텐서(CS-Sharpe 등) 생성 |
| `labels.py` | T+1 체결 기준 fee/slippage 제외 + funding 반영 레이블 생성 |
| `ranker.py` | CS-demeaned LightGBM Regressor 학습 및 상대 스코어 추론 |
| `calibrator.py` | 분위수 기반 동적 불확실성 조정 및 EV 보정 |

---

## 3. Data Flow

```text
[Data Maps] -> [Feature/Label Generation] -> [Double-Weighted Dataset] 
  -> [CS-Demeaned Training] -> [Quantile Calibration] 
  -> [Cost Barrier Gating] -> [Alpha Panel]
```

---

## 4. Business Rules

### Must Follow
- **Alpha Supplier Only:** pure expected return alpha(Bps)만 산출할 것.
- **Single-Order Weighting:** `labels.py`에서 1회만 가중치 계산 (`liq * (1+2|y_ev|)`). `dataset.py`에서 재곱 금지.
- **Calibrator/Ranker 타깃 분리:** calibrator는 `exec_net_ret`(pre-CS-demean), ranker는 `signed_net_ret`(CS-demeaned) 사용.
- **Dynamic Cost Barrier:** 거래 비용보다 작은 노이즈성 시그널은 `0.0`으로 소거.

### Must Not Do
- **Portfolio Control:** ML이 target weight, order, leverage를 직접 계산 금지.
- **Look-Ahead Leakage:** 미래 시점 데이터를 참조하여 Scaler/Imputer 피팅 금지.

### Internal Naming Convention
- 내부 로컬 변수명은 모델/알고리즘 고유명(`lgbm`, `ranker`, `calibrator`)보다 역할 중심 일반명(`model`, `fit_result`, `train_data`, `valid_data`, `score`, `estimate`, `lower/median/upper`)을 우선 사용한다.
- 공개 API(함수명, 클래스명, dataclass public field, strategy id, CLI arg)는 backward compatibility를 위해 기존 명칭을 유지한다.

---

## 5. Detailed Specifications

### 5.1 Feature Schema (50 Pillars)
- **Reversal/Momentum:** `ret_1` ~ `ret_36` 및 횡단면 랭크 팩터.
- **Volatility:** Realized Vol, Downside Vol, ATR Ratio.
- **Carry/Liquidity:** Funding Z-score, Volume Z-score, ADV Rank.
- **CS-Sharpe (High-Performance):** 개별 변동성 대비 기대수익률 강도를 크로스섹션 랭크화 한 `cs_sharpe_6` 및 `cs_sharpe_18`.

### 5.2 Single-Order Weighting (labels.py 1회 계산, dataset.py 재곱 금지)
무작위 노이즈 신호 배제를 위해 리턴 절대값($|y_{ev}|$)에 비례하여 샘플 가중치를 동적으로 부여합니다.
가중치는 `labels.py` 내에서 1회만 계산되며, `dataset.py`에서 재곱하지 않습니다.
$$\text{sample\_weight} = \text{original\_weight} \times (1.0 + 2.0 \times |y_{ev}|)$$
- `original_weight`: 유동성 기반 가중치 (`clip(log1p(volume), 0.25, 2.0)`).
- `y_ev`: `signed_net_ret` (CS-demeaned beta-residualized return; 비용 차감 없음, B1 유지).
- `eligible_mask=False` 구간은 `sample_weight=0.0`으로 유지.
- **단일 적용 원칙:** `dataset.py`의 `build_long_matrix()`는 `labels.sample_weight`를 그대로 사용. `(1+2|y_ev|)` 재곱 금지.

### 5.3 Quantile EV Calibration — Calibrator/Ranker 타깃 분리
- **Quantile Loss:** `q10`, `q50`, `q90` 분위수 예측기를 동시 학습.
- **Calibrator Label Target (`exec_net_ret`):** calibrator는 `exec_net_ret`(CS-demean 적용 전 beta-잔차화 수익률)을 타깃으로 학습.
  CS-demean은 크로스섹션 평균을 0으로 소거하므로 absolute EV가 제거되어 24bps cost wall 통과 불가.
  `exec_net_ret`는 demean 직전 `long_net.copy()`로 생성되며 절대 EV를 보존한다.
- **Ranker Label Target (`signed_net_ret`):** ranker는 CS-demeaned `signed_net_ret`(= `long_net` after demean)을 사용.
  `ranker.py` 내부에서 `_cs_demean(train.y_ev, ...)` 재적용하므로 상대 순위 학습에 문제 없음.
- **Uncertainty Adjustment:** 예측 불확실성 폭($q_{90} - q_{10}$)에 따라 알파 강도를 조절하여 꼬리 위험 방어.

**Conservative EV 수식 (`compute_conservative_ev`):**
```
uncertainty[i]   = max(q90[i] - q10[i], ε)
med_unc          = median(uncertainty)
lam_dynamic[i]   = clip(lambda_tail * uncertainty[i] / med_unc, 0, lambda_tail * 2.0)

# Sign-symmetric Multiplicative Penalty (q50 magnitude proportional)
penalty_ratio[i] = (q50[i] - q10[i]) / uncertainty[i] if q50[i] >= 0 else (q90[i] - q50[i]) / uncertainty[i]
penalty_term[i]  = clip(lam_dynamic[i] * penalty_ratio[i], 0.0, 0.99)

ev[i]            = q50[i] * (1.0 - penalty_term[i])
```
- **`lambda_tail`:** 기본값 `0.10` (범위: 0.05–0.30). 꼬리 위험 페널티 강도.
- **`lam_dynamic` 상한 캡 (`2 × lambda_tail`):** OOS regime shift로 인한 고변동성 구간에서 페널티 폭주를 방지.
- **Sign-symmetric Multiplicative 설계:** 페널티가 $q_{50}$의 절대 크기에 비례하여 부과되므로 CS-demean 환경 하에서 $q_{50} \approx 0$으로 수축하더라도 페널티가 $ev$를 불필요하게 음수 영역으로 끌고 가는 부호 반전 현상이 원천 차단됩니다. 또한 `penalty_term`을 최대 `0.99`로 제한하여 신호 보존력을 최대화합니다.

> **적용 변경 (2026-05-24):** EV 출력 경로의 fold-level group centering을 제거하여 absolute EV 크기 소거를 방지.

### 5.3.1 Calibrator Target Options (`calibrator_target`)

`StrategyMLConfig.calibrator_target`은 calibrator의 학습 타깃 `y_ev`를 결정한다.

| 옵션 | 설명 | 용도 |
|------|------|------|
| `"beta_residualized"` (default) | beta-잔차화 후, CS-demean 전 스냅샷 | 기존 B2 동작 유지 |
| `"gross"` | 원시 로그수익률 - funding (beta 제거 없음, fee 없음) | B2 magnitude 손실 가설 A/B 검증 |

**§5.3/§5.5 모순 해소 (2026-05-25):**  
§5.5(B2) beta-잔차화는 시장 드리프트 성분을 제거하여 CS-IC를 개선하지만, 동시에 calibrator 타깃의 절대 크기를 수축시켜 cost wall 통과가 어려워질 수 있다. `calibrator_target` 토글을 통해 ranker(잔차화+CS-demean 유지)와 calibrator 타깃을 독립적으로 제어하여 이 상충을 A/B 측정으로 결정한다.

### 5.4 Output Contract (`alpha_panel`)
- **Index:** `MultiIndex(datetime, symbol)`
- **Columns:** `alpha_long` (Bps), `alpha_short` (Bps)
- **Zero-filling:** 미매칭 구간은 반드시 `0.0`으로 치환하여 계좌 오염 방지.

### 5.5 B2 - Beta-Residualized Labels
`src/domain/futures/strategy/labels.py`의 `build_label_panel()`은 수익률에서 시장 베타 성분을 제거하여 특정 종목 고유의 알파를 분리합니다.  
라벨 계약은 **fee/slippage 제외 + funding 반영**입니다.

**Core Algorithm:**
- **_compute_trailing_beta():** 각 종목별로 120-bar 롤링 OLS 회귀를 통해 등가중 크로스섹션 시장 수익률에 대한 베타 추정
  - 윈도우: `_BETA_WINDOW = 120` bars
  - 최소 관측값: `_BETA_MIN_PERIODS = 20`
  - Look-Ahead Free: 시점 t의 베타는 [t-120, t) 범위 과거 데이터만 사용하여 미래 정보 누설 방지
  - 루프 정책: N개 종목(~20-50) 단위 루프, T 길이 루프 없음 (Zero-Loop Policy 준수)
  
- **Residualization Formula:**
  ```
  long_net[t,i] = gross_long[t,i] - beta[t,i] * market_fwd_ret[t] - cost - funding[t,i]
  short_net[t,i] = gross_short[t,i] + beta[t,i] * market_fwd_ret[t] - cost + funding[t,i]
  ```

- **CS-Demean (post-residualization, 2026-05-25):**
  ```
  # Vectorized [T-h, N] operation applied after main loop, before signed = long_net.copy()
  row_mean = nanmean(long_net[t_valid, eligible], axis=1, keepdims=True)  # [T-h, 1]
  long_net[t, mask]  -= row_mean   # zero-center long labels
  short_net[t, mask] += row_mean   # restore anti-symmetry: short_net ≈ -long_net_demeaned
  ```
  OLS beta != 1 이거나 eligible 종목이 부분 집합일 때 (예: 비대칭 funding rate) 잔차화만으로는
  시점별 CS 평균이 0을 보장하지 못한다. `long_net`과 `short_net` 모두 demean하여
  `short_net ≈ -long_net` anti-symmetry 계약을 보존한다.
  행당 eligible 종목이 <2인 경우 row_mean=0으로 처리(no-op).

**Benefit:** 시장 공통 인수 제거 + CS 평균 소거로 순수 크로스섹션 알파만 모델에 노출, bull/bear 시장 편향 제거

**Cost Contract:**
- 라벨은 fee/slippage를 차감하지 않는다.
- objective friction/hurdle 레이어에서 fee+slippage+EV_HURDLE를 한 번만 반영한다.

### 5.6 Track A - Diagnostic Logging
저노이즈 경로 추적을 위해 소수 trial에서만 compact 로그를 출력합니다.

**[ML-ALPHA-IC]**: 크로스섹션 IC(Information Coefficient) 기반 신호 유효성 판정
```
[ML-ALPHA-IC] mean_ic=0.0112 icir=0.42 t_stat=1.02 hit_ratio=0.55 n_obs=180
```
- `mean_ic`: 평균 Spearman Rank IC (이상적: ≥0.02)
- `icir`: IC Ratio = mean_ic / std(ic) (이상적: ≥0.5)
- `t_stat`: t-통계량 = mean_ic / (std_ic / √n_obs) (이상적: ≥2.0, p<0.05)
- `hit_ratio`: IC > 0인 기간 비율 (이상적: ≥0.45)
- `n_obs`: 유효 크로스섹션 시점 개수

**[ML-COST-WALL]**: 거래 비용 대비 예상 수익의 유효성 진단
```
[ML-COST-WALL] alpha_p95=10.04bps friction=14bps hurdle_bps=10.0 floor=24.0 status=WARN
```
- `alpha_p95`: 모델 예측 알파의 95분위수(단위: bps)
- `friction`: 편도 거래 비용(수수료+슬리피지, 기본값: ~14bps)
- `hurdle_bps`: EV_HURDLE 임계값(기본값: 10bps, 설정 범위: [3.0, 20.0])
- `floor`: 유효 비용 벽 = friction + hurdle_bps
- `status`: 신호 통과 여부
  - `OK`: alpha_p95 > floor
  - `WARN`: floor/2 < alpha_p95 ≤ floor (한계 신호, 주의 필요)
  - `FAIL`: alpha_p95 ≤ floor/2 (실패, 개선 필요)

**[RAW-SIGNAL-DIAG]**: beta-잔차화가 cross-sectional 신호 분산에 미치는 영향 진단
```
[RAW-SIGNAL-DIAG] raw_cs_std=0.0123 resid_cs_std=0.0045 var_retention=0.366 n_ts=720 raw_nz=0.891 resid_nz=0.743
```
- `raw_cs_std`: 잔차화 전 gross return의 평균 횡단면 표준편차
- `resid_cs_std`: 잔차화 후의 평균 횡단면 표준편차
- `var_retention`: `resid_cs_std / raw_cs_std` — 1.0에 가까울수록 잔차화 손실 적음
- `n_ts`: 유효 timestep 수 (유효 심볼 ≥5인 시점)
- WARNING: `n_len=1`(단일 종목)이면 beta-잔차화가 가격 신호를 0으로 수축시키므로, 단일 종목 경로에서 IC/EV 숫자는 무의미함

**[STRAT-PATH]**: 실행 가능한 신호 경로 연결 진단 (`objectives.py`, trial<5, leg 단위)
```
[STRAT-PATH] trial=1 leg=0 range=(0,720) bars=720 alpha_nz=0.8123 merge_nz=0.1450 xs_nz=0.2321 trades=84 long=43 short=41
```
- `alpha_nz`: alpha(`alpha_long/short`) 비영 비율
- `merge_nz`: membership/entry-block 반영 후 `target_weights` 비영 비율
- `xs_nz`: `xs_score_long/short` union 비영 비율
- `trades/long/short`: 실제 백테스트 체결 결과 기반 카운트 (추정치 아님)

### 5.7 B4 - IC Quality Gate (config-driven, 결선 완료)
신호 품질 게이트: `src/domain/futures/strategy/diagnostics.py` 함수 `passes_ic_gate()` 및 `ml_builder.py` 결선.

**Gate Thresholds (`StrategyMLConfig` 기본값, 완화된 초기값):**
- `ic_gate_min_mean_ic = 0.01` (B2 uplift 확인 후 0.02로 강화 예정)
- `ic_gate_min_t_stat = 1.5` (B2 uplift 확인 후 2.0으로 강화 예정)
- `ic_gate_min_hit_ratio = 0.45`
- `ic_gate_warn_only = True` (True=경고만, False=RuntimeError)

**Behavior:**
- `ic_gate_warn_only=True`: Gate 미만족 시 `[ML-IC-GATE] WARN: ...` 로그 출력, 진행 계속
- `ic_gate_warn_only=False`: Gate 미만족 시 `RuntimeError` 발생하여 파이프라인 차단
- 동전던지기 모델(IC=0) 통과 방지: `passes_quality_gate`의 `spearman_rank_ic >= 0.0` 조건에 추가하여 이중 게이팅

### 5.8 Directional Signal-Preservation Diagnostics & Viability Gate (2026-05-25)
- 전략 합성 경로(`src/domain/futures/optimization/objectives.py`)에서 `alpha -> xs_score` 방향성 보존 비율을 산출한다.
  - `xs_long_preservation_ratio = xs_long_nz_ratio / alpha_long_nz_ratio`
  - `xs_short_preservation_ratio = xs_short_nz_ratio / alpha_short_nz_ratio`
- 위 비율은 `_strategy_compose_diag` 및 `_strategy_signal_path_diag`에 함께 기록되며, `[COMPOSE-DIAG]`, `[STRAT-PATH]` 로그에 노출된다.
- directional viability helper(`src/domain/futures/strategy/diagnostics.py::passes_directional_viability_gate`)는 `alpha_long_non_zero_ratio`/`alpha_short_non_zero_ratio`만 평가한다.
  - 기본 threshold `0.0/0.0`로 backward-compatible 동작을 유지한다.
- signal-preservation helper(`src/domain/futures/strategy/diagnostics.py::passes_signal_preservation_gate`)는 `xs_long_preservation_ratio`/`xs_short_preservation_ratio`만 평가한다.
- ML builder hard gate 결선은 이번 스코프에서 미적용이며, 현재는 diagnostics/warn 기반 모니터링 계약으로 운영한다(추후 toggle 결선 예정).

### 5.9 EV Hurdle & OOS Trade Activation
과도하게 높았던 진입 장벽을 하향하여 모델의 유효한 상대적 랭크 예측(Rank IC ~0.027)이 정상 거래로 실현되도록 보정합니다.
- **최적화 탐색공간 하향:** `EV_HURDLE_BPS` 튜닝 범위를 `[5.0, 100.0]`에서 `[3.0, 20.0]`으로 조정하여 극단적인 신호 소거를 차단.
- **기본 허들 완화:** 기본값 `40.0 bps`를 `10.0 bps`로 경감하여 OOS에서의 `oos_zero_trades=0` 달성 및 유효 거래 빈도 확보.

### 5.10 Model Contract Alignment (Approved ADR)
- **Compatibility Name:** 전략 식별자 `ml_lambdamart_v1`는 호환성 목적으로 유지.
- **Implementation Reality:** 실제 학습기는 `CS-demeaned LightGBM regression` + quantile EV calibrator.
- **Score Split:** `rank_score`는 상대 순위 품질(IC/NDCG), `ev_score`는 절대 실행 가능성(cost wall 통과) 검증에 사용.
- **No Repeated Centering:** `ev_score`는 calibrator 출력 이후 추가 group-centering을 수행하지 않는다.
- **CS-Demean Source:** centering은 `build_label_panel()` (라벨 빌드 시점)에서 1회만 적용한다. calibrator fit/predict 경로에서 재적용 금지.

### 5.11 Virtual OOS Refit Normalization Consistency
- `build_ml_strategy_alpha`의 virtual OOS fill(refit) fold는 regular fold의 마지막 정규화 상태를 재사용하지 않는다.
- virtual fold의 `train` 구간(`[v_train_start, v_train_end)`)에서 `fit_robust_bounds` 및 train median imputer를 다시 피팅한다.
- 해당 virtual-train 기반 정규화를 virtual `train/valid/test` 전 경로에 동일 적용하여 fold 간 normalization leakage를 방지한다.

### 5.12 Final Evaluation Ensemble Cache Contract (2026-05-24)
- `run_final_oos_evaluation()`의 top-K ensemble 루프는 멤버별 `m_params` 시그니처(JSON sorted key)를 계산한다.
- 동일 시그니처가 재등장하면 `build_strategy_alpha` + merge + OOS backtest를 재실행하지 않고 캐시된 멤버 포트(`equity_curve` 포함)를 재사용한다.
- 캐시 적중 멤버도 결과 집계(멤버 수, 메타 allocator 입력)에 동일하게 포함되어 기존 public contract를 유지한다.
- 성능 목표: 중복 멤버가 많을 때 최종 평가 복잡도를 `O(K * C_eval)`에서 `O(U * C_eval + (K-U) * O(1))`로 축소 (`U`: unique params).
- 진단 로그:
  - 멤버 단위: `[ENSEMBLE-PROF] member=... cache=hit`
  - 요약: `[ENSEMBLE-PROF] summary ... cache_hits=... unique_engine_evals=...`

### 5.13 Bottleneck Profiling Contract (2026-05-24)
- `run_optimization` 병목 분해를 위해 run-level/leg-level 프로파일 로그를 표준화한다.
- `precompute_ml_optimization_context()`는 아래 구간을 모두 로깅한다.
  - `align`, `covariance`, `awf_refit_total`, `calibrator_total`, `prebuilt_total`, `total`
- AWF anchored refit 경로는 leg별 총 시간과 구간 정보를 로깅한다.
  - `[AWF-REFIT-PROF] leg=i/n total=... bars=... train_end=... test=[s,e)`
- `build_ml_strategy_alpha_anchored()`는 내부 학습 파이프라인 시간을 분해한다.
  - `feature_label`, `matrix`, `fit_predict`, `calibrator`, `total`

### 5.14 Step4 - Label Contract v2 (2026-05-25)
- `LabelPanel`은 기존 public field를 유지하면서 `rank_target`, `magnitude_target`, `cost_clearance_target`를 선택 필드로 추가한다.
- `rank_target`는 `signed_net_ret`, `magnitude_target`는 `exec_net_ret`를 가리키며, `metadata`에 target key/mode/cost-floor를 기록한다.
- B1/B2 불변식은 유지된다: fee/slippage는 label에서 차감하지 않고, beta-residualized + CS-demean 계약은 기존과 동일.

### 5.15 Step5 - Feature Registry & Group Ablation (2026-05-25)
- `features.py`는 단일 리스트 조립에서 group registry 조립 구조로 전환되었다.
- 지원 그룹: `trend`, `reversal`, `volatility`, `carry`, `liquidity`, `market_context`, `microstructure`, `missingness`.
- `StrategyMLConfig.feature_groups_enabled`로 그룹 단위 실험/ablation을 제어한다(기본값: 전체 활성화).

### 5.16 Step6 - Missing Data Handling (2026-05-25)
- `FeaturePanel.valid_mask`는 `all-finite` 강제 대신 거래 eligibility 기반으로 유지하고, 결측은 train-only 경로에서 처리한다.
- 정규화는 fold별 `fit_robust_bounds(train)` 이후 `fit_missing_value_imputer(train)`를 적용한다.
- missingness indicator 피처(`*_missing_ind`)를 PIT-safe하게 제공하며, imputer fit은 train split에만 수행되어 leakage를 방지한다.
- `run_ml_pipeline_for_universe()`는 anchored/non-anchored 호출 총 시간 로그를 남긴다.
  - `[ML-PIPE-PROF] anchored=... symbols=... tf=... elapsed=... alpha_rows=...`
- final ensemble은 기존 캐시 로그 외에 아래 분해를 추가한다.
  - `unique_engine_evals`, `unique_alpha_builds`, `alpha_build_count`
- `opt_main_futures` 프로파일 요약은 trial 내부 평균(ms)와 별도로 run-level hidden overhead를 출력한다.
  - `trial_elapsed_sum`, `run_optimization`, `hidden_overhead`

### 5.14 Performance Verification Snapshot (2026-05-24)
- 실행 조건:
  - `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode strategy --skip-universe --skip-data-sync --symbols BTCUSDT --trials 5 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1 --seed 7`
- 핵심 관측:
  - `run_optimization=15.29s`, `trial_elapsed_sum=0.27s`, `hidden_overhead=15.02s`
  - `ml_precompute total=9.93s` (`awf_refit=1.41s`)로 precompute 지연이 크게 감소
  - final ensemble summary(단일 멤버 실행): `build_alpha=15.05s`, `total=15.67s`
- 결론:
  - AWF leg 반복의 `feature/label` 재생성 병목은 panel cache로 완화됨.
  - 잔여 병목은 single-member 기준 final alpha build 경로가 지배적이며, 멤버 수 증가 시 ensemble cache 효과가 재확대된다.

### 5.15 Anchored Panel Cache Contract (2026-05-24)
- AWF leg refit는 `precompute_anchored_ml_panels()`로 causal `FeaturePanel/LabelPanel`을 1회 생성해 재사용한다.
- 재사용 범위는 raw panel까지만 허용한다. 아래 train-derived 단계는 leg별로 계속 재계산한다.
  - `fit_robust_bounds` / train median imputer / calibrator fitting
- `build_ml_strategy_alpha_anchored(..., precomputed_panels=...)` 옵션은 backward-compatible이며, 미지정 시 기존 경로를 사용한다.
- 진단 로그:
  - `[ML-PANEL-CACHE] scope=awf_precompute hit=true build=... rows=... symbols=... features=...`
  - anchored 내부 `[AWF-REFIT-PROF] feature_label=0.00s`로 cache 재사용 여부를 확인한다.

### 5.16 Alpha Override Prebuilt Contract (2026-05-24)
- AWF leg 경로는 `data_maps` clone/merge 대신 leg별 `alpha_overrides` aligned array를 구성해 prebuilt builder에 주입한다.
- OOS Platt calibrator fitting도 동일한 `alpha_overrides`를 참조하여 alpha source 일관성을 유지한다.
- 목표:
  - pandas DataFrame clone/merge overhead 감소
  - `O(L * S * T)` object copy 비용을 array 경로로 완화
- 계약:
  - override 미제공 시 기존 `data_maps` alpha column 경로를 fallback으로 유지한다.

### 5.17 Calibrator Target Configuration (2026-05-25)
- `StrategyMLConfig.calibrator_target`으로 calibrator `y_ev` 타깃을 config-driven 제어.
- 기본값 `"beta_residualized"` — B2 이전 동작과 완전 동일.
- `"gross"` 옵션은 A/B 측정 전용. OOS 검증 완료 전 프로덕션 사용 금지.
- **B1 불변 계약:** 라벨 레이어(labels.py)는 fee/slippage를 차감하지 않는다. 비용 차감은 objectives 레이어에서만 1회 수행. `"funding_net"` 옵션은 이 계약을 위반하므로 삭제됨.
- Ranker 타깃(`signed_net_ret` = CS-demeaned beta-residualized)은 `calibrator_target` 무관하게 불변.
- 관련 진단: `[RAW-SIGNAL-DIAG]` (§5.6) — 잔차화로 인한 분산 손실 정량화.

### 5.18 Step1~3 Implementation Contract (2026-05-25)
- **Step1 / Baseline Harness Metadata Standardization**
  - `build_ml_strategy_alpha` 출력 `panel.attrs`에 `baseline_harness` 표준 메타데이터를 기록한다.
  - 공통 필드: `version`, `mode`, `selected_horizon`, `cost_floor_bps`, `candidate_count`.
  - horizon 실험 모드에서는 `horizon_experiment`(candidate별 `alpha_p95_bps`, `score_bps`, `clears_cost_wall`)를 추가 기록한다.
- **Step2 / Config SSOT**
  - `ranker.py`, `calibrator.py`의 학습 하이퍼파라미터 하드코딩을 제거하고 `StrategyMLConfig` 필드를 단일 진실원(SSOT)으로 사용한다.
  - 기본값은 기존 동작과의 호환성을 위해 이전 하드코딩 값(예: lr 0.02, feature_fraction 0.70, bagging_fraction 0.75)을 반영한다.
- **Step3 / Horizon Experiment Axis**
  - `StrategyMLConfig`에 `horizon_experiment_enabled`, `horizon_candidates`를 추가한다.
  - 활성화 시 candidate horizon별로 alpha를 생성하고 `alpha_p95_bps - (friction + hurdle)` 점수(실행 가능성, cost-wall aware)로 최고 horizon을 선택한다.

### 5.19 Step7 - Model Family Split (2026-05-25)
- `StrategyMLConfig.model_family`는 `lgbm_regression`(baseline)과 `lgbm_huber`를 지원한다.
- `ranker.py::_build_ranker_model()`는 family별 LightGBM objective를 분기해 생성한다.
- 기본값은 `lgbm_regression`으로 유지되어 backward compatibility를 보장한다.

### 5.20 Step8 - Calibration v2 EV Mode (2026-05-25)
- `StrategyMLConfig.ev_mode`는 `quantile`과 `prob_x_magnitude`를 지원한다.
- `quantile`: 기존 sign-symmetric tail penalty 경로.
- `prob_x_magnitude`: directional probability(`q10/q50/q90`)와 magnitude(`|q50|`)를 결합한 EV 경로.

### 5.21 Step9 - Alpha Gate Diagnostics & Contract (2026-05-25)
- `diagnostics.py::alpha_gate_diagnostics()`는 아래 조건의 fail reason을 구조적으로 반환한다.
- 조건: `alpha_p95_bps` vs `friction + hurdle`, `long_nz`, `short_nz`, `xs preservation`.
- 기본 계약은 hard wall 유지: `alpha_gate_cost_wall_tolerance_bps=0.0`.
- tolerance는 실험/디버그용 override이며 기본 동작을 완화하지 않는다.
  - 외부 API(`build_ml_strategy_alpha` 시그니처/반환 타입)는 변경하지 않는다.

### 5.22 Step10 - LambdaMART Rank Extraction Path (2026-05-25)
- `StrategyMLConfig.ranking_mode`는 `group_ndcg`(default)와 `pointwise`를 지원한다.
- `ranker.py::fit_ranker()`는 `ranking_mode=group_ndcg`에서 `train.y_rank` relevance label과 `train.group`을 사용해 `lightgbm.LGBMRanker`를 학습한다.
- validation 데이터가 존재하면 `eval_set` + `eval_group`을 함께 전달하고, metric은 `ndcg`, `eval_at=(3, 5)` 계약을 적용한다.
- `pointwise` 모드에서는 기존 `model_family`(`lgbm_regression`, `lgbm_huber`)가 그대로 동작하며 backward compatibility를 유지한다.
- long/short는 별도 rank target + magnitude target + cost-clearance gate를 거쳐 `alpha_long/alpha_short`를 독립 구성한다.
- horizon candidate 선택 점수는 `in_fold_valid_alpha_p95_bps`(train/valid only) 기반으로 계산하며 test/OOS는 사용하지 않는다.

### 5.23 Dual-Side Virtual OOS Fill Contract (2026-05-25)
- virtual OOS fill은 마지막 fold 이후 커버되지 않는 live 구간을 채우기 위한 실제 OOS extension이다.
- long/short를 각각 독립 경로로 계산한다: `v_train_long/v_valid_long/v_test_long`, `v_train_short/v_valid_short/v_test_short`.
- 각 side마다 동일한 절차를 수행한다: feature normalizer fit → rank model fit → quantile calibrator fit → EV inference.
- 결과는 `ev_long_grid`, `ev_short_grid`에 **직접** 누적한다 (signed `ev_grid`를 경유하지 않는다).
- gate 로직은 regular fold와 동일하게 `cost_clearance_target_long` / `cost_clearance_target_short`를 사용한다.
- Invariant: virtual fill 완료 후 `alpha_long_nz > 0 AND alpha_short_nz > 0`이 동시에 보장되어야 한다.

### 5.24 Dynamic Cost Gate Contract (2026-05-25)
- gate 판정식: `gate_margin_bps = side_ev_bps - (dynamic_cost_bps + hurdle_bps)`, `gate_margin_bps > 0`이면 통과.
- 비용 입력 우선순위: 1. `execution_cost_bps_2d` (per-symbol, per-timestep) 2. `round_trip_cost_bps()` (global fallback).
- `hurdle_bps`는 `OPT_FUTURES_CONFIG["FUTURES_DEFAULT_EV_HURDLE_BPS"]` (default 10bps)에서 읽는다.
- long/short는 독립적으로 gate를 적용한다 (`cost_clearance_target_long`, `cost_clearance_target_short` 별도 계산).
- `LabelPanel.metadata["execution_cost_bps_source"]`로 per_symbol/fallback_global 구분 로깅한다.

### 5.25 group_ndcg → pointwise Auto-Fallback Contract (2026-05-25)
- `ranking_mode = "group_ndcg"`가 default이며, group cardinality 부족 시 자동 pointwise 전환한다.
- fallback 조건: `len(train.group) < _MIN_GROUPS_FOR_NDCG` (현재 5).
- fallback 시 `WARNING [RANKER] group_ndcg fallback to pointwise: n_groups=X < min=5` 로깅 후 `lgbm_regression` objective를 사용한다.
- `model_family == "lgbm_lambdarank"` + `force_pointwise=True` 조합에서도 regression으로 안전하게 fallback한다.
- HMM은 core alpha path의 필수 구성요소가 아니며, `RegimeConfig.enabled=False`(default)로 비활성화 상태를 유지한다.

---

## 6. Examples
- **Input:** Predicted EV 10bps, Round-trip Cost 14bps
- **Output:** Gated EV 0bps (Cost Barrier 적용으로 노이즈 소거)

---

## 7. Testing Expectations
- **Spearman IC Test:** 3 fold 연속 음수 기록 시 학습 하드 페일 판정.
- **Inference Integrity:** 추론 결과에 NaN이 포함되지 않았는지, 롱/숏 양방향 신호가 존재하는지 확인.
- **PIT Test:** 피처 연산 시 미래 데이터 참조(Look-ahead)가 없는지 검증.
