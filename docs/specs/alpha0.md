---
title: Futures ML Alpha rebuild - simple rank-native architecture
domain: futures-alpha
type: refactor
status: proposal
priority: critical
ai_read_policy: when_related
last_verified: 2026-05-31
related_paths:
  - src/domain/futures/universe/config.py
  - src/domain/futures/universe/selection.py
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/features.py
  - src/domain/futures/strategy/labels.py
  - src/domain/futures/strategy/ranker.py
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/strategy/diagnostics.py
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/domain/futures/optimization/objectives.py
---

# Futures ML Alpha Rebuild: Easy-to-Read Overview

## 1. What This Change Is About

현재 futures ML alpha는 여러 겹의 모델과 보정 로직을 쌓아 올린 구조다.
이번 개편의 목표는 복잡도를 줄이고, 실제로 의미 있는 alpha만 남기는 것이다.

한 줄로 요약하면:

- `dual-side ranker + quantile calibrator + conservative EV + rank-sized emit + alpha_p95 wall`
  구조를 걷어내고,
- `LightGBM LGBMRanker` 하나로 cross-sectional ranking을 학습한 뒤,
- 비용과 체결 현실을 반영한 gate로만 alpha를 통과시키는 구조로 단순화한다.

## 2. 왜 바꾸는가

현재 구조는 신호 스킬은 어느 정도 보여도 최종적으로는 비용과 sweep 기준을 넘지 못한다.
즉, 모델이 복잡해진 것에 비해 실전 alpha로 이어지는 효율이 낮다.

문제의 핵심은 다음과 같다.

- 절대 수익 크기를 맞추는 문제보다, 같은 bar 안에서 어떤 심볼이 더 나은지 순위를 맞추는 문제가 더 중요하다.
- q10/q50/q90 같은 quantile 보정은 복잡도만 늘리고, C3 portfolio IC 개선으로 연결되지 않았다.
- 24bps는 label을 바꾸는 기준이 아니라, 실제 turnover와 slippage를 반영한 admission gate로 써야 한다.

## 3. 무엇을 만들려고 하는가

### 3.1 모델은 하나만 쓴다

선정 알고리즘은 `LightGBM LGBMRanker`의 LambdaMART 단일 ranker이다.

이 선택의 이유는 단순하다.

- 유니버스가 작고 tabular panel 구조라서 sequence model보다 ranking model이 맞다.
- 목표는 return magnitude 예측이 아니라 cross-sectional ordering이다.
- group 단위 ranking은 시장 beta와 scale drift에 더 견고하다.

### 3.2 신호는 return unit으로 만든다

학습된 score는 그대로 쓰지 않고, return unit alpha로 바꾼다.

```python
alpha_signed[t, i] = score_z[t, i] * sigma_resid_trailing[t] * ic_lcb_fold
alpha_long = max(alpha_signed, 0.0)
alpha_short = max(-alpha_signed, 0.0)
```

의미는 다음과 같다.

- `score_z`: timestamp 내부에서 정규화한 rank score
- `sigma_resid_trailing`: 미래를 보지 않는 trailing residual volatility
- `ic_lcb_fold`: validation fold에서 얻은 conservative IC

즉, Grinold 식에 맞춰 score를 실제 수익 단위로 해석한다.

### 3.3 피처는 적고 단단하게 간다

초기 feature set은 24-32개 안쪽으로 제한한다.

살릴 것:

- price momentum / reversal 계열
- volatility / liquidity / execution cost 계열
- funding / basis / crowding 계열
- market context 계열
- BTC 대비 상대 움직임 같은 idiosyncratic feature

버릴 것:

- future/OOS를 보고 고르는 target-driven feature selection
- 근거 없는 sector/theme taxonomy
- 기본값으로 켜진 missingness shortcut

### 3.4 학습은 단순하게 간다

학습은 chronological fold 기반으로만 진행한다.

- train slice에서만 scaler/imputer를 fit한다.
- validation slice는 early stopping과 emission 선택에만 쓴다.
- test slice는 scoring만 한다.
- horizon sweep은 학습 과정에서 없애고 평가에서만 본다.

### 3.5 24bps는 이렇게 쓴다

24bps는 여전히 중요하다.
다만 의미는 “alpha가 24bps보다 커야 한다”가 아니라,
“실제 turnover와 slippage를 반영했을 때도 살아남는가”이다.

그래서 다음을 gate로 본다.

- `rank_ic_lcb > 0`
- `rank_ic_lcb >= breakeven_ic_eff`
- `basket_net_bps_lcb_24bps > 0`
- `awf_pos_frac >= 0.60`
- `awf_worst_leg_log_tw >= 0.0`
- `ev_cost_ratio >= 1.5`
- `turnover_cost_ratio <= 0.35`

## 4. 무엇을 남기고 무엇을 없애는가

### 유지

- Universe Stage2-Stage6 filtering
- `historical_stage5_union` training / Stage6 trading mask 분리
- `build_label_panel()`의 residualization과 purge/embargo
- chronological walk-forward fold
- IC, breakeven, breadth 계산
- AWF backtest와 실제 cost accounting

### 제거

- dual-side ranker
- quantile calibrator
- EVQuantiles
- alpha build-time hard fail on `alpha_p95_below_cost_wall`
- horizon experiment inside alpha builder
- target-driven OOS feature selection
- regime exposure scaling in the first rebuild

## 5. 검증은 무엇을 보나

핵심 평가지표는 다음이다.

- OOS Spearman Rank IC
- Newey-West IC t-stat
- IC LCB
- Effective Breadth / correlation-adjusted `N_eff`
- Breakeven IC
- Top-bottom spread
- 24bps net basket spread
- AWF stability
- Deflated Sharpe
- Regime IC, 특히 bear regime

지표 해석은 간단하다.

- 평균 IC만 좋고 LCB가 나쁘면 불안정한 신호다.
- basket net이 24bps 조건에서 살아남지 못하면 실전 alpha가 아니다.
- bear regime에서 음수면 시장이 어려울 때 망가지는 신호다.

## 6. 구현 위치

이번 개편은 아래 파일들을 중심으로 진행한다.

- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/features.py`
- `src/domain/futures/strategy/labels.py`
- `src/domain/futures/strategy/ranker.py`
- `src/domain/futures/strategy/ml_builder.py`
- `src/domain/futures/strategy/diagnostics.py`
- `src/domain/futures/strategy/alpha_evaluation.py`
- `src/domain/futures/optimization/objectives.py`

관련 테스트도 함께 정리한다.

- `tests/unit/domain/futures/strategy/test_ml_config.py`
- `tests/unit/domain/futures/strategy/test_ml_builder.py`
- `tests/unit/domain/futures/strategy/test_ml_ranker.py`
- `tests/unit/domain/futures/strategy/test_alpha_evaluation.py`
- `tests/unit/domain/futures/optimization/test_strategy_signal_path.py`

## 7. 구현 메모

### `src/domain/futures/strategy/config.py`

기존의 복잡한 active knob를 줄이고, simple rank-native default만 남긴다.

### `src/domain/futures/strategy/features.py`

feature set을 whitelist 기반으로 제한하고, BTC 대비 상대 feature 같은 idiosyncratic feature만 추가한다.

### `src/domain/futures/strategy/ranker.py`

단일 `LGBMRanker` path만 노출한다.

### `src/domain/futures/strategy/ml_builder.py`

fold 단위 학습, validation IC LCB 계산, emission 선택, alpha assembly를 한 번에 이어준다.

### `src/domain/futures/strategy/diagnostics.py`

IC LCB와 top-bottom spread diagnostics를 추가한다.

### `src/domain/futures/strategy/alpha_evaluation.py`

pass gate를 LCB와 실제 post-cost basket metric 중심으로 바꾼다.

### `src/domain/futures/optimization/objectives.py`

24bps cost는 backtest에서만 반영하고, alpha build gate에서는 diagnostic로만 남긴다.

## 8. 확인 방법

검증은 다음 순서로 본다.

```bash
uv run ruff check --fix src/domain/futures/strategy/config.py src/domain/futures/strategy/features.py src/domain/futures/strategy/ranker.py src/domain/futures/strategy/ml_builder.py src/domain/futures/strategy/diagnostics.py src/domain/futures/strategy/alpha_evaluation.py src/domain/futures/optimization/objectives.py
uv run mypy src/domain/futures/strategy/config.py src/domain/futures/strategy/features.py src/domain/futures/strategy/ranker.py src/domain/futures/strategy/ml_builder.py src/domain/futures/strategy/diagnostics.py src/domain/futures/strategy/alpha_evaluation.py src/domain/futures/optimization/objectives.py
uv run pytest tests/unit/domain/futures/strategy/test_ml_config.py tests/unit/domain/futures/strategy/test_ml_ranker.py tests/unit/domain/futures/strategy/test_ml_builder.py tests/unit/domain/futures/strategy/test_alpha_evaluation.py tests/unit/domain/futures/optimization/test_strategy_signal_path.py --tb=short
PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01
```

## 9. Acceptance

- `alpha_contract == "return_unit_grinold_rank"`
- active alpha path에서 q10/q50/q90 dependency가 없다
- `alpha_p95_below_cost_wall`만으로 build fail하지 않는다
- `rank_ic_lcb`, `breakeven_ic_eff`, `basket_net_bps_lcb_24bps`, `awf_pos_frac`, `ev_cost_ratio`, `turnover_cost_ratio`가 기록된다
- `ALPHA_PASS=false`면 blocker가 rank skill, breadth, cost/turnover, regime stability 중 어디인지 구분된다

#### `src/domain/futures/strategy/config.py`

Action: REPLACE `StrategyMLConfig` active knobs with simple rank-native defaults.

Required fields:

```python
model_family: Literal["lgbm_lambdarank"] = "lgbm_lambdarank"
ranking_mode: Literal["group_ndcg"] = "group_ndcg"
ranker_enabled: bool = True
rank_target_mode: Literal["cs_residual"] = "cs_residual"
label_horizon_bars: int = 12
purge_bars: int = 12
embargo_bars: int = 12
max_features: int = 32
rank_select_quantiles: tuple[float, ...] = (0.25, 0.35, 0.45)
target_breadth: int = 8
ic_lcb_z: float = 1.0
```

Deprecate active use of:

```python
calibrator_target
calibrator_target_mode
ev_mode
alpha_emit_mode
post_cost_admission_mode
horizon_experiment_enabled
horizon_candidates
```

#### `src/domain/futures/strategy/features.py`

Action: REPLACE feature group default by static whitelist and add idiosyncratic features.

Pseudo-code:

```python
btc_ret_6 = repeat(ret_6[:, btc_idx])
btc_ret_12 = repeat(ret_12[:, btc_idx])
ret_6_minus_btc_ret_6 = ret_6 - btc_ret_6
ret_12_minus_btc_ret_12 = ret_12 - btc_ret_12
funding_6_minus_cs_median = funding_mean_6 - nanmedian(funding_mean_6, axis=1)
basis_6_minus_cs_median = basis_mean_6 - nanmedian(basis_mean_6, axis=1)

if aligned.symbol_meta and "cluster_id" in aligned.symbol_meta:
    cluster_rel_ret_6 = ret_6 - per_bar_cluster_median(ret_6, cluster_id)
else:
    skip and metadata["cluster_rel_ret_6_enabled"] = False
```

No target-aware feature selection in this function.

#### `src/domain/futures/strategy/ranker.py`

Action: REPLACE active API with a single signed ranker path.

Required signatures:

```python
@dataclass(slots=True, frozen=True)
class RankerFitResult:
    model: lgb.LGBMRanker
    fit_mode: Literal["lambdarank"]
    best_iteration: int | None = None

def fit_ranker(
    train: LongMatrixDataset,
    valid: LongMatrixDataset,
    cfg: StrategyMLConfig,
) -> RankerFitResult: ...

def predict_rank_score(
    model: lgb.LGBMRanker,
    dataset: LongMatrixDataset,
) -> NDArray[np.float32]: ...
```

Remove ensemble models in the first rebuild. Reintroduce only after single-seed architecture passes objective gates.

#### `src/domain/futures/strategy/ml_builder.py`

Action: REPLACE `build_ml_strategy_alpha()` internals with simple signed flow.

Step-by-step:

1. Align data.
2. Build feature panel and label panel.
3. Build chronological folds.
4. For each fold:
   - fit train-only scaler/imputer.
   - build single `train`, `valid`, `test` matrix using `labels.relevance` and `labels.signed_net_ret`.
   - train `fit_ranker(train, valid, cfg)`.
   - predict valid/test score.
   - compute validation IC and HAC LCB.
   - choose quantile from `(0.25, 0.35, 0.45)` using validation 24bps net spread LCB.
   - write test score to `score_grid`.
5. Convert score to return-unit alpha:
   - per timestamp robust z-score.
   - trailing residual sigma from past labels only.
   - alpha = `score_z * sigma * max(valid_ic_lcb, 0.0)`.
6. Emit top/bottom selected quantile only.
7. Apply `trading_symbols` mask.
8. Assemble `alpha_panel` with `alpha_long`, `alpha_short`, `rank_score_long`, `rank_score_short`.
9. Attach attrs:
   - `model_family="lightgbm_simple_lambdarank"`
   - `alpha_contract="return_unit_grinold_rank"`
   - `fold_validation_ic`
   - `fold_ic_lcb`
   - `fold_selected_quantile`
   - `quality_report`

Do not raise on `alpha_p95_below_cost_wall`.

#### `src/domain/futures/strategy/diagnostics.py`

Action: ADD fold-level LCB and basket spread diagnostics.

Required signatures:

```python
def ic_lcb_hac(ic_series: np.ndarray, *, horizon_bars: int, z: float = 1.0) -> float: ...

def top_bottom_spread_bps(
    score_2d: np.ndarray,
    realized_2d: np.ndarray,
    eligible_2d: np.ndarray,
    *,
    quantile: float,
    cost_bps: float,
) -> dict[str, float]: ...
```

#### `src/domain/futures/strategy/alpha_evaluation.py`

Action: REPLACE pass gate to use LCB and actual post-cost basket metrics.

Add fields to `AlphaEvaluationReport`:

```python
rank_ic_lcb: float
basket_net_bps_lcb_24bps: float
top_bottom_spread_bps: float
turnover_proxy: float
```

Pass gate:

```python
rank_ic_lcb >= breakeven_ic_eff
ic_t_stat_nw >= 3.0
basket_net_bps_lcb_24bps > 0.0
deflated_sharpe >= 0.95
bear_ic >= 0.0
```

#### `src/domain/futures/optimization/objectives.py`

Action: KEEP cost subtraction in compose/backtest, but demote `[ML-COST-WALL] alpha_p95` from gate logic to diagnostic.

Invariant:

```python
mu_l_pre = beta_alpha * alpha_long - execution_cost_fraction
mu_s_pre = beta_alpha * alpha_short - execution_cost_fraction
```

No cost amortization toggle in alpha evaluation. If `COST_GATE_AMORTIZE` remains for backward compatibility, default must be `False` and tests must assert it.

## 8. Verification

L1:

```bash
uv run ruff check --fix src/domain/futures/strategy/config.py src/domain/futures/strategy/features.py src/domain/futures/strategy/ranker.py src/domain/futures/strategy/ml_builder.py src/domain/futures/strategy/diagnostics.py src/domain/futures/strategy/alpha_evaluation.py src/domain/futures/optimization/objectives.py
uv run mypy src/domain/futures/strategy/config.py src/domain/futures/strategy/features.py src/domain/futures/strategy/ranker.py src/domain/futures/strategy/ml_builder.py src/domain/futures/strategy/diagnostics.py src/domain/futures/strategy/alpha_evaluation.py src/domain/futures/optimization/objectives.py
```

Unit:

```bash
uv run pytest tests/unit/domain/futures/strategy/test_ml_config.py tests/unit/domain/futures/strategy/test_ml_ranker.py tests/unit/domain/futures/strategy/test_ml_builder.py tests/unit/domain/futures/strategy/test_alpha_evaluation.py tests/unit/domain/futures/optimization/test_strategy_signal_path.py --tb=short
```

Smoke (current CLI):

```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01
```

Acceptance:

- `alpha_contract == "return_unit_grinold_rank"`
- no q10/q50/q90 dependency in active alpha path.
- no alpha build failure caused only by `alpha_p95_below_cost_wall`.
- `rank_ic_lcb`, `breakeven_ic_eff`, `basket_net_bps_lcb_24bps`, `awf_pos_frac`, `ev_cost_ratio`, `turnover_cost_ratio` are logged and stored.
- If `ALPHA_PASS` remains false, fail reason must identify whether the blocker is rank skill, breadth, cost/turnover, or regime stability.
