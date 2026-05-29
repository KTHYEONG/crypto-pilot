---
title: Alpha ML 일반화 재구축 최종 개선안
domain: strategy-ml
type: prd
status: ready_for_implementation
priority: critical
ai_read_policy: when_related
related_paths:
  - docs/specs/alpha_breadth_decontamination.md
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/contracts.py
  - src/domain/futures/strategy/labels.py
  - src/domain/futures/strategy/dataset.py
  - src/domain/futures/strategy/ranker.py
  - src/domain/futures/strategy/calibrator.py
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/forecast/compose.py
  - src/domain/futures/optimization/objectives.py
change_triggers:
  - src/domain/futures/strategy/**
  - src/domain/futures/forecast/**
  - src/domain/futures/optimization/objectives.py
last_verified: 2026-05-29
---

# Alpha ML 일반화 재구축 최종 개선안

## 1. 냉정한 판정

`docs/specs/alpha_breadth_decontamination.md`의 핵심 진단인 "EV magnitude IS->OOS 붕괴"와 "신호가 비용벽보다 작다"는 타당하다. 그러나 문서의 다음 단계는 그대로 구현하면 과소 설계다.

1. D1은 필요했지만 충분하지 않다. 현재 기본값은 `StrategyMLConfig.ranker_enabled=False`이고, 이 경우 `score_grid`는 실제 ranker 점수가 아니라 `_rank_score(None, dataset)` 경로의 0 점수다. 따라서 `[OOS-RANKIC] ic=0.0000`은 "현재 배출 alpha가 수익성이 없다"는 증거로는 강하지만, "ranker가 OOS edge가 없다"는 최종 판정으로 쓰면 안 된다.
2. D3 rank portfolio는 근본 해결이 아니다. `compose_mu()`는 여전히 `alpha_long/alpha_short - cost - hurdle`의 절대 EV 크기를 통과 조건으로 쓴다. OOS p95가 4.9bps이고 비용벽이 24bps라면 top-k 재배열만으로 net mu가 양수가 되지 않는다.
3. D4 Maker 비용 인하는 연구 병목이 아니다. 비용을 24bps에서 4bps로 낮춰도, OOS IC가 0에 가깝거나 방향성이 불안정하면 Kelly/vol targeting 이후 실현 손익은 노이즈다.
4. D2는 방향은 맞지만 너무 넓다. "capacity↓, regularization↑, sign/rank target"을 한 번에 바꾸면 어떤 원인이 개선됐는지 분해할 수 없다.

최종 결론: 다음 구현은 D3/D4가 아니라 `측정 SSOT -> forward-return rank target -> capacity-controlled model -> post-cost admission gate` 순서로 재구축해야 한다. 목표는 EV magnitude를 키우는 것이 아니라, OOS에서 비용 차감 전에도 재현되는 cross-sectional ordering edge를 먼저 확보하는 것이다.

## 2. Target Files

- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/contracts.py`
- `src/domain/futures/strategy/labels.py`
- `src/domain/futures/strategy/dataset.py`
- `src/domain/futures/strategy/ranker.py`
- `src/domain/futures/strategy/calibrator.py`
- `src/domain/futures/strategy/ml_builder.py`
- `src/domain/futures/forecast/compose.py`
- `src/domain/futures/optimization/objectives.py`
- `tests/unit/domain/futures/strategy/test_ml_config.py`
- `tests/unit/domain/futures/strategy/test_ml_labels.py`
- `tests/unit/domain/futures/strategy/test_ml_ranker.py`
- `tests/unit/domain/futures/strategy/test_ml_builder.py`
- `tests/unit/domain/futures/optimization/test_strategy_signal_path.py`

## 3. Contract

### 3.1 Existing contracts that must stay backward-compatible

```python
def build_ml_strategy_alpha(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
    trading_symbols: tuple[str, ...] | None = None,
) -> pd.DataFrame: ...
```

```python
def build_long_matrix(
    features: FeaturePanel,
    labels: LabelPanel,
    start: int | None = None,
    end: int | None = None,
    min_group_size: int = 1,
    *,
    fold: FoldSpec | None = None,
    split: Literal["train", "valid", "test"] | None = None,
    rank_target_override: np.ndarray | None = None,
    relevance_override: np.ndarray | None = None,
    ev_target_override: np.ndarray | None = None,
) -> LongMatrixDataset: ...
```

```python
def fit_ranker(
    train: LongMatrixDataset,
    valid: LongMatrixDataset,
    cfg: StrategyMLConfig,
) -> RankerFitResult: ...
```

```python
def fit_quantile_calibrators(
    train: LongMatrixDataset,
    valid: LongMatrixDataset,
    cfg: StrategyMLConfig,
    rank_score_train: np.ndarray | None = None,
    rank_score_valid: np.ndarray | None = None,
) -> CalibratorFitResult: ...
```

```python
def compose_mu(
    alpha: AlphaForecast,
    cost: CostForecast,
    params: dict[str, Any],
    *,
    holding_bars: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
```

### 3.2 New config fields

Add to `StrategyMLConfig`:

```python
rank_target_mode: Literal["cs_residual", "forward_gross_rank"] = "forward_gross_rank"
calibrator_target_mode: Literal["signed_ev", "rank_confidence"] = "rank_confidence"
post_cost_admission_mode: Literal["ev_gate", "rank_then_ev_gate"] = "rank_then_ev_gate"
rank_portfolio_top_k: int = 4
rank_portfolio_min_score_spread_bps: float = 0.0
oos_ic_target_source: Literal["signed_net_ret", "forward_gross_ret"] = "forward_gross_ret"
```

Validation:

- `rank_portfolio_top_k >= 1`
- `rank_portfolio_min_score_spread_bps >= 0.0`
- all literals must reject unknown values with `ValueError`

### 3.3 LabelPanel extension

Add optional fields to `LabelPanel`:

```python
forward_gross_ret: np.ndarray | None = None
forward_gross_rank_target: np.ndarray | None = None
forward_gross_relevance: np.ndarray | None = None
```

Shape contract: every non-None array must be `[T, N]` and aligned to the same entry timing as `signed_net_ret`: entry at `open[t + 1]`, exit at `close[t + horizon]`.

### 3.4 ML output metadata

`build_ml_strategy_alpha()` must set:

```python
panel.attrs["oos_forward_rank_ic"] = {
    "mean_ic": float,
    "t_stat": float,
    "hit_ratio": float,
    "n_obs": int,
    "cofinite_p50": float,
    "bars_ge5_ratio": float,
    "target_source": "forward_gross_ret",
}
panel.attrs["generalization_report"] = {
    "is_rank_ic": float,
    "valid_rank_ic": float,
    "test_rank_ic": float,
    "oos_rank_ic": float,
    "retention_ratio": float,
    "decision": Literal["continue", "no_edge"],
}
```

## 4. Step-by-Step Logic

### Phase 1: 측정 SSOT 고정

1. `labels.py`에서 `gross_long_2d`를 버리지 말고 `LabelPanel.forward_gross_ret`로 반환한다.
2. `forward_gross_ret`는 beta residualization, CS demean, fee/slippage 차감 전 값이어야 한다.
3. `forward_gross_rank_target`은 bar별 cross-sectional rank 중심화 값으로 만든다.
   - eligible and finite symbols only
   - group size `< cfg.min_group_size`이면 해당 row는 NaN
   - rank scale은 `[-1.0, 1.0]`
4. `forward_gross_relevance`는 같은 row에서 5-bucket relevance `0..4`로 만든다.
5. `ml_builder.py`의 OOS IC는 기본적으로 `forward_gross_ret` 기준 Spearman IC를 계산한다.
6. 기존 `signed_net_ret` 기반 IC는 보조 진단으로만 유지한다.

Rationale: 현재 `signed_net_ret`는 OOS에서 NaN 비율이 높고, beta/CS 처리 후 측정 대상이 학습 타깃과 실행 수익 사이를 혼동한다. forward gross return을 SSOT로 두어 model edge와 cost/execution edge를 분리한다.

### Phase 2: target을 magnitude에서 ordering으로 전환

1. `build_long_matrix()` 호출 시 `rank_target_mode == "forward_gross_rank"`이면:
   - `rank_target_override=labels.forward_gross_rank_target`
   - `relevance_override=labels.forward_gross_relevance`
2. `ranker_enabled` 기본값은 `True`로 전환하되, 기존 ablation 테스트는 명시적으로 `False`를 사용하게 유지한다.
3. `fit_ranker()`의 기본 경로는 `group_ndcg` 유지.
4. pointwise fallback은 유지하되, fallback 발생 시 `RankerFitResult`에 `fit_mode: Literal["lambdarank", "pointwise"]`를 추가해 기록한다.
5. `score_grid`에는 ranker가 꺼진 경우 0 점수를 넣지 말고 NaN을 유지한다. ranker-disabled 상태의 OOS rank IC는 `n_obs=0`, `decision="not_measured"`가 되어야 한다.

Rationale: 문서의 D1 판정은 ranker-disabled 기본값 때문에 오염될 수 있다. 개선의 첫 가설은 "EV 크기 회귀"가 아니라 "횡단면 순서 학습"이다.

### Phase 3: calibrator를 alpha generator가 아닌 admission layer로 축소

1. `calibrator_target_mode == "rank_confidence"`이면 quantile calibrator는 EV magnitude 예측기가 아니라 score confidence 예측기로 사용한다.
2. 이 모드에서 `alpha_long/alpha_short` 생성은 다음 순서를 따른다.
   - raw score를 bar별 z-score 또는 percentile score로 정규화
   - long candidates: top-k positive rank score
   - short candidates: bottom-k negative rank score
   - candidate가 아닌 symbol은 alpha 0
   - candidate alpha magnitude는 fixed admission EV floor 이상일 때만 배출
3. 단, 실제 `compose_mu()`는 계속 비용을 차감한다. rank top-k는 비용벽을 우회하지 않는다.
4. `rank_then_ev_gate` 모드의 기본 후보 수는 `rank_portfolio_top_k=4`로 시작한다.

Rationale: rank로 breadth를 강제로 만들 수는 있지만 비용을 이길 수는 없다. 따라서 rank는 후보 선택, calibrator/EV는 admission gate로 분리해야 한다.

### Phase 4: capacity와 leakage 통제

1. 기본 모델 capacity를 낮춘다.
   - `ranker_n_estimators`: 800 -> 300
   - `calibrator_n_estimators`: 600 -> 200
   - `num_leaves`: 31 -> 15
   - `max_depth`: 6 -> 4
   - `min_data_in_leaf`: 50 -> 100
   - `ranker_lambda_l2`: 5.0 -> 20.0
   - `ranker_reg_alpha`: 1.5 -> 5.0
2. `build_ml_strategy_alpha()` 내부의 unconditional `replace(cfg.ml, min_data_in_leaf=30, num_leaves=31)`는 제거한다. 이 코드는 config-level capacity control을 무력화한다.
3. `purge_bars >= label_horizon_bars`, `embargo_bars >= label_horizon_bars`는 현재 검증을 유지한다.
4. horizon experiment는 p95 EV 기준이 아니라 `oos_forward_rank_ic.t_stat`, `retention_ratio`, post-cost paper return 기준으로 선택한다.

Rationale: 현재 구조는 IS quantile EV를 크게 만드는 방향으로 쉽게 과적합된다. capacity를 낮추고 선택 기준을 OOS ordering으로 바꾸지 않으면 p95 EV만 다시 커진다.

### Phase 5: Go/No-Go gate

Implement hard decision rules:

```text
continue if:
  oos_forward_rank_ic.mean_ic >= 0.015
  and oos_forward_rank_ic.t_stat >= 2.0
  and generalization_report.retention_ratio >= 0.50
  and post_cost paper spread return > 0

no_edge if:
  oos_forward_rank_ic.mean_ic < 0.005
  or oos_forward_rank_ic.t_stat < 1.0
```

If `no_edge`, do not implement D3/D4 as production features. Switch research to feature/horizon/universe redesign.

## 5. Surgical Plan

### `src/domain/futures/strategy/contracts.py`

Action: REPLACE `LabelPanel`

Instruction:

- Add optional fields listed in section 3.3 after `exec_net_ret`.
- Update `validate_label_panel()` if it checks field shapes. Each non-None extra label must match `[T, N]`.

### `src/domain/futures/strategy/config.py`

Action: REPLACE `StrategyMLConfig`

Instruction:

- Add fields from section 3.2.
- Change `ranker_enabled` default to `True`.
- Apply capacity defaults from Phase 4.
- Remove or update tests that assert `ranker_enabled is False`; new default must be `True`.

### `src/domain/futures/strategy/labels.py`

Action: REPLACE label construction tail

Instruction:

- Preserve `gross_long_2d`.
- Add helper:

```python
def _build_centered_rank_target(
    signed_ret: np.ndarray,
    eligible: np.ndarray,
    *,
    min_group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return centered rank target [-1, 1] and relevance 0..4 per row."""
```

- Use `scipy` only if already present in `pyproject.toml`; otherwise implement with pandas/numpy existing dependencies.
- Return `forward_gross_ret=gross_long_2d`, `forward_gross_rank_target=...`, `forward_gross_relevance=...`.

### `src/domain/futures/strategy/dataset.py`

Action: REPLACE rank target assembly

Instruction:

- If `rank_target_override` is provided, use its actual continuous value for `y_ev` only when an explicit `ev_target_override` is not provided.
- For `y_rank`, convert `relevance_override` directly when provided.
- Do not infer `y_rank = 4 if rank_target > 0 else 0` when `forward_gross_relevance` is available.

### `src/domain/futures/strategy/ranker.py`

Action: REPLACE `RankerFitResult` and fit metadata

Instruction:

- Change dataclass to:

```python
@dataclass(slots=True, frozen=True)
class RankerFitResult:
    """Ranker fit output."""

    model: lgb.LGBMRegressor | lgb.LGBMRanker
    fit_mode: Literal["lambdarank", "pointwise"]
```

- Return `fit_mode="lambdarank"` when `LGBMRanker` path is used, else `"pointwise"`.

### `src/domain/futures/strategy/ml_builder.py`

Action: REPLACE target resolution and OOS IC diagnostics

Instruction:

- `_resolve_side_targets(labels)` must choose forward gross rank/relevance when `ml_cfg.rank_target_mode == "forward_gross_rank"`.
- Remove unconditional capacity override:

```python
ml_cfg = replace(
    cfg.ml,
    min_data_in_leaf=30,
    num_leaves=31,
)
```

Use `ml_cfg = cfg.ml` unless only `n_jobs` is being resolved.

- In `_fit_predict_fold_dual_side()`, if `ranker_enabled=False`, score arrays must be NaN arrays, not zero arrays.
- OOS rank IC must use `labels.forward_gross_ret` when `oos_ic_target_source == "forward_gross_ret"`.
- Set `panel.attrs["oos_forward_rank_ic"]` and `panel.attrs["generalization_report"]`.

### `src/domain/futures/forecast/compose.py`

Action: ADD optional rank admission support

Instruction:

- Keep default `ev_gate` behavior numerically identical.
- For `params["POST_COST_ADMISSION_MODE"] == "rank_then_ev_gate"`, do not bypass cost. Only apply top-k candidate masking before hurdle gate if rank score arrays are available in `AlphaForecast` metadata. If metadata is unavailable, fallback to current behavior.

### `src/domain/futures/optimization/objectives.py`

Action: REPLACE strategy compose params propagation

Instruction:

- Pass `POST_COST_ADMISSION_MODE`, `RANK_PORTFOLIO_TOP_K`, and `RANK_PORTFOLIO_MIN_SCORE_SPREAD_BPS` from params into `compose_mu()`.
- Add diagnostics:
  - `rank_candidate_nz`
  - `rank_candidate_to_xs_preservation`
  - `post_cost_admission_mode`

## 6. Verification

Fast unit loop:

```bash
uv run ruff check src/domain/futures/strategy src/domain/futures/forecast src/domain/futures/optimization tests/unit/domain/futures/strategy tests/unit/domain/futures/optimization
uv run mypy src/domain/futures/strategy src/domain/futures/forecast src/domain/futures/optimization
uv run pytest tests/unit/domain/futures/strategy/test_ml_config.py tests/unit/domain/futures/strategy/test_ml_labels.py tests/unit/domain/futures/strategy/test_ml_ranker.py tests/unit/domain/futures/strategy/test_ml_builder.py tests/unit/domain/futures/optimization/test_strategy_signal_path.py --tb=short
```

Smoke verification:

```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode strategy-smoke --skip-universe --skip-data-sync --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,LTCUSDT --trials 1 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1
```

Expected outcomes:

- `panel.attrs["oos_forward_rank_ic"]` exists.
- ranker-enabled default produces nonzero/finite `score_grid` in test windows.
- ranker-disabled ablation reports IC as not measured, not zero edge.
- `generalization_report.decision` is either `continue` or `no_edge` based on explicit thresholds.
- No look-ahead: labels still use entry `open[t + 1]` and exit `close[t + horizon]`; purge/embargo remain `>= label_horizon_bars`.

## 7. Implementation Priority

1. Phase 1 and Phase 2 first. Without clean forward-return rank IC, all later changes are unverifiable.
2. Phase 4 capacity override removal in the same patch. Otherwise config experiments are silently invalid.
3. Phase 3 only after OOS forward rank IC is positive and statistically nontrivial.
4. D4 Maker-cost work remains out of scope until model edge passes the gate.

