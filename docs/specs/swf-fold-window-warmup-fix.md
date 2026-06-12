# 🎯 Objective

SWF Layer 1의 시간 경계, warm-up, point-in-time universe, fold 학습 유효성 계약을 바로잡아 모든 fold가 과거 정보만으로 학습되고 유효한 심볼별 OOS signal을 산출하도록 한다.

## 확인된 원인

### 1. Tiered window가 실행 기준일보다 한 분기 뒤처짐

- 일반 실행 window는 `run_config.date=2026-05-01` 기준으로 `2022-10-01 ~ 2026-03-31`을 사용한다.
- Tiered는 `get_layered_window(reference_date=window.end_date)`를 호출한다.
- `window.end_date=2026-03-31`은 분기 마지막 날이므로 Tiered 계산은 다시 이전 분기 끝인 `2025-12-31`을 holdout 종료일로 선택한다.
- 실제 Tiered L1 범위는 `2023-01-01 ~ 2024-07-01`이 되었고, 의도한 동일 기준일 계산 결과인 `2023-04-01 ~ 2024-10-01`과 다르다.

### 2. Universe timeline 시작일과 L1 시작일이 다름

- universe timeline은 일반 window의 `is_start=2023-10-01`부터 생성된다.
- Tiered L1은 `2023-01-01`부터 fold를 생성한다.
- `build_rule_signal_panels()`의 유효 마스크는 universe membership mask를 포함하므로 `2023-10-01` 이전 event는 모두 제거된다.
- 현재 fold 구간은 다음과 같다.
  - Fold 1 OOS: `2023-01-01 ~ 2023-04-20`, event 0
  - Fold 2 OOS: `2023-04-20 ~ 2023-08-07`, event 0
  - Fold 3 OOS: `2023-08-07 ~ 2023-11-25`, `2023-10-01` 이후 event만 존재
  - Fold 4 OOS: `2023-11-25 ~ 2024-03-13`
  - Fold 5 OOS: `2024-03-13 ~ 2024-07-01`

### 3. Warm-up 데이터와 ML fit 데이터가 같은 구간으로 취급됨

- `build_l1_swf_folds()`는 `fit_start=0`으로 고정한다.
- `warmup_bars`라는 인자는 실제 warm-up 길이가 아니라 첫 OOS 시작 위치인 `l1_start` index다.
- pre-L1 데이터는 indicator 계산 전용이어야 하지만 현재는 ML fit 후보 기간에도 포함된다.
- 반대로 첫 OOS를 `l1_start`에서 바로 시작하므로 L1 내부의 명시적인 initial training 구간이 없다.
- 현재 로드 시작일은 `2022-10-01`이지만 잘못 계산된 Tiered window의 요구 fetch 시작일은 `2022-01-01`이다. 의도한 365일 warm-up 중 273일이 누락됐다.

### 4. Fold 1~4가 실제로 학습되지 않음

- 재현 실행 로그에서 Fold index `0, 1, 2, 3` 모두 `fit=0`으로 Ensemble 학습이 생략됐다.
- Fold 4의 `fit_end`는 `2023-11-21`이지만 early-stop tail을 제외한 실제 train 종료는 universe timeline 시작일 `2023-10-01`보다 앞선다.
- Fold 3과 Fold 4는 OOS event가 각각 649개, 2759개 존재해도 학습 결과 대신 다음 prior fallback을 반환한다.
  - `expected_net_bps = 0`
  - `p_pass = 0.5`
  - `edge_source = DISABLED`
- 예측이 상수이므로 fold Spearman IC는 `n/a`가 된다.

### 5. 학습 실패 fold가 pooled IC에 포함됨

- fold IC 계산은 상수 예측을 `n/a`로 처리한다.
- pooled IC 집계는 fold 학습 상태와 예측 분산을 확인하지 않고 finite 값만 concat한다.
- 따라서 Fold 3과 Fold 4의 zero fallback 3408개가 Fold 5의 실제 예측과 함께 pooled IC에 포함된다.
- 현재 `Pooled IC=-0.095`, `N=5576`은 유효 OOS 예측만의 통계가 아니므로 alpha 부재를 확정하는 근거로 사용할 수 없다.

### 6. Current snapshot scope와 전기간 promotion 결과가 L1에 유입됨

- Tiered aligned scope는 과거 fold별 universe가 아니라 현재 OOS snapshot selected 심볼로 고정된다.
- 과거에는 거래 가능했지만 현재 snapshot에서 빠진 심볼이 L1 평가에서 제거되어 survivorship bias가 발생한다.
- Tiered 입력은 `ml_out.labeled`이며, 이는 bridge가 전체 추천 구간으로 계산한 promotion filter 적용 결과다.
- 초기 fold 시점에 알 수 없는 미래 구간의 promotion 결과가 과거 fold event 선택에 사용될 수 있다.

## Contract Changes

### Execution window

```python
def _resolve_layered_window(reference_date: str | None) -> LayeredWindow:
    ...
```

- `get_layered_window()`에는 `window.end_date`가 아니라 `run_config.date`와 동일한 기준일을 전달한다.
- `reference_date=None`이면 일반 quarterly window와 동일한 현재 날짜를 사용한다.

```python
def _run_universe_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    *,
    layered_window: LayeredWindow | None = None,
) -> tuple[
    list[str],
    dict[date, frozenset[str]],
    tuple[str, ...],
    tuple[str, ...],
    UniverseSnapshot,
    dict[date, frozenset[str]],
]:
    ...
```

- Tiered 활성 시 universe 시작일은 `min(window.is_start_date, layered_window.l1_start)`다.
- OOS snapshot 기준일은 기존 `window.oos_start_date`를 유지한다.

```python
def _run_data_stage(
    run_config: FuturesRunConfig,
    window: QuarterlyWindow,
    discovered_symbols: list[str],
    timeline: dict[date, frozenset[str]],
    inference_panel: tuple[str, ...] = (),
    live_inference_panel: tuple[str, ...] = (),
    inference_timeline: dict[date, frozenset[str]] | None = None,
    *,
    layered_window: LayeredWindow | None = None,
) -> DataStageResult:
    ...
```

- Tiered 활성 시 fetch 시작일은 `min(window.fetch_start_date, layered_window.fetch_start)`다.
- `aligned.datetimes[0] > layered_window.fetch_start`이면 조용히 진행하지 않고 명시적인 coverage 오류를 발생시킨다.

### SWF fold

```python
def build_l1_swf_folds(
    *,
    n_bars: int,
    n_folds: int = 5,
    l1_start_bars: int,
    l1_end_bars: int,
    purge_bars: int,
    embargo_bars: int,
    cal_fraction: float = 0.15,
) -> tuple[WFFold, ...]:
    ...
```

- `warmup_bars`를 `l1_start_bars`로 변경해 의미를 명확히 한다.
- `[l1_start_bars, l1_end_bars)`를 `n_folds + 1`개 순차 block으로 나눈다.
- 첫 block은 initial training 전용이며 OOS로 사용하지 않는다.
- fold `k` 계약:
  - `fit_start = l1_start_bars`
  - `oos_start = l1_start_bars + (k + 1) * block_len`
  - `fit_end = oos_start - purge_bars`
  - `oos_end = next_oos_start`, 마지막 fold는 `l1_end_bars`
  - `cal_len = floor((fit_end - fit_start) * cal_fraction)`
  - `cal_start = fit_end - max(1, cal_len)`
- pre-L1 구간 `[0, l1_start_bars)`은 feature 계산에만 사용하고 event label 학습에는 사용하지 않는다.
- `fit_end <= fit_start` 또는 OOS 길이가 1 bar 미만이면 fallback fold를 만들지 않고 `ValueError`를 발생시킨다.

### Fold training state

```python
FoldFitStatus = Literal[
    "trained",
    "insufficient_fit",
    "empty_oos",
    "constant_prediction",
    "failed",
]
```

```python
@dataclass(slots=True, frozen=True)
class CandidateFoldOutput:
    fold_id: int
    oos_start: int
    oos_end: int
    model_output: CandidateModelOutput
    selected_events: pd.DataFrame
    gate_report: GateValidationReport
    edge_report: EdgeValidationReport
    fit_status: FoldFitStatus
    n_fit: int
    skip_reason: str | None
    gate_model: Any | None = None
    edge_models: Any | None = None
    fit_set: Any | None = None
    calibration_set: Any | None = None
    oos_set: Any | None = None
    timing_profile: dict[str, float] | None = None
```

- `n_fit < cfg.min_fit_obs`이면 `fit_status="insufficient_fit"`이다.
- prior fallback output은 관측용으로 유지할 수 있지만 L1 통계와 SymbolSignal 집계에는 사용하지 않는다.

```python
@dataclass(slots=True, frozen=True)
class FoldDiagnostic:
    fold: int
    ic: float | None
    breadth: float
    n_valid: int
    n_eligible: int
    n_events: int
    n_fit: int
    fit_status: FoldFitStatus
    passed: bool
```

### Point-in-time scope

```python
def _fold_eligible_symbol_mask(
    *,
    aligned: AlignedMarketData,
    fold: WFFold,
    min_bar_coverage: float = 0.80,
) -> NDArray[np.bool_]:
    ...
```

- Tiered aligned symbol scope는 current snapshot selected가 아니라 historical universe union과 data-ready 심볼의 교집합이다.
- fold별 denominator는 다음 조건을 OOS bar의 80% 이상 충족한 심볼 수다.
  - `active_mask`
  - `inference_entry_warm_mask` 또는 `warm_mask`
  - `~entry_block_mask`
  - `~kill_mask`
- `breadth = n_valid / max(1, n_eligible)`로 계산한다.

### Tiered event source

```python
def _tiered_labeled_events(output: CandidatePipelineOutput) -> pd.DataFrame:
    ...
```

- `output.labeled_unfiltered`만 반환한다.
- `labeled_unfiltered`가 없으면 빈 frame fallback 대신 `ValueError("tiered requires unfiltered labeled events")`를 발생시킨다.
- bridge의 전기간 promotion 결과인 `output.labeled`는 Tiered L1 입력으로 사용하지 않는다.
- Tiered 활성 시 bridge symbol scope는 current selected가 아니라 `data_stage.valid_symbols`를 사용해 historical union event를 생성한다.

## Surgical Plan

### `[src/execution/opt_main_futures.py]`

- `[TARGET_FUNCTION_OR_CLASS]`: `run_pipeline`, `_run_universe_stage`, `_run_data_stage`, `_run_strategy_stage`, 신규 `_resolve_layered_window`, 신규 `_tiered_labeled_events`
- `[ALGORITHMIC_FLOW]`:
  1. `run_config.date`로 quarterly window와 layered window를 한 번만 계산한다.
  2. Tiered 활성 시 universe 시작일과 fetch 시작일을 layered 경계까지 확장한다.
  3. bridge signal 생성 scope를 historical data-ready union으로 확장한다.
  4. Tiered에는 unfiltered labeled event와 동일 union aligned data를 전달한다.
  5. current snapshot selected는 live allocation scope로만 유지한다.

### `[src/domain/futures/strategy/walk_forward.py]`

- `[TARGET_FUNCTION_OR_CLASS]`: `build_l1_swf_folds`
- `[ALGORITHMIC_FLOW]`:
  1. L1 기간을 initial train 1개와 OOS `K`개 block으로 분할한다.
  2. 모든 fold에서 `fit_start=l1_start_bars`를 강제한다.
  3. `fit_end < oos_start`와 purge gap을 검증한다.
  4. pre-L1 warm-up bar는 fold label 범위에서 제외한다.
  5. 불충분한 기간에는 오염된 fallback을 생성하지 않는다.
- 시간 복잡도 `O(K)`, 공간 복잡도 `O(K)`.

### `[src/domain/futures/strategy/candidate_contracts.py]`

- `[TARGET_FUNCTION_OR_CLASS]`: `CandidateFoldOutput`, 신규 `FoldFitStatus`
- `[ALGORITHMIC_FLOW]`: 학습 여부, fit 관측 수, skip 이유를 결과 계약에 포함한다.

### `[src/domain/futures/strategy/candidate_workflow.py]`

- `[TARGET_FUNCTION_OR_CLASS]`: `_fit_and_predict_single_fold`
- `[ALGORITHMIC_FLOW]`:
  1. fit dataset 생성 후 `n_fit`을 계산한다.
  2. 최소 fit 관측 수 미달이면 명시적 `insufficient_fit` 상태를 반환한다.
  3. OOS event가 없으면 `empty_oos` 상태를 반환한다.
  4. 학습 완료 후 예측 표준편차가 0이면 `constant_prediction`으로 표시한다.
  5. 상태가 `trained`인 fold만 downstream 통계 후보가 된다.

### `[src/domain/futures/strategy/tiered_workflow.py]`

- `[TARGET_FUNCTION_OR_CLASS]`: `FoldDiagnostic`, `run_l1_swf`, `run_tiered_pipeline`, 신규 `_fold_eligible_symbol_mask`
- `[ALGORITHMIC_FLOW]`:
  1. `fit_status != "trained"` fold는 `compose_symbol_signals()`에 전달하지 않는다.
  2. pooled IC 입력은 trained fold이며 예측과 실현값 길이가 같고 둘 다 분산이 양수인 경우만 허용한다.
  3. invalid fold의 zero fallback은 pooled sample 수에 포함하지 않는다.
  4. fold별 PIT eligible symbol denominator로 breadth를 계산한다.
  5. `trained_fold_coverage < 0.80`이면 alpha gate 계산 전에 L1을 `BLOCKED: insufficient_trained_folds`로 종료한다.
  6. event shapes는 `prediction: [E_f]`, `realized: [E_f]`, market arrays는 `[T, N]`을 유지한다.
- 시간 복잡도 `O(K * (T * N + E_f))`, 공간 복잡도 `O(T * N + sum(E_f))`.

### `[src/domain/futures/strategy/tiered_logging.py]`

- `[TARGET_FUNCTION_OR_CLASS]`: `format_layer1_table`
- `[ALGORITHMIC_FLOW]`: fold table에 `Fit N`, `Eligible N`, `Status/Reason`, 실제 OOS 날짜를 표시한다.

### `[src/domain/futures/optimization/workflow.py]`

- `[TARGET_FUNCTION_OR_CLASS]`: `objective_l1_ic`
- `[ALGORITHMIC_FLOW]`: production pipeline과 동일한 layered window, fold builder, unfiltered event 계약을 사용한다.

### Documentation correction

- `[docs/architecture/allocation.md]`: CPCV 설명을 현재 SWF 계약으로 교체한다.
- `[docs/decisions/signal-eh.md]`: `Fold 1-2 N=0은 버그 아님` 결론을 폐기하고 window mismatch와 `fit=0` 재현 근거를 기록한다.
- `[docs/results/result.md]`: 수정 전 `Pooled IC=-0.095`를 신뢰 가능한 alpha 결론이 아닌 invalid-fold 혼입 결과로 표시하고 재실행 전 상태를 `PENDING_REVALIDATION`으로 변경한다.

# 🧪 Test Scenario Design

### Scenario 1: 동일 기준일 window

- Given: `reference_date="2026-05-01"`
- When: quarterly window와 layered window를 계산
- Then: 두 window의 최종 종료일은 `2026-03-31`, layered L1은 `2023-04-01`, L2는 `2024-10-01`, fetch 시작일은 `2022-04-01`

### Scenario 2: Universe timeline이 L1 전체를 포함

- Given: quarterly IS 시작 `2023-10-01`, layered L1 시작 `2023-04-01`
- When: `_run_universe_stage()` 호출
- Then: `discover_universe_timeline(is_start=2023-04-01, ...)` 호출, 첫 membership window가 첫 L1 OOS보다 늦지 않음

### Scenario 3: Warm-up과 fit 분리

- Given: `l1_start_bars=1000`, `l1_end_bars=7000`, `n_folds=5`, `purge_bars=50`
- When: `build_l1_swf_folds()` 호출
- Then: 첫 fold `fit_start=1000`, 첫 OOS는 initial train block 이후 시작, 모든 fold에서 pre-L1 event index `<1000`은 fit dataset에 포함되지 않음

### Scenario 4: 초기 fold 정상 학습

- Given: L1 시작부터 PIT active인 3개 심볼, initial train block에 심볼별 충분한 labeled event, OOS에 비상수 예측 가능한 synthetic dataset
- When: `run_l1_swf()` 실행
- Then: Fold 1 `fit_status="trained"`, `n_fit >= cfg.min_fit_obs`, `N Events > 0`, IC가 `n/a`가 아님

### Scenario 5: 학습 실패 fold 통계 제외

- Given: `n_fit=0`, OOS event 649개, fallback `expected_net_bps=zeros(649)`
- When: L1 집계
- Then: fold status는 `insufficient_fit`, `signals_per_fold`에 해당 zero signal 없음, pooled sample 수 증가 없음, fold IC `None`

### Scenario 6: 상수 예측 제외

- Given: 학습 상태는 완료됐지만 OOS prediction variance가 0
- When: fold 진단과 pooled IC 계산
- Then: status는 `constant_prediction`, pooled IC 입력에서 제외, 명시적 reason 출력

### Scenario 7: PIT universe breadth

- Given: historical union 6개 심볼 중 Fold 1 OOS에서 4개만 80% 이상 active/warm, valid signal 2개
- When: fold breadth 계산
- Then: `n_eligible=4`, `breadth=0.5`; current snapshot selected 수는 denominator에 영향 없음

### Scenario 8: Current snapshot survivorship 방지

- Given: 과거 Fold 1에서 active였으나 현재 snapshot에서 제외된 심볼 A
- When: Tiered bridge와 aligned scope 구성
- Then: A의 과거 event가 L1 입력에 존재하며 현재 snapshot 제외만으로 제거되지 않음

### Scenario 9: Global promotion leakage 방지

- Given: `labeled_unfiltered`에는 Fold 1 event가 있고 `labeled`에는 미래 promotion 결과로 해당 event가 제거됨
- When: `_tiered_labeled_events()` 호출
- Then: Fold 1 event가 유지된 unfiltered frame 반환

### Scenario 10: Warm-up 데이터 부족 오류

- Given: layered fetch 시작일 `2022-04-01`, 실제 aligned 첫 시각 `2022-10-01`
- When: Tiered coverage 검증
- Then: 요구 시작일과 실제 시작일을 포함한 `ValueError` 발생, 부분 fold 실행 금지

### Scenario 11: 실패 fold 비율 gate

- Given: 5개 fold 중 trained fold 1개
- When: Layer 1 gate 평가
- Then: `trained_fold_coverage=0.20`, 상태 `BLOCKED: insufficient_trained_folds`, alpha 품질 결론 미출력

### Scenario 12: 재현 회귀

- Given: `2026-05-01`, `4h`, 기존 cache, `--phase signal --sync skip`
- When: 수정 후 실행
- Then: Fold 1~5 모두 실제 window와 `Fit N`을 출력하고, `insufficient_fit` fold의 zero fallback이 pooled N에 포함되지 않음

## Verification

```bash
uv run pytest tests/unit/domain/futures/strategy/test_walk_forward.py -k "l1_swf" --tb=short
uv run pytest tests/unit/domain/futures/strategy/test_tiered_workflow.py -k "l1_swf or pooled or eligible" --tb=short
uv run pytest tests/unit/domain/futures/strategy/test_candidate_workflow.py -k "insufficient_fit or constant_prediction" --tb=short
uv run pytest tests/unit/execution/test_opt_main_futures_strategy_mode.py -k "tiered or layered or unfiltered" --tb=short
uv run pytest tests/unit/application/futures/optimization/test_universe_service.py -k "layered or timeline" --tb=short
uv run pytest tests/unit/domain/futures/strategy/test_walk_forward.py tests/unit/domain/futures/strategy/test_tiered_workflow.py tests/unit/domain/futures/strategy/test_candidate_workflow.py --cov=src/domain/futures/strategy --cov-report=term-missing
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl LOG_LEVEL=INFO PYTHONPATH=. timeout 240 uv run python src/execution/opt_main_futures.py --phase signal --sync skip --timeframe 4h --date 2026-05-01 --trials 1
```

## Acceptance Criteria

- 모든 L1 fold의 universe timeline과 loaded data가 해당 fold 시작 이전부터 존재한다.
- indicator warm-up 데이터는 feature 계산에는 사용되지만 ML label fit에는 포함되지 않는다.
- initial training 없는 OOS fold를 만들지 않는다.
- 학습 실패 및 상수 예측 fold는 SymbolSignal, pooled IC, NW t-stat 표본에서 제외된다.
- current snapshot 선택으로 historical symbol을 제거하지 않는다.
- 전기간 promotion 결과를 초기 fold event 선택에 사용하지 않는다.
- 수정 후 통계로 재검증하기 전에는 기존 `Pooled IC=-0.095`를 alpha 결론으로 사용하지 않는다.
