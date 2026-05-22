# Futures Optimization Architecture Correction Spec

**작성일**: 2026-05-22  
**목적**: 현재 futures 최적화 경로에서 확인된 아키텍처 drift와 논리 오류를 코드로 바로 수정할 수 있도록 구현 사양을 고정한다.

---

## 1. 문제 정의

현재 구현은 아래 4개 축에서 문서 설계와 실제 동작이 어긋나 있다.

1. **분기 유니버스 멤버십 미반영**
   - 실제 최적화/백테스트는 quarter별 snapshot membership이 아니라 `union_symbols` 고정 집합으로 동작한다.
   - 결과적으로 퇴출 심볼 강제 청산, 신규 심볼 warm-up 후 진입이 구현되지 않는다.

2. **유니버스 상태 전이 미적용**
   - `previous_selection`이 snapshot build에 전달되지 않아 hysteresis / `k_in` / `k_out` / `min_dwell_days`가 실제로 작동하지 않는다.

3. **수집 기간과 실효 평가 기간의 계약 불명확**
   - orchestration은 `fetch_start -> IS start -> OOS start -> OOS end`를 계산하지만,
     실제 symbol별 usable bars가 문서상 24M IS + 6M OOS + warmup를 충족하는지 hard-check 하지 않는다.

4. **Optuna 실행 계약 문서화 부족**
   - `--trials`가 phase 총량이 아니라 A1/A2/B 각각에 적용되는 구조,
     sampler/pruner/storage/worker 정책,
     phase B single-worker 결정론 정책이 문서에 고정되어 있지 않다.

---

## 2. 구현 목표

수정 후 시스템은 반드시 아래를 만족해야 한다.

1. 백테스트/최적화는 **quarterly evolving universe**로 동작한다.
2. 각 decision bar에서 **해당 시점 active membership**만 거래 가능해야 한다.
3. 신규 심볼은 **warm-up 충족 전까지 거래 금지**되어야 한다.
4. 퇴출 심볼은 **다음 executable 시점에 target 0 및 강제 청산**되어야 한다.
5. 각 symbol은 **필요 수집 기간 전체**를 확보하지 못하면 최적화 universe에서 제외되어야 한다.
6. 각 phase와 membership 전이에 대해 **추적 가능한 상세 logging**이 남아야 한다.

---

## 3. 용어 및 시간 계약

### 3.1 기준 창

`get_quarterly_window(reference_date)`가 반환하는 기간을 canonical로 유지한다.

- `fetch_start`: `is_start - 365일`
- `is_start`: `oos_start - 24개월`
- `oos_start`: 현재 분기 시작 6개월 전
- `oos_end`: 현재 분기 시작 직전일

### 3.2 추가 계약

실제 구현에서는 아래 개념을 명시적으로 분리한다.

- `fetch_window`: 원천 수집 범위
- `usable_window`: 실제 symbol별로 결측 없이 사용할 수 있는 범위
- `warmup_bars_required`: signal / risk / covariance / embargo / calibration에 필요한 최소 선행 bar
- `membership_effective_from`: 분기 snapshot이 실제 거래에 효력을 갖는 첫 decision bar

---

## 4. 핵심 수정 사양

### 4.1 Quarterly membership timeline 도입

#### 대상 파일

- `src/execution/opt_main_futures.py`
- `src/domain/futures/universe/pipeline.py`
- `src/domain/futures/optimization/optimizer.py`
- `src/domain/futures/optimization/data_aligner.py`
- `src/domain/futures/backtest_engine.py`

#### 신규 개념

`UniverseMembershipTimeline`

```python
@dataclass(slots=True, frozen=True)
class UniverseMembershipWindow:
    effective_from: pd.Timestamp
    effective_to: pd.Timestamp | None
    snapshot_as_of: str
    active_symbols: tuple[str, ...]
    entry_symbols: tuple[str, ...]
    exit_symbols: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class UniverseMembershipTimeline:
    tf: str
    windows: tuple[UniverseMembershipWindow, ...]
```

#### 동작 규칙

1. `is_start ~ oos_end`를 덮는 quarter sequence를 생성한다.
2. 각 quarter snapshot build/load 시 `previous_selection`을 반드시 전달한다.
3. 각 snapshot의 selected symbols로 `effective_from ~ next_effective_from` membership window를 만든다.
4. optimizer/backtest는 union universe를 로드하되, 실제 거래 가능 여부는 membership timeline으로 제어한다.

#### 필수 구현 포인트

- `opt_main_futures.py`의 `_discover_evolution_symbols()`는
  단순 `union_symbols` 반환이 아니라 아래를 반환해야 한다.

```python
tuple[list[str], UniverseMembershipTimeline, list[UniverseSnapshot], pd.DataFrame]
```

- `latest_snapshot`만 남기는 현재 구조를 제거한다.
- quarter별 snapshot 객체 목록을 유지하여 audit 가능하게 한다.

---

### 4.2 Hysteresis / Dwell 실제 반영

#### 대상 파일

- `src/execution/opt_main_futures.py`
- `src/domain/futures/universe/pipeline.py`
- `src/domain/futures/universe/selection.py`

#### 수정 규칙

1. snapshot을 quarter 순회하며 build할 때 직전 quarter selected symbols를 `previous_selection`으로 전달한다.
2. `selection.py`의 dwell 판단에 필요한 컬럼이 없다면 임시로라도 아래 둘 중 하나를 넣는다.
   - `membership_days`
   - `dwell_days`
3. dwell source가 없으면 현재처럼 사실상 `k_out`만 적용되므로, 이를 금지한다.
4. dwell 계산은 snapshot 시점 기준 `first_selected_date`를 바탕으로 결정해야 하며, 추정치로 대체하지 않는다.

#### 추가 계약

- `logs/futures/universe/membership_state.parquet`
  - columns:
    - `quarter_start`
    - `symbol`
    - `is_selected`
    - `selection_reason`
    - `rank`
    - `dwell_days`
    - `was_prev_member`

---

### 4.3 Membership mask를 kill_signal로 주입

#### 대상 파일

- `src/domain/futures/optimization/optimizer.py`
- `src/domain/futures/optimization/data_aligner.py`
- `src/domain/futures/backtest_engine.py`
- `src/domain/futures/portfolio/execution_sim.py`

#### 핵심 원칙

union universe는 **데이터 적재 범위**일 뿐이고,
실제 체결 가능성은 `universe_active_mask_2d`로 제어한다.

#### 신규 2D 배열 계약

```python
universe_active_mask_2d: np.ndarray  # shape [B4h, N], 1.0=tradeable, 0.0=inactive
universe_entry_warm_mask_2d: np.ndarray  # shape [B4h, N], 1.0=warmup done
membership_kill_signal_2d: np.ndarray  # shape [B4h, N], 1.0=next bar force exit
```

#### 조합 규칙

기존 `kill_signal`이 있으면 아래처럼 합친다.

```python
effective_kill_signal = np.maximum(
    raw_kill_signal,
    membership_kill_signal_2d,
)
```

#### 거래 규칙

1. `universe_active_mask_2d[t, i] == 0` 이면 신규 진입 금지
2. 직전 bar active, 현재 bar inactive가 되면 `membership_kill_signal_2d[t, i] = 1`
3. 신규 active 심볼은 `warmup_bars_required` 충족 전까지 진입 금지
4. warm-up 중에는 signal 계산은 허용하되 `target_weight=0` 이어야 한다

#### 구현 방법

- timeline을 4h decision index 기준으로 펼쳐 symbol별 boolean matrix 생성
- 해당 matrix를 `aligned_data`에 넣고
- `_cached_kill_fund_lev()` 또는 `_run_portfolio_numba_block()` 이전 단계에서 effective kill로 합성

---

### 4.4 Warm-up 및 데이터 sufficiency hard gate

#### 대상 파일

- `src/execution/opt_main_futures.py`
- `src/domain/futures/optimization/opt_data_utils.py`
- `src/domain/futures/optimization/optimizer.py`

#### 신규 계산 항목

`warmup_bars_required`는 아래 최댓값으로 계산한다.

```python
max(
    strategy_lookback_bars,
    covariance_lookback_bars,
    composer_sigma_lookback_bars,
    atr_period,
    embargo_bars,
    platt_min_train_bars,
    minimum_membership_warm_bars,
)
```

#### hard gate 규칙

각 symbol은 최소한 아래를 만족해야 한다.

1. `fetch_start ~ oos_end` 전체 원천 수집 존재
2. `is_start` 이전에 `warmup_bars_required` 확보
3. `is_start ~ oos_end` 구간에서 required timeframe 연속성 만족
4. 1m intrabar 모드면 `exec_1m`도 동일 창을 충족

#### 실패 시 처리

- 해당 symbol은 `valid_symbols`에서 제거
- 제거 사유를 별도 로그와 parquet로 남긴다
- 제거 후 `eff_ref_len`이 목표 최소 기간 미만이면 optimization 중단

#### 추가 hard gate

현재 `eff_ref_len < 200` 같은 약한 조건 대신, 실제 월수 기준 검사를 넣는다.

예시:

- `4h` 기준
  - IS usable bars >= `24개월 * 30일 * 6`
  - OOS usable bars >= `6개월 * 30일 * 6`

윤년/월길이 오차는 허용하되 95% 미만이면 fail 처리한다.

---

### 4.5 Data collection 범위 보강

#### 대상 파일

- `src/domain/futures/universe/sync_utils.py`
- `src/domain/futures/data_loader.py`
- 필요 시 `src/core/utils/binance_vision.py`

#### 기존 문제

현재 universe sync는 실시간 `ticker24hr` 상위 40% 후보 중심이라
역사 시점의 유니버스 후보군을 완전하게 복원하지 못한다.

#### 수정 사양

1. universe ledger sync는 **historical symbol master** 기반으로 동작해야 한다.
2. symbol source 우선순위:
   - Vision XML all symbols
   - FAPI exchangeInfo onboardDate / status
   - 기존 ledger existing symbols
3. “현재 volume 상위 40%”는 sync 범위 축소 용도가 아니라
   **optional acceleration mode**로만 남긴다.
4. 기본 모드는 quarter 기간 전체에 대해 필요한 모든 symbol을 수집한다.

#### 필수 옵션

- `--sync-mode full_history_master` 기본
- `--sync-mode elite_fast` 선택

#### full_history_master 규칙

- 전체 symbol master를 확보
- 각 quarter snapshot에 한 번이라도 등장 가능한 symbol은 수집 대상에 포함
- delisted symbol도 제외하지 않는다

---

### 4.6 Optuna 실행 계약 명문화 및 코드 정렬

#### 대상 파일

- `docs/futures/*` 후속 반영
- `src/execution/opt_main_futures.py`
- `src/domain/futures/optimization/phase_runner.py`
- `src/domain/futures/optimization/phase_samplers.py`
- `src/domain/futures/optimization/run_tracker.py`

#### 문서와 코드가 맞춰야 할 내용

1. `--trials` 의미
   - 현재는 A1/A2/B 각각에 동일 적용
   - 총 실행 trial 수는 phase 합계
2. phase별 sampler
   - A1: TPE or NSGA-II
   - A2: BoTorchSampler fallback TPE
   - B: CMA-ES fallback
3. pruner
   - WilcoxonPruner
4. storage
   - SQLite WAL
5. phase B worker
   - 항상 single-worker

#### 코드 보완 요구

- CLI help와 로그에 아래를 명시한다.

```text
[OPTUNA-CONTRACT] requested_trials_per_phase=100 total_planned_trials=300 phases=A1,A2,B
```

- run summary에 아래를 포함한다.
  - `trials_per_phase`
  - `planned_total_trials`
  - `completed_trials_per_phase`
  - `sampler_by_phase`
  - `worker_by_phase`
  - `storage_url`

---

## 5. 상세 로깅 사양

이번 수정에서 추가하는 모든 기능은 세부 로깅이 반드시 있어야 한다.

### 5.1 공통 원칙

- `print()` 금지
- `logging` 사용
- event 이름은 검색 가능한 고정 prefix 사용
- key-value 형태로 남긴다

### 5.2 필수 로그 이벤트

#### A. Universe timeline 생성

logger: `opt_futures`

```text
[UNIVERSE-TIMELINE] quarter=2024-10-01 snapshot_as_of=2024-10-01 selected=18 new=3 dropped=2 retained=15
```

```text
[UNIVERSE-TIMELINE] symbol=SOLUSDT effective_from=2024-10-01 effective_to=2025-01-01 state=entry
```

#### B. Hysteresis / dwell 적용

logger: `src.domain.futures.universe.selection`

```text
[UNIVERSE-HYST] symbol=ADAUSDT prev_member=true rank=27 k_in=20 k_out=35 dwell_days=54 retained=true reason=retained_min_dwell
```

#### C. Data sufficiency 검사

logger: `opt_data_utils`

```text
[DATA-SUFFICIENCY] symbol=XRPUSDT tf=4h fetch_ok=true warmup_bars=252 required_is_bars=4320 actual_is_bars=4388 required_oos_bars=1080 actual_oos_bars=1092 pass=true
```

실패 예:

```text
[DATA-SUFFICIENCY] symbol=LINKUSDT tf=4h pass=false reason=missing_exec_1m coverage=0.81
```

#### D. Membership mask 적용

logger: `optimizer`

```text
[MEMBERSHIP-MASK] symbol=DOGEUSDT active_ratio=0.42 warm_ratio=0.37 forced_exit_count=3 blocked_entry_count=118
```

#### E. Forced exit 실행

logger: `backtest_engine`

```text
[MEMBERSHIP-EXIT] symbol=APTUSDT bar=184 reason=universe_dropout next_open_forced=true
```

#### F. Optuna 계약

logger: `run_tracker`

```text
[OPTUNA-CONTRACT] phase=phase_a1 sampler=TPESampler pruner=WilcoxonPruner workers=4 trials=100
```

```text
[OPTUNA-CONTRACT] phase=phase_b sampler=CmaEsSampler workers=1 trials=100 rationale=sqlite_complete_trial_race_prevention
```

### 5.3 영속 로그 산출물

아래 파일들을 남긴다.

- `logs/futures/universe/universe_timeline.parquet`
- `logs/futures/universe/membership_state.parquet`
- `logs/futures/data/data_sufficiency_report.parquet`
- `logs/futures/optimization/optuna_contract.json`
- `logs/futures/optimization/membership_mask_stats.parquet`

---

## 6. 모듈별 구현 작업

### 6.1 `src/execution/opt_main_futures.py`

#### 작업

1. `_discover_evolution_symbols()`를 `timeline builder`로 확장
2. quarter 순회 시 `previous_selection` 전달
3. `snapshot` 단일 객체 의존 제거
4. `validate_universe_quality()` 입력을 latest snapshot 하나가 아니라
   `timeline + latest snapshot + prior snapshot` 기반으로 정리
5. `load_futures_data_maps_for_symbols()` 호출 후 data sufficiency hard gate 추가
6. `MLPhaseDContext`에 timeline 전달

#### 신규 context 필드

```python
universe_timeline: UniverseMembershipTimeline | None = None
warmup_bars_required: int = 0
data_sufficiency_report: dict[str, Any] | None = None
```

---

### 6.2 `src/domain/futures/universe/pipeline.py`

#### 작업

1. `build_universe(... previous_selection=...)`를 실제 quarter loop에서 사용
2. snapshot metadata에 아래 추가 검토
   - `prev_selected_count`
   - `new_entry_count`
   - `dropout_count`
   - `retained_count`

---

### 6.3 `src/domain/futures/optimization/opt_data_utils.py`

#### 작업

1. symbol별 수집 충족도 계산 함수 추가
2. `exec_1m`, `4h`, `1h`, funding 각각 coverage 산출
3. `is_start` 이전 warm-up 확보 여부 판단
4. 부족한 symbol 제거 + audit 저장

#### 신규 함수

```python
def evaluate_symbol_data_sufficiency(...) -> dict[str, Any]: ...
def filter_symbols_by_data_sufficiency(...) -> tuple[...]: ...
```

---

### 6.4 `src/domain/futures/optimization/optimizer.py`

#### 작업

1. timeline -> per-bar membership masks 생성
2. `aligned_data`에 membership arrays 주입
3. strategy mode / normal mode 공통으로 `effective_kill_signal` 사용
4. first-leg diagnostics에 membership stats 포함

#### 신규 함수

```python
def build_universe_membership_arrays(...) -> dict[str, np.ndarray]: ...
def merge_membership_constraints_into_aligned(...) -> None: ...
```

---

### 6.5 `src/domain/futures/backtest_engine.py`

#### 작업

1. `kill_signal` fallback zero 처리 전 membership kill 우선 합성
2. `target_weights` 생성 후 inactive / warm-up incomplete symbol weight를 0으로 clamp
3. membership forced exit count를 diag로 반환

---

### 6.6 `src/domain/futures/universe/sync_utils.py`

#### 작업

1. `smart_filter_symbols()`를 기본 sync source로 쓰지 않도록 구조 변경
2. historical master mode 추가
3. delisted / inactive historical symbols 포함 가능하도록 symbol source 확대
4. sync coverage report 저장

---

## 7. 검증 기준

### 7.1 단위 검증

필수 테스트 추가:

- timeline이 quarter 경계에서 올바르게 생성되는지
- `previous_selection` 전달 시 hysteresis 유지가 되는지
- dropout 시 next bar forced exit가 걸리는지
- 신규 심볼이 warm-up 전 진입하지 않는지
- data sufficiency 실패 symbol이 제거되는지
- `--trials` 로그가 per-phase / total planned 값을 정확히 보여주는지

### 7.2 시나리오 검증

1. `reference_date` 하나를 고정하고
2. 2개 이상 quarter 변화를 포함한 구간에서
3. 특정 symbol이 진입/퇴출되는 케이스를 골라
4. 로그와 membership parquet가 실제 백테스트 체결과 일치해야 한다

### 7.3 Hard acceptance

아래 중 하나라도 실패하면 merge 금지.

1. membership timeline은 있으나 backtest kill 반영이 안 됨
2. warm-up 미충족 symbol이 실제 trade 발생
3. `--trials` 해석이 로그/summary에서 불명확
4. data sufficiency audit 파일 미생성
5. dropout symbol이 union universe 내에서 계속 trade 가능

---

## 8. 구현 순서

1. `UniverseMembershipTimeline` 계약 추가
2. quarter snapshot build에 `previous_selection` 연결
3. timeline parquet/log 생성
4. data sufficiency evaluator 추가
5. optimizer aligned_data에 membership masks 주입
6. backtest effective kill / warm-up clamp 적용
7. Optuna contract logging 추가
8. 관련 테스트 및 문서 후속 갱신

---

## 9. 비범위

이번 작업에서 아래는 직접 해결 대상이 아니다.

1. alpha 품질 자체 개선
2. HMM regime 모델 성능 향상
3. 새로운 cost model 발명
4. 전략 파라미터 튜닝

이번 작업의 목적은 **아키텍처 계약이 실제 실행 경로에 정확히 반영되도록 만드는 것**이다.

---

## 10. 최종 산출물

구현 완료 후 최소 산출물:

1. 분기별 유니버스 상태가 실제 백테스트에 반영된 코드
2. 데이터 sufficiency hard gate 코드
3. Optuna 실행 계약 logging 코드
4. 관련 unit/integration tests
5. `docs/futures/*` 후속 정리 문서
6. 아래 로그 산출물 생성 확인
   - `universe_timeline.parquet`
   - `membership_state.parquet`
   - `data_sufficiency_report.parquet`
   - `optuna_contract.json`
   - `membership_mask_stats.parquet`
