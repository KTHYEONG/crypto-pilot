# Spec: Fix realized return shifting bug in OOS evaluation

## 1. 문제 분석 및 배경

현재 `opt_main_futures.py`의 OOS 평가 단계에서 `ALPHA_PASS`가 실패하며 `portfolio_ic=0.0000`, `basket_net_bps=nan`, `multi_horizon_sweep_passes=0/3` 등 모든 사후 경제성 지표가 비정상적으로 무너지는 현상이 발생했습니다.

### 원인 분석 (Reindexing-before-shifting Bug)
- **현상:** Dense ranker 자체의 검증 스킬(`SCORE-IC`)은 `ic=0.0347`, `t=4.51`로 우수하게 측정되었으나, 사후 포트폴리오 평가(`evaluate_alpha`, `sweep_horizon_breakeven`) 진입 시점에 모든 성과 지표가 0 또는 NaN으로 파괴됨.
- **버그 코드 위치 1 (Primary Evaluation):**
  ```python
  # src/execution/opt_main_futures.py (Lines 926-931)
  for sym in common_syms:
      df = data_stage.data_maps[sym][tf].set_index("datetime")
      close = df["close"].reindex(pivot_long.index) # <-- OOS 인덱스로 먼저 필터링
      fwd_ret = np.log(close.shift(-horizon) / close) # <-- 필터링된 짧은 인덱스 상에서 shift 실행 (데이터 유실 및 look-forward 왜곡)
  ```
- **버그 코드 위치 2 (Multi-Horizon Sweep):**
  ```python
  # src/execution/opt_main_futures.py (Lines 1274-1280)
  for h in [6, 12, 18]:
      for sym in common_syms:
          df = data_stage.data_maps[sym][tf].set_index("datetime")
          close = df["close"].reindex(common_idx) # <-- 먼저 필터링
          fwd = np.log(close.shift(-h) / close) # <-- shift 실행
  ```
- **해결 방안:**
  전체 연속된 시계열 데이터 상에서 `shift(-h)`를 적용하여 물리적 시간 상의 정확한 forward return을 구한 뒤, 평가 대상 인덱스(`pivot_long.index` 또는 `common_idx`)로 `reindex`를 수행해야 시계열 무결성이 보존됩니다.

---

## 2. Target Files
- `src/execution/opt_main_futures.py`

---

## 3. Surgical Plan

### `src/execution/opt_main_futures.py` [REPLACE]

#### [Chunk 1: Primary realized return calculation]
**AS-IS:**
```python
    realized_rows: dict[str, pd.Series] = {}
    for sym in common_syms:
        df = data_stage.data_maps[sym][tf].set_index("datetime")
        close = df["close"].reindex(pivot_long.index)
        fwd_ret = np.log(close.shift(-horizon) / close)
        realized_rows[sym] = fwd_ret
    realized_df = pd.DataFrame(realized_rows, index=pivot_long.index)
```

**TO-BE:**
```python
    realized_rows: dict[str, pd.Series] = {}
    for sym in common_syms:
        df = data_stage.data_maps[sym][tf].set_index("datetime")
        # 전체 시계열에서 shift 후 OOS 인덱스로 reindex 수행
        fwd_ret = np.log(df["close"].shift(-horizon) / df["close"]).reindex(pivot_long.index)
        realized_rows[sym] = fwd_ret
    realized_df = pd.DataFrame(realized_rows, index=pivot_long.index)
```

#### [Chunk 2: Multi-horizon sweep realized return calculation]
**AS-IS:**
```python
    # sweep horizons
    realized_map: dict[int, np.ndarray] = {}
    alpha_long_map: dict[int, np.ndarray] = {}
    alpha_short_map: dict[int, np.ndarray] = {}
    for h in [6, 12, 18]:
        r_rows: dict[str, pd.Series] = {}
        for sym in common_syms:
            df = data_stage.data_maps[sym][tf].set_index("datetime")
            close = df["close"].reindex(common_idx)
            fwd = np.log(close.shift(-h) / close)
            r_rows[sym] = fwd
        r_df = pd.DataFrame(r_rows, index=common_idx)
        r_arr = r_df.iloc[:-h].to_numpy(dtype=np.float64)
        clip = min(len(al), len(r_arr))
        realized_map[h] = r_arr[:clip]
        alpha_long_map[h] = al[:clip]
        alpha_short_map[h] = as_[:clip]
```

**TO-BE:**
```python
    # sweep horizons
    realized_map: dict[int, np.ndarray] = {}
    alpha_long_map: dict[int, np.ndarray] = {}
    alpha_short_map: dict[int, np.ndarray] = {}
    for h in [6, 12, 18]:
        r_rows: dict[str, pd.Series] = {}
        for sym in common_syms:
            df = data_stage.data_maps[sym][tf].set_index("datetime")
            # 전체 시계열에서 shift 후 OOS 인덱스로 reindex 수행
            fwd = np.log(df["close"].shift(-h) / df["close"]).reindex(common_idx)
            r_rows[sym] = fwd
        r_df = pd.DataFrame(r_rows, index=common_idx)
        r_arr = r_df.iloc[:-h].to_numpy(dtype=np.float64)
        clip = min(len(al), len(r_arr))
        realized_map[h] = r_arr[:clip]
        alpha_long_map[h] = al[:clip]
        alpha_short_map[h] = as_[:clip]
```

---

## 4. Verification

수정 후 아래 테스트 명령어를 사용하여 `opt_main_futures.py` 실행이 정상적으로 통과하고 지표 왜곡이 해소되는지 확인합니다.

```bash
uv run python src/execution/opt_main_futures.py --mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01
```

**기대 결과:**
- `SCORE-IC` 뿐만 아니라 사후 포트폴리오 IC인 `net_ic` 및 `resid_ic`가 정상 수치(예: > 0.01)로 복원됩니다.
- `basket_net_bps`가 NaN이 아닌 실수값으로 출력되며 통과 여부가 결정됩니다.
- `📈 SWEEP`의 각 horizon 별 `ic`가 0.000이 아닌 유의미한 상관계수 값을 나타냅니다.
