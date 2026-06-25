# 🎯 Objective
L2가 매 시점마다 "지금 시장 상황에 맞는 최적 sleeve를 선택"하도록 재설계한다.
기존 **고정 평균 풀링**(`_combine_sleeve_signals_to_symbol`)을 **regime×family×TF 버킷별 동적 라우팅**으로 전환.

**근거 (실측)**: regime×family×TF 버킷 causal fit→oos corr = +0.14~+0.33 (4 split × 2 min_n, 8/8 양수, 7/8 p<0.05). 무조건(P2a) = 음수였으나 조건부에서 sign-flip → 조건부 신호 실재 확정.

---

# 📦 Context & Dependencies

## 핵심 타입 (기존, 변경 없음)
```python
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    L2SimulationCache,        # pre-computed sleeve matrices
    Layer2AllocationConfig,   # L2 config (필드 추가)
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.tiered_workflow.l2_meta import _parse_meta_group_ids
from numpy.typing import NDArray
import numpy as np
```

## L2SimulationCache 핵심 필드 (읽기 전용)
| 필드 | Shape | 의미 |
|---|---|---|
| `signal_mask_2d` | `[T, S] bool` | bar t에 sleeve j 활성여부 |
| `side_2d` | `[T, S] float64` | sleeve 방향 (+1/-1) |
| `holding_bars_2d` | `[T, S] float64` | sleeve 예상 holding bars |
| `sleeve_ids` | `tuple[(sym, strat_id), ...]` len=S | sleeve 식별자 |
| `sleeve_to_sym` | `[S] int64` | sleeve j → symbol col |
| `sleeve_to_tf` | `tuple[str, ...]` len=S | sleeve native TF (예: "4h") |

## 기존 함수: `_parse_meta_group_ids` (l2_meta.py:45)
```python
def _parse_meta_group_ids(strategy_id: str) -> tuple[str, str]:
    # "donchian_72_4h" → ("donchian_72", "4h")
    # "trend_pullback_continuation:tpc_10_50_8h" → (..., "8h")
```

## 수정 대상: OOS loop 핵심 anchor (awf_sim.py:1608)
```python
# 현재 코드 (변경 대상):
valid_signals, friction_by_symbol = _combine_sleeve_signals_to_symbol(
    _oos_sleeve_sigs,
    method=config.l2_sleeve_combine_method,
    conviction_cap_mult=config.l2_sleeve_conviction_cap_mult,
    sleeve_edges=_oos_sleeve_edges,
)
```

---

# ✍️ Contract Changes

## C-B1: `compute_bucket_realized_edges` (신규, `l2_meta.py`)
```python
def compute_bucket_realized_edges(
    cache: L2SimulationCache,
    aligned: AlignedMarketData,
    fit_start: int,
    fit_end: int,
    regime_code_1d: NDArray[np.int8],
    *,
    cost_bps: float = 6.0,
    min_n: int = 30,
    shrinkage: float = 0.3,
) -> dict[tuple[int, str, str], float]:
    """fit-leg [fit_start, fit_end) 구간에서 버킷별 실현 순엣지 계산.

    버킷 = (regime_code, family, TF) triplet.
    실현엣지 = side_j * fwd_ret(sym_j) * 10000 - cost_bps, bar 단위 평균.
    min_n 미달 버킷은 family prior로 shrinkage 보정:
      bucket_edge = (1-shrinkage)*raw_edge + shrinkage*family_prior.

    Returns:
        {(regime, family, TF): edge_bps}. 미관측 버킷은 포함 안 됨(KeyError → 0).
    """
```

**알고리즘**:
1. `close_2d`에서 `fwd_ret[t, sym] = (close[t+1] - close[t]) / close[t] * 10000` 사전계산 (단위: bps)
2. `t in [fit_start, fit_end)`, 활성(`signal_mask_2d[t,j]`) sleeve j 순회
3. `regime = int(regime_code_1d[t])`, `family, tf = _parse_meta_group_ids(sleeve_ids[j][1])`
4. `edge = side_2d[t,j] * fwd_ret[t, sleeve_to_sym[j]] - cost_bps`
5. 버킷별 `sum_edge`, `count` 누적
6. `raw_edge = sum_edge / count` (count>0), family prior = 같은 family 전체 raw_edge 평균
7. shrinkage 적용: count < min_n → `(1-shrinkage)*raw_edge + shrinkage*family_prior`
8. return dict

## C-B2: `filter_sleeves_by_bucket` (신규, `l2_meta.py`)
```python
def filter_sleeves_by_bucket(
    sleeve_sigs: dict[tuple[str, str], "SymbolSignal"],
    bucket_edges: dict[tuple[int, str, str], float],
    regime_now: int,
    *,
    edge_floor_bps: float = 100.0,
) -> dict[tuple[str, str], "SymbolSignal"]:
    """현재 regime의 버킷 엣지로 sleeve 필터링.

    (sym, strat_id) → 해당 버킷 edge > edge_floor_bps 인 sleeve만 통과.
    버킷 미관측(KeyError) sleeve는 edge=0 처리 → 통상 제거됨.
    sleeve_sigs가 비어있으면 빈 dict 반환.

    Returns:
        통과한 sleeve만 담긴 dict (순서 보존).
    """
```

**알고리즘**:
1. 빈 입력 early-return
2. 각 `(sym, strat_id)` → `family, tf = _parse_meta_group_ids(strat_id)`
3. `key = (regime_now, family, tf)`
4. `edge = bucket_edges.get(key, 0.0)`
5. `edge > edge_floor_bps` → 통과

## C-B3: `Layer2AllocationConfig` 신규 필드 (`dataclasses.py`)
```python
# 기존 필드들 다음에 추가
l2_routing_mode: Literal["pool", "bucket"] = "pool"
# "pool": 기존 _combine_sleeve_signals_to_symbol (하위호환 기본값)
# "bucket": regime×family×TF 버킷 필터 → pool

l2_bucket_min_n: int = 30
# 버킷 안정성: fit-leg 최소 event 수. 미달 시 shrinkage 적용

l2_bucket_shrinkage: float = 0.3
# 과적합 방지: raw_edge → family prior 방향 30% 축소
# bucket_edge = 0.7 * raw + 0.3 * family_prior

l2_bucket_edge_floor_bps: float = 100.0
# 필터 임계값 (bps). 이 이하 버킷은 OOS에서 배치 안 함
```

## C-B4: OOS 루프 수정 (`awf_sim.py`, `_run_awf_simulation` 내부)

**삽입 위치**: fit-leg에서 1회 계산(fold 시작), OOS bar t마다 필터 적용.

**fit-leg 계산 anchor** (oos_start 진입 직전, `_tf_inclusion_enabled` 블록 근처):
```python
# 신규: bucket routing이 켜진 경우 fit-leg 실현엣지 계산 (1회)
_bucket_edges: dict[tuple[int, str, str], float] = {}
if config.l2_routing_mode == "bucket":
    from src.domain.futures.strategy.tiered_workflow.l2_meta import (
        compute_bucket_realized_edges,
    )
    _bucket_edges = compute_bucket_realized_edges(
        cache=shared_l2_cache,
        aligned=aligned,
        fit_start=fold.fit_start,
        fit_end=fold.oos_start,
        regime_code_1d=_regime_code_1d,  # 기존 compute_market_regime_context 결과
        cost_bps=config.l2_bucket_cost_bps,
        min_n=config.l2_bucket_min_n,
        shrinkage=config.l2_bucket_shrinkage,
    )
```

**OOS bar t 필터 anchor** (현재 anchor 직전 삽입):
```python
# 현재 코드 (변경 대상 바로 위에 삽입):
# valid_signals, friction_by_symbol = _combine_sleeve_signals_to_symbol(...)

# 신규: bucket 모드면 먼저 sleeve 필터링
if config.l2_routing_mode == "bucket" and _bucket_edges:
    from src.domain.futures.strategy.tiered_workflow.l2_meta import filter_sleeves_by_bucket
    _regime_now = int(_regime_code_1d[t]) if t < len(_regime_code_1d) else 0
    _oos_sleeve_sigs = filter_sleeves_by_bucket(
        _oos_sleeve_sigs,
        _bucket_edges,
        _regime_now,
        edge_floor_bps=config.l2_bucket_edge_floor_bps,
    )
```

---

# 🧪 TDD Test Scenario Matrix

**파일**: `tests/unit/domain/futures/strategy/tiered_workflow/test_l2_bucket_routing.py`

## Fixtures (공통)
```python
import numpy as np, pytest
from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    compute_bucket_realized_edges, filter_sleeves_by_bucket,
)

# minimal mock cache
def _make_cache(T=10, S=2):
    from unittest.mock import MagicMock
    cache = MagicMock()
    cache.signal_mask_2d = np.zeros((T, S), dtype=bool)
    cache.side_2d = np.ones((T, S), dtype=np.float64)
    cache.holding_bars_2d = np.ones((T, S), dtype=np.float64)
    cache.sleeve_to_sym = np.array([0, 0], dtype=np.int64)  # 두 sleeve 모두 sym=0
    cache.sleeve_ids = (("BTC", "donchian_72_4h"), ("BTC", "trend_pullback_4h"))
    cache.sleeve_to_tf = ("4h", "4h")
    return cache

def _make_aligned(T=10, N=1):
    from unittest.mock import MagicMock
    aligned = MagicMock()
    aligned.close_2d = np.ones((T, N), dtype=np.float64)
    return aligned
```

## S1 — 방향정합: 상승 장, side=+1 → edge > 0
```python
def test_compute_bucket_edges_positive_when_price_rises():
    T, S = 6, 1
    cache = _make_cache(T, S)
    cache.signal_mask_2d[1:5, 0] = True
    cache.side_2d[:] = 1.0
    aligned = _make_aligned(T, 1)
    # 가격 1→2 상승 (fwd_ret = +100%)
    aligned.close_2d = np.array([[1.0],[1.5],[2.0],[2.0],[2.0],[2.0]])
    regime = np.array([1]*T, dtype=np.int8)

    result = compute_bucket_realized_edges(cache, aligned, 0, T, regime, cost_bps=1.0)

    assert len(result) > 0
    key = list(result.keys())[0]
    assert result[key] > 0.0
```

## S2 — 방향역전: side=+1, 가격 하락 → edge < 0
```python
def test_compute_bucket_edges_negative_when_price_falls():
    T, S = 6, 1
    cache = _make_cache(T, S)
    cache.signal_mask_2d[1:5, 0] = True
    cache.side_2d[:] = 1.0
    aligned = _make_aligned(T, 1)
    aligned.close_2d = np.array([[2.0],[1.5],[1.0],[1.0],[1.0],[1.0]])
    regime = np.array([1]*T, dtype=np.int8)

    result = compute_bucket_realized_edges(cache, aligned, 0, T, regime, cost_bps=1.0, min_n=1)

    assert len(result) > 0
    key = list(result.keys())[0]
    assert result[key] < 0.0
```

## S3 — shrinkage: count < min_n → family prior 방향 축소
```python
def test_compute_bucket_edges_shrinkage_toward_family_prior():
    # regime=0: family=donchian, TF=4h (count=3, min_n=10)
    # family prior = raw_edge (유일 버킷이므로 동일 — shrinkage 수식 검증)
    T, S = 6, 1
    cache = _make_cache(T, S)
    cache.signal_mask_2d[1:4, 0] = True   # count=3 < min_n=10
    cache.side_2d[:] = 1.0
    aligned = _make_aligned(T, 1)
    aligned.close_2d = np.array([[1.0],[2.0],[3.0],[4.0],[4.0],[4.0]])  # 큰 상승
    regime = np.array([0]*T, dtype=np.int8)

    raw_result = compute_bucket_realized_edges(
        cache, aligned, 0, T, regime, cost_bps=1.0, min_n=1, shrinkage=0.0
    )
    shrunk_result = compute_bucket_realized_edges(
        cache, aligned, 0, T, regime, cost_bps=1.0, min_n=10, shrinkage=0.3
    )

    # shrinkage 적용 시 raw와 다를 수 있지만 부호는 유지
    key = list(raw_result.keys())[0]
    assert key in shrunk_result
    assert shrunk_result[key] > 0.0  # 방향 보존
```

## S4 — 빈 입력: active sleeve 없음 → 빈 dict 반환
```python
def test_compute_bucket_edges_empty_when_no_active_sleeves():
    T, S = 6, 1
    cache = _make_cache(T, S)
    # signal_mask_2d all False (default)
    aligned = _make_aligned(T, 1)
    regime = np.array([0]*T, dtype=np.int8)

    result = compute_bucket_realized_edges(cache, aligned, 0, T, regime)
    assert result == {}
```

## S5 — filter: edge > floor 만 통과
```python
def test_filter_sleeves_by_bucket_passes_above_floor():
    from unittest.mock import MagicMock
    bucket_edges = {
        (1, "donchian_72", "4h"): 500.0,    # 통과
        (1, "trend_pullback", "4h"): 50.0,  # 탈락
    }
    sig_a = MagicMock()
    sig_b = MagicMock()
    sleeve_sigs = {
        ("BTC", "donchian_72_4h"): sig_a,
        ("BTC", "trend_pullback_4h"): sig_b,
    }
    result = filter_sleeves_by_bucket(sleeve_sigs, bucket_edges, regime_now=1, edge_floor_bps=100.0)

    assert ("BTC", "donchian_72_4h") in result
    assert ("BTC", "trend_pullback_4h") not in result
```

## S6 — filter: 미관측 버킷은 edge=0 → 탈락
```python
def test_filter_sleeves_by_bucket_unknown_bucket_treated_as_zero():
    bucket_edges = {}  # 아무 버킷도 학습 안 됨
    from unittest.mock import MagicMock
    sleeve_sigs = {("BTC", "donchian_72_4h"): MagicMock()}

    result = filter_sleeves_by_bucket(sleeve_sigs, bucket_edges, regime_now=1, edge_floor_bps=100.0)

    assert len(result) == 0
```

## S7 — filter: 빈 sleeve_sigs → 빈 dict
```python
def test_filter_sleeves_by_bucket_empty_input():
    result = filter_sleeves_by_bucket({}, {(1, "x", "4h"): 500.0}, regime_now=1, edge_floor_bps=100.0)
    assert result == {}
```

---

# 🛠️ 알고리즘 계획

## C-B1: `compute_bucket_realized_edges`

```
Input shapes:
  signal_mask_2d: [T, S] bool
  side_2d:        [T, S] float64  (+1 or -1)
  close_2d:       [T, N] float64  (N = n_symbols)
  sleeve_to_sym:  [S]    int64
  regime_code_1d: [T]    int8

Logic:
  1. fwd_bps[t, sym] = (close[t+1]-close[t]) / close[t] * 10000  (t < T-1)
     edge case: close[t]=0 → fwd_bps=0
  2. For t in [fit_start, fit_end-1]:
       active_js = where(signal_mask_2d[t])
       For j in active_js:
         regime = int(regime_code_1d[t])   # 경계 체크 필수
         family, tf = _parse_meta_group_ids(sleeve_ids[j][1])
         sym_col = sleeve_to_sym[j]
         raw_bps = side_2d[t,j] * fwd_bps[t, sym_col] - cost_bps
         accumulate bucket_sum[(regime,family,tf)], bucket_cnt[(regime,family,tf)]
  3. raw_edge per bucket = bucket_sum / bucket_cnt
  4. family_prior[family] = mean(raw_edge over all (regime, tf) in same family)
  5. If bucket_cnt < min_n:
       bucket_edge = (1-shrinkage)*raw_edge + shrinkage*family_prior[family]
     Else: bucket_edge = raw_edge
  6. Return {(regime,family,tf): bucket_edge for buckets with cnt>0}
```

## C-B2: `filter_sleeves_by_bucket`

```
Logic:
  1. if not sleeve_sigs: return {}
  2. result = {}
  3. for (sym, strat_id), sig in sleeve_sigs.items():
       family, tf = _parse_meta_group_ids(strat_id)
       key = (regime_now, family, tf)
       edge = bucket_edges.get(key, 0.0)
       if edge > edge_floor_bps:
           result[(sym, strat_id)] = sig
  4. return result
```

## C-B4: OOS 루프 삽입 순서 (awf_sim.py)

```
fold 시작:
  if config.l2_routing_mode == "bucket":
    _bucket_edges = compute_bucket_realized_edges(
        cache, aligned, fold.fit_start, fold.oos_start,
        regime_code_1d, cost_bps=6.0,
        min_n=config.l2_bucket_min_n, shrinkage=config.l2_bucket_shrinkage
    )

bar t (OOS):
  # 기존: _oos_sleeve_sigs 준비 완료
  if config.l2_routing_mode == "bucket" and _bucket_edges:
    _regime_now = int(regime_code_1d[t]) if t < len(regime_code_1d) else 0
    _oos_sleeve_sigs = filter_sleeves_by_bucket(
        _oos_sleeve_sigs, _bucket_edges, _regime_now,
        edge_floor_bps=config.l2_bucket_edge_floor_bps
    )
  # 기존 pooling:
  valid_signals, friction_by_symbol = _combine_sleeve_signals_to_symbol(...)
```

---

# ✅ 안전장치

| 항목 | 방어 방식 |
|---|---|
| **Look-ahead** | `compute_bucket_realized_edges`는 `fit_end=fold.oos_start`만 접근 |
| **min_n 미달** | shrinkage로 family prior에 축소 (0으로 붕괴 방지) |
| **미관측 버킷** | `bucket_edges.get(key, 0.0)` → floor 미달 → 자동 제외 |
| **하위호환** | `l2_routing_mode="pool"` 기본 → 기존 동작 완전 보존 |
| **close=0 분모** | `max(abs(close[t]), 1e-12)` 방어 |
| **regime 경계** | `t < len(regime_code_1d)` 체크, 초과 시 0 사용 |

---

# 🔐 한계 & 후속

- **약신호 (ρ≈0.25)**: 집중(K_RANK↓) 아닌 breadth 수확 필수.  
- **param 민감도** (`min_n`, `shrinkage`, `edge_floor`): 배포 전 embargoed stability selection으로 고정 (Stage C).  
- **net-of-cost 검증**: bucket_edge는 cost 차감 후 계산 → 경제성 내장.  
- **regime 전환기**: bucket 학습 구간 내 regime 분포가 OOS와 다를 수 있음 → shrinkage가 완충.
