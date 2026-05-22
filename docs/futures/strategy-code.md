# Strategy 구현 사양서 (P0: XS Momentum)

**대상 문서**: `docs/futures/strategy.md` v2.0
**최종 업데이트**: 2026-05-22
**목적**: legacy alpha/HMM 미사용 상태에서 `--quick-backtest` 대비 의미있는 신호를 가진 백테스트를 완주시킨다.

---

## 0. 사전 확인된 코드 사실

이 사양서는 다음 코드 상태를 전제로 한다. 구현 전 변경되었으면 사양서 먼저 수정한다.

| 항목 | 현재 상태 | 출처 |
|---|---|---|
| `run_ml_pipeline_for_universe()` | 빈 `MLPipelineOutput()` 반환 | `strategy_runtime/bridge.py:27` |
| `merge_ml_output_into_data_maps()` | **no-op stub** (`del ml_out, data_maps, ...`) | `strategy_runtime/bridge.py:63` |
| `optimizer.raw_full["alpha_long"]` | per-symbol DataFrame 컬럼에서 직접 읽음 | `optimization/optimizer.py:458, 563` |
| `MLPipelineOutput.alpha_panel` | `pd.DataFrame` (기본 빈 DF) | `strategy_runtime/bridge.py:14` |
| `inject_cs_momentum_ranks()` | 이미 존재 — XS rank 주입 패턴의 reference | `optimization/optimizer.py:78` |

**핵심 함의**: alpha_panel을 만들어 `MLPipelineOutput`에 실어도, 현재 `merge_ml_output_into_data_maps`가 no-op이라 데이터가 per-symbol df로 흘러가지 않는다. **두 함수 모두 구현 필수**.

---

## 1. 디렉토리 및 파일 구조

```text
src/domain/futures/strategy/
├── __init__.py             # public API: build_strategy_alpha
├── config.py               # StrategyConfig dataclass
├── momentum.py             # XS momentum 신호 코어
└── builder.py              # data_maps → alpha_panel 조립
```

**수정 파일**:
- `src/domain/futures/strategy_runtime/bridge.py` — `run_ml_pipeline_for_universe`, `merge_ml_output_into_data_maps` 실구현
- `src/execution/opt_main_futures.py` — `--strategy` CLI 플래그 추가, quick-backtest 경로와 분기

---

## 2. 모듈별 API 사양

### 2.1 `strategy/config.py`

```python
from dataclasses import dataclass, field

@dataclass(slots=True, frozen=True)
class MomentumConfig:
    """XS momentum sleeve parameters. Frozen for hash-based reproducibility."""
    lookback_bars: int = 6            # 4h × 6 = 24h
    top_ratio: float = 0.30           # 상위 30% → long
    bottom_ratio: float = 0.30        # 하위 30% → short
    min_symbols_for_xs: int = 5       # 미만이면 alpha = 0 (XS rank 무의미)
    edge_scale_per_bar: float = 1e-3  # rank [0,1] → simple return per bar 변환 스케일

@dataclass(slots=True, frozen=True)
class StrategyConfig:
    """Top-level strategy switch. Extend with carry/etc. in P1."""
    name: str = "momentum_v0"
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
```

**검증 규칙**:
- `0 < top_ratio <= 0.5`, `0 < bottom_ratio <= 0.5` — 위반 시 `ValueError`.
- `lookback_bars >= 1` — 위반 시 `ValueError`.

### 2.2 `strategy/momentum.py`

```python
def compute_xs_momentum_alpha(
    close_2d: np.ndarray,       # shape [T, N], NaN 허용
    cfg: MomentumConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-sectional momentum alpha.

    Args:
        close_2d: 4h closed-bar close panel, time-aligned across symbols.
                  NaN은 해당 (t, i) 셀 제외.
        cfg: MomentumConfig.

    Returns:
        (alpha_long, alpha_short) — 각 shape [T, N], dtype float64.
        단위: simple return per bar.
        warm-up 구간 `t < lookback_bars`: 전부 0.0.

    No look-ahead: alpha[t, :]는 close[t, :]까지만 사용.
    """
```

**의사코드**:

```python
T, N = close_2d.shape
L = cfg.lookback_bars
alpha_long = np.zeros((T, N), dtype=np.float64)
alpha_short = np.zeros((T, N), dtype=np.float64)

# 1. log momentum score
prev = close_2d[:-L]
curr = close_2d[L:]
with np.errstate(divide="ignore", invalid="ignore"):
    mom = np.log(curr / np.maximum(prev, 1e-12))
mom = np.where(np.isfinite(mom), mom, np.nan)
# shape [T-L, N] → place into [L:T, :]

# 2. XS percentile rank per row (NaN 제외)
for t_idx in range(mom.shape[0]):
    row = mom[t_idx]
    valid_mask = np.isfinite(row)
    n_valid = valid_mask.sum()
    if n_valid < cfg.min_symbols_for_xs:
        continue
    ranks = scipy.stats.rankdata(row[valid_mask], method="average") / n_valid
    # 3. tail-cut + normalize → [0, 1]
    al = np.maximum(ranks - (1.0 - cfg.top_ratio), 0.0) / cfg.top_ratio
    as_ = np.maximum(cfg.bottom_ratio - ranks, 0.0) / cfg.bottom_ratio
    # 4. scale → simple return per bar 단위
    alpha_long[L + t_idx, valid_mask] = al * cfg.edge_scale_per_bar
    alpha_short[L + t_idx, valid_mask] = as_ * cfg.edge_scale_per_bar

return alpha_long, alpha_short
```

**복잡도**: O(T × N × log N). T=수천, N=수십에서 충분히 빠르다(Numba 불필요).

**Look-ahead 가드**: `mom[t_idx]`는 `close_2d[L + t_idx]` 까지만 의존하고, output도 동일 row에 기록한다. 명시적으로 row 단위로 처리하여 broadcast 실수 방지.

### 2.3 `strategy/builder.py`

```python
def build_strategy_alpha(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """data_maps에서 close 패널을 조립하고 alpha_panel을 산출.

    Returns:
        DataFrame with MultiIndex (datetime, symbol), columns ["alpha_long", "alpha_short"].
        symbol 정합성: 입력 symbols 중 tf 데이터가 있는 심볼만 포함.
        시간 정합성: compute_multi_alignment_info 와 동일한 common_is_start_dt 기준.

    Raises:
        ValueError: 유효 심볼 < cfg.momentum.min_symbols_for_xs
    """
```

**의사코드**:

```python
from src.domain.futures.optimization.optimizer import compute_multi_alignment_info

info = compute_multi_alignment_info(data_maps, symbols, tf, embargo=0)
if info is None:
    return pd.DataFrame(columns=["alpha_long", "alpha_short"])

eff_len = info["eff_ref_len"]
offsets = info["alignment_offsets"]
valid_symbols = [s for s in symbols if s in offsets]

if len(valid_symbols) < cfg.momentum.min_symbols_for_xs:
    raise ValueError(f"strategy needs >= {cfg.momentum.min_symbols_for_xs} symbols, got {len(valid_symbols)}")

# 1. close 2D 패널 조립 (time × symbol)
close_2d = np.zeros((eff_len, len(valid_symbols)), dtype=np.float64)
datetimes = None
for j, sym in enumerate(valid_symbols):
    df = data_maps[sym][tf]
    s = offsets[sym]
    close_2d[:, j] = df["close"].iloc[s : s + eff_len].to_numpy(dtype=np.float64)
    if datetimes is None:
        datetimes = df["datetime"].iloc[s : s + eff_len].to_numpy()

# 2. momentum
alpha_long, alpha_short = compute_xs_momentum_alpha(close_2d, cfg.momentum)

# 3. long-format DataFrame
idx = pd.MultiIndex.from_product([datetimes, valid_symbols], names=["datetime", "symbol"])
panel = pd.DataFrame({
    "alpha_long":  alpha_long.reshape(-1),
    "alpha_short": alpha_short.reshape(-1),
}, index=idx).sort_index()
panel.attrs["strategy_name"] = cfg.name
panel.attrs["lookback_bars"] = cfg.momentum.lookback_bars
return panel
```

### 2.4 `strategy/__init__.py`

```python
from src.domain.futures.strategy.builder import build_strategy_alpha
from src.domain.futures.strategy.config import MomentumConfig, StrategyConfig

__all__ = ["StrategyConfig", "MomentumConfig", "build_strategy_alpha"]
```

---

## 3. bridge.py 패치 사양

`strategy_runtime/bridge.py`의 두 stub 함수를 다음과 같이 교체한다.

### 3.1 `run_ml_pipeline_for_universe` 교체

```python
def run_ml_pipeline_for_universe(
    symbols: list[str],
    tf: str,
    fetch_start: str | None,
    end_date: str | None,
    opt_config: dict[str, Any],
    *,
    strategy_cfg: StrategyConfig | None = None,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> MLPipelineOutput:
    """Strategy 주입 모드: data_maps에서 XS momentum alpha_panel 생성.

    strategy_cfg=None 이면 기존처럼 빈 MLPipelineOutput을 반환한다
    (--quick-backtest 동작 호환).
    """
    del fetch_start, end_date, opt_config, kwargs

    if strategy_cfg is None or preloaded_data_maps is None:
        return MLPipelineOutput()

    alpha_panel = build_strategy_alpha(
        data_maps=preloaded_data_maps,
        symbols=symbols,
        tf=tf,
        cfg=strategy_cfg,
    )
    return MLPipelineOutput(alpha_panel=alpha_panel)
```

### 3.2 `merge_ml_output_into_data_maps` 실구현

```python
def merge_ml_output_into_data_maps(
    ml_out: MLPipelineOutput,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    log_tag: str = "",
) -> None:
    """alpha_panel의 long-format을 per-symbol df의 컬럼으로 풀어준다.

    optimizer는 raw_full["alpha_long"]/raw_full["alpha_short"]를 직접 읽으므로
    alpha_panel만 만들고 끝나면 신호가 흘러가지 않는다.
    """
    panel = getattr(ml_out, "alpha_panel", None)
    if panel is None or panel.empty:
        return
    if not {"alpha_long", "alpha_short"}.issubset(panel.columns):
        _logger.warning("[%s] alpha_panel missing required columns; skip merge", log_tag)
        return

    # panel index: (datetime, symbol)
    by_sym = panel.reset_index().groupby("symbol", sort=False)
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        try:
            sym_rows = by_sym.get_group(sym)
        except KeyError:
            continue
        df = data_maps[sym][tf]
        # datetime 기준 left-join, 누락 row는 0.0
        merged = df[["datetime"]].merge(
            sym_rows[["datetime", "alpha_long", "alpha_short"]],
            on="datetime", how="left",
        )
        df["alpha_long"]  = merged["alpha_long"].fillna(0.0).to_numpy(dtype=np.float64)
        df["alpha_short"] = merged["alpha_short"].fillna(0.0).to_numpy(dtype=np.float64)
```

**검증 가드**:
- 컬럼이 이미 있으면 덮어쓴다(로그 경고).
- `datetime` dtype mismatch는 fail-fast (`assert df["datetime"].dtype == sym_rows["datetime"].dtype`).

---

## 4. opt_main_futures.py 패치 사양

### 4.1 CLI 플래그 추가

`argparse` 영역에 추가:

```python
parser.add_argument(
    "--strategy",
    type=str,
    default=None,
    choices=[None, "momentum_v0"],
    help="신규 strategy 주입. None이면 기존 quick/full 경로 유지.",
)
parser.add_argument(
    "--mom-lookback",
    type=int,
    default=6,
    help="XS momentum lookback (4h bars). Default 6 = 24h.",
)
```

### 4.2 ml_out 분기 패치

기존 `if args.quick_backtest:` 블록 직후, 다음 추가:

```python
elif args.strategy == "momentum_v0":
    OPT_FUTURES_CONFIG["FUTURES_WF_HMM_LEG_REFIT"] = False
    OPT_FUTURES_CONFIG["FUTURES_USE_META_LABELER"] = False
    OPT_FUTURES_CONFIG["FUTURES_REGIME_POLICY_ENABLED"] = False
    _logger.warning(" [STEP 2/4] STRATEGY mode=%s (lookback=%d).", args.strategy, args.mom_lookback)

    strategy_cfg = StrategyConfig(
        name=args.strategy,
        momentum=MomentumConfig(lookback_bars=args.mom_lookback),
    )
    ml_out = run_ml_pipeline_for_universe(
        valid_symbols, args.tf, fetch_start_date, end_date, OPT_FUTURES_CONFIG,
        strategy_cfg=strategy_cfg,
        preloaded_data_maps=oos_data_maps if not pre_args.skip_universe else None,
    )
    # G-ALPHA hard-kill 우회 (strategy_v0는 alpha component filter 거치지 않음)
```

`if not args.hmm_only and not args.quick_backtest:` 의 G-ALPHA hard-kill 조건문도 `and args.strategy is None` 으로 확장하여 strategy 모드에서도 우회한다.

### 4.3 data_maps merge 호출

`precompute_ml_optimization_context` 호출 직전에 다음이 이미 있어야 한다 (없으면 추가):

```python
merge_ml_output_into_data_maps(ml_out, oos_data_maps, valid_symbols, args.tf, log_tag="strategy")
```

기존 코드 `optimizer.py:908` 의 호출과 동일하지만, strategy 모드에서도 반드시 실행되어야 한다.

---

## 5. 테스트 사양

테스트는 3개 계층으로 분리한다. 각 계층이 검증하는 것이 다르며 백테스트만으로는 대체할 수 없다.

| 계층 | 목적 | 속도 | 위치 |
|---|---|---|---|
| **단위 테스트** | 코드 정확성, NaN/look-ahead | 초 단위 | `tests/unit/` |
| **신호 진단 테스트** | alpha 통계적 유효성 (IC, 분포) | 수 초 | `tests/signal/` |
| **E2E 백테스트** | 파이프라인 통합 + 경제적 유효성 | 수 분~시간 | 섹션 6 |

> **왜 세 계층이 모두 필요한가**:
> - look-ahead bias는 백테스트에서 오히려 성과가 좋게 나온다 — 단위 테스트로만 잡을 수 있다.
> - 신호가 완전히 0이어도 `--quick-backtest`와 동일한 E2E 경로를 완주한다 — 신호 기여를 백테스트 숫자로 분리하기 어렵다.
> - 백테스트는 `BETA_ALPHA` 같은 파라미터에 종속적 — 신호 자체가 나쁜 건지 파라미터 문제인지 구분이 안 된다.

```text
tests/
├── unit/domain/futures/strategy/
│   ├── test_momentum.py        # compute_xs_momentum_alpha
│   └── test_builder.py         # build_strategy_alpha
└── signal/
    └── test_strategy_ic.py     # alpha IC, 분포, look-ahead 진단
```

### 5.1 `test_momentum.py` 케이스 (단위 테스트)

| 테스트명 | 내용 |
|---|---|
| `test_warmup_zero` | `t < lookback`은 alpha=0 |
| `test_top_only_long` | 단조 증가 path: 최상위 심볼만 alpha_long > 0 |
| `test_bottom_only_short` | 단조 감소 path: 최하위 심볼만 alpha_short > 0 |
| `test_nan_skip` | 일부 심볼 NaN: rank 계산에서 제외, alpha=0 |
| `test_min_symbols_gate` | `n_valid < min_symbols_for_xs`: 전부 0 |
| `test_no_lookahead` | `close[t+1]` 변경이 `alpha[t]`에 영향 없음 (불변성) |
| `test_dtype_shape` | 출력 shape == 입력 shape, dtype float64 |

### 5.2 `test_builder.py` 케이스 (단위 테스트)

| 테스트명 | 내용 |
|---|---|
| `test_panel_index` | MultiIndex `(datetime, symbol)`, sorted |
| `test_panel_columns` | `["alpha_long", "alpha_short"]` 정확히 |
| `test_alignment_consistency` | compute_multi_alignment_info와 동일 base 사용 |
| `test_insufficient_symbols_raises` | min_symbols 미달 시 `ValueError` |

### 5.3 `test_strategy_ic.py` 케이스 (신호 진단 테스트)

**목적**: 합성 데이터로 alpha의 통계적 성질이 경제적으로 의미있는지 확인한다.  
실제 시장 데이터 불필요 — 합성 OHLCV 패널로 충분하다.

#### 신호 분포 검사

```python
def test_alpha_nonzero_ratio():
    """alpha_long > 0인 비율이 top_ratio ± 5% 범위에 있는지 확인.
    
    warm-up 이후 구간에서:
    - alpha_long > 0 비율 ≈ top_ratio (0.30 기준 → 25~35%)
    - alpha_short > 0 비율 ≈ bottom_ratio
    - alpha_long > 0 AND alpha_short > 0 같은 symbol: 0 (상호 배타적)
    """

def test_alpha_scale_range():
    """edge_scale_per_bar 기준으로 alpha 값 범위 확인.
    
    max(alpha_long) ≈ edge_scale_per_bar (= 1e-3 기본값)
    min > 0인 값들의 평균 > 0
    """
```

#### Forward IC 검사 (look-ahead bias 탐지)

```python
def test_forward_ic_positive():
    """합성 데이터에서 alpha_long이 t+1 수익률과 양의 상관을 갖는지 확인.
    
    설계:
    - 미래 수익률이 높은 심볼에 인위적으로 높은 past momentum을 부여한 합성 패널 생성
    - Spearman rank IC(alpha_long[t, :], ret[t+1, :]) > 0 검증
    
    이 테스트가 실패하면 momentum 수식 자체가 방향이 반대임을 의미.
    """

def test_lookahead_ic_suspicious():
    """look-ahead bias 탐지: alpha[t]와 ret[t-1] 상관이 ret[t+1]보다 높으면 warn.
    
    정상적인 momentum:
      IC(alpha[t], ret[t+1]) > IC(alpha[t], ret[t-1])  ← 미래 예측이 과거 예측보다 강함
    
    look-ahead가 있으면:
      IC(alpha[t], ret[t+1]) >> IC(alpha[t], ret[t])   ← 비정상적으로 높음 (> 0.3)
    
    pytest.warns(UserWarning) 로 처리.
    """
```

#### 신호 안정성 검사

```python
def test_alpha_turnover_finite():
    """연속 bar 간 alpha_long 변화율(turnover proxy) 이 유한하고 0이 아닌지 확인.
    
    turnover = mean(|alpha_long[t] - alpha_long[t-1]| > 0) per symbol
    
    기대: warm-up 이후 구간에서 0 < turnover < 1.0
    turnover = 0 이면 신호가 frozen 상태 (버그 가능성)
    turnover = 1.0 이면 매 bar 완전 교체 (비용 불감 신호)
    """

def test_alpha_not_constant():
    """동일 lookback 내 alpha가 모든 t에서 같은 값을 갖지 않는지 확인.
    
    합성 데이터에서 std(alpha_long, axis=0) > 0 (심볼별 시계열 분산 > 0)
    """
```

**실행 명령**:
```bash
uv run pytest tests/unit/domain/futures/strategy/ -v --tb=short
uv run pytest tests/signal/test_strategy_ic.py -v --tb=short
```

두 스위트 모두 합성 데이터만 사용하므로 외부 데이터 불필요, 수 초 내 완료.

---

## 6. End-to-End 검증 절차

### 6.1 Baseline (zero alpha) 먼저

```bash
uv run python -m src.execution.opt_main_futures \
  --skip-universe --skip-data-sync \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT \
  --trials 5 --tf 4h --quick-backtest \
  2>&1 | tee logs/baseline_quick.log
```

기록할 지표: `CAGR`, `MDD`, `EV/Cost`, `funding_drag`, `turnover`.

### 6.2 Strategy 모드 실행

```bash
uv run python -m src.execution.opt_main_futures \
  --skip-universe --skip-data-sync \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT \
  --trials 5 --tf 4h --strategy momentum_v0 --mom-lookback 6 \
  2>&1 | tee logs/momentum_v0_lb6.log
```

`--mom-lookback 18` 으로도 1회 반복.

### 6.3 PASS 조건

| 조건 | 기준 |
|---|---|
| RuntimeError/TypeError | 없음 |
| optimizer가 trial >= 1 완료 | 필수 |
| alpha_panel non-empty | `log_alpha_component_summary` 로그에서 확인 |
| `positive_leg_ratio` | baseline 대비 동등 이상 |
| `EV/Cost` | baseline 대비 동등 이상 (signal이 cost를 못 이기면 fail) |

PASS 실패 시 우선 점검 순서:
1. `merge_ml_output_into_data_maps` 후 `data_maps[sym][tf]["alpha_long"]` non-zero 비율
2. `signal_composer.apply_linear_signal_composer_scores` 의 `mu_l` 분포 — friction이 신호보다 큰지
3. `edge_scale_per_bar` 값 — 너무 작으면 EV hurdle 통과 못함

---

## 7. Fail-Fast 가드레일

구현 중 다음 조건 위반 시 `RuntimeError` 또는 `ValueError`로 즉시 중단한다.

| 위반 | 위치 | 메시지 |
|---|---|---|
| legacy 모듈 import 발견 | `strategy/*.py` | `"legacy import forbidden in strategy module"` |
| `alpha_panel` 의 NaN/Inf | `build_strategy_alpha` 출력 직전 | `"alpha_panel contains NaN/Inf at idx=..."` |
| MultiIndex 순서 비정렬 | `build_strategy_alpha` 출력 직전 | `"alpha_panel must be sorted by (datetime, symbol)"` |
| `merge_ml_output_into_data_maps`의 datetime dtype mismatch | 머지 직전 | `"datetime dtype mismatch: ..."` |
| symbol 수 < `min_symbols_for_xs` | `build_strategy_alpha` 진입 | `ValueError` (4.3.1) |

---

## 8. 작업 순서 (체크리스트)

1. `strategy/config.py` — dataclass 작성 + 검증 규칙
2. `strategy/momentum.py` — `compute_xs_momentum_alpha` + 단위 테스트 통과
3. `strategy/builder.py` — `build_strategy_alpha` + 단위 테스트 통과
4. `strategy/__init__.py` — public API 노출
5. `bridge.py` — `run_ml_pipeline_for_universe` / `merge_ml_output_into_data_maps` 교체
6. `opt_main_futures.py` — CLI 플래그 + 분기 추가
7. baseline (`--quick-backtest`) 1회 실행
8. strategy (`--strategy momentum_v0`) lookback=6, 18 각 1회 실행
9. PASS 조건 점검 + 로그 비교

각 단계 종료 시 `uv run ruff check src/domain/futures/strategy/ src/domain/futures/strategy_runtime/bridge.py` 및 `uv run mypy` 통과 필수.

---

## 9. P1 확장 포인트 (참고용, 구현 X)

- `strategy/carry.py`: funding_rate_sum 기반 negative-funding long / positive-funding short.
- `strategy/blend.py`: 복수 lookback (6, 18) IC 가중 blend.
- `strategy/diagnostics.py`: rolling rank IC, EV/Cost decay, sleeve attribution.
- `strategy/regime_gate.py`: drawdown/realized_vol 기반 gross multiplier (HMM 없이).

P1 진입은 P0의 8번 PASS 조건이 안정적으로 통과한 뒤로 한다.
