"""Linear alpha + regime signal composition for portfolio weights (no CS rank / HMM multiply)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.optimization.opt_config import (
    OPT_FUTURES_CONFIG,
)
from src.domain.futures.strategy.cs_rank import VOL_FLOOR, SymbolSignal

_logger = logging.getLogger(__name__)


def hours_per_bar_tf(tf: str) -> float:
    t = str(tf).strip().lower()
    if t.endswith("h"):
        return float(t.replace("h", "") or 4)
    if t.endswith("d"):
        return float(t.replace("d", "") or 1) * 24.0
    if t.endswith("m"):
        return float(t.replace("m", "") or 1) / 60.0
    return 4.0


def composer_sigma_lookback_bars(tf: str, opt_cfg: dict[str, Any] | None = None) -> int:
    """Get ~8 calendar days of bars for simple per-bar return std (sigma_t,i)."""
    cfg = opt_cfg or OPT_FUTURES_CONFIG
    by_tf = cfg.get("FUTURES_COMPOSER_SIGMA_LOOKBACK_BY_TF")
    key = str(tf).strip().lower()
    if isinstance(by_tf, dict) and key in by_tf:
        return max(3, int(by_tf[key]))
    days = float(cfg.get("FUTURES_COMPOSER_SIGMA_CALENDAR_DAYS", 8.0))
    hpb = hours_per_bar_tf(tf)
    return max(3, int(days * 24.0 / max(hpb, 1e-9)))


def rolling_per_bar_return_std(close_1d: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling std of simple returns r_t = (c_t - c_{t-1}) / |c_{t-1}| (causal)."""
    c = np.asarray(close_1d, dtype=np.float64).ravel()
    n = c.size
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    r = np.zeros(n, dtype=np.float64)
    r[1:] = (c[1:] - c[:-1]) / np.maximum(np.abs(c[:-1]), 1e-12)
    rw = max(2, int(window))
    s = pd.Series(r).rolling(rw, min_periods=2).std(ddof=1)
    v = s.to_numpy(dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(v, 1e-12)  # type: ignore[no-any-return]


def compose_symbol_signals(
    *,
    model_output: Any,
    close_2d: NDArray[np.float64],
    symbols: tuple[str, ...],
    tf: str,
    min_obs: int,
    t_stat_floor: float,
    beta_vs_market_1d: NDArray[np.float64] | None = None,
    opt_cfg: dict[str, Any] | None = None,
) -> dict[str, SymbolSignal]:
    """CandidateModelOutput(L1 OOS 예측)을 SymbolSignal 매핑으로 변환하는 어댑터.

    model_output.events DataFrame(심볼별 OOS 이벤트)과 expected_net_bps를 소비해
    심볼별 HAC(Newey-West) t-stat, per-bar 변동성, BTC 베타를 집계한다.

    Args:
        model_output: CandidateModelOutput (Any — 순환 import 방지).
            .events: pd.DataFrame, 'symbol' 컬럼 필수.
            .expected_net_bps: NDArray[float64], len == len(events).
        close_2d: 종가 행렬. Shape: [T, N], float64.
        symbols: close_2d 열 순서와 동일한 심볼 튜플. len == N.
        tf: 타임프레임 문자열 (예: '4h', '1d').
        min_obs: QC 최소 관측 수. n_obs < min_obs이면 valid=False.
        t_stat_floor: QC 최소 |t-stat|. |t_stat| < t_stat_floor이면 valid=False.
        beta_vs_market_1d: BTC 베타 배열. Shape: [N], float64. None이면 beta_btc=None.
        opt_cfg: 변동성 lookback 오버라이드용 config dict.

    Returns:
        심볼 → SymbolSignal 매핑. 방어 실패 시 빈 dict.

    Time Complexity: O(N * T) — 심볼별 rolling std 지배.
    Space Complexity: O(N * T) — close_2d 참조 (복사 없음).
    """
    # --- 방어 체크 ---
    try:
        events: pd.DataFrame = model_output.events
        net_bps_raw: Any = model_output.expected_net_bps
    except AttributeError:
        _logger.warning("compose_symbol_signals: model_output 속성 누락, 빈 dict 반환")
        return {}

    if not isinstance(events, pd.DataFrame) or "symbol" not in events.columns:
        _logger.warning("compose_symbol_signals: events에 'symbol' 컬럼 없음, 빈 dict 반환")
        return {}

    net_bps: NDArray[np.float64] = np.asarray(net_bps_raw, dtype=np.float64)
    if len(net_bps) != len(events):
        _logger.warning(
            "compose_symbol_signals: net_bps 길이(%d) != events 길이(%d), 빈 dict 반환",
            len(net_bps),
            len(events),
        )
        return {}

    if len(events) == 0:
        return {}

    # --- per-symbol 집계 DataFrame ---
    df = events[["symbol"]].copy()
    df["net_bps"] = net_bps

    sym_to_idx: dict[str, int] = {s: i for i, s in enumerate(symbols)}
    lookback: int = composer_sigma_lookback_bars(tf, opt_cfg)

    result: dict[str, SymbolSignal] = {}

    for sym in df["symbol"].unique():
        grp_bps: NDArray[np.float64] = df[df["symbol"] == sym]["net_bps"].to_numpy(dtype=np.float64)
        n_obs: int = len(grp_bps)
        raw_mu: float = float(grp_bps.mean())

        # --- HAC(Newey-West) t-stat ---
        if n_obs < 4:
            t_stat: float = 0.0
        elif float(np.std(grp_bps)) < 1e-6:
            t_stat = 0.0
        else:
            # embargo_bars: walk-forward 기준 ~5% 샘플 크기
            embargo_bars: int = max(1, n_obs // 20)
            m: int = min(n_obs - 1, max(1, embargo_bars))
            mu_arr: NDArray[np.float64] = grp_bps - raw_mu  # demeaned
            gamma0: float = float(np.dot(mu_arr, mu_arr)) / n_obs
            gamma_sum: float = gamma0
            for j in range(1, m + 1):
                w: float = 1.0 - j / (m + 1)  # Bartlett kernel
                gamma_j: float = float(np.dot(mu_arr[j:], mu_arr[:-j])) / n_obs
                gamma_sum += 2.0 * w * gamma_j
            se_hac: float = float(np.sqrt(max(gamma_sum, 1e-20) / n_obs))
            t_stat = raw_mu / se_hac if se_hac > 1e-20 else 0.0

        # --- per-bar 변동성 ---
        vol: float
        sym_idx: int | None = sym_to_idx.get(sym)
        if sym_idx is not None and sym_idx < close_2d.shape[1]:
            close_col: NDArray[np.float64] = close_2d[:, sym_idx]
            n_close = close_col.size
            if n_close < 2:
                last_vol = 0.0
            else:
                rw = max(2, int(lookback))
                r = (close_col[1:] - close_col[:-1]) / np.maximum(np.abs(close_col[:-1]), 1e-12)
                if n_close - 1 < rw:
                    r_sub = np.zeros(n_close, dtype=np.float64)
                    r_sub[1:] = r
                    last_vol = float(np.std(r_sub, ddof=1)) if len(r_sub) >= 2 else 0.0
                else:
                    last_vol = float(np.std(r[-rw:], ddof=1))
            vol = float(max(last_vol, VOL_FLOOR))
        else:
            vol = float(VOL_FLOOR)

        # --- BTC 베타 ---
        beta_btc: float | None = None
        if beta_vs_market_1d is not None and sym_idx is not None and sym_idx < len(beta_vs_market_1d):
            beta_btc = float(beta_vs_market_1d[sym_idx])

        # --- valid QC 게이트 ---
        valid: bool = (
            n_obs >= min_obs and abs(t_stat) >= t_stat_floor and bool(np.isfinite(raw_mu)) and bool(np.isfinite(vol))
        )

        result[sym] = SymbolSignal(
            raw_mu=raw_mu,
            volatility=vol,
            n_obs=n_obs,
            t_stat=t_stat,
            valid=valid,
            beta_btc=beta_btc,
        )

    _logger.debug(
        "compose_symbol_signals: n_symbols=%d, n_valid=%d",
        len(result),
        sum(1 for s in result.values() if s.valid),
    )
    return result
