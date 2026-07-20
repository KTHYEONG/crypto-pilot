from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.timeframe_contracts import scale_bar_count

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import RegimeConfig

_logger = logging.getLogger(__name__)

_REGIME_NAMES = (
    "bull_quiet",
    "bull_volatile",
    "bear_quiet",
    "bear_volatile",
    "transition",
    "crash",
)
# Compressed regime: 3-state mapping
_REGIME_COMPRESSED_NAMES = ("bull", "bear", "crisis")
# 6-state → 3-state: bull(0+1) / bear(2+3) / crisis(4+5)
_REGIME_COMPRESSION_MAP: dict[int, int] = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}


def compress_regime_codes(code_1d: NDArray[np.int8]) -> NDArray[np.int8]:
    """6-state regime codes → 3-state compressed codes (bull/bear/crisis).

    Args:
        code_1d: 6-state regime codes [T], dtype=np.int8, range [0..5].

    Returns:
        Compressed 3-state codes [T], dtype=np.int8, range [0..2].
        0=bull, 1=bear, 2=crisis.
    """
    compressed = np.full_like(code_1d, 2, dtype=np.int8)  # default crisis
    for src, dst in _REGIME_COMPRESSION_MAP.items():
        compressed[code_1d == src] = dst
    return compressed


_DEFAULT_BARS_PER_YEAR = 365.0 * 6.0
_EPS = 1e-12


def _ema_1d(values: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for idx in range(1, values.shape[0]):
        cur = values[idx]
        prev = out[idx - 1]
        out[idx] = cur if not np.isfinite(prev) else (alpha * cur) + ((1.0 - alpha) * prev)
    return out


def _rolling_std_1d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        finite = values[start : idx + 1]
        finite = finite[np.isfinite(finite)]
        if finite.size > 0:
            out[idx] = float(np.std(finite, ddof=0))
    return out


def _zscore_1d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    mean = np.full(values.shape[0], np.nan, dtype=np.float64)
    std = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        finite = values[start : idx + 1]
        finite = finite[np.isfinite(finite)]
        if finite.size > 0:
            mean[idx] = float(np.mean(finite))
            std[idx] = float(np.std(finite, ddof=0))
    return (values - mean) / np.maximum(std, _EPS)


def _expanding_quantile_causal(values: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """Causal expanding q-th quantile — no lookahead. q in [0, 1].

    Args:
        values: 1-D float64 array of input values.
        q: Quantile to compute, in [0, 1]. q=0.5 is equivalent to the median.

    Returns:
        Array of same shape where out[i] = q-th percentile of values[:i+1]
        ignoring non-finite. Entries remain NaN if no finite value has been seen.

    Time complexity: O(T²·logT) — expanding prefix sort per step.
    Space complexity: O(T).
    """
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        sample = values[: idx + 1]
        finite = sample[np.isfinite(sample)]
        if finite.size > 0:
            out[idx] = float(np.percentile(finite, q * 100.0))
    return out


def _infer_bars_per_year(datetimes: NDArray[np.datetime64]) -> float:
    if datetimes.shape[0] < 2:
        return _DEFAULT_BARS_PER_YEAR
    seconds = datetimes.astype("datetime64[s]").astype(np.int64)
    diffs = np.diff(seconds)
    finite = diffs[diffs > 0]
    if finite.size == 0:
        return _DEFAULT_BARS_PER_YEAR
    median_seconds = float(np.median(finite))
    if not np.isfinite(median_seconds) or median_seconds <= 0.0:
        return _DEFAULT_BARS_PER_YEAR
    return (365.0 * 24.0 * 60.0 * 60.0) / median_seconds


def _btc_index(symbols: tuple[str, ...]) -> int:
    """[ADR_20260717_L2_CRISIS_BTC_REGIME_DATA_INTEGRITY_FIX] fail-closed BTC anchor lookup."""
    for idx, symbol in enumerate(symbols):
        if "BTC" in symbol.upper():
            return idx
    raise ValueError(f"no BTC-named symbol found in {symbols!r}; market regime classification requires a BTC anchor")


def _btc_log_returns(aligned: AlignedMarketData) -> NDArray[np.float64]:
    close = np.asarray(aligned.close_2d, dtype=np.float64)
    btc_close = np.maximum(close[:, _btc_index(aligned.symbols)], _EPS)
    returns = np.zeros(btc_close.shape[0], dtype=np.float64)
    returns[1:] = np.diff(np.log(btc_close))
    return returns


def _expanding_robust_location_scale(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    location = np.zeros(values.shape[0], dtype=np.float64)
    scale = np.ones(values.shape[0], dtype=np.float64)
    for idx in range(values.shape[0]):
        sample = values[: idx + 1]
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        location[idx] = median
        scale[idx] = max(1.4826 * mad, _EPS)
    return location, scale


def _cusum_thresholds(target_arl_bars: int) -> tuple[float, float, int]:
    tail_z = float(norm.isf(1.0 / (2.0 * max(target_arl_bars, 2))))
    drift = max(0.25 * tail_z, 0.1)
    threshold = max(1.8 * tail_z, 1.0)
    hold_bars = max(6, round(math.log1p(float(target_arl_bars))))
    return drift, threshold, hold_bars


def _compute_dispersion_z(aligned: AlignedMarketData, window: int) -> NDArray[np.float64]:
    close = np.asarray(aligned.close_2d, dtype=np.float64)
    log_ret = np.zeros_like(close, dtype=np.float64)
    log_ret[1:] = np.diff(np.log(np.maximum(close, _EPS)), axis=0)
    dispersion = np.nanstd(log_ret, axis=1, ddof=0)
    return _zscore_1d(dispersion, window)


def _schmitt_directional_state(
    trend_snr: NDArray[np.float64],
    enter_theta: float,
    exit_theta: float,
    enter_band: NDArray[np.float64] | None = None,
) -> NDArray[np.int8]:
    """Stateful Schmitt hysteresis trigger for directional state.

    Eliminates chatter near zero by requiring a larger threshold to enter a
    directional state (``enter_theta``) than to exit it (``exit_theta``).

    When ``enter_band`` is provided, the per-bar enter threshold is taken from
    the self-calibrating persistence-targeted band instead of the scalar
    ``enter_theta`` (the scalar pair is then used only to derive the
    exit/enter ratio).  This replaces the arbitrary hardcoded threshold with a
    data-driven one while preserving the hysteresis (exit < enter) invariant.

    States:
        0 = NEUTRAL, 1 = BULL, 2 = BEAR

    Args:
        trend_snr: Trend signal-to-noise ratio array [T].
        enter_theta: Scalar absolute SNR enter threshold (fallback / ratio base).
        exit_theta: Scalar absolute SNR exit threshold.  Must be < ``enter_theta``.
        enter_band: Optional per-bar adaptive enter threshold [T].  When finite
            and positive at bar t, enter=enter_band[t] and exit=ratio·enter_band[t]
            with ratio = exit_theta/enter_theta.

    Returns:
        int8 array [T]: 0=NEUTRAL, 1=BULL, 2=BEAR.

    Time complexity: O(T) sequential.
    Space complexity: O(T).
    """
    t_len: int = trend_snr.shape[0]
    result: NDArray[np.int8] = np.zeros(t_len, dtype=np.int8)
    ratio: float = exit_theta / enter_theta if enter_theta > 0.0 else 0.4
    state: int = 0  # 0=NEUTRAL
    for t in range(t_len):
        snr = trend_snr[t]
        if not np.isfinite(snr):
            result[t] = np.int8(state)
            continue
        if enter_band is not None and np.isfinite(enter_band[t]) and enter_band[t] > 0.0:
            e_th = float(enter_band[t])
            x_th = ratio * e_th
        else:
            e_th = enter_theta
            x_th = exit_theta
        if state == 0:  # NEUTRAL
            if snr >= e_th:
                state = 1  # → BULL
            elif snr <= -e_th:
                state = 2  # → BEAR
        elif state == 1:  # BULL
            if snr <= -x_th:
                state = 0  # → NEUTRAL
        else:  # BEAR (state == 2)
            if snr >= x_th:
                state = 0  # → NEUTRAL
        result[t] = np.int8(state)
    return result


def _persistence_targeted_band(
    snr_abs: NDArray[np.float64],
    target_dwell: float,
    min_n_eff: int = 60,
) -> NDArray[np.float64]:
    """Causal persistence-targeted transition band via Markov p_ii inversion.

    Derives a per-bar band threshold such that the fraction of bars classified
    as decisive (``|snr| >= band``) approximates the target steady-state
    probability implied by the desired dwell time.

    Math:
        E[dwell] = 1 / (1 - p_ii)  →  p_ii = 1 - 1 / target_dwell
        decisive_fraction ≈ 1 - p_ii = 1 / target_dwell
        band[t] = quantile(|snr|[0..t], 1 - 1/target_dwell)

    Args:
        snr_abs: Absolute SNR values [T] (non-negative, finite or NaN).
        target_dwell: Target expected dwell in bars. Clamped to >= 2.
        min_n_eff: Bars required before band is updated from default (0.5).

    Returns:
        float64 band array [T].

    Time complexity: O(T²·logT) causal expanding quantile.
    Space complexity: O(T).
    """
    t_len: int = snr_abs.shape[0]
    band: NDArray[np.float64] = np.full(t_len, 0.5, dtype=np.float64)
    safe_dwell = max(float(target_dwell), 2.0)
    target_p_ii = 1.0 - 1.0 / safe_dwell
    # Decisive_fraction (|snr| >= band) should equal 1 - target_p_ii = 1/target_dwell.
    # → band must be the target_p_ii-th quantile so only the top 1/dwell fraction exceeds it.
    for t in range(min_n_eff, t_len):
        sample = snr_abs[: t + 1]
        finite = sample[np.isfinite(sample)]
        if finite.size > 0:
            band[t] = float(np.percentile(finite, target_p_ii * 100.0))
    return band


def _continuous_regime_codes(
    *,
    trend_snr: NDArray[np.float64],
    vol_scale: NDArray[np.float64],
    crisis_active: NDArray[np.bool_],
    trend_band_arr: NDArray[np.float64] | None = None,
    vol_threshold: NDArray[np.float64] | None = None,
    schmitt_state: NDArray[np.int8] | None = None,
) -> NDArray[np.int8]:
    """Assign 6-state discrete regime codes from continuous overlay signals.

    Args:
        trend_snr: Trend signal-to-noise ratio array [T].
        vol_scale: Volatility scaling factor array [T].
        crisis_active: Boolean crisis flag array [T].
        trend_band_arr: Per-bar transition half-width array [T]. Used only
            when ``schmitt_state`` is None. Bars where
            ``|trend_snr[t]| < trend_band_arr[t]`` are assigned transition (code 4).
            If None, falls back to fixed 0.5 for all bars (conservative default).
        vol_threshold: Per-bar adaptive threshold for quiet/volatile split [T].
            If None, falls back to fixed value 1.0 (original behaviour).
        schmitt_state: Optional stateful Schmitt hysteresis state array [T].
            int8: 0=NEUTRAL, 1=BULL, 2=BEAR. When provided, replaces the
            stateless sign-cut decisiveness/direction logic; ``trend_band_arr``
            is ignored. Backward compatible: if None, original stateless logic runs.

    Returns:
        int8 array of regime codes [T]:
            0=bull_quiet, 1=bull_volatile, 2=bear_quiet, 3=bear_volatile,
            4=transition, 5=crash.

    Time complexity: O(T). Space complexity: O(T).
    """
    code = np.full(trend_snr.shape[0], 4, dtype=np.int8)  # transition = default
    finite = np.isfinite(trend_snr) & np.isfinite(vol_scale)
    if schmitt_state is not None:
        # Schmitt stateful path: 0=NEUTRAL → transition, 1=BULL, 2=BEAR
        decisive = finite & (schmitt_state != 0)
        bull = schmitt_state == 1
    else:
        # Stateless path (backward compat): band-based decisiveness + sign direction
        tb: NDArray[np.float64] = (
            trend_band_arr if trend_band_arr is not None else np.full(trend_snr.shape[0], 0.5, dtype=np.float64)
        )
        decisive = finite & (np.abs(trend_snr) >= tb)
        bull = trend_snr >= 0.0
    # adaptive threshold: vol_threshold 제공 시 사용, 아니면 1.0 고정
    vt: NDArray[np.float64] = vol_threshold if vol_threshold is not None else np.ones_like(vol_scale)
    quiet = vol_scale >= vt
    code[decisive & bull & quiet] = 0
    code[decisive & bull & ~quiet] = 1
    code[decisive & ~bull & quiet] = 2
    code[decisive & ~bull & ~quiet] = 3
    # decisive==False인 finite 구간은 code 4(transition) 유지
    code[crisis_active] = 5
    return code


def _dwell_median(code_1d: NDArray[np.int8]) -> float:
    if code_1d.size == 0:
        return 0.0
    dwell: list[int] = []
    run = 1
    for idx in range(1, code_1d.shape[0]):
        if int(code_1d[idx]) == int(code_1d[idx - 1]):
            run += 1
            continue
        dwell.append(run)
        run = 1
    dwell.append(run)
    return float(np.median(np.asarray(dwell, dtype=np.float64)))


def _weighted_tstat(diff: NDArray[np.float64], min_n_eff: int) -> float:
    finite = diff[np.isfinite(diff)]
    n_eff = finite.shape[0]
    if n_eff < min_n_eff or n_eff < 2:
        return 0.0
    std = float(np.std(finite, ddof=1))
    if std <= _EPS:
        return 0.0
    return float(np.mean(finite) / (std / math.sqrt(float(n_eff))))


def _safe_regime_cfg(cfg: RegimeConfig | None) -> RegimeConfig:
    if cfg is not None:
        return cfg
    from src.domain.futures.strategy.config import RegimeConfig as RuntimeRegimeConfig

    return RuntimeRegimeConfig()


def _clone_aligned_with_close(
    aligned: AlignedMarketData,
    close_2d: NDArray[np.float64],
) -> AlignedMarketData:
    return AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=aligned.active_mask,
        warm_mask=aligned.warm_mask,
        entry_block_mask=aligned.entry_block_mask,
        kill_mask=aligned.kill_mask,
        basis_2d=aligned.basis_2d,
        oi_2d=aligned.oi_2d,
        lsr_2d=aligned.lsr_2d,
        adv_usdt_2d=aligned.adv_usdt_2d,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
        inference_active_mask=aligned.inference_active_mask,
        inference_entry_warm_mask=aligned.inference_entry_warm_mask,
        vol_30d_1d=aligned.vol_30d_1d,
        friction_score_1d=aligned.friction_score_1d,
        alpha_capacity_score_1d=aligned.alpha_capacity_score_1d,
        diversification_score_1d=aligned.diversification_score_1d,
        tradeable_score_1d=aligned.tradeable_score_1d,
        cluster_id_1d=aligned.cluster_id_1d,
        beta_vs_market_1d=aligned.beta_vs_market_1d,
        cluster_size_1d=aligned.cluster_size_1d,
        anchor_cluster_1d=aligned.anchor_cluster_1d,
        symbol_meta=aligned.symbol_meta,
    )


@dataclass(slots=True, frozen=True)
class RiskOverlayContext:
    vol_scale_1d: NDArray[np.float64]
    trend_scale_1d: NDArray[np.float64]
    crisis_active_1d: NDArray[np.bool_]
    overlay_mult_1d: NDArray[np.float64]


def compute_trend_efficiency_1d(
    close_1d: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    t = close_1d.shape[0]
    er = np.full(t, np.nan, dtype=np.float64)
    if t < window + 1:
        return er
    diff = np.abs(np.diff(close_1d, prepend=close_1d[:1]))
    cumsum = np.cumsum(diff)
    path = np.empty(t, dtype=np.float64)
    path[:window] = np.nan
    for i in range(window, t):
        path[i] = cumsum[i] - cumsum[i - window]
    net = np.abs(close_1d - np.roll(close_1d, window))
    net[:window] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.abs(path) > _EPS, net / path, 0.0)
    er[window:] = ratio[window:]
    return np.clip(er, 0.0, 1.0)


def compute_positioning_crowding_z_2d(
    aligned: AlignedMarketData,
    *,
    tf: str,
    oi_change_window_base: int = 6,
    z_window_base: int = 42,
    base_tf: str = "4h",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """[ADR_20260703_L1_CROWD] 심볼별 OI 빌드업/LSR 쏠림 z-score 시계열 (TF-aware).

    aligned.oi_2d/lsr_2d가 None이면 해당 배열은 전량 NaN으로 반환.

    Args:
        aligned: 정렬 시장 데이터.
        tf: 대상 timeframe (예: "4h", "8h").
        oi_change_window_base: OI 변화량 window (base_tf 기준).
        z_window_base: z-score window (base_tf 기준).
        base_tf: 기준 timeframe.

    Returns:
        (oi_build_z_2d, lsr_log_z_2d) 각각 [T, N] float64.
    """
    from src.domain.futures.signals.rules import _zscore_2d

    close = aligned.close_2d
    oi = aligned.oi_2d
    lsr = aligned.lsr_2d
    oi_change_window = scale_bar_count(oi_change_window_base, tf, base_tf)
    z_window = scale_bar_count(z_window_base, tf, base_tf)

    if oi is None:
        oi_build_z_2d = np.full_like(close, np.nan, dtype=np.float64)
    else:
        oi_valid = np.isfinite(oi) & (oi > 0)
        with np.errstate(divide="ignore"):
            oi_log = np.where(oi_valid, np.log(oi), np.nan)
        oi_log_change = oi_log - np.roll(oi_log, oi_change_window, axis=0)
        oi_log_change[:oi_change_window] = np.nan
        oi_build_z_2d = _zscore_2d(oi_log_change, window=z_window)

    if lsr is None:
        lsr_log_z_2d = np.full_like(close, np.nan, dtype=np.float64)
    else:
        lsr_valid = np.isfinite(lsr) & (lsr > 0)
        with np.errstate(divide="ignore"):
            lsr_log = np.where(lsr_valid, np.log(lsr), np.nan)
        lsr_log_z_2d = _zscore_2d(lsr_log, window=z_window)

    return oi_build_z_2d, lsr_log_z_2d


def compute_crowding_persistent_mask_2d(
    oi_build_z_2d: NDArray[np.float64],
    lsr_log_z_2d: NDArray[np.float64],
    trend_sign_2d: NDArray[np.float64],
    *,
    oi_threshold: float = 0.5,
    lsr_threshold: float = 1.0,
    persistence_bars: int = 3,
    recovery_cooldown_bars: int = 3,
) -> NDArray[np.bool_]:
    """[ADR_20260703_L1_CROWD] 심볼별 positioning-crowding 지속성 마스크 [T, N].

    NaN 입력은 raw condition에서 False로 간주 (nan_to_num).
    각 컬럼(심볼)별 독립 상태기계 적용.

    Args:
        oi_build_z_2d: OI build z-score [T, N].
        lsr_log_z_2d: LSR log z-score [T, N].
        trend_sign_2d: 트렌드 베팅 부호 (+1/-1/0) [T, N].
        oi_threshold: OI build z-score 임계값.
        lsr_threshold: LSR log z-score 절대값 임계값.
        persistence_bars: 연속 발화 필요 bar 수.
        recovery_cooldown_bars: raw-off 후 상태 유지 bar 수.

    Returns:
        [T, N] bool, 1-bar 시차 포함.

    Raises:
        ValueError: persistence_bars < 1.
    """
    if persistence_bars < 1:
        raise ValueError("persistence_bars must be >= 1")

    raw_long = (trend_sign_2d > 0) & (oi_build_z_2d >= oi_threshold) & (lsr_log_z_2d >= lsr_threshold)
    raw_short = (trend_sign_2d < 0) & (oi_build_z_2d >= oi_threshold) & (lsr_log_z_2d <= -lsr_threshold)
    raw_2d = np.nan_to_num(raw_long | raw_short, nan=False)

    n_cols = raw_2d.shape[1]
    mask_2d = np.empty_like(raw_2d)
    for j in range(n_cols):
        mask_2d[:, j] = _apply_persistence_and_cooldown_1d(
            raw_2d[:, j],
            persistence_bars,
            recovery_cooldown_bars,
        )
    return mask_2d


def compute_crowding_dampener_mult(
    is_crowded: bool,
    *,
    floor_mult: float,
) -> float:
    """[ADR_20260703_L1_CROWD] persistence-gated crowding 상태에 따른 raw_mu 감쇠 승수.

    floor_mult는 기본값 없음(필수 키워드 인자) — 매직넘버 방지.

    Args:
        is_crowded: persistence-gated crowding 활성 상태.
        floor_mult: crowded 시 곱할 승수 (0 < floor_mult <= 1.0).

    Returns:
        floor_mult if is_crowded else 1.0.
    """
    return floor_mult if is_crowded else 1.0


@dataclass(slots=True, frozen=True)
class MarketRegimeContext:
    code_1d: NDArray[np.int8]
    name_by_code: tuple[str, ...]
    trend_score_1d: NDArray[np.float64]
    vol_z_1d: NDArray[np.float64]
    dispersion_z_1d: NDArray[np.float64]
    trend_efficiency_1d: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    vol_scale_1d: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    crisis_active_1d: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, dtype=np.bool_))

    def names(self) -> NDArray[np.object_]:
        return np.asarray([self.name_by_code[int(code)] for code in self.code_1d], dtype=object)


@dataclass(slots=True, frozen=True)
class RegimeQualityReport:
    persistence_dwell: float
    leakage_ok: bool
    overlay_lift_bps: float
    overlay_lift_tstat: float
    crisis_precision_ok: bool
    passed: bool
    reasons: tuple[str, ...]


def compute_risk_overlay(
    *,
    aligned: AlignedMarketData,
    cfg: RegimeConfig | None = None,
) -> RiskOverlayContext:
    """Return a causal, self-calibrating continuous risk overlay."""
    regime_cfg = _safe_regime_cfg(cfg)
    btc_log_ret = _btc_log_returns(aligned)
    bars_per_year = _infer_bars_per_year(aligned.datetimes)

    vol_mean = _ema_1d(btc_log_ret, regime_cfg.overlay_vol_ewma_span)
    vol_var = _ema_1d(np.square(btc_log_ret - vol_mean), regime_cfg.overlay_vol_ewma_span)
    realized_vol = np.sqrt(np.maximum(vol_var, 0.0) * bars_per_year)
    realized_vol = np.maximum(realized_vol, _EPS)
    vol_scale = np.clip(
        regime_cfg.overlay_target_vol_ann / realized_vol,
        regime_cfg.overlay_vol_scale_clip[0],
        regime_cfg.overlay_vol_scale_clip[1],
    )

    btc_close = np.maximum(aligned.close_2d[:, _btc_index(aligned.symbols)], _EPS)
    trend_anchor = _ema_1d(np.log(btc_close), regime_cfg.overlay_trend_snr_span)
    trend_score = np.log(btc_close) - trend_anchor
    trend_std = _rolling_std_1d(trend_score, regime_cfg.overlay_trend_snr_span)
    trend_snr = trend_score / np.maximum(trend_std, _EPS)
    trend_scale = 0.5 * (1.0 + np.tanh(np.nan_to_num(trend_snr, nan=0.0, posinf=6.0, neginf=-6.0)))

    robust_mu, robust_sigma = _expanding_robust_location_scale(btc_log_ret)
    standardized = (btc_log_ret - robust_mu) / np.maximum(robust_sigma, _EPS)
    standardized = np.clip(np.nan_to_num(standardized, nan=0.0), -10.0, 10.0)
    drift, threshold, hold_bars = _cusum_thresholds(regime_cfg.crisis_target_arl_bars)

    pos_cusum = 0.0
    neg_cusum = 0.0
    cooldown = 0
    crisis_active = np.zeros(standardized.shape[0], dtype=bool)
    for idx, residual in enumerate(standardized):
        pos_cusum = max(0.0, pos_cusum + residual - drift)
        neg_cusum = max(0.0, neg_cusum - residual - drift)
        if pos_cusum > threshold or neg_cusum > threshold:
            cooldown = hold_bars
            pos_cusum = 0.0
            neg_cusum = 0.0
        if cooldown > 0:
            crisis_active[idx] = True
            cooldown -= 1

    overlay_raw = vol_scale * trend_scale
    overlay_mult = np.where(
        crisis_active,
        regime_cfg.crisis_gross_floor,
        overlay_raw,
    )
    return RiskOverlayContext(
        vol_scale_1d=vol_scale.astype(np.float64, copy=False),
        trend_scale_1d=trend_scale.astype(np.float64, copy=False),
        crisis_active_1d=crisis_active.astype(np.bool_, copy=False),
        overlay_mult_1d=overlay_mult.astype(np.float64, copy=False),
    )


# In-memory memoization for compute_market_regime_context.
# Cache key: (aligned_content_fp, cfg_hash). Uses content-based fingerprint to avoid
# false hits from id() reuse across short-lived test objects. In production, the same
# aligned object persists throughout a run, so the content fp is stable.
_REGIME_MEMO: dict[tuple[str, str], MarketRegimeContext] = {}
_REGIME_MEMO_MAX = 8


def _aligned_fingerprint(aligned: AlignedMarketData) -> str:
    _c = np.asarray(aligned.close_2d, dtype=np.float64)
    _sig = (_c.shape, float(_c[0, 0]), float(_c[-1, 0]), float(_c[0, -1] if _c.shape[1] > 1 else 0))
    return str(hash(_sig))


def compute_market_regime_context(
    *,
    aligned: AlignedMarketData,
    cfg: RegimeConfig | None = None,
    overlay: RiskOverlayContext | None = None,
) -> MarketRegimeContext:
    if cfg is not None:
        try:
            _cfg_src = repr(sorted(vars(cfg).items()))
        except TypeError:
            _cfg_src = str(cfg)
        _cfg_hash = str(hash(_cfg_src))
    else:
        _cfg_hash = "default"
    _aligned_fp = _aligned_fingerprint(aligned)
    _key = (_aligned_fp, _cfg_hash)
    _cached = _REGIME_MEMO.get(_key)
    if _cached is not None:
        return _cached

    close = np.asarray(aligned.close_2d, dtype=np.float64)
    if close.ndim != 2 or close.shape[0] == 0:
        raise ValueError("aligned.close_2d must be non-empty 2D array")

    regime_cfg = _safe_regime_cfg(cfg)
    overlay = compute_risk_overlay(aligned=aligned, cfg=regime_cfg) if overlay is None else overlay
    btc_close = np.maximum(close[:, _btc_index(aligned.symbols)], _EPS)
    trend_anchor = _ema_1d(np.log(btc_close), regime_cfg.overlay_trend_snr_span)
    trend_score = np.log(btc_close) - trend_anchor
    trend_std = _rolling_std_1d(trend_score, regime_cfg.overlay_trend_snr_span)
    trend_snr = trend_score / np.maximum(trend_std, _EPS)

    btc_log_ret = _btc_log_returns(aligned)
    bars_per_year = _infer_bars_per_year(aligned.datetimes)
    vol_mean = _ema_1d(btc_log_ret, regime_cfg.overlay_vol_ewma_span)
    vol_var = _ema_1d(np.square(btc_log_ret - vol_mean), regime_cfg.overlay_vol_ewma_span)
    realized_vol = np.sqrt(np.maximum(vol_var, 0.0) * bars_per_year)
    vol_z = _zscore_1d(np.log(np.maximum(realized_vol, _EPS)), regime_cfg.overlay_trend_snr_span)
    dispersion_z = _compute_dispersion_z(aligned, regime_cfg.overlay_trend_snr_span)

    # causal expanding median of vol_scale as adaptive threshold for quiet/volatile split
    vol_median = _expanding_quantile_causal(overlay.vol_scale_1d, 0.5)
    # fallback to 1.0 where insufficient data (< regime_min_n_eff bars)
    min_n = regime_cfg.regime_min_n_eff
    vol_threshold = np.where(
        np.arange(vol_median.shape[0]) >= min_n,
        vol_median,
        np.ones_like(vol_median),
    ).astype(np.float64)

    # --- P1: Schmitt hysteresis + persistence-targeted band ---
    # RegimeConfig에 속성이 없는 경우 폴백 기본값 사용
    enter_theta: float = float(getattr(regime_cfg, "trend_hysteresis_enter", 0.35))
    exit_theta: float = float(getattr(regime_cfg, "trend_hysteresis_exit", 0.15))
    target_dwell: float = float(getattr(regime_cfg, "persistence_target_dwell", 6.0))

    trend_snr_clean: NDArray[np.float64] = np.nan_to_num(trend_snr, nan=0.0)

    # persistence-targeted band: self-calibrating adaptive enter threshold
    abs_snr: NDArray[np.float64] = np.abs(trend_snr_clean)
    trend_band_arr: NDArray[np.float64] = _persistence_targeted_band(abs_snr, target_dwell, min_n)

    # Schmitt stateful directional state (hysteresis) driven by the adaptive band;
    # scalar enter/exit thetas only set the exit/enter ratio (and pre-min_n fallback).
    schmitt_state: NDArray[np.int8] = _schmitt_directional_state(
        trend_snr_clean, enter_theta, exit_theta, enter_band=trend_band_arr
    )

    code = _continuous_regime_codes(
        trend_snr=trend_snr_clean,
        vol_scale=overlay.vol_scale_1d,
        crisis_active=overlay.crisis_active_1d,
        trend_band_arr=trend_band_arr,
        vol_threshold=vol_threshold,
        schmitt_state=schmitt_state,
    )

    # Step F: regime distribution DEBUG log
    if _logger.isEnabledFor(logging.DEBUG):
        _unique_r, _counts_r = np.unique(code, return_counts=True)
        _n_total_r = int(code.shape[0])
        _summary = "; ".join(
            f"{_REGIME_NAMES[int(r)]}({int(r)}):{float(c) / float(_n_total_r) * 100:.1f}%"
            for r, c in sorted(zip(_unique_r.tolist(), _counts_r.tolist(), strict=True))
        )
        _logger.debug("[REGIME-DIST] total_bars=%d %s", _n_total_r, _summary)

    trend_efficiency_1d = compute_trend_efficiency_1d(
        btc_close,
        regime_cfg.trend_efficiency_window,
    )

    _result = MarketRegimeContext(
        code_1d=code,
        name_by_code=_REGIME_NAMES,
        trend_score_1d=trend_snr.astype(np.float64, copy=False),
        vol_z_1d=vol_z.astype(np.float64, copy=False),
        dispersion_z_1d=dispersion_z.astype(np.float64, copy=False),
        trend_efficiency_1d=trend_efficiency_1d,
        vol_scale_1d=overlay.vol_scale_1d,
        crisis_active_1d=overlay.crisis_active_1d,
    )
    if len(_REGIME_MEMO) >= _REGIME_MEMO_MAX:
        _REGIME_MEMO.pop(next(iter(_REGIME_MEMO)))
    _REGIME_MEMO[_key] = _result
    return _result


def _rolling_max_1d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    out = np.empty(values.shape[0], dtype=np.float64)
    for i in range(values.shape[0]):
        start = max(0, i - window + 1)
        out[i] = float(np.max(values[start : i + 1]))
    return out


def _apply_persistence_and_cooldown_1d(
    raw: NDArray[np.bool_],
    persistence_bars: int,
    recovery_cooldown_bars: int,
) -> NDArray[np.bool_]:
    """Persistence + recovery cooldown 상태기계 + 1-bar 시차 (심볼 독립 1-D).

    compute_reversal_risk_off_1d L641-664에서 추출한 공용 헬퍼.
    NaN 입력은 False로 간주.

    Args:
        raw: 원시 조건 [T], bool.
        persistence_bars: 연속 발화 필요 bar 수 (>=1).
        recovery_cooldown_bars: raw-off 후 상태 유지 bar 수.

    Returns:
        [T] bool, 1-bar 시차 포함 (out[0] == False).
    """
    t = raw.shape[0]
    if t == 0:
        return np.empty(0, dtype=np.bool_)
    if persistence_bars > 1:
        run_count = 0
        persistent = np.zeros_like(raw)
        for i in range(t):
            run_count = run_count + 1 if bool(raw[i]) else 0
            persistent[i] = run_count >= persistence_bars
    else:
        persistent = raw
    if recovery_cooldown_bars == 0:
        return np.concatenate([[False], persistent[:-1]]).astype(np.bool_)
    s_on = False
    off_run = 0
    state: NDArray[np.bool_] = np.zeros(t, dtype=np.bool_)
    for i in range(t):
        if bool(persistent[i]):
            s_on = True
            off_run = 0
        elif s_on:
            off_run = off_run + 1 if not bool(raw[i]) else 0
            if off_run >= max(recovery_cooldown_bars, 1) and not bool(raw[i]):
                s_on = False
        state[i] = s_on
    return np.concatenate([[False], state[:-1]]).astype(np.bool_)


def apply_regime_cap_release_cooldown(
    code_1d: NDArray[np.int8],
    *,
    cooldown_bars: int,
) -> NDArray[np.int8]:
    """[ADR_20260718_L2_CRISIS_REGIME_CAP_RELEASE_COOLDOWN] 레짐 캡 전용 파생 코드.

    code_1d 자체(방향 alpha 신호)는 변경하지 않는다 — 오직 캡 계산에만 쓰는
    파생 배열을 반환한다. bear/crisis(1,2) 진입은 항상 즉시 반영되고, bull(0)
    으로의 "복귀"만 최근 cooldown_bars 이내에 bear/crisis가 있었으면 지연된다
    (대체 상태는 항상 bear=1, crisis로 승격하지 않음).

    Args:
        code_1d: 원본 레짐 코드 [T] ∈ {0=bull, 1=bear, 2=crisis}.
        cooldown_bars: bear/crisis 이후 bull 복귀를 지연시킬 최소 bar 수.
            0이면 원본과 동일(no-op).

    Returns:
        [T] int8, 캡 계산 전용 파생 코드. code_1d와 shape 동일.

    Raises:
        ValueError: cooldown_bars < 0.
    """
    if cooldown_bars < 0:
        raise ValueError(f"cooldown_bars must be >= 0, got {cooldown_bars}")
    if cooldown_bars == 0 or code_1d.size == 0:
        return code_1d.copy()
    defensive_raw = code_1d != 0
    sticky_defensive = _apply_persistence_and_cooldown_1d(
        defensive_raw, persistence_bars=1, recovery_cooldown_bars=cooldown_bars
    )
    out = code_1d.copy()
    out[sticky_defensive & (code_1d == 0)] = 1
    return out



def compute_risk_severity_code(
    vol_scale_1d: NDArray[np.float64],
    crisis_active_1d: NDArray[np.bool_],
    *,
    elevated_vol_quantile: float = 0.35,
    min_n_eff: int = 60,
) -> NDArray[np.int8]:
    """방향(trend) 신뢰도와 완전히 독립적인 리스크 심각도 코드.

    0=calm, 1=elevated(causal expanding quantile 이하 vol_scale — 고변동성),
    2=crash(CUSUM crisis_active, 항상 최우선 오버라이드). transition(방향
    불확실)은 더 이상 자동으로 crisis 취급되지 않고 자신의 실제 변동성
    수준에 따라 calm 또는 elevated로 분류된다.

    Args:
        vol_scale_1d: target_vol/realized_vol [T] (낮을수록 고변동성).
        crisis_active_1d: CUSUM 급변 플래그 [T].
        elevated_vol_quantile: elevated 판정 causal quantile.
        min_n_eff: 최소 관측치(이전은 보수적으로 calm 처리).

    Returns:
        int8 [T], {0,1,2}.
    """
    vs = np.asarray(vol_scale_1d, dtype=np.float64)
    ca = np.asarray(crisis_active_1d, dtype=np.bool_)
    if vs.size == 0:
        return np.array([], dtype=np.int8)

    elevated_threshold = _expanding_quantile_causal(vs, elevated_vol_quantile)
    if elevated_threshold.size == 0:
        elevated_threshold = np.full(vs.shape, 0.5)

    code = np.zeros(vs.shape[0], dtype=np.int8)
    for t in range(vs.shape[0]):
        if t < min_n_eff:
            code[t] = 0
        elif ca[t]:
            code[t] = 2
        elif vs[t] < elevated_threshold[t]:
            code[t] = 1
        else:
            code[t] = 0
    return code


def compute_reversal_risk_off_1d(
    btc_close_1d: NDArray[np.float64],
    *,
    dd_window: int,
    dd_threshold: float,
    mom_fast: int,
    mom_slow: int,
    persistence_bars: int = 1,
    recovery_cooldown_bars: int = 0,
) -> NDArray[np.bool_]:
    """Trailing drawdown + momentum 기반 시장반전 risk-off 마스크 [T]."""
    if persistence_bars < 1:
        raise ValueError("persistence_bars must be >= 1")
    t = btc_close_1d.shape[0]
    if t == 0:
        return np.empty(0, dtype=np.bool_)
    high_water = _rolling_max_1d(btc_close_1d, dd_window)
    dd = 1.0 - btc_close_1d / np.maximum(high_water, _EPS)
    mom_fast_ema = _ema_1d(btc_close_1d, mom_fast)
    mom_slow_ema = _ema_1d(btc_close_1d, mom_slow)
    mom = mom_fast_ema - mom_slow_ema
    raw = (dd >= dd_threshold) & (mom < 0.0)
    return _apply_persistence_and_cooldown_1d(raw, persistence_bars, recovery_cooldown_bars)


def _synthetic_ath_decline_path() -> NDArray[np.float64]:
    rise = np.linspace(100.0, 110.0, 20, dtype=np.float64)
    fall = np.linspace(110.0, 85.0, 30, dtype=np.float64)
    return np.concatenate([rise, fall])


def synthetic_crash_defense_verdict(
    *,
    dd_window: int = 90,
    dd_threshold: float = 0.12,
    mom_fast: int = 20,
    mom_slow: int = 120,
    persistence_bars: int = 3,
    recovery_cooldown_bars: int = 0,
) -> tuple[bool, int]:
    close = _synthetic_ath_decline_path()
    risk_off = compute_reversal_risk_off_1d(
        close,
        dd_window=dd_window,
        dd_threshold=dd_threshold,
        mom_fast=mom_fast,
        mom_slow=mom_slow,
        persistence_bars=persistence_bars,
        recovery_cooldown_bars=recovery_cooldown_bars,
    )
    fires = bool(risk_off[20:].any())
    risk_off_bar_count = int(risk_off.sum())
    return (fires, risk_off_bar_count)


def compute_xs_downside_breadth_1d(
    universe_close_2d: NDArray[np.float64],
    *,
    mom_window: int,
) -> NDArray[np.float64]:
    """각 t에서 mom_window-bar 로그수익 < 0 인 심볼 비율(유효심볼 기준).

    r[t,i] = log(close[t,i] / close[t-mom_window,i]); 유효 = finite & 양수가격.
    neg_frac[t] = (#유효∧r<0) / max(#유효, 1); t<mom_window 또는 #유효=0 → 0.0.
    look-ahead 없음(close[t]까지만 사용). 소비측에서 1-bar shift 책임.

    Args:
        universe_close_2d: [T, N] 종가 배열.
        mom_window: 모멘텀 산술 기간(bar).

    Returns:
        [T] float64, 값 ∈ [0,1]. NaN-safe, zero-division safe.
    """
    prev = np.full_like(universe_close_2d, np.nan, dtype=np.float64)
    prev[mom_window:] = universe_close_2d[:-mom_window]
    valid = np.isfinite(universe_close_2d) & (universe_close_2d > _EPS) & np.isfinite(prev) & (prev > _EPS)
    r = np.where(valid, np.log(universe_close_2d / np.maximum(prev, _EPS)), np.nan)
    neg = valid & (r < 0.0)
    denom = valid.sum(axis=1)
    neg_frac = np.where(denom > 0, neg.sum(axis=1) / np.maximum(denom, 1), 0.0)
    return neg_frac


def compute_market_state_risk_off_1d(
    btc_close_1d: NDArray[np.float64],
    universe_close_2d: NDArray[np.float64],
    *,
    dd_window: int,
    dd_threshold: float,
    mom_fast: int,
    mom_slow: int,
    breadth_mom_window: int,
    breadth_neg_frac_enter: float,
    breadth_neg_frac_exit: float,
    persistence_bars: int = 1,
    recovery_cooldown_bars: int = 0,
) -> NDArray[np.bool_]:
    """BTC-axis ∧ breadth-axis AND-게이트 + 비대칭 hysteresis de-gross 마스크.

    btc_off = (trailing_dd >= dd_threshold) & (ema_fast - ema_slow < 0)
    breadth = compute_xs_downside_breadth_1d(universe_close_2d, mom_window=breadth_mom_window)
    breadth_on[t] = hysteresis(breadth, enter=breadth_neg_frac_enter, exit=breadth_neg_frac_exit)
    raw_on[t]  = btc_off[t] & breadth_on[t]
    persist_on = persistence(raw_on, persistence_bars)
    state_on   = recovery_hysteresis(persist_on, raw_on, recovery_cooldown_bars)
    return shift1(state_on)

    Args:
        btc_close_1d: BTC 종가 [T].
        universe_close_2d: 유니버스 종가 [T, N].
        dd_window: trailing drawdown window.
        dd_threshold: drawdown 임계.
        mom_fast: fast EMA span.
        mom_slow: slow EMA span (must be > mom_fast).
        breadth_mom_window: breadth 모멘텀 window.
        breadth_neg_frac_enter: breadth 진입 임계.
        breadth_neg_frac_exit: breadth 해제 임계 (hysteresis).
        persistence_bars: 연속 발화 요구 bar 수.
        recovery_cooldown_bars: raw-off 해제 대기 bar 수.

    Returns:
        [T] bool, 1-bar shift 적용 (t에서 t-1 결정).
    """
    t = btc_close_1d.shape[0]
    if t == 0:
        return np.empty(0, dtype=np.bool_)

    high_water = _rolling_max_1d(btc_close_1d, dd_window)
    dd = 1.0 - btc_close_1d / np.maximum(high_water, _EPS)
    mom_fast_ema = _ema_1d(btc_close_1d, mom_fast)
    mom_slow_ema = _ema_1d(btc_close_1d, mom_slow)
    mom = mom_fast_ema - mom_slow_ema
    btc_off = (dd >= dd_threshold) & (mom < 0.0)

    breadth = compute_xs_downside_breadth_1d(universe_close_2d, mom_window=breadth_mom_window)

    b_on = False
    breadth_on: NDArray[np.bool_] = np.zeros(t, dtype=np.bool_)
    for i in range(t):
        if not b_on and breadth[i] >= breadth_neg_frac_enter:
            b_on = True
        elif b_on and breadth[i] < breadth_neg_frac_exit:
            b_on = False
        breadth_on[i] = b_on

    raw_on = btc_off & breadth_on

    if persistence_bars > 1:
        run_count = 0
        persist_on: NDArray[np.bool_] = np.zeros_like(raw_on)
        for i in range(raw_on.shape[0]):
            run_count = run_count + 1 if bool(raw_on[i]) else 0
            persist_on[i] = run_count >= persistence_bars
    else:
        persist_on = raw_on

    s_on = False
    off_run = 0
    state: NDArray[np.bool_] = np.zeros(t, dtype=np.bool_)
    for i in range(t):
        if bool(persist_on[i]):
            s_on = True
            off_run = 0
        elif s_on:
            off_run = off_run + 1 if not bool(raw_on[i]) else 0
            if off_run >= max(recovery_cooldown_bars, 1) and not bool(raw_on[i]):
                s_on = False
        state[i] = s_on

    return np.concatenate([[False], state[:-1]]).astype(np.bool_)


def evaluate_regime_quality(
    *,
    aligned: AlignedMarketData,
    cfg: RegimeConfig | None = None,
    base_edge_1d: NDArray[np.float64],
    cal_eval_mask: NDArray[np.bool_],
    is_mask: NDArray[np.bool_] | None = None,
) -> RegimeQualityReport:
    """Evaluate regime overlay quality on the cal-eval slice only."""
    del is_mask
    regime_cfg = _safe_regime_cfg(cfg)
    overlay = compute_risk_overlay(aligned=aligned, cfg=regime_cfg)
    regime_ctx = compute_market_regime_context(aligned=aligned, cfg=regime_cfg)

    cal_mask = np.asarray(cal_eval_mask, dtype=bool)
    if cal_mask.shape[0] != base_edge_1d.shape[0]:
        raise ValueError("cal_eval_mask and base_edge_1d must have identical length")

    base = np.asarray(base_edge_1d, dtype=np.float64)
    base_cal = base[cal_mask]
    overlaid_cal = base_cal * overlay.overlay_mult_1d[cal_mask]

    def _safe_sharpe(arr: NDArray[np.float64]) -> float:
        """Compute mean/std Sharpe; returns 0.0 if insufficient finite data."""
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            return 0.0
        return float(np.mean(finite) / max(float(np.std(finite, ddof=1)), _EPS))

    sharpe_base = _safe_sharpe(base_cal)
    sharpe_overlaid = _safe_sharpe(overlaid_cal)
    # risk-adjusted lift: Sharpe(overlaid) - Sharpe(base), scaled to bps for reporting
    overlay_lift_bps = (sharpe_overlaid - sharpe_base) * 1e4
    # tstat: directional significance of (overlaid - base) difference series
    overlay_diff = overlaid_cal - base_cal
    overlay_lift_tstat = _weighted_tstat(overlay_diff, regime_cfg.regime_min_n_eff)

    crisis_mask = cal_mask & overlay.crisis_active_1d
    normal_mask = cal_mask & ~overlay.crisis_active_1d
    crisis_mean = float(np.nanmean(base[crisis_mask])) if np.any(crisis_mask) else 0.0
    normal_mean = float(np.nanmean(base[normal_mask])) if np.any(normal_mask) else 0.0
    crisis_precision_ok = crisis_mean < normal_mean

    pivot = max(1, base.shape[0] // 2)
    perturbed_close = np.array(aligned.close_2d, dtype=np.float64, copy=True)
    perturbed_close[pivot + 1 :, :] *= 1.15
    perturbed = _clone_aligned_with_close(aligned, perturbed_close)
    leak_check = compute_risk_overlay(aligned=perturbed, cfg=regime_cfg)
    leakage_ok = bool(
        np.allclose(
            overlay.overlay_mult_1d[: pivot + 1],
            leak_check.overlay_mult_1d[: pivot + 1],
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        )
    )

    regime_dwell = _dwell_median(regime_ctx.code_1d[cal_mask])
    crisis_dwell = _dwell_median(overlay.crisis_active_1d[cal_mask].astype(np.int8, copy=False))
    persistence_dwell = max(regime_dwell, crisis_dwell)
    reasons: list[str] = []
    if persistence_dwell < 6.0:
        reasons.append("dwell_below_threshold")
    if not leakage_ok:
        reasons.append("leakage_detected")
    if overlay_lift_bps <= 0.0:
        reasons.append("overlay_lift_non_positive")
    if overlay_lift_tstat < regime_cfg.regime_overlay_min_lift_tstat:
        reasons.append("overlay_lift_below_threshold")
    if not crisis_precision_ok:
        reasons.append("crisis_precision_failed")
    if np.isfinite(base[cal_mask]).sum() < regime_cfg.regime_min_n_eff:
        reasons.append("insufficient_cal_eval_obs")

    passed = not reasons if regime_cfg.regime_quality_gate_enabled else True
    report = RegimeQualityReport(
        persistence_dwell=persistence_dwell,
        leakage_ok=leakage_ok,
        overlay_lift_bps=overlay_lift_bps,
        overlay_lift_tstat=overlay_lift_tstat,
        crisis_precision_ok=crisis_precision_ok,
        passed=passed,
        reasons=tuple(reasons),
    )
    from src.domain.futures.strategy.rule_diagnostics import log_regime_quality_report

    log_regime_quality_report(report)
    return report
