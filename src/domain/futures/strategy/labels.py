from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.forecast.diagnostics import LabelDiagnostics
from src.domain.futures.portfolio.friction_model import resolve_cost_snapshot
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.contracts import LabelPanel
from src.domain.futures.strategy.diagnostics import gross_return_diagnostics

_logger = logging.getLogger(__name__)

_BETA_WINDOW: int = 120
_BETA_MIN_PERIODS: int = 20


def _build_relevance(
    signed_ret: NDArray[np.float32],
    eligible: NDArray[np.bool_],
    min_group_size: int,
) -> NDArray[np.int32]:
    rel = np.zeros(signed_ret.shape, dtype=np.int32)
    for t in range(signed_ret.shape[0]):
        idx = np.flatnonzero(eligible[t] & np.isfinite(signed_ret[t]))
        if idx.size < min_group_size:
            continue
        vals = signed_ret[t, idx]
        q15 = float(np.nanpercentile(vals, 15))
        q35 = float(np.nanpercentile(vals, 35))
        q65 = float(np.nanpercentile(vals, 65))
        q85 = float(np.nanpercentile(vals, 85))
        rel[t, idx] = np.where(
            vals >= q85,
            4,
            np.where(vals >= q65, 3, np.where(vals >= q35, 2, np.where(vals >= q15, 1, 0))),
        )
    return np.asarray(rel, dtype=np.int32)


def _compute_trailing_beta(
    close_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute per-symbol rolling OLS beta against cross-sectional market return.

    All computation uses only past bars — no look-ahead.
    beta_2d defaults to 1.0 when insufficient history exists.

    Args:
        close_2d: Close price array of shape [T, N].

    Returns:
        beta_2d: Rolling beta array of shape [T, N].
        Time complexity: O(T * N). Space complexity: O(T * N).

    """
    t_len, n_len = close_2d.shape
    # 1-bar spot log returns: shape [T, N]
    spot_ret: NDArray[np.float64] = np.full((t_len, n_len), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = close_2d[1:] / close_2d[:-1]
        log_r = np.where((ratio > 0) & np.isfinite(ratio), np.log(ratio), np.nan)
    spot_ret[1:] = log_r

    # Equal-weighted cross-sectional market return per bar: shape [T]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        market_ret: NDArray[np.float64] = np.nanmean(spot_ret, axis=1)

    # Rolling OLS beta per symbol (loop over N ~20-50 symbols, not T). Zero-Loop policy compliance.
    mkt_series: pd.Series = pd.Series(market_ret)
    beta_2d: NDArray[np.float64] = np.ones((t_len, n_len), dtype=np.float64)
    for col in range(n_len):
        sym_series = pd.Series(spot_ret[:, col])
        cov = sym_series.rolling(_BETA_WINDOW, min_periods=_BETA_MIN_PERIODS).cov(mkt_series)
        var = mkt_series.rolling(_BETA_WINDOW, min_periods=_BETA_MIN_PERIODS).var()
        beta = (cov / var.clip(lower=1e-12)).fillna(1.0).clip(-5.0, 5.0)
        beta_2d[:, col] = beta.to_numpy()

    return beta_2d


def build_label_panel(aligned: AlignedMarketData, cfg: StrategyMLConfig) -> LabelPanel:
    """Build t+1 execution aligned label tensors with beta-residualized returns.

    Uses trailing rolling OLS beta (look-ahead free) to remove cross-sectional
    market factor from gross log returns before training signal generation.
    Cost deduction is deferred to the objectives layer (canonical B1 fix).

    Args:
        aligned: Aligned market data tensors.
        cfg: Strategy ML configuration.

    Returns:
        LabelPanel with beta-residualized net returns and relevance scores.

    """
    t_len, n_len = aligned.close_2d.shape
    if n_len == 1:
        _logger.warning(
            "[RAW-SIGNAL-DIAG] WARNING: single-symbol universe (n_len=1). "
            "Beta-residualization collapses price signal to ~0 (market == symbol). "
            "Cross-sectional IC is undefined. Use N>=5 for valid diagnostics."
        )
    horizon = cfg.label_horizon_bars
    long_net = np.full((t_len, n_len), np.nan, dtype=np.float32)
    short_net = np.full((t_len, n_len), np.nan, dtype=np.float32)
    # Pre-allocated capture of raw gross log returns (before beta-residualization).
    # Shape: [T, N]. Used for RAW-SIGNAL-DIAG and "gross" calibrator_target branch.
    gross_long_2d: NDArray[np.float32] = np.full((t_len, n_len), np.nan, dtype=np.float32)
    # Phase D: use_inference_active_mask 분기 — C1 학습 시 Stage5 timeline 마스크 우선 사용.
    # inference_active_mask가 None이면(데이터에 컬럼 없음) 기존 universe 마스크로 fallback.
    _use_inf_mask: bool = (
        getattr(cfg, "use_inference_active_mask", True)
        and aligned.inference_active_mask is not None
        and aligned.inference_entry_warm_mask is not None
    )
    if _use_inf_mask:
        # inference 학습: entry_block_mask는 Stage6 trading 전용이므로 미적용.
        # inference_entry_warm_mask가 warm-up 기간 완료 여부를 나타냄.
        assert aligned.inference_active_mask is not None  # _use_inf_mask 조건으로 보장
        assert aligned.inference_entry_warm_mask is not None  # _use_inf_mask 조건으로 보장
        eligible = (
            aligned.inference_active_mask
            & aligned.inference_entry_warm_mask
            & ~aligned.kill_mask
        )
    else:
        eligible = (
            aligned.active_mask
            & aligned.warm_mask
            & ~aligned.entry_block_mask
            & ~aligned.kill_mask
        )
    # Gross alpha: cost deducted once at objectives layer (B1 canonical fix).
    cost = np.float64(0.0)

    # --- Beta-residualization (B2) ---
    # beta_2d[t, i]: trailing OLS beta at bar t — pure past data, no look-ahead.
    beta_2d: NDArray[np.float64] = _compute_trailing_beta(aligned.close_2d)

    # Market forward return for the same label horizon (equal-weighted, vectorized).
    # Computed for bars t using [t+1 .. t+horizon] — identical indexing to gross_long.
    market_fwd_ret: NDArray[np.float64] = np.full(t_len, np.nan, dtype=np.float64)
    if t_len > horizon:
        entry_mkt = aligned.open_2d[1 : t_len - horizon + 1]  # shape [T-h, N]
        exit_mkt = aligned.close_2d[horizon:t_len]  # shape [T-h, N]
        valid_mkt = (
            (entry_mkt > 0.0)
            & (exit_mkt > 0.0)
            & np.isfinite(entry_mkt)
            & np.isfinite(exit_mkt)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rets = np.where(valid_mkt, np.log(exit_mkt / entry_mkt), np.nan)
        market_fwd_ret[: t_len - horizon] = np.nanmean(log_rets, axis=1)

    for t in range(t_len - horizon):
        entry = aligned.open_2d[t + 1]
        exit_ = aligned.close_2d[t + horizon]
        valid_px = (entry > 0.0) & (exit_ > 0.0) & np.isfinite(entry) & np.isfinite(exit_)
        row_ok = eligible[t] & valid_px
        if not np.any(row_ok):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            gross_long = np.log(exit_ / entry)
            gross_short = np.log(entry / exit_)
        funding = aligned.funding_2d[t]
        # Capture raw gross log return before any residualization (diagnostic + gross calibrator).
        gross_long_2d[t, row_ok] = gross_long[row_ok].astype(np.float32)
        mkt_ret_t = float(market_fwd_ret[t]) if np.isfinite(market_fwd_ret[t]) else 0.0
        beta_t: NDArray[np.float64] = beta_2d[t]  # shape [N]
        # Residualize: remove estimated market beta component
        residual_adj = beta_t * mkt_ret_t
        long_net[t, row_ok] = (
            gross_long[row_ok] - residual_adj[row_ok] - cost - funding[row_ok]
        ).astype(np.float32)
        short_net[t, row_ok] = (
            gross_short[row_ok] + residual_adj[row_ok] - cost + funding[row_ok]
        ).astype(np.float32)

    # --- RAW-SIGNAL-DIAG: quantify variance shrinkage from beta-residualization ---
    # Compute before CS-demean so resid_long_2d = long_net (post-resid, pre-demean).
    # Only emitted when enough timesteps exist to be informative (n_timesteps >= 5).
    _diag = gross_return_diagnostics(
        gross_long_2d,
        long_net,
        eligible,
        min_symbols=5,
    )
    if _diag["n_timesteps"] >= 5:
        _logger.info(
            "[RAW-SIGNAL-DIAG] raw_cs_std=%.4f resid_cs_std=%.4f var_retention=%.3f"
            " n_ts=%d raw_nz=%.3f resid_nz=%.3f",
            _diag["raw_cs_std_mean"],
            _diag["resid_cs_std_mean"],
            _diag["variance_retention_ratio"],
            int(_diag["n_timesteps"]),
            _diag["raw_nonzero_ratio"],
            _diag["resid_nonzero_ratio"],
        )

    # Snapshot calibrator target: configurable via cfg.calibrator_target.
    # "beta_residualized" (default): pre-CS-demean beta-residualized return (original behavior).
    # "gross": raw log return minus funding only — no beta removal (A/B test for magnitude loss).
    # Shape: [T, N], NaN where not computed. Time complexity: O(T*N).
    if cfg.calibrator_target == "gross":
        # Gross: raw log return minus funding only (no beta removal, no fee).
        exec_net_ret: NDArray[np.float32] = (gross_long_2d - aligned.funding_2d).astype(
            np.float32
        )
        exec_net_ret = np.where(
            eligible & np.isfinite(gross_long_2d),
            exec_net_ret,
            np.float32(np.nan),
        ).astype(np.float32)
    else:
        # Default: beta-residualized, pre-CS-demean snapshot (original B2 behavior).
        exec_net_ret = long_net.copy()

    # CS-demean: per-timestep subtract cross-sectional mean from long_net and short_net labels.
    # OLS beta residualization does not guarantee E_i[long_net[t,i]] = 0 when beta != 1
    # or when eligible is a proper subset of all symbols (e.g., differential funding rates).
    # Vectorized: [T-h, N] masked nanmean → subtract per-row to zero-center CS distribution.
    # short_net adds the same mean (≈ -long_net before demean -> 0 after demean),
    # preserving the long/short anti-symmetry contract.
    _n_t: int = t_len - horizon
    _cs_mask: NDArray[np.bool_] = np.isfinite(long_net[:_n_t]) & eligible[:_n_t]  # [T-h, N]
    _cs_count: NDArray[np.intp] = _cs_mask.sum(axis=1)  # [T-h]
    _masked_long: NDArray[np.float32] = np.where(_cs_mask, long_net[:_n_t], np.float32(np.nan))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _row_mean: NDArray[np.float32] = np.nanmean(_masked_long, axis=1, keepdims=True).astype(
            np.float32
        )  # [T-h, 1]
    # Skip rows with < 2 valid symbols (effective mean = 0 → no-op).
    _effective_mean: NDArray[np.float32] = np.where(
        (_cs_count >= 2)[:, np.newaxis], _row_mean, np.float32(0.0)
    )
    long_net[:_n_t] = np.where(_cs_mask, long_net[:_n_t] - _effective_mean, long_net[:_n_t])
    short_net[:_n_t] = np.where(_cs_mask, short_net[:_n_t] + _effective_mean, short_net[:_n_t])

    signed = long_net.copy()
    finite_long = np.isfinite(long_net)
    rel = _build_relevance(
        signed_ret=signed,
        eligible=eligible & finite_long,
        min_group_size=cfg.min_group_size,
    )
    rel_long = _build_relevance(
        signed_ret=long_net,
        eligible=eligible & finite_long,
        min_group_size=cfg.min_group_size,
    )
    rel_short = _build_relevance(
        signed_ret=short_net,
        eligible=eligible & finite_long,
        min_group_size=cfg.min_group_size,
    )

    liq_weight = np.clip(np.log1p(np.maximum(aligned.volume_2d, 0.0)), 0.25, 2.0)
    valid_mask = eligible & finite_long
    original_weight = np.where(valid_mask, liq_weight, 0.0).astype(np.float32)
    y_ev_abs = np.where(valid_mask, np.abs(signed), 0.0).astype(np.float32)
    sample_weight = (original_weight * (1.0 + 2.0 * y_ev_abs)).astype(np.float32)

    # Phase D: Quality factor — coverage_60d 기반 per-symbol 가중치 [N] → broadcast [T, N]
    # Time complexity: O(N). Space complexity: O(N).
    _sym_meta = aligned.symbol_meta or {}
    _cov_key = "coverage_60d" if "coverage_60d" in _sym_meta else "last_60d_coverage"
    if _cov_key in _sym_meta:
        quality_arr = _sym_meta[_cov_key]  # shape [N]
        quality_min_f = float(getattr(cfg, "sample_weight_quality_clip_min", 0.50))
        quality_clipped = np.clip(quality_arr, quality_min_f, 1.0).astype(np.float32)  # [N]
        # sample_weight shape: [T, N] → broadcast quality [N] across rows
        sample_weight = sample_weight * quality_clipped[np.newaxis, :]  # [T, N]

    # Phase D: Cluster balance factor — cluster_id 기반 1/sqrt(cluster_size) 역수 가중치
    # Time complexity: O(N). Space complexity: O(N).
    if (
        getattr(cfg, "sample_weight_cluster_balance_enabled", True)
        and "cluster_id" in _sym_meta
    ):
        cluster_ids_arr = _sym_meta["cluster_id"].astype(np.int32)  # [N]
        unique_cids, cid_counts = np.unique(
            cluster_ids_arr[cluster_ids_arr >= 0], return_counts=True
        )
        _size_map: dict[int, int] = dict(
            zip(unique_cids.tolist(), cid_counts.tolist(), strict=False)
        )
        cluster_w = np.array(
            [
                1.0 / math.sqrt(_size_map.get(int(c), 1)) if c >= 0 else 1.0
                for c in cluster_ids_arr
            ],
            dtype=np.float32,
        )  # [N]
        sample_weight = sample_weight * cluster_w[np.newaxis, :]  # [T, N]

    if cfg.sample_weight_time_decay_halflife_bars is not None:
        # P2: Exponential time-decay weighting — 최신 데이터에 더 높은 가중치 부여.
        # shape: sample_weight [T, N], time_decay [T] (최신 t=T-1 → 1.0, 과거로 갈수록 감소)
        hl = float(cfg.sample_weight_time_decay_halflife_bars)
        lam = math.log(2.0) / max(hl, 1.0)
        t_len_sw = sample_weight.shape[0]
        time_decay = np.exp(
            -lam * np.arange(t_len_sw - 1, -1, -1, dtype=np.float64)
        ).astype(np.float32)  # [T]
        sample_weight = (sample_weight.T * time_decay).T  # broadcast [T, N]
    # Dynamic cost: execution_cost_bps_2d (per-symbol) preferred; fallback to global round-trip.
    # cost_clearance_target gates on actual label return vs dynamic_cost only (no hurdle here).
    # hurdle is applied at EV inference time in ml_builder.py: ev_bps - (dynamic_cost + hurdle) > 0.
    cost_snapshot = resolve_cost_snapshot(
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
        shape=(t_len, n_len),
    )
    dynamic_cost_2d = np.asarray(cost_snapshot.execution_cost_bps_2d, dtype=np.float32)
    if cost_snapshot.execution_cost_bps_source == "fallback_global":
        _logger.debug(
            "[LABEL-GATE] execution_cost_bps missing — fallback to global round_trip_cost=%.1f",
            cost_snapshot.round_trip_cost_bps_fallback,
        )
    else:
        _logger.debug(
            "[LABEL-GATE] Using per-symbol execution_cost_bps (mean=%.1fbps)",
            float(np.nanmean(dynamic_cost_2d)),
        )

    # Signed magnitude targets: the calibrator receives the full distribution
    # including negative losses. This allows it to learn that a long trade with
    # negative exec_net_ret is a BAD long, improving directional discrimination.
    # Censored max(x,0) targets remove direction signal and consistently collapse
    # fold OOS IC to negative values, causing quality gate failure.
    magnitude_target_long = np.where(
        valid_mask,
        exec_net_ret,
        0.0,
    ).astype(np.float32)
    magnitude_target_short = np.where(
        valid_mask,
        -exec_net_ret,
        0.0,
    ).astype(np.float32)

    # cost_clearance 3필드 → LabelDiagnostics로 격리 (LabelPanel 훈련 계약에서 제외)
    _label_diag = LabelDiagnostics(
        cost_clearance_target=np.where(
            valid_mask,
            (np.abs(exec_net_ret) * 1e4) - dynamic_cost_2d,
            np.float32(0.0),
        ).astype(np.float32),
        cost_clearance_target_long=np.where(
            valid_mask,
            (magnitude_target_long * 1e4) - dynamic_cost_2d,
            np.float32(0.0),
        ).astype(np.float32),
        cost_clearance_target_short=np.where(
            valid_mask,
            (magnitude_target_short * 1e4) - dynamic_cost_2d,
            np.float32(0.0),
        ).astype(np.float32),
    )

    metadata = {
        "rank_target_key": "signed_net_ret",
        "rank_target_long_key": "long_net_ret",
        "rank_target_short_key": "short_net_ret",
        "magnitude_target_key": "exec_net_ret",
        "magnitude_target_long_key": "exec_net_ret",
        "magnitude_target_short_key": "-exec_net_ret",
        "calibrator_target_mode": cfg.calibrator_target,
        "round_trip_cost_bps": cost_snapshot.round_trip_cost_bps_fallback,
        "execution_cost_bps_source": cost_snapshot.execution_cost_bps_source,
        "label_horizon_bars": int(cfg.label_horizon_bars),
        "_label_diagnostics_summary": {
            "cost_clearance_mean": float(
                np.nanmean(_label_diag.cost_clearance_target)
            ),
        },
    }
    return LabelPanel(
        long_net_ret=long_net,
        short_net_ret=short_net,
        signed_net_ret=signed,
        exec_net_ret=exec_net_ret,
        relevance=rel,
        sample_weight=sample_weight,
        eligible_mask=eligible & finite_long,
        rank_target=signed,
        magnitude_target=exec_net_ret,
        rank_target_long=long_net,
        rank_target_short=short_net,
        magnitude_target_long=magnitude_target_long,
        magnitude_target_short=magnitude_target_short,
        relevance_long=rel_long,
        relevance_short=rel_short,
        dynamic_cost_bps_2d=dynamic_cost_2d,
        metadata=metadata,
    )
