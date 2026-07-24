from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.admission import (
    combine_admitted_forecasts,
    evaluate_signal_admission,
)
from src.domain.futures.compound.bar_engine import build_multi_timeframe_bars
from src.domain.futures.compound.baseline_alloc import solve_baseline_weights
from src.domain.futures.compound.calibration import (
    build_folds_4h,
    build_multi_horizon_targets,
    calibrate_signals,
)
from src.domain.futures.compound.config import (
    AdmissionConfig,
    BaselineAllocConfig,
    CalibrationConfig,
    DenseSimConfig,
    LadderConfig,
    RiskModelConfig,
)
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    LadderStageResult,
    MarketFeatureCube,
    MultiTimeframeBars,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.dense_simulator import simulate_dense_portfolio
from src.domain.futures.compound.risk_model import estimate_covariance_path
from src.domain.futures.compound.signal_bank import build_raw_signal_panel

_logger = logging.getLogger(__name__)


def _l1_catalog(stage_id: str) -> tuple[SignalDescriptor, ...] | None:
    if stage_id == "L1-0":
        return (SignalDescriptor("trend_ema:medium", "trend_ema", "medium", 72, "4h"),)
    if stage_id == "L1-1":
        return (
            SignalDescriptor("trend_ema:fast", "trend_ema", "fast", 24, "4h"),
            SignalDescriptor("trend_ema:medium", "trend_ema", "medium", 72, "4h"),
            SignalDescriptor("trend_ema:slow", "trend_ema", "slow", 216, "4h"),
            SignalDescriptor("momentum_ts:fast", "momentum_ts", "fast", 24, "4h"),
            SignalDescriptor("momentum_ts:medium", "momentum_ts", "medium", 72, "4h"),
            SignalDescriptor("momentum_ts:slow", "momentum_ts", "slow", 216, "4h"),
        )
    if stage_id == "L1-2":
        return (
            SignalDescriptor("trend_ema:fast", "trend_ema", "fast", 24, "4h"),
            SignalDescriptor("trend_ema:medium", "trend_ema", "medium", 72, "4h"),
            SignalDescriptor("trend_ema:slow", "trend_ema", "slow", 216, "4h"),
            SignalDescriptor("momentum_ts:fast", "momentum_ts", "fast", 24, "4h"),
            SignalDescriptor("momentum_ts:medium", "momentum_ts", "medium", 72, "4h"),
            SignalDescriptor("momentum_ts:slow", "momentum_ts", "slow", 216, "4h"),
            SignalDescriptor("breakout_donchian:fast", "breakout_donchian", "fast", 24, "4h"),
            SignalDescriptor("breakout_donchian:medium", "breakout_donchian", "medium", 72, "4h"),
        )
    return None  # L1-3 uses default catalog


def _build_l1_forecast(
    market: MarketFeatureCube,
    stage_id: str,
    bars: MultiTimeframeBars,
    panel: RawSignalPanel,
    config: LadderConfig,
) -> CalibratedForecastPanel:
    if stage_id != "L1-3":
        bars_4h = bars.cubes["4h"]
        n_bars_4h = bars_4h.timestamps_ns.size
        n_syms = len(bars_4h.symbols)
        ts_4h = bars_4h.timestamps_ns

        catalog = _l1_catalog(stage_id)
        sig_indices = [
            i for i, d in enumerate(panel.descriptors)
            if catalog is not None and any(
                d.signal_id == c.signal_id for c in catalog
            )
        ]

        if sig_indices:
            valid_scores = []
            for idx in sig_indices:
                z_slice = panel.z_3d[:, :, idx]
                valid_scores.append(np.where(np.isfinite(z_slice), z_slice, 0.0))
            stacked = np.stack(valid_scores, axis=-1)
            z_mean = np.mean(stacked, axis=-1).astype(np.float32)
            sigma = np.where(np.isfinite(panel.sigma_2d), panel.sigma_2d, 1e-6)
            mu_2d = z_mean * sigma
            n_sig = max(len(sig_indices), 1)
            se_2d = sigma / np.sqrt(n_sig)
            se_2d = np.where(np.isfinite(se_2d), se_2d, np.nan).astype(np.float32)
        else:
            mu_2d = np.zeros((n_bars_4h, n_syms), dtype=np.float32)
            se_2d = np.full((n_bars_4h, n_syms), np.nan, dtype=np.float32)

        return CalibratedForecastPanel(
            decision_timestamps_ns=ts_4h,
            symbols=bars_4h.symbols,
            mu_2d=mu_2d,
            se_2d=se_2d,
            family_mu_3d=np.zeros((n_bars_4h, n_syms, 1), dtype=np.float32),
            family_ids=(),
            admitted_signal_ids=(),
            fold_manifest_hash="",
        )

    try:
        calib_config = CalibrationConfig()
        admit_config = AdmissionConfig(n_bootstrap=min(config.n_bootstrap, 100))
        horizons = tuple(sorted({d.target_horizon_hours for d in panel.descriptors}))
        targets = build_multi_horizon_targets(bars, panel.sigma_2d, horizons)
        max_horizon_bars = max(horizons) // 4 if horizons else 0
        folds = build_folds_4h(panel.z_3d.shape[0], calib_config, max_target_horizon_bars=max_horizon_bars)
        calibs = calibrate_signals(panel, targets, folds, calib_config)
        evidence = evaluate_signal_admission(
            panel, targets, calibs, folds,
            market.execution_cost_bps_2d, admit_config,
        )
        forecast = combine_admitted_forecasts(panel, calibs, evidence, folds)
        if len(forecast.admitted_signal_ids) > 0:
            return forecast
        _logger.info("[L1-3] no signals admitted, using zero-mu fallback")
    except Exception as exc:
        _logger.warning("[L1-3] calibration/admission failed: %s", exc)

    bars_4h = bars.cubes["4h"]
    n_bars_4h_fb = bars_4h.timestamps_ns.size
    n_syms_fb = len(bars_4h.symbols)
    return CalibratedForecastPanel(
        decision_timestamps_ns=bars_4h.timestamps_ns,
        symbols=bars_4h.symbols,
        mu_2d=np.zeros((n_bars_4h_fb, n_syms_fb), dtype=np.float32),
        se_2d=np.full((n_bars_4h_fb, n_syms_fb), np.nan, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars_4h_fb, n_syms_fb, 1), dtype=np.float32),
        family_ids=(),
        admitted_signal_ids=(),
        fold_manifest_hash="",
    )


def _subsample_to_4h(hourly_2d: NDArray[np.bool_]) -> NDArray[np.bool_]:
    n_1h = hourly_2d.shape[0]
    n_4h = n_1h // 4
    usable = n_4h * 4
    result: NDArray[np.bool_] = np.any(hourly_2d[:usable].reshape(n_4h, 4, hourly_2d.shape[1]), axis=1)
    return result


def _compute_returns_4h(
    bars_4h: TimeframeBarCube,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    close = bars_4h.close_2d.astype(np.float64)
    n_bars = close.shape[0]
    n_syms = close.shape[1]
    ret = np.zeros((n_bars, n_syms), dtype=np.float32)
    valid = np.zeros((n_bars, n_syms), dtype=np.bool_)
    for t in range(1, n_bars):
        prev = close[t - 1]
        curr = close[t]
        mask = (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        log_ret = np.full(n_syms, 0.0, dtype=np.float64)
        log_ret[mask] = np.log(curr[mask] / prev[mask])
        ret[t] = log_ret.astype(np.float32)
        valid[t, mask] = True
    return ret, valid


def _compute_ladder_metrics(
    ledger_returns: NDArray[np.float64],
    bars_per_year: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    n = ledger_returns.size
    if n < 2:
        return 0.0, 0.0, 0.0, float("inf")

    log_equity = np.log(np.maximum(1.0 + ledger_returns, 1e-12))
    ann_factor = bars_per_year / n
    total_log_growth = float(np.sum(log_equity))
    oos_log_growth = total_log_growth * ann_factor

    mean_ret = float(np.mean(ledger_returns))
    std_ret = float(np.std(ledger_returns, ddof=1))
    sharpe = mean_ret / std_ret * np.sqrt(bars_per_year) if std_ret > 1e-15 else 0.0

    equity_curve = np.cumprod(np.maximum(1.0 + ledger_returns, 1e-12))
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = float(np.min(dd)) if dd.size > 0 else 0.0

    boot_growths = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_log_growth = float(np.sum(log_equity[idx])) * ann_factor
        boot_growths[i] = boot_log_growth
    boot_growths.sort()
    lcb90 = float(boot_growths[int(n_bootstrap * 0.05)])

    return oos_log_growth, lcb90, sharpe, max_dd


def run_experiment_ladder(
    market: MarketFeatureCube,
    eligible_2d: NDArray[np.bool_],
    config: LadderConfig,
    rng_seed: int = 42,
) -> tuple[LadderStageResult, ...]:
    close = market.fields_2d.get("close")
    if close is None:
        raise ValueError("market cube missing close field")

    bars = build_multi_timeframe_bars(market)
    bars_4h = bars.cubes["4h"]
    eligible_4h = _subsample_to_4h(eligible_2d)

    returns_4h, valid_4h = _compute_returns_4h(bars_4h)
    returns_4h = np.where(eligible_4h, returns_4h, 0.0).astype(np.float32)
    valid_4h = valid_4h & eligible_4h

    l1_stages = ["L1-0", "L1-1", "L1-2", "L1-3"]
    l2_modes: list[Literal["inverse_vol", "risk_scaled"]] = ["inverse_vol", "risk_scaled"]

    risk_config = RiskModelConfig()
    alloc_config = BaselineAllocConfig()
    sim_config = DenseSimConfig()
    bars_per_year = sim_config.bars_per_year

    funding_field = bars.aux_1h_fields.get("funding", None)
    if funding_field is not None:
        funding_upsampled = funding_field.astype(np.float32)
    else:
        funding_upsampled = np.zeros((bars_4h.timestamps_ns.size * 4, len(market.symbols)), dtype=np.float32)

    cost_bps_arr = eligible_2d.astype(np.float32) * config.cost_bps
    rng = np.random.default_rng(rng_seed)
    results: list[LadderStageResult] = []

    for l1_id in l1_stages:
        try:
            catalog = _l1_catalog(l1_id)
            panel = build_raw_signal_panel(bars, eligible_4h, catalog=catalog)
        except Exception as exc:
            _logger.error("[EVAL] stage %s signal panel build failed: %s", l1_id, exc)
            for l2_mode in l2_modes:
                stage_id = f"{l1_id}|L2-{'0' if l2_mode == 'inverse_vol' else '1'}"
                results.append(_error_result(stage_id))
            continue

        forecast = _build_l1_forecast(market, l1_id, bars, panel, config)

        for l2_mode in l2_modes:
            stage_id = f"{l1_id}|L2-{'0' if l2_mode == 'inverse_vol' else '1'}"
            try:
                cov_path = estimate_covariance_path(returns_4h, valid_4h, risk_config)

                mu_2d = forecast.mu_2d
                w = solve_baseline_weights(mu_2d, cov_path, alloc_config, l2_mode)

                ledger = simulate_dense_portfolio(
                    bars_4h=bars_4h,
                    target_weights_2d=w,
                    funding_1h_2d=funding_upsampled,
                    cost_bps=cost_bps_arr,
                    config=sim_config,
                )

                oos_growth, lcb90, sharpe, mdd = _compute_ladder_metrics(
                    ledger.net_returns_1d, bars_per_year, config.n_bootstrap, rng,
                )

                ledger_2x = simulate_dense_portfolio(
                    bars_4h=bars_4h,
                    target_weights_2d=w,
                    funding_1h_2d=funding_upsampled,
                    cost_bps=config.cost_bps * 2.0,
                    config=sim_config,
                )
                growth_2x, _, _, _ = _compute_ladder_metrics(
                    ledger_2x.net_returns_1d, bars_per_year, config.n_bootstrap, rng,
                )

                turnover = _compute_turnover(w, bars_per_year)

                _logger.info(
                    "[EVAL] stage=%s growth=%.6f lcb90=%.6f sharpe=%.3f mdd=%.4f to=%.2f 2x=%.6f",
                    stage_id, oos_growth, lcb90, sharpe, mdd, turnover, growth_2x,
                )

                promoted = (
                    growth_2x > 0
                    and (not results or oos_growth >= results[-1].oos_log_growth or lcb90 >= results[-1].oos_growth_lcb90)
                )
                results.append(LadderStageResult(
                    stage_id=stage_id,
                    oos_log_growth=oos_growth,
                    oos_growth_lcb90=lcb90,
                    sharpe=sharpe,
                    max_drawdown=mdd,
                    turnover_per_year=turnover,
                    growth_2x_cost=growth_2x,
                    status="ok",
                    promoted=promoted,
                ))
            except Exception as exc:
                _logger.error("[EVAL] stage %s failed: %s", stage_id, exc)
                results.append(_error_result(stage_id))

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_dir = Path(f"logs/futures/redesign_ladder/{ts}")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ladder_report.json"

    report_data = [
        {
            "stage_id": r.stage_id,
            "oos_log_growth": r.oos_log_growth,
            "oos_growth_lcb90": r.oos_growth_lcb90,
            "sharpe": r.sharpe,
            "max_drawdown": r.max_drawdown,
            "turnover_per_year": r.turnover_per_year,
            "growth_2x_cost": r.growth_2x_cost,
            "status": r.status,
            "promoted": r.promoted,
        }
        for r in results
    ]
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    return tuple(results)


def _compute_turnover(
    weights_2d: NDArray[np.float64],
    bars_per_year: float,
) -> float:
    n_bars = weights_2d.shape[0]
    if n_bars < 2:
        return 0.0
    turnovers = np.zeros(n_bars - 1, dtype=np.float64)
    for t in range(1, n_bars):
        turnovers[t - 1] = float(np.sum(np.abs(weights_2d[t] - weights_2d[t - 1])))
    mean_turnover = float(np.mean(turnovers))
    return mean_turnover * bars_per_year


def _error_result(stage_id: str) -> LadderStageResult:
    return LadderStageResult(
        stage_id=stage_id,
        oos_log_growth=0.0,
        oos_growth_lcb90=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        turnover_per_year=0.0,
        growth_2x_cost=0.0,
        status="error",
        promoted=False,
    )
