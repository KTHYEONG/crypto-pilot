from __future__ import annotations

import numpy as np

from src.domain.futures.optimization.ml_context import _attach_execution_cost_bps_2d
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG


def _base_inputs() -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], list[str]]:
    t, n = 12, 2
    symbols = ["BTCUSDT", "ETHUSDT"]
    close = np.full((t, n), 100.0, dtype=np.float64)
    high = close * 1.002
    low = close * 0.998
    volume = np.full((t, n), 2_000_000.0, dtype=np.float64)
    funding = np.full((t, n), 0.0001, dtype=np.float64)
    aligned = {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "funding_rate_sum": funding,
    }
    prebuilt = {
        "BTCUSDT": {"execution_cost_bps": np.full(t, 12.0, dtype=np.float64)},
        "ETHUSDT": {"execution_cost_bps": np.full(t, 14.0, dtype=np.float64)},
    }
    return aligned, prebuilt, symbols


def test_attach_execution_cost_static_path_sets_fraction() -> None:
    aligned, prebuilt, symbols = _base_inputs()
    old = OPT_FUTURES_CONFIG.get("COST_FORECAST_DYNAMIC", False)
    OPT_FUTURES_CONFIG["COST_FORECAST_DYNAMIC"] = False
    try:
        _attach_execution_cost_bps_2d(
            aligned=aligned,
            prebuilt_arrays=prebuilt,
            symbols=symbols,
            slice_start=2,
            slice_end=10,
        )
    finally:
        OPT_FUTURES_CONFIG["COST_FORECAST_DYNAMIC"] = old

    bps = np.asarray(aligned["execution_cost_bps_2d"], dtype=np.float64)
    frac = np.asarray(aligned["execution_cost_fraction_2d"], dtype=np.float64)
    assert bps.shape == (8, 2)
    np.testing.assert_allclose(bps[:, 0], 12.0, rtol=1e-9)
    np.testing.assert_allclose(bps[:, 1], 14.0, rtol=1e-9)
    np.testing.assert_allclose(frac, bps / 10000.0, rtol=1e-9)
    assert float(aligned["_cost_forecast_dynamic"]) == 0.0


def test_attach_execution_cost_dynamic_path_overrides_static_floor() -> None:
    aligned, prebuilt, symbols = _base_inputs()
    backup = {
        "COST_FORECAST_DYNAMIC": OPT_FUTURES_CONFIG.get("COST_FORECAST_DYNAMIC", False),
        "FUTURES_COST_VOL_BUFFER_COEF": OPT_FUTURES_CONFIG.get("FUTURES_COST_VOL_BUFFER_COEF", 0.0),
        "FUTURES_COST_LATENCY_BUFFER_BPS": OPT_FUTURES_CONFIG.get("FUTURES_COST_LATENCY_BUFFER_BPS", 0.5),
        "FUTURES_COST_ORDER_NOTIONAL_USDT": OPT_FUTURES_CONFIG.get("FUTURES_COST_ORDER_NOTIONAL_USDT", 0.0),
        "FUTURES_COST_FUNDING_EVENT_BUFFER_BPS": OPT_FUTURES_CONFIG.get("FUTURES_COST_FUNDING_EVENT_BUFFER_BPS", 0.0),
    }
    OPT_FUTURES_CONFIG["COST_FORECAST_DYNAMIC"] = True
    OPT_FUTURES_CONFIG["FUTURES_COST_VOL_BUFFER_COEF"] = 0.2
    OPT_FUTURES_CONFIG["FUTURES_COST_LATENCY_BUFFER_BPS"] = 0.5
    OPT_FUTURES_CONFIG["FUTURES_COST_ORDER_NOTIONAL_USDT"] = 50_000.0
    OPT_FUTURES_CONFIG["FUTURES_COST_FUNDING_EVENT_BUFFER_BPS"] = 0.2
    try:
        _attach_execution_cost_bps_2d(
            aligned=aligned,
            prebuilt_arrays=prebuilt,
            symbols=symbols,
            slice_start=0,
            slice_end=12,
        )
    finally:
        for k, v in backup.items():
            OPT_FUTURES_CONFIG[k] = v

    bps = np.asarray(aligned["execution_cost_bps_2d"], dtype=np.float64)
    assert bps.shape == (12, 2)
    assert np.all(bps[:, 0] >= 12.0)
    assert np.all(bps[:, 1] >= 14.0)
    assert float(aligned["_cost_forecast_dynamic"]) == 1.0
    assert "execution_cost_fraction_2d" in aligned
