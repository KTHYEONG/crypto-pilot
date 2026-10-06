"""Contract scenarios XSC-03 and SCENARIO_COSTFIX_01 for the cross-sectional module.

XSC-03-SPEC-FROZEN-BOUNDS and SCENARIO_COSTFIX_01_LEDGER_PNL_REGRESSION.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.quant.technical_experts.cross_sectional import (
    XsCompositeSpec,
    _ledger_pnl,
    run_xs_composite_ledger,
)


class TestCompositeSpec:
    def test_xsc_03_frozen_defaults_and_cost_rate(self) -> None:
        spec = XsCompositeSpec()
        assert (spec.halflife_bars, spec.no_trade_band, spec.execution_delay_bars) == (
            6, 0.05, 1,
        )
        assert abs(spec.round_trip_cost_rate() - 0.0008) < 1e-12
        assert dataclasses.is_dataclass(spec)

    def test_xsc_03_out_of_range_fields_fail_closed(self) -> None:
        with pytest.raises(ValueError, match="no_trade_band"):
            XsCompositeSpec(no_trade_band=1.0)
        with pytest.raises(ValueError, match="no_trade_band"):
            XsCompositeSpec(no_trade_band=-0.1)
        with pytest.raises(ValueError, match="halflife_bars"):
            XsCompositeSpec(halflife_bars=-1)
        with pytest.raises(ValueError, match="execution_delay_bars"):
            XsCompositeSpec(execution_delay_bars=-1)
        with pytest.raises(ValueError, match="fee_rate"):
            XsCompositeSpec(fee_rate=-0.1)
        with pytest.raises(ValueError, match="slippage_rate"):
            XsCompositeSpec(slippage_rate=-0.1)


class TestCostRepricing:
    """SCENARIO_COSTFIX_01: honest turnover-cost repricing of the overlay stack."""

    def _sizing_inputs(
        self, rows: int = 300, crash_factor: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        idx = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
        rng = np.random.default_rng(3)
        closes = pd.DataFrame({
            "A": 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.008, rows))),
            "B": 100 * np.exp(np.cumsum(rng.normal(-0.0015, 0.008, rows))),
        }, index=idx)
        opens = closes.shift(1).bfill()
        opens.loc[idx[40], "A"] = opens.loc[idx[39], "A"] * crash_factor
        funding = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
        weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
        return weights, opens, funding, idx

    # SCENARIO_COSTFIX_01_LEDGER_PNL_REGRESSION
    def test_costfix_01_ledger_pnl_extraction_is_regression_free(self) -> None:
        weights, opens, funding, _ = self._sizing_inputs()
        spec = XsCompositeSpec()
        equity, turnover_series = run_xs_composite_ledger(weights, opens, funding, spec)
        lag = 1 + spec.execution_delay_bars
        lagged = weights.shift(lag).fillna(0.0).to_numpy(dtype=np.float64)
        o = opens.to_numpy(dtype=np.float64)
        f = funding.to_numpy(dtype=np.float64)
        o2o = np.zeros_like(o)
        with np.errstate(divide="ignore", invalid="ignore"):
            o2o[1:] = o[1:] / o[:-1] - 1.0
        net_returns, turnover = _ledger_pnl(lagged, o2o, f, spec.round_trip_cost_rate())
        assert np.allclose(turnover, turnover_series.to_numpy())
        assert np.allclose(
            net_returns, equity.pct_change().fillna(0.0).to_numpy(), atol=1e-12,
        )


def test_composite_ledger_import_is_quant_isolated() -> None:
    """The composite ledger module must not pull the research gate stack."""
    import ast
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    code = (
        "import src.quant.technical_experts.cross_sectional, sys;"
        "loaded = sorted(k for k in sys.modules if k.startswith('src.quant.'));"
        "print(loaded)"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(repo_root),
    }
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"cross_sectional import failed: {completed.stderr}"
    )
    loaded = ast.literal_eval(completed.stdout.strip())
    assert "src.quant.evaluation.reliability" not in loaded
    assert "src.quant.risk.growth_sizing" not in loaded
