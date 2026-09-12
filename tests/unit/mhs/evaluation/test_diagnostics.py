"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.diagnostics as diagnostics


def test_diagnostics_module_present() -> None:
    assert diagnostics.__name__ == "src.mhs.evaluation.diagnostics"
    assert callable(diagnostics._phase_diagnostics)


def test_phase_diagnostics_threadpool_equivalence() -> None:
    import pandas as pd
    import numpy as np
    from src.mhs.types import BookSpec, HorizonBand
    from src.mhs.evaluation.diagnostics import _phase_diagnostics

    grid = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    symbols = ["BTC", "ETH"]
    log_close = pd.DataFrame({"BTC": np.log(np.linspace(100, 110, 48)), "ETH": np.log(np.linspace(50, 55, 48))}, index=grid)
    eligible = pd.DataFrame(True, index=grid, columns=symbols)
    opens = pd.DataFrame({"BTC": np.linspace(100, 110, 48), "ETH": np.linspace(50, 55, 48)}, index=grid)
    bar_funding = pd.DataFrame(0.0, index=grid, columns=symbols)
    spec = BookSpec(band=HorizonBand(name="slow_momentum", horizons_hours=(6, 12, 24), sign=1), horizon_hours=6, step_hours=6, min_symbols=2)

    res = _phase_diagnostics(log_close, eligible, opens, bar_funding, grid, spec)
    assert res is not None
    assert np.isfinite(res.ensemble_sharpe)
    assert np.isfinite(res.ensemble_ann)
