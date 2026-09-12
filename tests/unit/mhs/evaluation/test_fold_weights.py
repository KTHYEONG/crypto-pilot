"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.fold_weights as fold_weights


def test_fold_weights_module_present() -> None:
    assert fold_weights.__name__ == "src.mhs.evaluation.fold_weights"
    assert callable(fold_weights._build_fold_target_weights)


def test_slice_base_panel_in_memory_identity() -> None:
    import pandas as pd
    import numpy as np
    from src.mhs.panel import slice_base_panel

    dates = pd.date_range("2021-01-01", periods=3000, freq="1h", tz="UTC")
    close_df = pd.DataFrame({
        "SYM1": np.linspace(100.0, 200.0, len(dates)),
        "SYM2": np.linspace(50.0, 60.0, len(dates)),
        "SYM_SHORT": [10.0] * 500 + [np.nan] * (len(dates) - 500),
    }, index=dates)
    open_df = close_df * 0.99
    base_panel = {"close": close_df, "open": open_df}

    start = dates[100]
    end = dates[2500]
    sliced = slice_base_panel(base_panel, start, end, min_bars=2000)

    assert "SYM1" in sliced["close"].columns
    assert "SYM2" in sliced["close"].columns
    assert "SYM_SHORT" not in sliced["close"].columns
    assert sliced["close"].index[0] == start
    assert sliced["close"].index[-1] == end
    np.testing.assert_allclose(
        sliced["close"]["SYM1"].to_numpy(),
        close_df.loc[start:end, "SYM1"].to_numpy(),
    )


def test_slice_base_panel_validation_errors() -> None:
    import pytest
    import pandas as pd
    from src.mhs.panel import slice_base_panel

    start = pd.Timestamp("2021-01-01", tz="UTC")
    end = pd.Timestamp("2021-02-01", tz="UTC")

    with pytest.raises(ValueError, match="base_panel must be non-empty"):
        slice_base_panel({}, start, end)

    close_df = pd.DataFrame({"SYM1": [1.0] * 10}, index=pd.date_range("2021-01-01", periods=10, freq="1h", tz="UTC"))
    with pytest.raises(ValueError, match="no symbol survived the panel filters"):
        slice_base_panel({"close": close_df}, start, end, min_bars=100)

