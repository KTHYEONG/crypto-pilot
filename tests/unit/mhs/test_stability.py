from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mhs.stability import StabilityResult, regime_split_stability

_PPY = 365.0 * 24.0


def _pnl_series(values: list[float], start: str = "2021-01-01", freq: str = "1h") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


def test_regime_split_stability_flags_sign_flip() -> None:
    # SCENARIO_REGIME_SPLIT_STABILITY_FLAGS_SIGN_FLIP: a pnl series strongly
    # positive in window 1 and strongly negative in window 2 yields
    # sign_consistent=False, a negative min_window_sharpe, and a negative
    # decay -- the mom_168h case (+0.615 then -0.690). A series positive in
    # both windows yields sign_consistent=True with a positive
    # min_window_sharpe -- the mom_336h case (+0.651 then +0.742).
    n = 365 * 24
    rng = np.random.default_rng(0)
    flip = _pnl_series(
        list(rng.normal(0.0005, 0.01, n)) + list(rng.normal(-0.0005, 0.01, n)),
    )
    split = pd.Timestamp("2022-01-01", tz="UTC")
    result = regime_split_stability(flip, (split,), _PPY)
    assert isinstance(result, StabilityResult)
    assert len(result.window_sharpes) == 2
    w1, w2 = result.window_sharpes[0][1], result.window_sharpes[1][1]
    assert np.isfinite(w1)
    assert np.isfinite(w2)
    assert w1 > 0
    assert w2 < 0
    assert result.sign_consistent is False
    assert result.min_window_sharpe < 0
    assert result.decay < 0
    assert result.decay == pytest.approx(w2 - w1)

    stable = _pnl_series(
        list(rng.normal(0.0005, 0.01, n)) + list(rng.normal(0.0004, 0.01, n)),
    )
    result2 = regime_split_stability(stable, (split,), _PPY)
    assert result2.sign_consistent is True
    assert result2.min_window_sharpe > 0
    assert result2.decay == pytest.approx(result2.window_sharpes[1][1] - result2.window_sharpes[0][1])


def test_regime_split_stability_degenerate_window_excluded_not_raised() -> None:
    # SCENARIO_REGIME_SPLIT_STABILITY_DEGENERATE_WINDOW_EXCLUDED_NOT_RAISED:
    # a window with fewer than 2 observations or zero variance produces a
    # non-finite Sharpe that still appears in window_sharpes but is excluded
    # from min_window_sharpe and sign_consistent; the call does not raise.
    # Empty pnl, empty split_points, non-monotonic split_points, and
    # periods_per_year <= 0 each raise ValueError.
    rng = np.random.default_rng(1)
    positive = _pnl_series(list(rng.normal(0.0005, 0.01, 365 * 24)))
    split = pd.Timestamp("2022-01-01", tz="UTC")  # far past the series end
    result = regime_split_stability(positive, (split,), _PPY)
    assert len(result.window_sharpes) == 2
    assert np.isfinite(result.window_sharpes[0][1])
    assert not np.isfinite(result.window_sharpes[1][1])
    assert np.isfinite(result.min_window_sharpe)
    assert result.min_window_sharpe > 0
    assert result.sign_consistent is True  # the only finite window is positive

    zero_var = _pnl_series([0.001] * 100)
    zero_var_split = pd.Timestamp(zero_var.index[50])
    result2 = regime_split_stability(zero_var, (zero_var_split,), _PPY)
    assert not np.isfinite(result2.window_sharpes[0][1])
    assert not np.isfinite(result2.window_sharpes[1][1])
    assert not np.isfinite(result2.min_window_sharpe)
    assert result2.sign_consistent is False

    with pytest.raises(ValueError, match="must not be empty"):
        regime_split_stability(pd.Series([], dtype=float), (split,), _PPY)
    with pytest.raises(ValueError, match="split_points"):
        regime_split_stability(positive, (), _PPY)
    with pytest.raises(ValueError, match="ascending"):
        regime_split_stability(
            positive, (pd.Timestamp("2022-06-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC")), _PPY,
        )
    with pytest.raises(ValueError, match="periods_per_year"):
        regime_split_stability(positive, (split,), 0.0)
