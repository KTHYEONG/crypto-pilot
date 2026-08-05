"""Contract scenarios XABB-01..XABB-04 (and the frozen python_assertions) for
the XS alpha x baseline blend primitives (``select_baseline_blend_weight`` /
``build_blended_ledger``).

Scenario coverage:
* XABB-01-DISCOVERY-ONLY-SELECTION
* XABB-02-ZERO-VARIANCE-SAFE-DIVISION
* XABB-03-FULL-HISTORY-APPLICATION
* XABB-04-FAIL-CLOSED-MISALIGNED-INDEX
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.research.technical_experts.xs_alpha_baseline_blend import (
    build_blended_ledger,
    select_baseline_blend_weight,
)

_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _antiseries(n: int = 25, start: str = "2022-01-01") -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """Perfectly anti-correlated pair whose 0.5 blend is a constant positive return."""
    idx = pd.date_range(start, periods=2 * n, freq="4h", tz="UTC")
    a = pd.Series([0.02, -0.01] * n, index=idx)
    b = pd.Series([-0.01, 0.02] * n, index=idx)
    return a, b, idx


def test_xabb_python_assertion_selects_zero_variance_blend() -> None:
    # Contract python_assertion: 50-bar anti-correlated pair, grid picks 0.5
    # (the exact blend where the pair collapses to a constant positive return).
    a, b, idx = _antiseries()
    assert select_baseline_blend_weight(a, b, idx[0], idx[-1], _GRID) == 0.5


def test_xabb_python_assertion_build_blended_ledger() -> None:
    # Contract python_assertion: 0.5 blend of the anti-correlated pair compounds
    # 1.005 over all 50 bars, and the weights frame columns are ordered.
    a, b, idx = _antiseries()
    aw = pd.DataFrame({"S": [1.0] * 50}, index=idx)
    bw = pd.Series([1.0] * 50, index=idx)
    equity, weights = build_blended_ledger(a, aw, b, bw, 0.5)
    assert abs(float(equity.iloc[-1]) - 1.005 ** 50) < 1e-9
    assert list(weights.columns) == ["xs_alpha", "baseline"]


def test_xabb_01_discovery_only_selection() -> None:
    # Appending qualification-window data with a different optimal weight to
    # either leg must not change the weight selected on the discovery window.
    disc_start = pd.Timestamp("2022-01-01", tz="UTC")
    disc_end = pd.Timestamp("2022-02-01", tz="UTC")
    idx = pd.date_range(disc_start, disc_end, freq="4h", tz="UTC")
    c = pd.Series(np.resize([0.02, -0.01], len(idx)), index=idx)

    # Identical discovery legs -> every grid blend has the same Sharpe -> the
    # deterministic lowest-weight tie-break selects 0.0.
    a_disc = c.copy()
    b_disc = c.copy()

    # If the appended qualification data were read, the blend would collapse to
    # a constant positive return at w=1.0 (0.0/1.0 anti-correlated constants),
    # so the unrestricted optimum would be 1.0 -- never 0.0.
    qual_idx = pd.date_range(
        disc_end + pd.Timedelta(hours=4), periods=40, freq="4h", tz="UTC",
    )
    a_qual = pd.Series(0.01, index=qual_idx)
    b_qual = pd.Series(-0.01, index=qual_idx)

    a_full = pd.concat([a_disc, a_qual])
    b_full = pd.concat([b_disc, b_qual])

    assert select_baseline_blend_weight(
        a_full, b_full, disc_start, disc_end, _GRID,
    ) == 0.0
    assert select_baseline_blend_weight(
        a_disc, b_disc, disc_start, disc_end, _GRID,
    ) == 0.0


def test_xabb_02_zero_variance_safe_division() -> None:
    # A grid point whose blended series has exactly zero std and a positive
    # mean is selected as +inf Sharpe without raising or emitting a warning.
    a, b, idx = _antiseries()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        weight = select_baseline_blend_weight(a, b, idx[0], idx[-1], _GRID)
    assert weight == 0.5


def test_xabb_03_full_history_application() -> None:
    # One fixed scalar weight is applied uniformly across the entire input
    # history, including bars outside the discovery window used to select it.
    a, b, idx = _antiseries()
    aw = pd.DataFrame({"S": [1.0] * 50, "T": [0.5] * 50}, index=idx)
    bw = pd.Series([1.0] * 50, index=idx)
    equity, weights = build_blended_ledger(a, aw, b, bw, 0.5)

    # 0.5*a + 0.5*b == 0.005 on every bar of the full 50-bar history.
    assert abs(float(equity.iloc[0]) - 1.005) < 1e-9
    assert abs(float(equity.iloc[-1]) - 1.005 ** 50) < 1e-9
    assert list(weights.columns) == ["xs_alpha", "baseline"]
    # xs_alpha = 0.5 * sum(abs([1.0, 0.5])) = 0.75; baseline = 0.5 * abs(1.0).
    assert np.allclose(weights["xs_alpha"].to_numpy(), 0.75)
    assert np.allclose(weights["baseline"].to_numpy(), 0.5)


def test_xabb_04_fail_closed_misaligned_index() -> None:
    # build_blended_ledger raises ValueError (never a silent reindex/fillna)
    # when the inputs do not share an identical DatetimeIndex.
    a, b, idx = _antiseries()
    aw = pd.DataFrame({"S": [1.0] * 50}, index=idx)
    bw = pd.Series([1.0] * 50, index=idx)
    shifted = b.copy()
    shifted.index = idx + pd.Timedelta(hours=4)
    with pytest.raises(ValueError, match="identical DatetimeIndex"):
        build_blended_ledger(a, aw, shifted, bw, 0.5)
    with pytest.raises(ValueError, match="identical DatetimeIndex"):
        build_blended_ledger(a, aw.iloc[:-1], b, bw, 0.5)


def test_xabb_fail_closed_selection_validation() -> None:
    a, b, idx = _antiseries()
    with pytest.raises(ValueError, match="must not be empty"):
        select_baseline_blend_weight(a, b, idx[0], idx[-1], ())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_baseline_blend_weight(a, b, idx[0], idx[-1], (0.0, 1.5))
    bad = a.copy()
    bad.iloc[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        select_baseline_blend_weight(bad, b, idx[0], idx[-1], _GRID)
    with pytest.raises(ValueError, match="fewer than 2"):
        select_baseline_blend_weight(a.iloc[:1], b.iloc[:1], idx[0], idx[-1], _GRID)


def test_xabb_fail_closed_build_validation() -> None:
    a, b, idx = _antiseries()
    aw = pd.DataFrame({"S": [1.0] * 50}, index=idx)
    bw = pd.Series([1.0] * 50, index=idx)
    with pytest.raises(ValueError, match="xs_alpha_weight"):
        build_blended_ledger(a, aw, b, bw, 1.5)
    with pytest.raises(ValueError, match="xs_alpha_weight"):
        build_blended_ledger(a, aw, b, bw, -0.1)
    bad_net = a.copy()
    bad_net.iloc[3] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        build_blended_ledger(bad_net, aw, b, bw, 0.5)
    bad_aw = aw.copy()
    bad_aw.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="realized-weight inputs must be finite"):
        build_blended_ledger(a, bad_aw, b, bw, 0.5)
    bad_bw = bw.copy()
    bad_bw.iloc[4] = np.inf
    with pytest.raises(ValueError, match="realized-weight inputs must be finite"):
        build_blended_ledger(a, aw, b, bad_bw, 0.5)
