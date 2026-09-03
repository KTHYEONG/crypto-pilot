"""Contract scenarios XSC-01..XSC-05, XSC-07, XSA-02, and XSV3-01 for the cross-sectional module.

XSC-01-NO-TRADE-BAND-STATEFUL, XSC-02-WEIGHTS-DOLLAR-NEUTRAL,
XSC-03-SPEC-FROZEN-BOUNDS, XSC-04-LEDGER-EXECUTION-LAG,
XSC-05-ADMISSION-SCALE-INVARIANT, XSC-07-COMPOSITE-BEATS-SINGLE-FAMILY,
XSA-02-COMPOSITE-PRESERVATION, XSV3-01-FAMILY-SUM,
SCENARIO_XSV5_01_DUAL_FAMILY_EXCLUDES_FUNDING,
SCENARIO_XSV6_01_CAUSAL_VOL_WEIGHTS_EXCLUDE_CURRENT_BAR,
SCENARIO_XSV6_02_VOL_WEIGHTED_MATCHES_MANUAL_RECOMPUTE,
SCENARIO_XS_POSITIONING_WEIGHTS_01,
SCENARIO_XSV6SIZE_01_DISCOVERY_ONLY_SIZING_NO_LEAKAGE,
SCENARIO_XSV6SIZE_02_INFEASIBLE_SIZING_FAILS_CLOSED, and
SCENARIO_COSTFIX_01..07 (honest turnover-cost repricing of the vol-target
overlay stack).
"""

from __future__ import annotations


import numpy as np
import pandas as pd



def _score_frame(rows: int = 40, cols: int = 5, seed: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(rows, cols)),
        index=index,
        columns=[chr(ord("A") + i) for i in range(cols)],
    )
















def _alpha_inputs(
    rows: int = 300, cols: int = 5, seed: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic strictly-positive closes, in-[0,1] taker ratios, finite funding."""
    index = pd.date_range("2024-01-01", periods=rows, freq="4h", tz="UTC")
    columns = [chr(ord("A") + i) for i in range(cols)]
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=(rows, cols)), axis=0))
    closes = pd.DataFrame(closes, index=index, columns=columns)
    taker = pd.DataFrame(
        0.5 + 0.1 * np.sin(np.arange(rows)[:, None] / 9.0 + np.arange(cols)),
        index=index, columns=columns,
    )
    funding = pd.DataFrame(
        0.0001 * np.cos(np.arange(rows)[:, None] / 5.0 + np.arange(cols)),
        index=index, columns=columns,
    )
    return closes, taker, funding


def _ref_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Reference finite-only cross-sectional z-score (see XSA-02 docstring)."""
    values = frame.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    mean = np.where(finite, values, 0.0).sum(axis=1, keepdims=True) / np.maximum(count, 1)[:, None]
    demeaned = np.where(finite, values - mean, 0.0)
    var = (demeaned ** 2).sum(axis=1, keepdims=True) / np.maximum(count - 1, 1)[:, None]
    std = np.sqrt(np.maximum(var, 0.0))
    out = np.zeros_like(values)
    np.divide(
        demeaned, std, out=out,
        where=(count[:, None] >= 2) & (std > 0.0),
    )
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)
























