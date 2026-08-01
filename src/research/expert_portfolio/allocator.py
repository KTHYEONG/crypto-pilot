from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.expert_portfolio.models import ExpertPortfolioSpec, lcb_z_score

_MAX_ACF_LAG = 20


def _validate_panel(component_returns: pd.DataFrame) -> None:
    """Fail-closed validation of the completed component-return panel."""
    if not isinstance(component_returns.index, pd.DatetimeIndex):
        raise ValueError("component_returns must have a DatetimeIndex")
    if not component_returns.index.is_monotonic_increasing:
        raise ValueError("component_returns index must be monotonic increasing")
    if component_returns.index.has_duplicates:
        raise ValueError("component_returns index must not contain duplicate timestamps")


def _validate_as_of(component_returns: pd.DataFrame, as_of: pd.Timestamp) -> None:
    if not isinstance(as_of, pd.Timestamp):
        raise ValueError(f"as_of must be a pd.Timestamp, got {type(as_of).__name__}")
    if as_of.tzinfo is not None and component_returns.index.tz is None:
        raise ValueError("as_of is tz-aware while returns index is tz-naive")
    if as_of.tzinfo is None and component_returns.index.tz is not None:
        raise ValueError("as_of is tz-naive while returns index is tz-aware")


def _validate_previous_weights(previous_weights: pd.Series, spec: ExpertPortfolioSpec) -> None:
    if not isinstance(previous_weights, pd.Series):
        raise ValueError(f"previous_weights must be a pd.Series, got {type(previous_weights).__name__}")
    expected = _weight_columns(spec)
    if list(previous_weights.index) != list(expected):
        raise ValueError(
            f"previous_weights must be aligned to weight columns {expected}, "
            f"got {list(previous_weights.index)}"
        )
    values = previous_weights.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("previous_weights must contain only finite values")
    if (values < 0.0).any():
        raise ValueError("previous_weights must be non-negative")
    if float(values.sum()) > spec.gross_exposure + 1e-12:
        raise ValueError(
            f"previous_weights sum {float(values.sum()):.6f} exceeds gross exposure "
            f"{spec.gross_exposure}"
        )


def _weight_columns(spec: ExpertPortfolioSpec) -> tuple[str, ...]:
    return (*tuple(e.expert_id for e in spec.experts), "CASH")


def _causal_mean_var(rets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal per-bar mean/variance over completed returns strictly earlier.

    For decision bar ``k`` the statistics use ``rets[0:k]`` only, so no future
    or current-bar return can influence the target.
    """
    n = len(rets)
    cum = np.concatenate([[0.0], np.cumsum(rets)])
    cum2 = np.concatenate([[0.0], np.cumsum(rets * rets)])
    counts = np.arange(n)
    mean = np.where(counts > 0, cum[:n] / np.maximum(counts, 1), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.where(
            counts > 1,
            (cum2[:n] - cum[:n] * cum[:n] / np.maximum(counts, 1)) / (counts - 1),
            np.nan,
        )
    return mean, var


def _causal_block_aware_inflation(rets: np.ndarray) -> np.ndarray:
    """Causal Bartlett (Newey-West) variance-inflation factor per decision bar.

    For decision bar ``k`` the factor inflates the mean's variance by the
    completed autocorrelation structure of ``rets[0:k]``:
    ``1 + 2 * sum_l (1 - l/(L+1)) * rho_l``. This is the block-aware variance of
    a serially dependent sample: white noise inflates by ~1 while positively
    autocorrelated returns inflate materially, so the lower confidence bound
    never overstates independence. Only rows strictly before ``k`` are used, and
    a negative inflation estimate is clipped to zero (fail closed).
    """
    n = len(rets)
    cum = np.concatenate([[0.0], np.cumsum(rets)])
    cum2 = np.concatenate([[0.0], np.cumsum(rets * rets)])
    k = np.arange(n)
    mean = cum[:n] / np.maximum(k, 1)
    denom = cum2[:n] - k * mean * mean
    inflation = np.ones(n, dtype=np.float64)
    for lag in range(1, _MAX_ACF_LAG + 1):
        pairs = k - lag
        mask = (k >= 2) & (pairs > 0)
        if not mask.any():
            break
        safe_pairs = np.clip(pairs, 0, None)
        cprod = np.concatenate([[0.0], np.cumsum(rets[lag:] * rets[:-lag])])
        r_lo = cum[safe_pairs]
        r_hi = cum[k] - cum[lag]
        cov = cprod[safe_pairs] - mean[k] * (r_lo + r_hi) + safe_pairs * mean[k] * mean[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            acf = np.where(mask & (denom[k] > 0), cov / np.where(denom[k] > 0, denom[k], 1.0), 0.0)
        weight = 1.0 - lag / (_MAX_ACF_LAG + 1)
        inflation += 2.0 * weight * np.where(mask, acf, 0.0)
    return np.maximum(inflation, 0.0)


def _raw_allocation(rets: np.ndarray, confidence: float, min_history_bars: int) -> np.ndarray:
    """Causal variance-normalized positive-LCB allocation for one expert column.

    Returns an array aligned to ``rets`` where decision bar ``k`` carries
    ``max(0, LCB_k) / var_k`` using completed log returns strictly before ``k``,
    with the LCB standard error inflated by the causal block-aware variance
    factor. Invalid data, insufficient completed history, zero variance, or a
    non-positive lower confidence bound all yield zero.
    """
    n = len(rets)
    finite = np.isfinite(rets)
    all_finite_before = np.concatenate([[True], np.cumprod(finite)])[:n]
    counts = np.arange(n)
    mean, var = _causal_mean_var(rets)
    with np.errstate(divide="ignore", invalid="ignore"):
        inflation = _causal_block_aware_inflation(rets)
        se = np.sqrt(var * inflation / np.maximum(counts, 1))
        lcb = mean - lcb_z_score(confidence) * se
        u = np.maximum(0.0, lcb) / var
    mask = (
        (counts >= min_history_bars)
        & all_finite_before
        & np.isfinite(var)
        & (var > 0.0)
        & np.isfinite(u)
    )
    return np.where(mask, u, 0.0)


def _group_matrices(spec: ExpertPortfolioSpec) -> tuple[np.ndarray, np.ndarray]:
    """One-hot family and underlying-symbol membership for the expert library.

    An expert belongs to exactly one family and to each of its underlying
    symbols, so overlapping exposures share the symbol budget across families.
    """
    families = sorted({e.family for e in spec.experts})
    symbols = sorted({s for e in spec.experts for s in e.symbols})
    family_index = {f: i for i, f in enumerate(families)}
    symbol_index = {s: i for i, s in enumerate(symbols)}
    m = len(spec.experts)
    family_mat = np.zeros((m, len(families)), dtype=np.float64)
    symbol_mat = np.zeros((m, len(symbols)), dtype=np.float64)
    for j, expert in enumerate(spec.experts):
        family_mat[j, family_index[expert.family]] = 1.0
        for symbol in expert.symbols:
            symbol_mat[j, symbol_index[symbol]] = 1.0
    return family_mat, symbol_mat


def _project_weights(
    raw: np.ndarray,
    spec: ExpertPortfolioSpec,
    family_mat: np.ndarray,
    symbol_mat: np.ndarray,
) -> np.ndarray:
    """Deterministic proportional projection onto the pre-registered constraint set.

    Non-negativity already holds. A single global scale factor is the largest
    factor that keeps total risky exposure, every source-family aggregate, and
    every underlying-symbol aggregate within their pre-registered limits;
    infeasible mass stays as cash. The scale never exceeds one, so weak
    evidence is never scaled up.
    """
    n = raw.shape[0]
    total = raw.sum(axis=1, keepdims=True)
    scales = np.ones((n, 1), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_total = np.where(total > 0, total, np.inf)
        scales = np.minimum(scales, np.where(np.isfinite(safe_total), spec.gross_exposure / safe_total, np.inf))
        family_sums = raw @ family_mat
        safe_family = np.where(family_sums > 0, family_sums, np.inf)
        family_scale = np.where(
            np.isfinite(safe_family), spec.family_exposure_limit / safe_family, np.inf,
        )
        scales = np.minimum(scales, family_scale.min(axis=1, keepdims=True))
        symbol_sums = raw @ symbol_mat
        safe_symbol = np.where(symbol_sums > 0, symbol_sums, np.inf)
        symbol_scale = np.where(
            np.isfinite(safe_symbol), spec.symbol_exposure_limit / safe_symbol, np.inf,
        )
        scales = np.minimum(scales, symbol_scale.min(axis=1, keepdims=True))
    return np.asarray(raw * scales, dtype=np.float64)


def _causal_lcb_weight_series(
    component_returns: pd.DataFrame,
    spec: ExpertPortfolioSpec,
) -> pd.DataFrame:
    """Vectorized causal LCB target-weight series for the master backtest.

    Processes one time-ordered panel in a single pass (cumulative statistics,
    no per-bar DataFrame copies). Row ``k`` is the target decided at the close
    of bar ``k`` from completed returns strictly before it; columns are the
    expert ids plus ``CASH``.
    """
    _validate_panel(component_returns)
    expert_ids = tuple(e.expert_id for e in spec.experts)
    missing = [e for e in expert_ids if e not in component_returns.columns]
    if missing:
        raise ValueError(f"experts missing from component_returns: {missing}")

    raw = np.zeros((len(component_returns), len(expert_ids)), dtype=np.float64)
    for j, expert_id in enumerate(expert_ids):
        rets = np.log1p(component_returns[expert_id].to_numpy(dtype=np.float64))
        raw[:, j] = _raw_allocation(rets, spec.confidence, spec.min_history_bars)

    family_mat, symbol_mat = _group_matrices(spec)
    risky = _project_weights(raw, spec, family_mat, symbol_mat)
    cash = spec.gross_exposure - risky.sum(axis=1, keepdims=True)
    weights = np.concatenate([risky, cash], axis=1)
    return pd.DataFrame(weights, index=component_returns.index, columns=[*list(expert_ids), "CASH"])


def compute_causal_lcb_weights(
    component_returns: pd.DataFrame,
    spec: ExpertPortfolioSpec,
    *,
    as_of: pd.Timestamp,
    previous_weights: pd.Series,
) -> pd.Series:
    """Causal block-aware lower-confidence-bound target weights for one decision bar.

    Only rows strictly before ``as_of`` influence the returned target. Every
    risky weight is finite and non-negative, risky gross exposure is at most
    ``spec.gross_exposure``, and invalid, unavailable, or non-positive-LCB
    experts receive zero allocation with the remainder in ``CASH``.
    ``previous_weights`` is validated for ledger continuity and fails closed on
    any malformed prior allocation. Malformed panels raise ``ValueError``;
    valid-but-weak evidence returns an exact all-cash allocation.
    """
    _validate_panel(component_returns)
    _validate_as_of(component_returns, as_of)
    _validate_previous_weights(previous_weights, spec)
    if as_of not in component_returns.index:
        raise ValueError(f"as_of {as_of} is not in the component_returns index")
    series = _causal_lcb_weight_series(component_returns, spec)
    return series.loc[as_of]


def _check_contract() -> None:
    """Executable assertions locking the frozen allocator surface."""
    from src.research.expert_portfolio.models import ExpertDefinition  # noqa: PLC0415

    assert compute_causal_lcb_weights.__name__ == "compute_causal_lcb_weights"
    panel = pd.DataFrame(
        {"A": [0.001] * 40, "B": [0.001] * 40},
        index=pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC"),
    )
    spec = ExpertPortfolioSpec(experts=(
        ExpertDefinition("A", "src", "f", ("S1",), "run", "hash"),
        ExpertDefinition("B", "src", "f", ("S2",), "run", "hash2"),
    ))
    weights = compute_causal_lcb_weights(
        panel,
        spec,
        as_of=panel.index[30],
        previous_weights=pd.Series({"A": 0.0, "B": 0.0, "CASH": 1.0}),
    )
    assert set(weights.index) == {"A", "B", "CASH"}
    assert float(weights["CASH"]) == 1.0 - float(weights.drop("CASH").sum())


_check_contract()
