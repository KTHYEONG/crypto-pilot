"""Deterministic contextual winner router for the expert library.

A pre-registered router classifies each decision bar into one of six market
states (trend up/down/flat times volatility high/low) and, at every decision
bar, allocates the full gross exposure to the single expert whose strictly
positive conditional lower confidence bound over completed samples of the
current state is greatest.  Contexts without enough completed evidence or
without a strictly positive winner allocate ``CASH``.  No model is fitted and
no future row ever influences a decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.expert_portfolio.allocator import (
    _causal_block_aware_inflation,
    _validate_panel,
)
from src.research.expert_portfolio.contracts import (
    ContextualRouterSpec,
    ExpertDefinition,
    ExpertPortfolioSpec,
    lcb_z_score,
)

_STATE_TRENDS = ("down", "flat", "up")
_STATE_VOLS = ("low", "high")
_UNAVAILABLE = "unavailable"
_FLAT_TREND_BAND = 0.0005  # pre-registered flat band: |log change| <= band => flat


def state_labels() -> list[str]:
    """The six pre-registered contextual states."""
    return [f"{trend}_{vol}_vol" for trend in _STATE_TRENDS for vol in _STATE_VOLS]


def build_causal_context_labels(
    close: pd.Series,
    spec: ContextualRouterSpec,
) -> pd.Series:
    """Classify every bar into a causal six-state context label.

    At decision bar ``t`` the trend uses closes no later than ``t`` (the log
    change over ``trend_lookback_bars``) and volatility uses completed returns
    no later than ``t`` (the rolling standard deviation over
    ``volatility_lookback_bars`` completed returns), so no future row can
    influence a label.  A state is ``high`` volatility when the completed
    rolling volatility is at least its expanding median computed through ``t``.
    Insufficient history returns the label ``unavailable``.  The returned
    ``Series`` preserves ``close``'s index exactly.
    """
    if not isinstance(close, pd.Series):
        raise ValueError(f"close must be a pd.Series, got {type(close).__name__}")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("close must have a DatetimeIndex")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be monotonic increasing")
    if close.index.has_duplicates:
        raise ValueError("close index must not contain duplicate timestamps")
    values = close.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("close must contain only finite values")
    if (values <= 0.0).any():
        raise ValueError("close must contain only strictly positive values")

    n = len(values)
    labels = np.full(n, _UNAVAILABLE, dtype=object)

    log_change = np.full(n, np.nan, dtype=np.float64)
    if n > spec.trend_lookback_bars:
        log_change[spec.trend_lookback_bars:] = np.log(
            values[spec.trend_lookback_bars:] / values[:-spec.trend_lookback_bars]
        )
    trend = np.full(n, None, dtype=object)
    trend[log_change > _FLAT_TREND_BAND] = "up"
    trend[log_change < -_FLAT_TREND_BAND] = "down"
    trend[(np.abs(log_change) <= _FLAT_TREND_BAND) & np.isfinite(log_change)] = "flat"

    volatility = np.full(n, np.nan, dtype=np.float64)
    if n > spec.volatility_lookback_bars:
        logret = np.diff(np.log(values))
        rolling = (
            pd.Series(logret)
            .rolling(
                spec.volatility_lookback_bars,
                min_periods=spec.volatility_lookback_bars,
            )
            .std(ddof=0)
            .to_numpy()
        )
        # close row t uses the completed window of returns ending at t: rolling[t-1]
        volatility[spec.volatility_lookback_bars :] = rolling[
            spec.volatility_lookback_bars - 1 :
        ]
        median = pd.Series(volatility).expanding(min_periods=1).median().to_numpy()
        high = volatility >= median
        ready = np.isfinite(volatility) & np.isfinite(log_change)
        labels[ready] = np.array(
            [
                f"{t}_{'high' if h else 'low'}_vol"
                for t, h in zip(trend[ready], high[ready], strict=True)
            ],
            dtype=object,
        )
    return pd.Series(labels, index=close.index, name="context")


def compute_causal_contextual_winner_weights(
    component_returns: pd.DataFrame,
    decision_context: pd.Series,
    portfolio_spec: ExpertPortfolioSpec,
    router_spec: ContextualRouterSpec,
) -> pd.DataFrame:
    """Causal single-winner target-weight series for one contextual master ledger.

    Row ``t`` is the target decided at the close of bar ``t``.  The only
    completed evidence is ``(C_i, r_{i+1})`` for ``i < t - 1``: a target made
    at bar ``i`` earns ``r_{i+1}`` and a sample may influence the decision only
    after that round trip has fully completed strictly before ``t``.  For the
    current context ``decision_context[t]`` every expert's conditional log-return
    lower confidence bound is computed over matching completed samples only; the
    single expert with the greatest strictly positive LCB receives the full
    ``gross_exposure`` (declaration order breaks exact ties) and every other row
    is ``CASH``.  A context that is unavailable, has fewer than
    ``min_context_history_bars`` completed samples, or whose winner score is not
    strictly positive and finite allocates ``CASH``.  The existing causal LCB-mix
    weights are never modified.
    """
    _validate_panel(component_returns)
    if not isinstance(decision_context, pd.Series):
        raise ValueError(
            f"decision_context must be a pd.Series, got {type(decision_context).__name__}"
        )
    if not decision_context.index.equals(component_returns.index):
        raise ValueError(
            "decision_context must be aligned to the component_returns index"
        )
    if decision_context.isna().any():
        raise ValueError("decision_context must not contain missing labels")
    if not all(isinstance(value, str) for value in decision_context.to_numpy()):
        raise ValueError("decision_context must contain only string labels")
    known = set(state_labels()) | {_UNAVAILABLE}
    unknown = sorted({value for value in decision_context.unique() if value not in known})
    if unknown:
        raise ValueError(f"decision_context contains unknown labels: {unknown}")

    expert_ids = tuple(e.expert_id for e in portfolio_spec.experts)
    missing = [e for e in expert_ids if e not in component_returns.columns]
    if missing:
        raise ValueError(f"experts missing from component_returns: {missing}")

    n = len(component_returns)
    z_score = lcb_z_score(router_spec.confidence)
    min_history = router_spec.min_context_history_bars
    labels = decision_context.to_numpy(dtype=object)
    returns = component_returns[list(expert_ids)].to_numpy(dtype=np.float64)
    logret = np.log1p(returns)
    if not np.isfinite(logret[1:]).all():
        raise ValueError(
            "component returns must be finite on every sample row; a missing "
            "return is never zero-filled"
        )

    weight_columns = [*list(expert_ids), "CASH"]
    risky = np.zeros((n, len(expert_ids)), dtype=np.float64)

    for state in state_labels():
        state_rows = np.flatnonzero(labels == state)
        # a state row must earn a return strictly inside the panel to be usable
        state_rows = state_rows[state_rows < n - 1]
        if state_rows.size == 0:
            continue
        # sample rows are the returns earned by each state row one bar later
        sample_rows = state_rows + 1
        samples = logret[sample_rows, :]
        csum = np.vstack([np.zeros((1, samples.shape[1])), np.cumsum(samples, axis=0)])
        csum2 = np.vstack([
            np.zeros((1, samples.shape[1])), np.cumsum(samples * samples, axis=0),
        ])
        # j(t): number of state rows strictly before row t-1 that are usable
        eligible = np.searchsorted(state_rows, np.arange(n) - 1)

        for t in np.flatnonzero(labels == state):
            j = int(eligible[t])
            if j < min_history or j < 2:
                continue
            mean = csum[j] / j
            # Append a sentinel so the existing causal inflation helper can
            # evaluate the variance of all j completed samples at index j;
            # the sentinel itself is excluded from that calculation.
            inflation = np.ones(len(expert_ids), dtype=np.float64)
            for column in range(len(expert_ids)):
                completed = np.concatenate([samples[:, column], np.array([0.0])])
                inflation[column] = _causal_block_aware_inflation(completed)[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                variance = (csum2[j] - csum[j] * csum[j] / j) / (j - 1)
                std_error = np.sqrt(np.maximum(variance, 0.0) * inflation / j)
                lcb = mean - z_score * std_error
            best = int(np.argmax(lcb))
            if not np.isfinite(lcb[best]) or lcb[best] <= 0.0:
                continue
            risky[t, best] = portfolio_spec.gross_exposure

    cash = portfolio_spec.gross_exposure - risky.sum(axis=1, keepdims=True)
    weights = np.concatenate([risky, cash], axis=1)
    return pd.DataFrame(
        weights, index=component_returns.index, columns=weight_columns,
    )


def _check_contract() -> None:
    """Executable assertions locking the frozen contextual-router surface."""
    close = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC"),
    )
    spec = ContextualRouterSpec("BTCUSDT", 1, 1, 1)
    labels = build_causal_context_labels(close, spec)
    assert labels.index.equals(close.index)
    assert labels.iloc[0] == _UNAVAILABLE
    assert set(labels.iloc[1:]).issubset(set(state_labels()))
    assert compute_causal_contextual_winner_weights.__name__ == (
        "compute_causal_contextual_winner_weights"
    )
    index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    panel = pd.DataFrame(
        {
            "A": [0.0, 0.02, 0.02, 0.02, 0.02, 0.02],
            "B": [0.0, -0.01, -0.01, -0.01, -0.01, -0.01],
        },
        index=index,
    )
    context = pd.Series(["up_low_vol"] * 6, index=index)
    portfolio_spec = ExpertPortfolioSpec(experts=(
        ExpertDefinition("A", "src", "f", ("S1",), "run", "h"),
        ExpertDefinition("B", "src", "f", ("S2",), "run", "h2"),
    ))
    weights = compute_causal_contextual_winner_weights(
        panel, context, portfolio_spec, spec,
    )
    assert weights.columns.tolist() == ["A", "B", "CASH"]


_check_contract()
