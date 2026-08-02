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
    _group_matrices,
    _project_weights,
    _validate_panel,
)
from src.research.expert_portfolio.models import (
    ContextualRouterSpec,
    ExpertDefinition,
    ExpertPortfolioSpec,
    lcb_z_score,
)

_STATE_TRENDS = ("down", "flat", "up")
_STATE_VOLS = ("low", "high")
_UNAVAILABLE = "unavailable"
_FLAT_TREND_QUANTILE = 0.10  # causal expanding quantile of |z| => flat boundary


def state_labels() -> list[str]:
    """The six pre-registered contextual states."""
    return [f"{trend}_{vol}_vol" for trend in _STATE_TRENDS for vol in _STATE_VOLS]


def build_causal_context_labels(
    close: pd.Series,
    spec: ContextualRouterSpec,
) -> pd.Series:
    """Classify every bar into a causal six-state context label.

    At decision bar ``t`` the trend is the volatility-normalized completed log
    change over ``trend_lookback_bars``,

        z_t = log(C_t / C_(t-L)) / (sigma_t * sqrt(L))

    where ``sigma_t`` is the completed rolling one-bar log-return standard
    deviation over ``volatility_lookback_bars`` returns ending at ``t``; no
    future row influences a label.  ``t`` is ``flat`` when ``abs(z_t)`` is at
    most its causal expanding ``_FLAT_TREND_QUANTILE`` of all finite ``|z|``
    values computed through ``t``, otherwise ``up``/``down`` by ``z_t``'s sign.
    A state is ``high`` volatility when the completed rolling volatility is at
    least its expanding median computed through ``t``.  The boundary is
    invariant to a positive scaling of ``close`` and to a proportional
    volatility change.  Insufficient finite history returns ``unavailable``.
    The returned ``Series`` preserves ``close``'s index exactly.
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

        with np.errstate(divide="ignore", invalid="ignore"):
            z = log_change / (volatility * np.sqrt(spec.trend_lookback_bars))
        abs_z = np.where(np.isfinite(z), np.abs(z), np.nan)
        flat_boundary = (
            pd.Series(abs_z).expanding(min_periods=1).quantile(_FLAT_TREND_QUANTILE)
        )
        flat = np.isfinite(abs_z) & (abs_z <= flat_boundary.to_numpy())
        trend = np.full(n, None, dtype=object)
        trend[flat] = "flat"
        trend[~flat & (z > 0.0)] = "up"
        trend[~flat & (z < 0.0)] = "down"

        ready = np.isfinite(volatility) & np.isfinite(z)
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


def compute_causal_per_symbol_contextual_weights(
    component_returns: pd.DataFrame,
    decision_context: pd.Series,
    portfolio_spec: ExpertPortfolioSpec,
    router_spec: ContextualRouterSpec,
) -> pd.DataFrame:
    """Causal per-symbol winner target-weight series for one contextual ledger.

    The v2 router picks, in every decision state, at most one strictly-positive
    conditional LCB winner from each symbol's distinct-family experts using only
    completed observations strictly before ``t`` (the identical attribution as
    :func:`compute_causal_contextual_winner_weights`); a symbol without a
    positive eligible winner is exactly CASH. The surviving symbol winners then
    receive inverse realized-volatility weight that is projected once onto
    ``gross_exposure``, ``family_exposure_limit``, and
    ``symbol_exposure_limit``, so two strategies on the same symbol are never
    concurrently exposed while different symbols can be held simultaneously.
    When no symbol has a positive eligible LCB the returned frame is exact
    all-CASH. Malformed panels, duplicate expert ids, unknown contexts, and
    misaligned inputs fail closed with ``ValueError``; the existing global
    single-winner path is never modified.
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
    if component_returns.columns.has_duplicates:
        raise ValueError("component_returns columns must be unique")
    missing = [e for e in expert_ids if e not in component_returns.columns]
    if missing:
        raise ValueError(f"experts missing from component_returns: {missing}")

    by_symbol: dict[str, list[int]] = {}
    for index, expert in enumerate(portfolio_spec.experts):
        if len(expert.symbols) != 1:
            raise ValueError(
                f"per-symbol router requires single-symbol experts, got "
                f"{expert.symbols} for {expert.expert_id}"
            )
        by_symbol.setdefault(expert.symbols[0], []).append(index)
    for symbol, group in by_symbol.items():
        families = [portfolio_spec.experts[i].family for i in group]
        if len(families) != len(set(families)):
            raise ValueError(
                f"duplicate family within symbol {symbol} for per-symbol router"
            )

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

    raw = np.zeros((n, len(expert_ids)), dtype=np.float64)
    for state in state_labels():
        state_rows = np.flatnonzero(labels == state)
        state_rows = state_rows[state_rows < n - 1]
        if state_rows.size == 0:
            continue
        sample_rows = state_rows + 1
        samples = logret[sample_rows, :]
        csum = np.vstack([np.zeros((1, samples.shape[1])), np.cumsum(samples, axis=0)])
        csum2 = np.vstack([
            np.zeros((1, samples.shape[1])), np.cumsum(samples * samples, axis=0),
        ])
        eligible = np.searchsorted(state_rows, np.arange(n) - 1)
        # The block-aware inflation of one expert column is identical for every
        # decision row of the same state, so it is evaluated once per column and
        # then indexed by ``j`` instead of being recomputed per row.
        inflation_by_column = []
        for column in range(len(expert_ids)):
            completed = np.concatenate([samples[:, column], np.array([0.0])])
            inflation_by_column.append(_causal_block_aware_inflation(completed))

        for t in np.flatnonzero(labels == state):
            j = int(eligible[t])
            if j < min_history or j < 2:
                continue
            for group in by_symbol.values():
                group_indexes = tuple(group)
                mean = csum[j, group_indexes] / j
                inflation = np.array(
                    [inflation_by_column[column][j] for column in group_indexes],
                    dtype=np.float64,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    variance = (
                        csum2[j, group_indexes]
                        - csum[j, group_indexes] * csum[j, group_indexes] / j
                    ) / (j - 1)
                    std_error = np.sqrt(np.maximum(variance, 0.0) * inflation / j)
                    lcb = mean - z_score * std_error
                best_local = int(np.argmax(lcb))
                if not np.isfinite(lcb[best_local]) or lcb[best_local] <= 0.0:
                    continue
                winner = group_indexes[best_local]
                vol = np.sqrt(max(float(variance[best_local]), 0.0))
                if vol > 0.0:
                    raw[t, winner] = 1.0 / vol

    family_mat, symbol_mat = _group_matrices(portfolio_spec)
    risky = _project_weights(raw, portfolio_spec, family_mat, symbol_mat)
    cash = portfolio_spec.gross_exposure - risky.sum(axis=1, keepdims=True)
    weights = np.concatenate([risky, cash], axis=1)
    return pd.DataFrame(
        weights, index=component_returns.index, columns=[*list(expert_ids), "CASH"],
    )


def _check_contract() -> None:
    """Executable assertions locking the frozen contextual-router surface."""
    close = pd.Series(
        [100.0, 101.0, 99.0, 102.0],
        index=pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC"),
    )
    spec = ContextualRouterSpec("BTCUSDT", 1, 2, 1)
    labels = build_causal_context_labels(close, spec)
    assert labels.index.equals(close.index)
    assert labels.iloc[0] == _UNAVAILABLE
    labeled = labels[labels != _UNAVAILABLE]
    assert not labeled.empty
    assert set(labeled).issubset(set(state_labels()))
    assert compute_causal_contextual_winner_weights.__name__ == (
        "compute_causal_contextual_winner_weights"
    )
    assert compute_causal_per_symbol_contextual_weights.__name__ == (
        "compute_causal_per_symbol_contextual_weights"
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
