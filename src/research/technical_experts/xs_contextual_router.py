"""Causal context-conditional family-sleeve selection for the XS alpha screen.

The research-only v3 successor to ``xs_alpha_multihorizon_v2`` keeps the three
economically distinct alpha-family sleeves (``trend``,
``funding_contrarian``, ``taker_imbalance``) as separate standalone books and,
at every decision bar, allocates to the single family whose conditional
block-aware lower confidence bound over fully completed same-state samples is
strictly positive and greatest.  A bar whose context is unavailable,
under-sampled, or without a positive finite winner holds ``CASH``.  The
six-state context classifier, ``state_labels``, the ``lcb_z_score`` quantile
constant, and the block-aware inflation helper are the pre-existing
expert-library components: no second state classifier and no duplicated
formula are introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.research.expert_portfolio.allocator import causal_block_aware_inflation
from src.research.expert_portfolio.contextual_router import state_labels
from src.research.expert_portfolio.models import ContextualRouterSpec, lcb_z_score

_FAMILY_ORDER = ("trend", "funding_contrarian", "taker_imbalance")
_CASH = "CASH"
_UNAVAILABLE = "unavailable"

__all__ = [
    "XsContextualAllocation",
    "XsScoreRoutedAllocation",
    "build_xs_causal_contextual_allocation",
    "build_xs_causal_score_selection",
    "build_xs_context_market",
]


def build_xs_context_market(closes: pd.DataFrame) -> pd.Series:
    """Cross-sectional equal-weight geometric market close.

    Returns ``exp(mean(log(close_i)))`` across the columns of a validated,
    strictly-positive close panel, so no single large-cap asset defines the
    decision state of a dollar-neutral cross-sectional book.
    """
    if not isinstance(closes, pd.DataFrame):
        raise DataIntegrityError(
            f"closes must be a DataFrame, got {type(closes).__name__}"
        )
    index = closes.index
    if not isinstance(index, pd.DatetimeIndex) or getattr(index, "tz", None) is None:
        raise DataIntegrityError("closes index must be a tz-aware UTC DatetimeIndex")
    if not index.is_unique:
        raise DataIntegrityError("closes index must be unique")
    if not index.is_monotonic_increasing:
        raise DataIntegrityError("closes index must be monotonic increasing")
    values = closes.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise DataIntegrityError("closes must be finite and strictly positive")
    return pd.Series(
        np.exp(np.log(values).mean(axis=1)),
        index=index,
        name="market",
    )


@dataclass(frozen=True, slots=True)
class XsContextualAllocation:
    """Deterministic outcome of one causal contextual family selection.

    ``target_weights`` is the combined target matrix (the winner family's banded
    sleeve weights per bar, all zero for ``CASH`` rows), ``decision_context``
    preserves the input context labels, ``selected_sleeve`` names the selected
    family or ``"CASH"`` per bar, and ``conditional_lcb`` carries the three
    per-family block-aware lower confidence bounds (``NaN`` where no eligible
    completed evidence exists).
    """

    target_weights: pd.DataFrame
    decision_context: pd.Series
    selected_sleeve: pd.Series
    conditional_lcb: pd.DataFrame


@dataclass(frozen=True, slots=True)
class XsScoreRoutedAllocation:
    """Deterministic outcome of one causal score-layer family selection.

    Field-for-field parallel to :class:`XsContextualAllocation` except
    ``target_weights`` is replaced by ``combined_score``: the winner family's
    raw (pre-EWMA, pre-demean, pre-band) score row per bar, all zero for
    ``CASH`` rows.  ``decision_context``, ``selected_sleeve``, and
    ``conditional_lcb`` carry identical semantics, so the application-layer
    router diagnostics consume either allocation type unchanged.
    """

    combined_score: pd.DataFrame
    decision_context: pd.Series
    selected_sleeve: pd.Series
    conditional_lcb: pd.DataFrame


def _validate_router_sleeve_inputs(
    sleeves: dict[str, pd.DataFrame],
    sleeve_returns: pd.DataFrame,
    decision_context: pd.Series,
    *,
    frame_kind: str,
) -> tuple[pd.DatetimeIndex, list[str]]:
    """Shared fail-closed validation for the causal router sleeve inputs.

    ``frame_kind`` is ``"weight"`` or ``"score"`` and only selects the wording
    of the error messages; every structural check is identical for the
    weight-layer and score-layer routers so the two public builders share one
    validation contract.  Returns the validated common index and ordered
    column set of the sleeve frames.
    """
    if not isinstance(sleeves, dict):
        raise DataIntegrityError(f"sleeve_{frame_kind}s must be a dict")
    if set(sleeves) != set(_FAMILY_ORDER):
        raise DataIntegrityError(
            f"sleeve_{frame_kind}s must map exactly the families {list(_FAMILY_ORDER)}, "
            f"got {sorted(sleeves)}"
        )
    for name in _FAMILY_ORDER:
        if not isinstance(sleeves[name], pd.DataFrame):
            raise DataIntegrityError(
                f"sleeve {frame_kind} frame {name!r} must be a DataFrame"
            )

    reference = sleeves[_FAMILY_ORDER[0]]
    index = reference.index
    if not isinstance(index, pd.DatetimeIndex) or getattr(index, "tz", None) is None:
        raise DataIntegrityError("sleeve index must be a tz-aware UTC DatetimeIndex")
    if not index.is_unique:
        raise DataIntegrityError("sleeve index must be unique")
    if not index.is_monotonic_increasing:
        raise DataIntegrityError("sleeve index must be monotonic increasing")
    for name in _FAMILY_ORDER:
        frame = sleeves[name]
        if not frame.index.equals(reference.index):
            raise DataIntegrityError(
                f"sleeve {frame_kind} frames must share an identical index"
            )
        if list(frame.columns) != list(reference.columns):
            raise DataIntegrityError(
                f"sleeve {frame_kind} frames must share an identical ordered column set"
            )
    symbol_columns = list(reference.columns)
    if len(symbol_columns) != len(set(symbol_columns)):
        raise DataIntegrityError(f"sleeve {frame_kind} columns must be unique")

    if not isinstance(sleeve_returns, pd.DataFrame):
        raise DataIntegrityError("sleeve_returns must be a DataFrame")
    if not sleeve_returns.index.equals(reference.index):
        raise DataIntegrityError("sleeve_returns must share the sleeve index")
    if list(sleeve_returns.columns) != list(_FAMILY_ORDER):
        raise DataIntegrityError(
            f"sleeve_returns columns must be {list(_FAMILY_ORDER)}"
        )

    if not isinstance(decision_context, pd.Series):
        raise DataIntegrityError("decision_context must be a pd.Series")
    if not decision_context.index.equals(reference.index):
        raise DataIntegrityError(
            "decision_context must be aligned to the sleeve index"
        )
    if decision_context.isna().any():
        raise DataIntegrityError("decision_context must not contain missing labels")
    if not all(isinstance(value, str) for value in decision_context.to_numpy()):
        raise DataIntegrityError("decision_context must contain only string labels")
    known = set(state_labels()) | {_UNAVAILABLE}
    unknown = sorted(
        {value for value in decision_context.unique() if value not in known}
    )
    if unknown:
        raise DataIntegrityError(f"decision_context contains unknown labels: {unknown}")

    return index, symbol_columns


def build_xs_causal_contextual_allocation(
    sleeve_weights: dict[str, pd.DataFrame],
    sleeve_returns: pd.DataFrame,
    decision_context: pd.Series,
    router_spec: ContextualRouterSpec,
) -> XsContextualAllocation:
    """Causal single-family target-weight series for the XS alpha contextual book.

    Row ``t`` is the target decided at the close of bar ``t``.  The only
    completed evidence is ``(label_i, r_{i+1})`` for ``i < t - 1``: a target
    made at bar ``i`` earns ``r_{i+1}`` and a sample may influence the decision
    only after that round trip has fully completed strictly before ``t``.  For
    the current context ``decision_context[t]`` every family's conditional
    log-return lower confidence bound is computed over matching completed
    samples only; the single family with the greatest strictly positive finite
    LCB (declaration order breaks exact ties) receives its banded sleeve target
    row and every other bar is ``CASH``.  A context that is unavailable, has
    fewer than ``min_context_history_bars`` completed samples, or whose winner
    score is not strictly positive and finite allocates ``CASH``.

    The combined ``target_weights`` frame is the sole input to the final
    composite ledger; the standalone sleeve return series are diagnostic
    evidence only.  Malformed input (shape/index/column disagreement, an
    unknown family, non-finite returns after the first mark, or an invalid
    label) raises :class:`DataIntegrityError` -- missing data is never
    zero-filled.
    """
    index, symbol_columns = _validate_router_sleeve_inputs(
        sleeve_weights, sleeve_returns, decision_context, frame_kind="weight",
    )
    family_matrices = {
        name: sleeve_weights[name].to_numpy(dtype=np.float64)
        for name in _FAMILY_ORDER
    }
    target, selected, lcb_out = _causal_family_lcb_selection(
        family_matrices, sleeve_returns, decision_context, router_spec,
    )
    return XsContextualAllocation(
        target_weights=pd.DataFrame(target, index=index, columns=symbol_columns),
        decision_context=decision_context,
        selected_sleeve=pd.Series(selected, index=index, name="selected_sleeve"),
        conditional_lcb=pd.DataFrame(lcb_out, index=index, columns=list(_FAMILY_ORDER)),
    )


def _causal_family_lcb_selection(
    matrices: dict[str, np.ndarray],
    sleeve_returns: pd.DataFrame,
    decision_context: pd.Series,
    router_spec: ContextualRouterSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block-aware-LCB winner selection over the six causal states (shared core).

    Selects, for every decision bar, the single family whose conditional
    block-aware lower confidence bound over fully completed same-state samples
    is strictly positive and greatest, and copies that family's input row (a
    banded weight row or a raw score row, whichever ``matrices`` holds) into
    the target row.  This is the per-state/per-bar loop body shared verbatim by
    :func:`build_xs_causal_contextual_allocation` (weight layer) and
    :func:`build_xs_causal_score_selection` (score layer); only the write
    target -- a bare ``ndarray`` row -- is generalized, so the selection logic
    is byte-for-byte identical across both construction layers.

    Returns ``(target, selected, lcb_out)`` where ``target`` is the
    ``[n, n_cols]`` winner-row matrix (all zero for ``CASH`` rows),
    ``selected`` is the ``[n]`` object array of family names or ``"CASH"``, and
    ``lcb_out`` is the ``[n, len(_FAMILY_ORDER)]`` float64 array of per-family
    lower confidence bounds (``NaN`` where no eligible completed evidence
    exists).  Inputs are assumed already validated by the public builders.
    """
    z_score = lcb_z_score(router_spec.confidence)
    min_history = router_spec.min_context_history_bars
    labels = decision_context.to_numpy(dtype=object)
    returns = sleeve_returns.to_numpy(dtype=np.float64)
    logret = np.log1p(returns)
    if not np.isfinite(logret[1:]).all():
        raise DataIntegrityError(
            "sleeve returns must be finite on every sample row; a missing "
            "return is never zero-filled"
        )

    n = len(labels)
    lcb_out = np.full((n, len(_FAMILY_ORDER)), np.nan, dtype=np.float64)
    selected = np.full(n, _CASH, dtype=object)
    target = np.zeros((n, matrices[_FAMILY_ORDER[0]].shape[1]), dtype=np.float64)

    for state in state_labels():
        state_rows = np.flatnonzero(labels == state)
        state_rows = state_rows[state_rows < n - 1]
        if state_rows.size == 0:
            continue
        sample_rows = state_rows + 1
        samples = logret[sample_rows, :]
        csum = np.vstack(
            [np.zeros((1, samples.shape[1])), np.cumsum(samples, axis=0)]
        )
        csum2 = np.vstack(
            [np.zeros((1, samples.shape[1])), np.cumsum(samples * samples, axis=0)]
        )
        eligible = np.searchsorted(state_rows, np.arange(n) - 1)
        inflation_by_column: list[np.ndarray] = []
        for column in range(len(_FAMILY_ORDER)):
            completed = np.concatenate(
                [samples[:, column], np.array([0.0])]
            )
            inflation_by_column.append(causal_block_aware_inflation(completed))

        for t in np.flatnonzero(labels == state):
            j = int(eligible[t])
            if j < min_history or j < 2:
                continue
            mean = csum[j] / j
            inflation = np.array(
                [inflation_by_column[column][j] for column in range(len(_FAMILY_ORDER))],
                dtype=np.float64,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                variance = (csum2[j] - csum[j] * csum[j] / j) / (j - 1)
                std_error = np.sqrt(np.maximum(variance, 0.0) * inflation / j)
                row_lcb = mean - z_score * std_error
            lcb_out[t, :] = row_lcb
            best = int(np.argmax(row_lcb))
            if not np.isfinite(row_lcb[best]) or row_lcb[best] <= 0.0:
                continue
            family = _FAMILY_ORDER[best]
            selected[t] = family
            target[t, :] = matrices[family][t, :]

    return target, selected, lcb_out


def build_xs_causal_score_selection(
    sleeve_scores: dict[str, pd.DataFrame],
    sleeve_returns: pd.DataFrame,
    decision_context: pd.Series,
    router_spec: ContextualRouterSpec,
) -> XsScoreRoutedAllocation:
    """Causal single-family score-row selection for the XS alpha score-layer router.

    Row ``t`` is the decision made at the close of bar ``t`` and consumes the
    identical six-state, block-aware-LCB winner rule as the weight-layer
    router, but the winner's *raw pre-normalization score row* (never an
    EWMA-smoothed, demeaned, or banded weight row) is written into
    ``combined_score``; a ``CASH`` bar is an all-zero row.  The caller is
    responsible for feeding ``combined_score`` through
    :func:`build_xs_neutral_weights` exactly once -- the construction-layer
    separation (smooth once after selection, never per family) is the entire
    point of the score-layer relocation.  No future row ever influences a
    decision.  Malformed input fails closed with
    :class:`DataIntegrityError`, identical to the weight-layer function.
    """
    index, symbol_columns = _validate_router_sleeve_inputs(
        sleeve_scores, sleeve_returns, decision_context, frame_kind="score",
    )
    family_matrices = {
        name: sleeve_scores[name].to_numpy(dtype=np.float64)
        for name in _FAMILY_ORDER
    }
    target, selected, lcb_out = _causal_family_lcb_selection(
        family_matrices, sleeve_returns, decision_context, router_spec,
    )
    return XsScoreRoutedAllocation(
        combined_score=pd.DataFrame(target, index=index, columns=symbol_columns),
        decision_context=decision_context,
        selected_sleeve=pd.Series(selected, index=index, name="selected_sleeve"),
        conditional_lcb=pd.DataFrame(lcb_out, index=index, columns=list(_FAMILY_ORDER)),
    )


def _check_contract() -> None:
    """Executable assertions locking the XS contextual-router surface."""
    closes = pd.DataFrame(
        {
            "A": [100.0, 101.0, 99.0, 102.0],
            "B": [200.0, 202.0, 198.0, 204.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC"),
    )
    market = build_xs_context_market(closes)
    assert market.index.equals(closes.index)
    expected = np.exp(np.log(closes.to_numpy(dtype=np.float64)).mean(axis=1))
    assert np.allclose(market.to_numpy(), expected, atol=1e-12)

    index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    columns = ["S1", "S2"]
    weights = {
        name: pd.DataFrame(0.1, index=index, columns=columns) for name in _FAMILY_ORDER
    }
    returns = pd.DataFrame(0.001, index=index, columns=list(_FAMILY_ORDER))
    context = pd.Series(["unavailable"] * len(index), index=index)
    spec = ContextualRouterSpec(
        "XS_EQUAL_WEIGHT_MARKET", 42, 42, 168, 0.90,
    )
    allocation = build_xs_causal_contextual_allocation(
        weights, returns, context, spec,
    )
    assert allocation.selected_sleeve.iloc[0] == _CASH
    assert float(allocation.target_weights.abs().to_numpy().max()) == 0.0

    score_allocation = build_xs_causal_score_selection(
        weights, returns, context, spec,
    )
    assert score_allocation.selected_sleeve.iloc[0] == _CASH
    assert float(score_allocation.combined_score.abs().to_numpy().max()) == 0.0
    assert score_allocation.selected_sleeve.equals(allocation.selected_sleeve)


_check_contract()
