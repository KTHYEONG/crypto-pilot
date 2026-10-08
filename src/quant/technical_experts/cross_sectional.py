"""Frozen target-weight composite ledger shared by the MHS pre-screen proxy.

:class:`XsCompositeSpec` freezes the execution convention (t+1+delay open
fills, per-unit-turnover fee + slippage, funding on the held book) and
:func:`run_xs_composite_ledger` / :func:`run_xs_composite_ledger_multi_tier`
compound a supplied weight panel into an equity ledger under it. Weight
construction lives with the callers; ``src.engine.execution.pnl`` is the only
production consumer. This module is imported by the live daemon chain, so its
import-time contract check must reference only symbols defined here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError

_INITIAL_EQUITY = 10_000.0


@dataclass(frozen=True, slots=True)
class XsCompositeSpec:
    """Frozen construction and execution contract of the XS composite profile.

    ``halflife_bars`` and ``no_trade_band`` are the only fitted parameters in
    the whole construction and were frozen on discovery data alone, so the
    qualification result stays an honest out-of-sample test. The remaining
    fields are the production execution convention shared with the TS screen.
    """

    halflife_bars: int = 6
    no_trade_band: float = 0.05
    execution_delay_bars: int = 1
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0003
    gap_carry: bool = False

    def __post_init__(self) -> None:
        if self.halflife_bars < 0:
            raise ValueError(
                f"halflife_bars must be >= 0, got {self.halflife_bars}"
            )
        if not 0.0 <= self.no_trade_band < 1.0:
            raise ValueError(
                f"no_trade_band must be in [0.0, 1.0), got {self.no_trade_band}"
            )
        if self.execution_delay_bars < 0:
            raise ValueError(
                f"execution_delay_bars must be >= 0, got {self.execution_delay_bars}"
            )
        if self.fee_rate < 0:
            raise ValueError(f"fee_rate must be >= 0, got {self.fee_rate}")
        if self.slippage_rate < 0:
            raise ValueError(f"slippage_rate must be >= 0, got {self.slippage_rate}")

    def round_trip_cost_rate(self) -> float:
        """Per-unit-turnover charge ``fee_rate + slippage_rate``."""
        return self.fee_rate + self.slippage_rate


def _ledger_components(
    lagged: np.ndarray,
    o2o: np.ndarray,
    funding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rate-independent core of the frozen ledger P&L formula.

    Computes the per-bar ``book_return`` (weighted open-to-open), the per-bar
    ``funding_charge`` (weighted funding-rate), and the per-bar ``turnover``
    (row sum of absolute lagged-weight changes, first row trades from zero)
    from the already-lagged weight matrix. This is the single source of truth
    for the round-trip cost formula -- callers must not reimplement it.
    ``_ledger_pnl`` applies a cost rate to these components; the multi-tier
    ledger reuses them once across many rates.
    """
    prev_lagged = np.zeros_like(lagged)
    prev_lagged[1:] = lagged[:-1]
    turnover = np.abs(lagged - prev_lagged).sum(axis=1)

    active = lagged != 0.0
    missing_active = active & (~np.isfinite(o2o) | ~np.isfinite(funding))
    if missing_active.any():
        row, col = np.argwhere(missing_active)[0]
        raise DataIntegrityError(
            "active ledger cell has non-finite open return/funding "
            f"(row={row}, column={col})"
        )
    # Lifecycle NaNs in inactive symbols are expected and must not poison the
    # cross-sectional sum through IEEE-754's 0 * NaN propagation.
    safe_o2o = np.where(np.isfinite(o2o), o2o, 0.0)
    safe_funding = np.where(np.isfinite(funding), funding, 0.0)
    book_return = (lagged * safe_o2o).sum(axis=1)
    funding_charge = (lagged * safe_funding).sum(axis=1)
    return book_return, funding_charge, turnover


def _ledger_pnl(
    lagged: np.ndarray,
    o2o: np.ndarray,
    funding: np.ndarray,
    cost_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen ledger P&L formula: per-bar net returns and turnover.

    ``lagged`` is the already-lagged weight matrix (row ``t`` is what is held
    against the ``t``-th open-to-open return), ``o2o`` the open-to-open return
    matrix, and ``funding`` the per-bar funding-rate matrix.  Turnover is the
    row sum of absolute lagged-weight changes (the first row trades from zero);
    each bar's net return is ``sum(lagged * o2o) - turnover * cost_rate -
    sum(lagged * funding)``.  This is the single source of truth for the round
    -trip cost formula -- callers must not reimplement it. The expression order
    ``(book_return - turnover * cost_rate) - funding_charge`` is frozen; the
    multi-tier ledger applies the identical order per rate.
    """
    book_return, funding_charge, turnover = _ledger_components(lagged, o2o, funding)
    net_returns = book_return - turnover * cost_rate - funding_charge
    return net_returns, turnover


def _xs_composite_inputs(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    spec: XsCompositeSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Shared array construction for the composite ledger.

    Validates the index/column alignment with the identical messages as
    ``run_xs_composite_ledger`` and builds ``(lagged, o2o, funding, index)``:
    ``lagged`` holds weights shifted by ``1 + execution_delay_bars`` so they are
    held against the ``open[t+1+delay] -> open[t+2+delay]`` return, ``o2o`` is
    the open-to-open return matrix, and ``funding`` the per-bar funding matrix.
    The single-call and multi-tier ledgers share this so they can never diverge
    in array construction.
    """
    if not (
        weights.index.equals(opens.index)
        and weights.index.equals(bar_funding.index)
    ):
        raise DataIntegrityError(
            "weights, opens, and bar_funding must share an identical index"
        )
    if not (
        list(weights.columns) == list(opens.columns)
        and list(weights.columns) == list(bar_funding.columns)
    ):
        raise DataIntegrityError(
            "weights, opens, and bar_funding must share an identical column set"
        )

    w = weights.to_numpy(dtype=np.float64)
    o = opens.ffill(axis=0).to_numpy(dtype=np.float64) if spec.gap_carry else opens.to_numpy(dtype=np.float64)
    f = bar_funding.to_numpy(dtype=np.float64)

    lag = 1 + spec.execution_delay_bars
    lagged = np.zeros_like(w)
    if lag < w.shape[0]:
        lagged[lag:] = w[: w.shape[0] - lag]

    o2o = np.zeros_like(o)
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o[1:] = o[1:] / o[:-1] - 1.0

    return lagged, o2o, f, weights.index


def run_xs_composite_ledger(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    spec: XsCompositeSpec,
) -> tuple[pd.Series, pd.Series]:
    """Compound the composite book into a total-equity ledger and turnover.

    Weights formed at close[t] are lagged by ``1 + execution_delay_bars`` bars
    so they are only held against the ``open[t+1+delay] -> open[t+2+delay]``
    return, matching the frozen target-weight execution convention. Each bar's net return is
    ``sum(w_lagged * open-to-open) - turnover * round_trip_cost_rate() -
    sum(w_lagged * bar_funding)`` where turnover is the row sum of absolute
    lagged-weight changes. Returns the strictly-positive equity ledger and the
    per-bar turnover series, both sharing the input index.
    """
    lagged, o2o, f, index = _xs_composite_inputs(weights, opens, bar_funding, spec)
    net_returns, turnover = _ledger_pnl(
        lagged, o2o, f, spec.round_trip_cost_rate()
    )

    equity_values = _INITIAL_EQUITY * np.cumprod(1.0 + net_returns)
    if not np.isfinite(equity_values).all():
        raise DataIntegrityError("xs composite equity became non-finite")
    if (equity_values <= 0.0).any():
        raise DataIntegrityError("xs composite equity would reach zero")

    equity = pd.Series(equity_values, index=index, name="equity", dtype=np.float64)
    turnover_series = pd.Series(turnover, index=index, name="turnover", dtype=np.float64)
    return equity, turnover_series


def run_xs_composite_ledger_multi_tier(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    base_spec: XsCompositeSpec,
    cost_rates: Sequence[float],
) -> list[tuple[pd.Series, pd.Series]]:
    """Single-pass multi-tier composite ledger sharing the array construction.

    Builds ``(lagged, o2o, funding)`` ONCE via ``_xs_composite_inputs`` (the
    identical construction and index/column validations as
    ``run_xs_composite_ledger``), computes the rate-independent ledger
    components ONCE via ``_ledger_components``, then applies the frozen net
    expression ``book_return - turnover * cost_rate - funding_charge`` per
    rate. Each returned ``(equity, turnover)`` pair is bit-identical to calling
    ``run_xs_composite_ledger`` with a spec whose ``round_trip_cost_rate()``
    equals that rate (same arrays, same expression order, same equity
    finite/positive validations and messages). Raises ``ValueError`` on an empty
    ``cost_rates`` or any negative rate.
    """
    if not cost_rates:
        raise ValueError("cost_rates must not be empty")
    if any(rate < 0.0 for rate in cost_rates):
        raise ValueError(f"cost_rates must be >= 0, got {cost_rates}")

    lagged, o2o, f, index = _xs_composite_inputs(weights, opens, bar_funding, base_spec)
    book_return, funding_charge, turnover = _ledger_components(lagged, o2o, f)

    results: list[tuple[pd.Series, pd.Series]] = []
    for cost_rate in cost_rates:
        net_returns = book_return - turnover * cost_rate - funding_charge
        equity_values = _INITIAL_EQUITY * np.cumprod(1.0 + net_returns)
        if not np.isfinite(equity_values).all():
            raise DataIntegrityError("xs composite equity became non-finite")
        if (equity_values <= 0.0).any():
            raise DataIntegrityError("xs composite equity would reach zero")
        equity = pd.Series(equity_values, index=index, name="equity", dtype=np.float64)
        turnover_series = pd.Series(turnover, index=index, name="turnover", dtype=np.float64)
        results.append((equity, turnover_series))
    return results


def _check_contract() -> None:
    """Executable assertions locking the frozen composite-ledger surface at import."""
    from inspect import signature

    assert list(signature(run_xs_composite_ledger).parameters) == [
        "weights", "opens", "bar_funding", "spec",
    ]
    assert list(signature(run_xs_composite_ledger_multi_tier).parameters) == [
        "weights", "opens", "bar_funding", "base_spec", "cost_rates",
    ]
    spec = XsCompositeSpec()
    assert (spec.halflife_bars, spec.no_trade_band, spec.execution_delay_bars) == (6, 0.05, 1)
    assert abs(spec.round_trip_cost_rate() - 0.0008) < 1e-12


_check_contract()
