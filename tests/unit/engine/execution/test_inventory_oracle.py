"""Independent-ledger oracle contract: streamed accumulator ledger vs single-panel recomputation.

The oracle in ``src.mhs/execution/ledger.py`` re-derives the six ledger series by a different
arithmetic path (per-symbol cumulative sums of fill deltas) than the causal window accumulator.
Agreement at 1e-12 is therefore evidence about window splitting, carry, and booking -- it only
holds when the oracle is told the two facts the accumulator records but a bare fill stream cannot
express: which bars had known funding, and that a fill's timestamp is a bar *availability* stamp.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.engine.execution import (
    ExecutionReplayWindow,
    ExecutionSpec,
    replay_execution_windows,
    simulated_inventory_ledger,
)

from tests.unit.engine.test_execution import _partition_windows

SPEC = ExecutionSpec()
# Extra recorded columns (decision-time pre_trade_equity and friends) must be ignored by the oracle.
FILL_COLUMNS = ["timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"]
LEDGER_SERIES = (
    "equity",
    "net_returns",
    "mark_to_market_pnl",
    "funding_charge",
    "fee_charge",
    "fill_turnover",
)
BOUNDS = (
    "OHLCV_STRICT_PROXY",
    "OHLCV_IMMEDIATE_TAKER",
    "OHLCV_LADDERED_PROXY",
    "OHLCV_PEG_CHASE_PROXY",
)
UNKNOWN_FIRST = 600
UNKNOWN_LAST = 624


def _workload(
    days: int = 12, n_symbols: int = 6, seed: int = 7, funding: float = 1.0e-5
) -> dict[str, object]:
    """Deterministic in-memory panel: 5m grid, 6h decisions, marks distinct from closes."""
    grid = pd.date_range("2021-01-01", periods=days * 24 * 12, freq="5min", tz="UTC")
    symbols = [f"SYM{i:03d}USDT" for i in range(n_symbols)]
    rng = np.random.default_rng(seed)
    closes = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(grid)))) for s in symbols},
        index=grid,
    )
    decision_grid = pd.date_range("2021-01-01", periods=days * 4, freq="6h", tz="UTC")
    weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
    rng_w = np.random.default_rng(seed + 1)
    for ts in decision_grid:
        active = rng_w.choice(symbols, size=4, replace=False)
        weights.loc[ts, active] = rng_w.uniform(0.05, 0.25, 4) * rng_w.choice([-1, 1], 4)
    return {
        "grid": grid,
        "highs": closes * 1.001,
        "lows": closes * 0.999,
        "closes": closes,
        "marks": closes * (1 + rng.normal(0, 3e-4, closes.shape)),
        "funding": pd.DataFrame(funding, index=grid, columns=symbols),
        "weights": weights,
        "signals": decision_grid + pd.Timedelta(hours=1),
    }


def _windows(workload: dict[str, object], n_windows: int = 3) -> list[ExecutionReplayWindow]:
    return _partition_windows(
        workload["grid"],
        workload["weights"],
        workload["signals"],
        workload["highs"],
        workload["lows"],
        workload["closes"],
        workload["marks"],
        workload["funding"],
        SPEC,
        n_windows=n_windows,
    )


def _replay(
    workload: dict[str, object],
    bound: str = "OHLCV_STRICT_PROXY",
    n_windows: int = 3,
    windows: list[ExecutionReplayWindow] | None = None,
) -> object:
    return replay_execution_windows(
        windows if windows is not None else _windows(workload, n_windows), 1.0, bound, SPEC
    )


def _oracle(workload: dict[str, object], result: object, **kwargs: object) -> object:
    return simulated_inventory_ledger(
        result.simulated_fills[FILL_COLUMNS],
        workload["marks"],
        workload["funding"],
        1.0,
        result.fill_source,
        "MARK_PRICE",
        **kwargs,
    )


def _assert_ledgers_agree(streamed: object, oracle: object) -> None:
    assert streamed.equity.index.equals(oracle.equity.index)
    for field in LEDGER_SERIES:
        np.testing.assert_allclose(
            getattr(streamed, field).to_numpy(),
            getattr(oracle, field).to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
    assert oracle.primary_valid == streamed.primary_valid
    assert oracle.invalid_reasons == streamed.invalid_reasons


def _unknown_funding_panel(workload: dict[str, object]) -> pd.DataFrame:
    known = pd.DataFrame(True, index=workload["grid"], columns=workload["weights"].columns)
    known.iloc[UNKNOWN_FIRST:UNKNOWN_LAST, :] = False
    return known


def _dust_anchor(
    units: pd.DataFrame, fill_ts: pd.DatetimeIndex, column: str, run_length: int = 20
) -> int:
    """Index of a mid-run bar whose cumsum units are dust yet nonzero."""
    values = units[column].to_numpy()
    dust = (values != 0.0) & (np.abs(values) < 1e-12)
    filled = set(fill_ts)
    run = 0
    for i, is_dust in enumerate(dust):
        run = run + 1 if is_dust and units.index[i] not in filled else 0
        if run >= run_length:
            return i - run_length // 2
    raise AssertionError(f"no dust run of >= {run_length} bars for {column}")


@pytest.fixture(scope="module")
def unknown_funding_run() -> tuple[dict[str, object], object, pd.DataFrame]:
    workload = _workload()
    known = _unknown_funding_panel(workload)
    windows = [
        dataclasses.replace(w, funding_known=known.loc[w.minute_grid]) for w in _windows(workload)
    ]
    return workload, _replay(workload, windows=windows), known


@pytest.mark.parametrize(("bound", "seed"), [(b, s) for b in BOUNDS for s in (7, 11)])
def test_oracle_matches_accumulator_across_bounds(bound: str, seed: int) -> None:
    workload = _workload(seed=seed)
    result = _replay(workload, bound=bound)
    _assert_ledgers_agree(result.ledger, _oracle(workload, result))


@pytest.mark.parametrize("n_windows", [1, 2, 3, 5])
def test_oracle_matches_accumulator_across_window_splits(n_windows: int) -> None:
    """The oracle is split-free, so every split agreeing certifies carry across boundaries."""
    workload = _workload()
    result = _replay(workload, n_windows=n_windows)
    _assert_ledgers_agree(result.ledger, _oracle(workload, result))


def test_oracle_matches_through_flat_nan_mark_holes() -> None:
    """Flat inventory over a NaN mark hole is valued at zero, not reported as missing data."""
    workload = _workload()
    workload["marks"].iloc[500:520, 0] = np.nan
    workload["marks"].iloc[0:3, 1] = np.nan
    result = _replay(workload)
    _assert_ledgers_agree(result.ledger, _oracle(workload, result))


def test_held_nan_mark_hole_invalidates_both() -> None:
    workload = _workload()
    baseline = _replay(workload)
    units = (
        baseline.simulated_fills.pivot_table(
            index="timestamp", columns="symbol", values="quantity_delta", aggfunc="sum"
        )
        .reindex(workload["grid"])
        .fillna(0.0)
        .cumsum()
    )
    symbol = units.columns[0]
    held = np.flatnonzero(units[symbol].abs().to_numpy() > 1e-6)
    assert len(held) > 1
    hole = int(held[len(held) // 2])
    marks = workload["marks"].copy()
    marks.iloc[hole : hole + 4, marks.columns.get_loc(symbol)] = np.nan
    workload["marks"] = marks

    result = _replay(workload)
    oracle = _oracle(workload, result)
    _assert_ledgers_agree(result.ledger, oracle)
    assert oracle.primary_valid is False
    assert oracle.invalid_reasons == ("MISSING_DATA",)
    first_mark_gap = next(g for g in oracle.data_gaps if g.code == "MISSING_HELD_MARK")
    assert (first_mark_gap.symbol, first_mark_gap.timestamp) == (symbol, workload["grid"][hole])


def test_unknown_funding_is_not_charged_and_invalidates_when_held(
    unknown_funding_run: tuple[dict[str, object], object, pd.DataFrame],
) -> None:
    workload, result, known = unknown_funding_run
    oracle = _oracle(workload, result, funding_known=known)
    _assert_ledgers_agree(result.ledger, oracle)
    bars = slice(UNKNOWN_FIRST, UNKNOWN_LAST)
    assert (oracle.funding_charge.to_numpy()[bars] == 0.0).all()
    assert (result.ledger.funding_charge.to_numpy()[bars] == 0.0).all()
    assert oracle.primary_valid is False
    assert result.ledger.primary_valid is False


def test_legacy_call_ignores_funding_knowledge(
    unknown_funding_run: tuple[dict[str, object], object, pd.DataFrame],
) -> None:
    """Without the knowledge input the oracle invents a funding charge and stays valid."""
    workload, result, _ = unknown_funding_run
    oracle = _oracle(workload, result)
    divergence = np.abs(oracle.funding_charge - result.ledger.funding_charge).to_numpy()[
        UNKNOWN_FIRST:UNKNOWN_LAST
    ]
    assert divergence.max() > 1e-6
    assert oracle.primary_valid is True
    assert result.ledger.primary_valid is False


def test_availability_stamps_map_to_bar_labels() -> None:
    """``simulated_fills.timestamp`` is the bar availability stamp, one step after the label."""
    workload = _workload()
    step = pd.Timedelta(minutes=5)
    windows = [
        dataclasses.replace(w, bar_available_at=w.minute_grid + step) for w in _windows(workload)
    ]
    result = _replay(workload, windows=windows)

    _assert_ledgers_agree(
        result.ledger, _oracle(workload, result, bar_available_at=workload["grid"] + step)
    )
    unmapped = _oracle(workload, result)
    divergence = np.abs(result.ledger.fill_turnover.to_numpy() - unmapped.fill_turnover.to_numpy())
    assert divergence.max() > 1e-3


def test_off_grid_availability_stamp_fails_closed(
    knowledge_baseline: tuple[dict[str, object], object]
) -> None:
    workload, result = knowledge_baseline
    with pytest.raises(DataIntegrityError, match="bar availability grid"):
        _oracle(workload, result, bar_available_at=workload["grid"] + pd.Timedelta(minutes=7))


def test_dust_tail_is_not_held() -> None:
    """Sub-epsilon cumsum dust left by a full close is flat, so a NaN mark hole over it is benign."""
    workload = _workload(days=20, seed=11)
    weights = workload["weights"].copy()
    weights.iloc[1::2] = 0.0
    workload["weights"] = weights

    baseline = _replay(workload, bound="OHLCV_LADDERED_PROXY")
    probe = _oracle(workload, baseline, retain_simulated_units=True)
    symbol = probe.simulated_units.columns[0]
    fill_ts = pd.DatetimeIndex(baseline.simulated_fills["timestamp"])
    anchor = _dust_anchor(probe.simulated_units, fill_ts, symbol)
    units = probe.simulated_units[symbol].iloc[anchor]
    assert 0.0 < abs(units) < 1e-12

    marks = workload["marks"].copy()
    marks.iloc[anchor : anchor + 3, marks.columns.get_loc(symbol)] = np.nan
    workload["marks"] = marks

    result = _replay(workload, bound="OHLCV_LADDERED_PROXY")
    oracle = _oracle(workload, result)
    _assert_ledgers_agree(result.ledger, oracle)
    assert oracle.primary_valid is True
    assert result.ledger.primary_valid is True


@pytest.fixture(scope="module")
def knowledge_baseline() -> tuple[dict[str, object], object]:
    workload = _workload()
    return workload, _replay(workload)


def _malformed_knowledge_inputs(
    workload: dict[str, object], case: str
) -> dict[str, object]:
    columns = workload["weights"].columns
    grid = workload["grid"]
    known = pd.DataFrame(True, index=grid, columns=columns)
    if case == "knowledge_shifted_index":
        return {"funding_known": known.iloc[:-1]}
    if case == "knowledge_reordered_columns":
        return {"funding_known": known[list(columns[::-1])]}
    if case == "knowledge_float_dtype":
        return {"funding_known": known.astype(float)}
    if case == "availability_not_monotonic":
        return {"bar_available_at": pd.DatetimeIndex([*grid[:-1], grid[-2]])}
    if case == "availability_wrong_length":
        return {"bar_available_at": grid[:-1]}
    if case == "availability_tz_naive":
        return {"bar_available_at": grid.tz_localize(None)}
    if case == "availability_precedes_label":
        return {"bar_available_at": grid - pd.Timedelta(minutes=1)}
    raise AssertionError(f"unregistered malformed-input case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "knowledge_shifted_index",
        "knowledge_reordered_columns",
        "knowledge_float_dtype",
        "availability_not_monotonic",
        "availability_wrong_length",
        "availability_tz_naive",
        "availability_precedes_label",
    ],
)
def test_malformed_knowledge_inputs_fail_closed(
    knowledge_baseline: tuple[dict[str, object], object], case: str
) -> None:
    workload, result = knowledge_baseline
    with pytest.raises(DataIntegrityError):
        _oracle(workload, result, **_malformed_knowledge_inputs(workload, case))