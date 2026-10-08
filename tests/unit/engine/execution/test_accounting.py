

def test_causal_state_marks_unrealized_pnl_once() -> None:
    import numpy as np
    from src.engine.execution.accounting import CausalPortfolioState
    state = CausalPortfolioState(cash=1000.0, units=np.zeros(1), last_marks=np.array([100.0]), last_event_ns=None)
    state.apply_fill(symbol_index=0, quantity_delta=10.0, fill_price=100.0, fee_bps=0.0)
    state.advance_to(event_ns=1, marks=np.array([110.0]), funding_rates=np.zeros(1), funding_known=np.ones(1, dtype=bool))
    assert state.cash == 0.0
    assert state.equity() == 1100.0


def test_causal_state_does_not_charge_pre_entry_funding() -> None:
    import numpy as np
    from src.engine.execution.accounting import CausalPortfolioState
    state = CausalPortfolioState(cash=1000.0, units=np.zeros(1), last_marks=np.array([100.0]), last_event_ns=None)
    charged = state.advance_to(event_ns=1, marks=np.array([100.0]), funding_rates=np.array([0.01]), funding_known=np.ones(1, dtype=bool))
    state.apply_fill(symbol_index=0, quantity_delta=10.0, fill_price=100.0, fee_bps=0.0)
    assert charged == 0.0
    assert state.equity() == 1000.0


def test_causal_state_rejects_unknown_funding_for_held_position() -> None:
    import numpy as np
    import pytest
    from src.common.errors import DataIntegrityError
    from src.engine.execution.accounting import CausalPortfolioState
    state = CausalPortfolioState(cash=0.0, units=np.array([10.0]), last_marks=np.array([100.0]), last_event_ns=0)
    with pytest.raises(DataIntegrityError, match='funding'):
        state.advance_to(event_ns=1, marks=np.array([100.0]), funding_rates=np.array([0.0]), funding_known=np.zeros(1, dtype=bool))


def test_causal_state_rejects_decreasing_event_time() -> None:
    import numpy as np
    import pytest
    from src.common.errors import DataIntegrityError
    from src.engine.execution.accounting import CausalPortfolioState
    state = CausalPortfolioState(cash=1000.0, units=np.zeros(1), last_marks=np.array([100.0]), last_event_ns=5)
    with pytest.raises(DataIntegrityError, match='monotonic'):
        state.advance_to(event_ns=4, marks=np.array([100.0]), funding_rates=np.zeros(1), funding_known=np.ones(1, dtype=bool))


def test_reconcile_causal_state_rejects_divergence() -> None:
    import numpy as np
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.engine.execution.accounting import CausalPortfolioState, reconcile_causal_state
    from src.engine.execution.contracts import SimulatedInventoryLedgerResult
    index = pd.date_range('2025-01-01', periods=2, freq='3min', tz='UTC')
    ledger = SimulatedInventoryLedgerResult(
        equity=pd.Series([1000.0, 1000.0], index=index, dtype='float64'),
        net_returns=pd.Series([0.0], index=index[1:], dtype='float64'),
        simulated_units=None,
        mark_to_market_pnl=pd.Series([0.0, 0.0], index=index, dtype='float64'),
        funding_charge=pd.Series([0.0, 0.0], index=index, dtype='float64'),
        fee_charge=pd.Series([0.0, 0.0], index=index, dtype='float64'),
        fill_turnover=pd.Series([0.0, 0.0], index=index, dtype='float64'),
        fill_source='OHLCV_IMMEDIATE_TAKER',
        mark_source='OHLCV_CLOSE_FALLBACK',
        primary_valid=True,
        invalid_reasons=(),
    )
    state = CausalPortfolioState(cash=0.0, units=np.zeros(1), last_marks=np.array([100.0]), last_event_ns=None)
    with pytest.raises(DataIntegrityError, match='diverged'):
        reconcile_causal_state(state, ledger)


def test_causal_state_treats_dust_units_as_flat_for_unknown_funding() -> None:
    import numpy as np
    import pytest
    from src.common.errors import DataIntegrityError
    from src.engine.execution.accounting import QTY_EPS, CausalPortfolioState
    assert QTY_EPS == 1e-12
    # Given: dust inventory below QTY_EPS and an unknown-funding bar
    state = CausalPortfolioState(cash=1.0, units=np.array([1e-15]), last_marks=np.array([np.nan]), last_event_ns=None)
    # When: must not raise (dust is flat)
    charged = state.advance_to(event_ns=1, marks=np.array([100.0]), funding_rates=np.array([0.0]), funding_known=np.array([False]))
    assert charged == 0.0
    # Then: a real position still fails closed
    held = CausalPortfolioState(cash=1.0, units=np.array([1e-6]), last_marks=np.array([np.nan]), last_event_ns=None)
    with pytest.raises(DataIntegrityError, match='unknown funding'):
        held.advance_to(event_ns=1, marks=np.array([100.0]), funding_rates=np.array([0.0]), funding_known=np.array([False]))


def _scalar_reference_settle(state, bar_ns, columns, marks, funding_rates, funding_known, fills, n_cols):  # noqa: ANN001, ANN202
    import numpy as np

    queue = sorted(fills, key=lambda e: e[0])
    qi = 0
    p0_ns = int(bar_ns[0])
    while qi < len(queue) and queue[qi][0] < p0_ns:
        state.units[int(queue[qi][1])] += float(queue[qi][2])
        qi += 1
    gmarks = np.full(n_cols, np.nan, dtype="float64")
    grates = np.zeros(n_cols, dtype="float64")
    gknown = np.ones(n_cols, dtype=bool)
    all_known = np.ones(n_cols, dtype=bool)
    gap_rows: list[tuple[int, int]] = []
    charges: list[float] = []
    for b in range(len(bar_ns)):
        bns = int(bar_ns[b])
        gmarks[columns] = marks[b]
        grates[columns] = funding_rates[b]
        gknown[columns] = funding_known[b]
        held = np.abs(state.units) >= 1e-12
        unpriceable = ~np.isfinite(gmarks) & (grates != 0.0)
        held_unknown = held & (~gknown | unpriceable)
        if bool(held_unknown.any()):
            gap_rows.append((b, int(np.flatnonzero(held_unknown)[0])))
            charged = state.advance_to(
                event_ns=bns, marks=gmarks,
                funding_rates=np.where(gknown, grates, 0.0), funding_known=all_known,
            )
        else:
            charged = state.advance_to(event_ns=bns, marks=gmarks, funding_rates=grates, funding_known=gknown)
        charges.append(charged)
        while qi < len(queue) and queue[qi][0] == bns:
            state.apply_fill(
                symbol_index=int(queue[qi][1]), quantity_delta=float(queue[qi][2]),
                fill_price=float(queue[qi][3]), fee_bps=float(queue[qi][4]),
            )
            qi += 1
    return np.asarray(charges, dtype="float64"), gap_rows


def _random_window_case(rng, n_cols):  # noqa: ANN001, ANN202
    import numpy as np

    cols = np.arange(n_cols, dtype=np.intp)
    nl = int(rng.integers(1, n_cols + 1)) if n_cols else 0
    roster = np.sort(rng.choice(cols, size=nl, replace=False)).astype(np.intp) if nl else np.empty(0, dtype=np.intp)
    nb = int(rng.integers(4, 12))
    start = int(rng.integers(1_000_000_000_000, 2_000_000_000_000))
    bar_ns = (start + np.arange(nb, dtype=np.int64) * 180_000_000_000).astype(np.int64)
    marks = 100.0 + rng.normal(0.0, 2.0, (nb, max(nl, 1)))[:, :nl] if nl else np.zeros((nb, 0))
    if nl:
        marks[rng.random((nb, nl)) < 0.3] = np.nan
        if rng.random() < 0.3:
            marks[:, int(rng.integers(0, nl))] = np.nan
    rates = rng.normal(0.0, 1e-4, (nb, nl)) if nl else np.zeros((nb, 0))
    known = rng.random((nb, nl) if nl else (nb, 0)) > 0.12
    units0 = np.zeros(n_cols, dtype="float64")
    if nl:
        for c in roster.tolist():
            units0[int(c)] = float(rng.normal(0.0, 1.5))
        for c in set(cols.tolist()) - set(roster.tolist()):
            if rng.random() < 0.5:
                units0[int(c)] = float(rng.uniform(-5e-13, 5e-13))
    last_marks = np.where(rng.random(n_cols) < 0.5, 100.0 + rng.normal(0, 1, n_cols), np.nan)
    fills: list[tuple[int, int, float, float, float]] = []
    if nl:
        c0 = int(roster[0])
        fills.append((int(bar_ns[0]) - 180_000_000_000, c0, 0.25, 100.0, 8.0))
        fills.append((int(bar_ns[0]), c0, 0.5, 101.0, 8.0))
        fills.append((int(bar_ns[-1]), int(roster[-1]), -0.3, 99.0, 5.0))
        mid = int(bar_ns[nb // 2])
        fills.extend([(mid, c0, 0.1, 100.0, 8.0), (mid, c0, -0.05, 100.5, 8.0), (mid, c0, 0.02, 99.5, 8.0)])
        if nl >= 2:
            fills.extend([(mid, int(roster[0]), 0.07, 100.0, 8.0), (mid, int(roster[1]), -0.04, 101.0, 8.0)])
        fills.append((int(bar_ns[0]), c0, -float(units0[c0]) - 0.25, 100.0, 0.0))
        fills.append((int(bar_ns[-1]), int(roster[-1]), 1e-13, 100.0, 0.0))
    return bar_ns, roster, marks, rates, known, units0, last_marks, fills


def test_settle_window_matches_scalar_reference_bit_for_bit() -> None:
    import numpy as np

    from src.engine.execution.accounting import CausalPortfolioState

    rng = np.random.default_rng(20261005)
    widths = [1, 7, 8, 9, 17, 33, 130]
    for i in range(200):
        n_cols = widths[i % len(widths)]
        bar_ns, roster, marks, rates, known, units0, last_marks, fills = _random_window_case(rng, n_cols)
        a = CausalPortfolioState(cash=1000.0, units=units0.copy(), last_marks=last_marks.copy(), last_event_ns=None)
        b = CausalPortfolioState(cash=1000.0, units=units0.copy(), last_marks=last_marks.copy(), last_event_ns=None)
        n = len(fills)
        out = a.settle_window(
            bar_ns=bar_ns, columns=roster, marks=marks, funding_rates=rates, funding_known=known,
            fill_ns=np.asarray([f[0] for f in fills], dtype="int64"),
            fill_columns=np.asarray([f[1] for f in fills], dtype=np.intp),
            fill_quantities=np.asarray([f[2] for f in fills], dtype="float64"),
            fill_prices=np.asarray([f[3] for f in fills], dtype="float64"),
            fill_fee_bps=np.asarray([f[4] for f in fills], dtype="float64"),
        )
        ref_charges, ref_gaps = _scalar_reference_settle(
            b, bar_ns, roster, marks, rates, known, fills, n_cols,
        )
        assert a.cash == b.cash
        np.testing.assert_array_equal(a.units, b.units)
        np.testing.assert_array_equal(a.last_marks, b.last_marks)
        assert a.last_event_ns == b.last_event_ns
        np.testing.assert_array_equal(out.funding_charged, ref_charges)
        assert out.gap_bar_offsets.tolist() == [g[0] for g in ref_gaps]
        assert out.gap_witness_columns.tolist() == [g[1] for g in ref_gaps]
        assert out.fills_applied + out.backlog_applied == n


def test_per_bar_charge_is_canonical_left_fold() -> None:
    import numpy as np

    from src.engine.execution.accounting import CausalPortfolioState

    rng = np.random.default_rng(11)
    for _ in range(200):
        units = rng.normal(0, 2, 9)
        marks = 100.0 + rng.normal(0, 1, 9)
        rates = rng.normal(0, 1e-3, 9)
        terms = [(float(r) * float(u)) * float(m) for r, u, m in zip(rates, units, marks, strict=True)]
        fold = 0.0
        for x in terms:
            fold += x
        if fold != float(np.sum(np.asarray(terms))):
            break
    else:
        raise AssertionError("need a case where sum differs from left fold")
    state = CausalPortfolioState(cash=0.0, units=units.copy(), last_marks=np.full(9, np.nan), last_event_ns=None)
    solo = state.advance_to(
        event_ns=7, marks=marks, funding_rates=rates, funding_known=np.ones(9, dtype=bool),
    )
    assert solo == fold
    assert solo != float(np.sum(np.asarray(terms)))
    vec = CausalPortfolioState(cash=0.0, units=units.copy(), last_marks=np.full(9, np.nan), last_event_ns=None)
    out = vec.settle_window(
        bar_ns=np.asarray([7], dtype="int64"), columns=np.arange(9, dtype=np.intp),
        marks=marks.reshape(1, 9), funding_rates=rates.reshape(1, 9),
        funding_known=np.ones((1, 9), dtype=bool),
        fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
        fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
        fill_fee_bps=np.empty(0, dtype="float64"),
    )
    assert float(out.funding_charged[0]) == fold
    assert float(out.funding_charged[0]) != float(np.sum(np.asarray(terms)))


def test_roster_width_equals_canonical_width() -> None:
    import numpy as np

    from src.engine.execution.accounting import CausalPortfolioState

    rng = np.random.default_rng(3)
    n_cols, nb, nl = 9, 6, 4
    roster = np.sort(rng.choice(n_cols, size=nl, replace=False)).astype(np.intp)
    bar_ns = (1_000_000_000 + np.arange(nb, dtype=np.int64) * 60_000_000_000).astype(np.int64)
    marks = 100.0 + rng.normal(0, 1, (nb, nl))
    rates = rng.normal(0, 1e-4, (nb, nl))
    known = rng.random((nb, nl)) > 0.2
    units0 = np.zeros(n_cols)
    units0[roster] = rng.normal(0, 1, nl)
    lm = np.full(n_cols, 100.0)
    a = CausalPortfolioState(cash=500.0, units=units0.copy(), last_marks=lm.copy(), last_event_ns=None)
    b = CausalPortfolioState(cash=500.0, units=units0.copy(), last_marks=lm.copy(), last_event_ns=None)
    out_a = a.settle_window(
        bar_ns=bar_ns, columns=roster, marks=marks, funding_rates=rates, funding_known=known,
        fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
        fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
        fill_fee_bps=np.empty(0, dtype="float64"),
    )
    full_marks = np.full((nb, n_cols), np.nan)
    full_rates = np.zeros((nb, n_cols))
    full_known = np.ones((nb, n_cols), dtype=bool)
    full_marks[:, roster] = np.where(np.isfinite(marks), marks, np.nan)
    full_rates[:, roster] = rates
    full_known[:, roster] = known
    out_b = b.settle_window(
        bar_ns=bar_ns, columns=np.arange(n_cols, dtype=np.intp), marks=full_marks,
        funding_rates=full_rates, funding_known=full_known,
        fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
        fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
        fill_fee_bps=np.empty(0, dtype="float64"),
    )
    assert a.cash == b.cash
    np.testing.assert_array_equal(a.units, b.units)
    np.testing.assert_array_equal(a.last_marks, b.last_marks)
    np.testing.assert_array_equal(out_a.funding_charged, out_b.funding_charged)
    assert out_a.gap_bar_offsets.tolist() == out_b.gap_bar_offsets.tolist()


def test_held_unknown_reports_gap_and_charges_known_only() -> None:
    import numpy as np

    from src.engine.execution.accounting import CausalPortfolioState

    state = CausalPortfolioState(cash=100.0, units=np.array([1.0, 2.0]), last_marks=np.array([np.nan, np.nan]), last_event_ns=None)
    bar_ns = np.asarray([10, 20, 30, 40], dtype="int64")
    marks = np.full((4, 2), 100.0)
    rates = np.full((4, 2), 1e-4)
    rates[3, 1] = 5e-4
    known = np.ones((4, 2), dtype=bool)
    known[3, 1] = False
    out = state.settle_window(
        bar_ns=bar_ns, columns=np.asarray([0, 1], dtype=np.intp), marks=marks,
        funding_rates=rates, funding_known=known,
        fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
        fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
        fill_fee_bps=np.empty(0, dtype="float64"),
    )
    assert out.gap_bar_offsets.tolist() == [3]
    assert out.gap_witness_columns.tolist() == [1]
    assert float(out.funding_charged[3]) == (1e-4 * 1.0) * 100.0


def test_flat_unknown_keeps_rate_and_is_not_gap() -> None:
    import numpy as np

    from src.engine.execution.accounting import CausalPortfolioState

    state = CausalPortfolioState(cash=50.0, units=np.array([1e-15, 1.0]), last_marks=np.array([np.nan, np.nan]), last_event_ns=None)
    bar_ns = np.asarray([10], dtype="int64")
    marks = np.asarray([[100.0, 100.0]])
    rates = np.asarray([[1e-4, 2e-4]])
    known = np.asarray([[False, True]])
    out = state.settle_window(
        bar_ns=bar_ns, columns=np.asarray([0, 1], dtype=np.intp), marks=marks,
        funding_rates=rates, funding_known=known,
        fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
        fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
        fill_fee_bps=np.empty(0, dtype="float64"),
    )
    assert out.gap_bar_offsets.size == 0
    dust_term = (1e-4 * 1e-15) * 100.0
    main_term = (2e-4 * 1.0) * 100.0
    assert float(out.funding_charged[0]) == dust_term + main_term


def test_non_monotone_bars_fail_closed_without_mutation() -> None:
    import numpy as np
    import pytest

    from src.common.errors import DataIntegrityError
    from src.engine.execution.accounting import CausalPortfolioState

    for bars, last in (
        (np.asarray([10, 20, 30], dtype="int64"), 15),
        (np.asarray([10, 10, 30], dtype="int64"), None),
    ):
        state = CausalPortfolioState(cash=9.0, units=np.array([1.0]), last_marks=np.array([5.0]), last_event_ns=last)
        snap = (state.cash, state.units.copy(), state.last_marks.copy(), state.last_event_ns)
        with pytest.raises(DataIntegrityError, match=r"monoton|increasing"):
            state.settle_window(
                bar_ns=bars, columns=np.asarray([0], dtype=np.intp),
                marks=np.full((3, 1), 100.0), funding_rates=np.zeros((3, 1)),
                funding_known=np.ones((3, 1), dtype=bool),
                fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
                fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
                fill_fee_bps=np.empty(0, dtype="float64"),
            )
        assert state.cash == snap[0]
        np.testing.assert_array_equal(state.units, snap[1])
        np.testing.assert_array_equal(state.last_marks, snap[2])
        assert state.last_event_ns == snap[3]


def test_off_grid_or_late_fill_fails_closed_without_mutation() -> None:
    import numpy as np
    import pytest

    from src.common.errors import DataIntegrityError
    from src.engine.execution.accounting import CausalPortfolioState

    bars = np.asarray([10, 20, 30], dtype="int64")
    for bad_ns in (21, 31, 10_000):
        state = CausalPortfolioState(cash=9.0, units=np.array([0.0]), last_marks=np.array([5.0]), last_event_ns=None)
        snap = (state.cash, state.units.copy(), state.last_marks.copy(), state.last_event_ns)
        with pytest.raises(DataIntegrityError):
            state.settle_window(
                bar_ns=bars, columns=np.asarray([0], dtype=np.intp),
                marks=np.full((3, 1), 100.0), funding_rates=np.zeros((3, 1)),
                funding_known=np.ones((3, 1), dtype=bool),
                fill_ns=np.asarray([bad_ns], dtype="int64"), fill_columns=np.asarray([0], dtype=np.intp),
                fill_quantities=np.asarray([1.0]), fill_prices=np.asarray([100.0]), fill_fee_bps=np.asarray([8.0]),
            )
        assert state.cash == snap[0]
        np.testing.assert_array_equal(state.units, snap[1])


def test_unsorted_roster_folds_and_witnesses_in_canonical_order() -> None:
    import numpy as np

    from src.engine.execution.accounting import CausalPortfolioState

    n_cols = 12
    columns = np.asarray([5, 2, 9], dtype=np.intp)
    bar_ns = np.asarray([10, 20], dtype="int64")
    marks = np.full((2, 3), 100.0)
    rates = np.full((2, 3), 1e-4)
    known = np.ones((2, 3), dtype=bool)
    known[1, 1] = False
    known[1, 2] = False
    units0 = np.zeros(n_cols)
    units0[2] = 1.0
    units0[9] = 2.0
    a = CausalPortfolioState(cash=0.0, units=units0.copy(), last_marks=np.full(n_cols, np.nan), last_event_ns=None)
    b = CausalPortfolioState(cash=0.0, units=units0.copy(), last_marks=np.full(n_cols, np.nan), last_event_ns=None)
    out = a.settle_window(
        bar_ns=bar_ns, columns=columns, marks=marks, funding_rates=rates, funding_known=known,
        fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
        fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
        fill_fee_bps=np.empty(0, dtype="float64"),
    )
    ref_charges, ref_gaps = _scalar_reference_settle(b, bar_ns, columns, marks, rates, known, [], n_cols)
    assert a.cash == b.cash
    np.testing.assert_array_equal(a.units, b.units)
    np.testing.assert_array_equal(out.funding_charged, ref_charges)
    assert out.gap_bar_offsets.tolist() == [1]
    assert out.gap_witness_columns.tolist() == [2]


def test_empty_roster_settles_free_bars() -> None:
    import numpy as np

    from src.engine.execution.accounting import CausalPortfolioState

    state = CausalPortfolioState(cash=77.0, units=np.zeros(3), last_marks=np.array([1.0, 2.0, 3.0]), last_event_ns=None)
    out = state.settle_window(
        bar_ns=np.asarray([5, 6, 7], dtype="int64"), columns=np.empty(0, dtype=np.intp),
        marks=np.zeros((3, 0)), funding_rates=np.zeros((3, 0)), funding_known=np.zeros((3, 0), dtype=bool),
        fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
        fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
        fill_fee_bps=np.empty(0, dtype="float64"),
    )
    np.testing.assert_array_equal(out.funding_charged, np.zeros(3))
    assert state.cash == 77.0
    assert state.last_event_ns == 7
    np.testing.assert_array_equal(state.last_marks, np.array([1.0, 2.0, 3.0]))


def test_settle_window_fail_closed_branches() -> None:
    import numpy as np
    import pytest

    from src.common.errors import DataIntegrityError
    from src.engine.execution.accounting import CausalPortfolioState

    def _base(n_cols=2, nb=3):
        return CausalPortfolioState(
            cash=10.0, units=np.zeros(n_cols), last_marks=np.full(n_cols, 1.0), last_event_ns=None,
        )

    with pytest.raises(DataIntegrityError):
        _base().settle_window(
            bar_ns=np.empty(0, dtype="int64"), columns=np.asarray([0], dtype=np.intp),
            marks=np.zeros((1, 1)), funding_rates=np.zeros((1, 1)), funding_known=np.ones((1, 1), dtype=bool),
            fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
            fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
            fill_fee_bps=np.empty(0, dtype="float64"),
        )
    with pytest.raises(DataIntegrityError):
        _base().settle_window(
            bar_ns=np.asarray([1, 2, 3], dtype="int64"), columns=np.asarray([[0]]),
            marks=np.zeros((3, 1)), funding_rates=np.zeros((3, 1)), funding_known=np.ones((3, 1), dtype=bool),
            fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
            fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
            fill_fee_bps=np.empty(0, dtype="float64"),
        )
    with pytest.raises(DataIntegrityError):
        _base().settle_window(
            bar_ns=np.asarray([1, 2, 3], dtype="int64"), columns=np.asarray([9], dtype=np.intp),
            marks=np.zeros((3, 1)), funding_rates=np.zeros((3, 1)), funding_known=np.ones((3, 1), dtype=bool),
            fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
            fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
            fill_fee_bps=np.empty(0, dtype="float64"),
        )
    with pytest.raises(DataIntegrityError):
        _base().settle_window(
            bar_ns=np.asarray([1, 2, 3], dtype="int64"), columns=np.asarray([0, 0], dtype=np.intp),
            marks=np.zeros((3, 2)), funding_rates=np.zeros((3, 2)), funding_known=np.ones((3, 2), dtype=bool),
            fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
            fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
            fill_fee_bps=np.empty(0, dtype="float64"),
        )
    with pytest.raises(DataIntegrityError):
        _base().settle_window(
            bar_ns=np.asarray([1, 2, 3], dtype="int64"), columns=np.asarray([0], dtype=np.intp),
            marks=np.zeros((2, 1)), funding_rates=np.zeros((3, 1)), funding_known=np.ones((3, 1), dtype=bool),
            fill_ns=np.empty(0, dtype="int64"), fill_columns=np.empty(0, dtype=np.intp),
            fill_quantities=np.empty(0, dtype="float64"), fill_prices=np.empty(0, dtype="float64"),
            fill_fee_bps=np.empty(0, dtype="float64"),
        )
    with pytest.raises(DataIntegrityError):
        _base().settle_window(
            bar_ns=np.asarray([1, 2, 3], dtype="int64"), columns=np.asarray([0], dtype=np.intp),
            marks=np.zeros((3, 1)), funding_rates=np.zeros((3, 1)), funding_known=np.ones((3, 1), dtype=bool),
            fill_ns=np.asarray([1, 2], dtype="int64"), fill_columns=np.asarray([0], dtype=np.intp),
            fill_quantities=np.asarray([1.0]), fill_prices=np.asarray([1.0]), fill_fee_bps=np.asarray([1.0]),
        )
    with pytest.raises(DataIntegrityError):
        _base().settle_window(
            bar_ns=np.asarray([1, 2, 3], dtype="int64"), columns=np.asarray([0], dtype=np.intp),
            marks=np.zeros((3, 1)), funding_rates=np.zeros((3, 1)), funding_known=np.ones((3, 1), dtype=bool),
            fill_ns=np.asarray([2], dtype="int64"), fill_columns=np.asarray([7], dtype=np.intp),
            fill_quantities=np.asarray([1.0]), fill_prices=np.asarray([1.0]), fill_fee_bps=np.asarray([1.0]),
        )
    with pytest.raises(DataIntegrityError):
        _base().settle_window(
            bar_ns=np.asarray([1, 2, 3], dtype="int64"), columns=np.asarray([0], dtype=np.intp),
            marks=np.zeros((3, 1)), funding_rates=np.zeros((3, 1)), funding_known=np.ones((3, 1), dtype=bool),
            fill_ns=np.asarray([2], dtype="int64"), fill_columns=np.asarray([1], dtype=np.intp),
            fill_quantities=np.asarray([1.0]), fill_prices=np.asarray([1.0]), fill_fee_bps=np.asarray([1.0]),
        )
    with pytest.raises(DataIntegrityError):
        _base().settle_window(
            bar_ns=np.asarray([10, 20], dtype="int64"), columns=np.asarray([0], dtype=np.intp),
            marks=np.zeros((2, 1)), funding_rates=np.zeros((2, 1)), funding_known=np.ones((2, 1), dtype=bool),
            fill_ns=np.asarray([1], dtype="int64"), fill_columns=np.asarray([9], dtype=np.intp),
            fill_quantities=np.asarray([1.0]), fill_prices=np.asarray([1.0]), fill_fee_bps=np.asarray([1.0]),
        )
    with pytest.raises(DataIntegrityError):
        CausalPortfolioState(
            cash=10.0, units=np.zeros(2), last_marks=np.full(2, 1.0), last_event_ns=None,
        ).settle_window(
            bar_ns=np.asarray([1, 2], dtype="int64"), columns=np.empty(0, dtype=np.intp),
            marks=np.zeros((2, 0)), funding_rates=np.zeros((2, 0)), funding_known=np.zeros((2, 0), dtype=bool),
            fill_ns=np.asarray([1], dtype="int64"), fill_columns=np.asarray([0], dtype=np.intp),
            fill_quantities=np.asarray([1.0]), fill_prices=np.asarray([1.0]), fill_fee_bps=np.asarray([1.0]),
        )


