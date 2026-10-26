"""Queue-aware tape fill model invariants: pure resting-order fill paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.engine.execution.tape_fill import (
    RestingOrder,
    TapeFillResult,
    TapeFillSpec,
    TapeTrades,
    post_only_limit,
    simulate_resting_fill,
    touch_proxy,
)

_COLUMNS = ("transact_time_ms", "agg_trade_id", "price", "quantity", "is_buyer_maker")


def _tape(rows: list[tuple[int, int, float, float, bool]], coverage: tuple[int, int] = (0, 10_000)) -> TapeTrades:
    frame = pd.DataFrame(rows, columns=list(_COLUMNS))
    return TapeTrades.from_slice_frame(frame, coverage_ms=coverage)


def _buy(
    limit: float = 100.0,
    qty: float = 5.0,
    queue: float | None = None,
    window: tuple[int, int] = (0, 10_000),
    fraction: float = 1.0,
    unknown: str = "through_only",
) -> tuple[RestingOrder, TapeFillSpec]:
    return (
        RestingOrder(
            side="BUY",
            limit_price=limit,
            quantity=qty,
            active_from_ms=window[0],
            active_until_ms=window[1],
            queue_ahead_qty=queue,
        ),
        TapeFillSpec(queue_ahead_fraction=fraction, unknown_queue=unknown),  # type: ignore[arg-type]
    )


def test_trade_through_fills_full_remainder_at_through_instant() -> None:
    tape = _tape([(100, 1, 100.0, 2.0, True), (200, 2, 99.9, 1.0, True)])
    order, spec = _buy()
    result = simulate_resting_fill(order, tape, spec)
    assert result.fills == ((200, 5.0),)
    assert result.filled_qty == 5.0
    assert result.through_ms == result.completed_ms == 200
    assert result.first_fill_ms == 200


def test_buyer_aggressor_print_below_limit_still_counts_as_through() -> None:
    tape = _tape([(100, 1, 99.95, 1.0, False)])
    order, spec = _buy()
    result = simulate_resting_fill(order, tape, spec)
    assert result.filled_qty == 5.0
    assert result.through_ms == result.completed_ms == 100


def test_at_price_volume_fills_only_after_queue_ahead_consumed() -> None:
    tape = _tape([(10, 1, 100.0, 2.0, True), (20, 2, 100.0, 3.0, True), (30, 3, 100.0, 5.0, True)])
    order, spec = _buy(qty=3.0, queue=4.0)
    result = simulate_resting_fill(order, tape, spec)
    assert result.fills == ((20, 1.0), (30, 2.0))
    assert result.filled_qty == 3.0
    assert result.completed_ms == 30
    assert result.through_ms is None


def test_queue_ahead_fraction_scales_the_queue() -> None:
    tape = _tape([(10, 1, 100.0, 2.0, True), (20, 2, 100.0, 3.0, True), (30, 3, 100.0, 5.0, True)])
    order, spec = _buy(qty=3.0, queue=4.0, fraction=0.5)
    result = simulate_resting_fill(order, tape, spec)
    assert result.fills == ((20, 3.0),)
    assert result.completed_ms == 20


def test_opposite_aggressor_at_price_prints_never_fill() -> None:
    tape = _tape([(10, 1, 100.0, 2.0, False), (20, 2, 100.0, 7.0, False)])
    order, spec = _buy()
    result = simulate_resting_fill(order, tape, spec)
    assert result.filled_qty == 0.0
    assert result.fills == ()
    assert result.first_fill_ms is None
    assert result.completed_ms is None
    assert result.through_ms is None


def test_unknown_queue_through_only_vs_front_of_queue_bounds() -> None:
    tape = _tape([(10, 1, 100.0, 1.0, True), (20, 2, 100.0, 1.0, True)])
    strict_order, strict_spec = _buy(qty=3.0, queue=None, unknown="through_only")
    loose_order, loose_spec = _buy(qty=3.0, queue=None, unknown="front_of_queue")
    strict = simulate_resting_fill(strict_order, tape, strict_spec)
    loose = simulate_resting_fill(loose_order, tape, loose_spec)
    assert strict.filled_qty == 0.0
    assert loose.filled_qty == 2.0
    assert loose.completed_ms is None
    rng = np.random.default_rng(11)
    for _ in range(25):
        n = 40
        times = np.sort(rng.integers(0, 9_000, size=n))
        ids = np.arange(n)
        prices = rng.choice([99.9, 100.0, 100.1], size=n).astype("float64")
        quantities = rng.uniform(0.1, 2.0, size=n)
        flags = rng.uniform(size=n) < 0.5
        fuzz = _tape(
            list(zip(times.tolist(), ids.tolist(), prices.tolist(), quantities.tolist(), flags.tolist(), strict=True))
        )
        qty = float(rng.uniform(1.0, 5.0))
        upper = simulate_resting_fill(
            _buy(qty=qty, queue=None, unknown="front_of_queue")[0], fuzz, TapeFillSpec(unknown_queue="front_of_queue")
        )
        lower = simulate_resting_fill(
            _buy(qty=qty, queue=None, unknown="through_only")[0], fuzz, TapeFillSpec(unknown_queue="through_only")
        )
        assert upper.filled_qty >= lower.filled_qty


def test_monotone_partial_fills_never_exceed_quantity() -> None:
    rng = np.random.default_rng(7)
    n = 200
    times = np.sort(rng.integers(0, 9_000, size=n))
    prices = rng.choice([99.9, 100.0, 100.1], size=n).astype("float64")
    quantities = rng.uniform(0.1, 2.0, size=n)
    flags = rng.uniform(size=n) < 0.5
    tape = _tape(
        list(
            zip(
                times.tolist(), np.arange(n).tolist(), prices.tolist(), quantities.tolist(), flags.tolist(), strict=True
            )
        )
    )
    order, spec = _buy(qty=7.0, queue=3.0)
    result = simulate_resting_fill(order, tape, spec)
    moments = [t for t, _ in result.fills]
    increments = [q for _, q in result.fills]
    assert moments == sorted(moments)
    assert len(set(moments)) == len(moments)
    assert all(q > 0.0 for q in increments)
    assert result.filled_qty <= 7.0 + 1e-9
    assert abs(result.filled_qty - sum(increments)) <= 1e-9 * 7.0
    assert (result.filled_qty == 7.0) == (result.completed_ms is not None)


def test_future_perturbation_leaves_result_identical() -> None:
    rows = [
        (500, 1, 100.0, 2.0, True),
        (1_000, 2, 100.0, 3.0, True),
        (1_500, 3, 99.9, 1.0, True),
        (2_500, 4, 100.0, 9.0, True),
        (3_000, 5, 101.0, 4.0, False),
    ]
    tape = _tape(rows)
    order, spec = _buy(qty=4.0, queue=1.0, window=(400, 2_000))
    before = simulate_resting_fill(order, tape, spec)
    rng = np.random.default_rng(3)
    perturbed = [
        r
        if r[0] < 2_000
        else (r[0], r[1], float(rng.uniform(50.0, 150.0)), float(rng.uniform(0.1, 5.0)), bool(rng.integers(0, 2)))
        for r in rows
    ]
    after = simulate_resting_fill(order, _tape(perturbed), spec)
    assert after == before


def test_pre_post_trades_are_ignored() -> None:
    tape = _tape([(100, 1, 99.0, 5.0, True), (900, 2, 101.0, 1.0, False)])
    order, spec = _buy(window=(500, 10_000))
    result = simulate_resting_fill(order, tape, spec)
    assert result.filled_qty == 0.0
    assert result.through_ms is None


def test_window_beyond_tape_coverage_fails_closed() -> None:
    tape = _tape([(100, 1, 100.0, 1.0, True)], coverage=(0, 1_000))
    late = RestingOrder(
        side="BUY", limit_price=100.0, quantity=1.0, active_from_ms=0, active_until_ms=1_001, queue_ahead_qty=None
    )
    early = RestingOrder(
        side="BUY", limit_price=100.0, quantity=1.0, active_from_ms=-1, active_until_ms=900, queue_ahead_qty=None
    )
    with pytest.raises(DataIntegrityError):
        simulate_resting_fill(late, tape, TapeFillSpec())
    with pytest.raises(DataIntegrityError):
        simulate_resting_fill(early, tape, TapeFillSpec())


def test_sell_mirror_symmetry() -> None:
    rng = np.random.default_rng(5)
    n = 60
    level = 100.0
    times = np.sort(rng.integers(0, 9_000, size=n))
    prices = rng.choice([99.9, 100.0, 100.1], size=n).astype("float64")
    quantities = rng.uniform(0.1, 2.0, size=n)
    flags = rng.uniform(size=n) < 0.5
    buy_tape = _tape(
        list(
            zip(
                times.tolist(), np.arange(n).tolist(), prices.tolist(), quantities.tolist(), flags.tolist(), strict=True
            )
        )
    )
    sell_prices = (2.0 * level - prices).tolist()
    sell_flags = (~flags).tolist()
    sell_tape = _tape(
        list(zip(times.tolist(), np.arange(n).tolist(), sell_prices, quantities.tolist(), sell_flags, strict=True))
    )
    buy_order, buy_spec = _buy(limit=level, qty=4.0, queue=2.0)
    sell_order = RestingOrder(
        side="SELL", limit_price=level, quantity=4.0, active_from_ms=0, active_until_ms=10_000, queue_ahead_qty=2.0
    )
    buy_result = simulate_resting_fill(buy_order, buy_tape, buy_spec)
    sell_result = simulate_resting_fill(sell_order, sell_tape, buy_spec)
    assert sell_result == buy_result


def test_touch_proxy_is_causal_and_rejects_crossed_proxies() -> None:
    tape = _tape(
        [
            (100, 1, 99.9, 1.0, True),
            (200, 2, 100.1, 1.0, False),
            (300, 3, 99.5, 1.0, True),
            (400, 4, 100.5, 1.0, False),
            (500, 5, 10.0, 1.0, True),
            (600, 6, 1000.0, 1.0, False),
        ]
    )
    assert touch_proxy(tape, 450, 400) == (99.5, 100.5)
    assert touch_proxy(tape, 300, 200) == (99.9, 100.1)
    crossed = _tape([(100, 1, 100.2, 1.0, True), (200, 2, 100.1, 1.0, False)])
    assert touch_proxy(crossed, 500, 500) is None
    one_sided = _tape([(100, 1, 99.9, 1.0, True)])
    assert touch_proxy(one_sided, 500, 500) is None
    with pytest.raises(ValueError, match="lookback_ms"):
        touch_proxy(tape, 500, 0)
    with pytest.raises(DataIntegrityError):
        touch_proxy(tape, 500, 600)


def test_post_only_limit_never_crosses_the_touch() -> None:
    assert post_only_limit("BUY", 100.2, (99.9, 100.1)) == 99.9
    assert post_only_limit("BUY", 99.5, (99.9, 100.1)) == 99.5
    assert post_only_limit("SELL", 99.8, (99.9, 100.1)) == 100.1
    with pytest.raises(ValueError, match="finite"):
        post_only_limit("BUY", float("nan"), (99.9, 100.1))
    with pytest.raises(ValueError, match="side"):
        post_only_limit("HOLD", 100.0, (99.9, 100.1))


def test_slice_frame_validation() -> None:
    good = [(200, 2, 100.0, 1.0, True), (100, 1, 99.9, 2.0, False), (100, 0, 99.9, 3.0, True)]
    tape = _tape(good)
    assert tape.time_ms.tolist() == [100, 100, 200]
    assert tape.price.tolist() == [99.9, 99.9, 100.0]
    assert tape.time_ms.dtype == np.int64
    assert tape.price.dtype == np.float64
    assert tape.qty.dtype == np.float64
    assert tape.buyer_maker.dtype == bool
    base = pd.DataFrame(good, columns=list(_COLUMNS))
    with pytest.raises(DataIntegrityError):
        TapeTrades.from_slice_frame(base.drop(columns=["price"]), coverage_ms=(0, 10_000))
    nan_price = base.copy()
    nan_price.loc[0, "price"] = float("nan")
    with pytest.raises(DataIntegrityError):
        TapeTrades.from_slice_frame(nan_price, coverage_ms=(0, 10_000))
    zero_qty = base.copy()
    zero_qty.loc[0, "quantity"] = 0.0
    with pytest.raises(DataIntegrityError):
        TapeTrades.from_slice_frame(zero_qty, coverage_ms=(0, 10_000))
    with pytest.raises(DataIntegrityError):
        TapeTrades.from_slice_frame(base, coverage_ms=(0, 100))
    with pytest.raises(DataIntegrityError):
        TapeTrades.from_slice_frame(base, coverage_ms=(10_000, 0))
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in _COLUMNS})
    blank = TapeTrades.from_slice_frame(empty, coverage_ms=(0, 10_000))
    assert blank.time_ms.size == 0
    order, spec = _buy(window=(0, 10_000))
    result = simulate_resting_fill(order, blank, spec)
    assert result == TapeFillResult(filled_qty=0.0, fills=(), first_fill_ms=None, completed_ms=None, through_ms=None)


def test_slice_frame_rejects_malformed_inputs() -> None:
    good = [(100, 1, 99.9, 2.0, False)]
    base = pd.DataFrame(good, columns=list(_COLUMNS))
    with pytest.raises(DataIntegrityError, match="DataFrame"):
        TapeTrades.from_slice_frame([good], coverage_ms=(0, 10_000))  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError, match="pair"):
        TapeTrades.from_slice_frame(base, coverage_ms=(0,))  # type: ignore[arg-type]
    bad_time = pd.DataFrame(
        [(float("nan"), 1.0, 99.9, 2.0, False)],
        columns=list(_COLUMNS),
    )
    with pytest.raises(DataIntegrityError, match="transact_time_ms"):
        TapeTrades.from_slice_frame(bad_time, coverage_ms=(0, 10_000))
    bad_id = pd.DataFrame(
        [(100.0, float("inf"), 99.9, 2.0, False)],
        columns=list(_COLUMNS),
    )
    with pytest.raises(DataIntegrityError, match="agg_trade_id"):
        TapeTrades.from_slice_frame(bad_id, coverage_ms=(0, 10_000))
    null_flag = pd.DataFrame(
        [(100.0, 1.0, 99.9, 2.0, None)],
        columns=list(_COLUMNS),
    )
    with pytest.raises(DataIntegrityError, match="is_buyer_maker"):
        TapeTrades.from_slice_frame(null_flag, coverage_ms=(0, 10_000))
    text_price = pd.DataFrame(
        [(100.0, 1.0, "nine", 2.0, False)],
        columns=list(_COLUMNS),
    )
    with pytest.raises(DataIntegrityError, match="non-numeric"):
        TapeTrades.from_slice_frame(text_price, coverage_ms=(0, 10_000))


def test_invalid_order_and_spec_rejected() -> None:
    with pytest.raises(ValueError, match="side"):
        RestingOrder(
            side="HOLD", limit_price=100.0, quantity=1.0, active_from_ms=0, active_until_ms=1, queue_ahead_qty=None
        )
    with pytest.raises(ValueError, match="limit_price"):
        RestingOrder(
            side="BUY", limit_price=-1.0, quantity=1.0, active_from_ms=0, active_until_ms=1, queue_ahead_qty=None
        )
    with pytest.raises(ValueError, match="quantity"):
        RestingOrder(
            side="BUY",
            limit_price=100.0,
            quantity=float("inf"),
            active_from_ms=0,
            active_until_ms=1,
            queue_ahead_qty=None,
        )
    with pytest.raises(ValueError, match="active window"):
        RestingOrder(
            side="BUY", limit_price=100.0, quantity=1.0, active_from_ms=5, active_until_ms=5, queue_ahead_qty=None
        )
    with pytest.raises(ValueError, match="queue_ahead_qty"):
        RestingOrder(
            side="BUY", limit_price=100.0, quantity=1.0, active_from_ms=0, active_until_ms=1, queue_ahead_qty=-0.5
        )
    with pytest.raises(ValueError, match="queue_ahead_fraction"):
        TapeFillSpec(queue_ahead_fraction=1.5)
    with pytest.raises(ValueError, match="unknown_queue"):
        TapeFillSpec(unknown_queue="mid_of_queue")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="qty_rtol"):
        TapeFillSpec(qty_rtol=0.0)


def test_int64_sorting_and_window_boundaries_preserve_precision() -> None:
    epoch = 2**53
    tape = _tape(
        [
            (epoch + 1, epoch + 1, 100.0, 1.0, True),
            (epoch, epoch + 1, 100.0, 2.0, True),
            (epoch, epoch, 100.0, 3.0, True),
        ],
        coverage=(epoch, epoch + 2),
    )
    assert tape.time_ms.tolist() == [epoch, epoch, epoch + 1]
    assert tape.qty.tolist() == [3.0, 2.0, 1.0]
    order, spec = _buy(qty=10.0, queue=0.0, window=(epoch + 1, epoch + 2))
    assert simulate_resting_fill(order, tape, spec).fills == ((epoch + 1, 1.0),)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("transact_time_ms", 10.5),
        ("agg_trade_id", 1.5),
        ("agg_trade_id", 2**63),
        ("agg_trade_id", -(2**63) - 1),
        ("agg_trade_id", "1"),
        ("is_buyer_maker", "False"),
    ],
)
def test_slice_rejects_lossy_coercion(column: str, value: object) -> None:
    data = dict(zip(_COLUMNS, [[10], [1], [100.0], [1.0], [True]], strict=True))
    frame = pd.DataFrame(data).astype({column: object})
    frame[column] = [value]
    with pytest.raises(DataIntegrityError, match=column):
        TapeTrades.from_slice_frame(frame, coverage_ms=(0, 100))


@pytest.mark.parametrize("coverage", [(float("nan"), 100), (0, float("inf")), (0.5, 100)])
def test_non_integral_coverage_is_rejected(coverage: tuple[float, float]) -> None:
    frame = pd.DataFrame([(10, 1, 100.0, 1.0, True)], columns=list(_COLUMNS))
    with pytest.raises(DataIntegrityError, match="coverage_ms"):
        TapeTrades.from_slice_frame(frame, coverage_ms=coverage)  # type: ignore[arg-type]


@pytest.mark.parametrize("rtol", [float("inf"), float("nan")])
def test_nonfinite_completion_tolerance_is_rejected(rtol: float) -> None:
    with pytest.raises(ValueError, match="qty_rtol"):
        TapeFillSpec(qty_rtol=rtol)


def test_completion_tolerance_snaps_and_merges_same_timestamp() -> None:
    tape = _tape(
        [
            (10, 1, 100.0, 0.4, True),
            (10, 2, 100.0, 0.3, True),
            (20, 3, 100.0, 0.3 - 5e-13, True),
            (30, 4, 99.9, 1.0, False),
        ]
    )
    order, spec = _buy(qty=1.0, queue=0.0)
    result = simulate_resting_fill(order, tape, spec)
    assert result.fills == ((10, 0.7), (20, 0.30000000000000004))
    assert result.filled_qty == sum(q for _, q in result.fills) == 1.0
    assert result.completed_ms == 20
    assert result.through_ms == 30


def test_partial_fill_then_through_fills_only_remainder() -> None:
    tape = _tape([(10, 1, 100.0, 2.0, True), (20, 2, 99.9, 0.1, False), (20, 3, 100.0, 8.0, True)])
    order, spec = _buy(qty=5.0, queue=0.0)
    assert simulate_resting_fill(order, tape, spec).fills == ((10, 2.0), (20, 3.0))


def test_slice_arrays_do_not_alias_the_frame() -> None:
    frame = pd.DataFrame([(10, 1, 100.0, 2.0, True)], columns=list(_COLUMNS))
    tape = TapeTrades.from_slice_frame(frame, coverage_ms=(0, 100))
    frame.loc[0, "price"] = 99.0
    assert tape.price.tolist() == [100.0]
    assert all(array.flags.c_contiguous for array in (tape.time_ms, tape.price, tape.qty, tape.buyer_maker))


def test_full_int64_identifier_range_is_preserved() -> None:
    tape = _tape([(10, 2**63 - 1, 100.0, 1.0, True), (10, -(2**63), 100.0, 2.0, True)])
    assert tape.qty.tolist() == [2.0, 1.0]
