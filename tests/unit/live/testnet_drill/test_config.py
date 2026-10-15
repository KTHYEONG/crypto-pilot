"""DrillConfig validation invariants."""

from __future__ import annotations


from decimal import Decimal

import pytest

from tests.unit.live.testnet_drill._stubs import make_config


@pytest.mark.parametrize("name", ["settle_wait_s", "settle_poll_s"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_settlement_durations_are_positive_and_finite(name, value):
    with pytest.raises(ValueError, match=name):
        make_config(**{name: value})


def test_valid_config_normalizes() -> None:
    config = make_config(symbols=["BTCUSDT", "ETHUSDT"])
    assert config.symbols == ("BTCUSDT", "ETHUSDT")
    assert config.probe_notional_usdt <= config.max_probe_notional_usdt


def test_rejects_empty_symbols() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_config(symbols=())


def test_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="unique"):
        make_config(symbols=("BTCUSDT", "BTCUSDT"))


def test_rejects_non_uppercase_symbol() -> None:
    with pytest.raises(ValueError, match="uppercase ASCII"):
        make_config(symbols=("btcusdt",))


def test_rejects_non_positive_notionals() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_config(probe_notional_usdt=Decimal(0))


def test_rejects_probe_above_cap() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        make_config(probe_notional_usdt=Decimal("30"))


def test_rejects_undecodable_decimal() -> None:
    with pytest.raises(ValueError, match="decimal-like"):
        make_config(probe_notional_usdt="nonsense")  # type: ignore[arg-type]


def test_rejects_non_finite_decimal() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_config(probe_notional_usdt=float("nan"))


def test_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_config(qty_tolerance=Decimal("-1"))


def test_rejects_non_positive_resting_distance() -> None:
    with pytest.raises(ValueError, match="resting_distance_bps"):
        make_config(resting_distance_bps=0)


def test_rejects_non_int_counts() -> None:
    with pytest.raises(ValueError, match="must be an int"):
        make_config(max_open_orders_probe=True)


def test_rejects_empty_open_order_probe() -> None:
    with pytest.raises(ValueError, match="max_open_orders_probe"):
        make_config(max_open_orders_probe=0)


def test_rejects_non_numeric_timeout() -> None:
    with pytest.raises(ValueError, match="scenario_timeout_s"):
        make_config(scenario_timeout_s="soon")  # type: ignore[arg-type]


def test_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="scenario_timeout_s"):
        make_config(scenario_timeout_s=0.0)


def test_rejects_recv_window_out_of_range() -> None:
    with pytest.raises(ValueError, match="recv_window_ms"):
        make_config(recv_window_ms=60001)


def test_rejects_empty_funding_soak() -> None:
    with pytest.raises(ValueError, match="funding_soak"):
        make_config(funding_soak_min_funding_events=0)


def test_rejects_empty_slate_attempts() -> None:
    with pytest.raises(ValueError, match="slate_max_attempts"):
        make_config(slate_max_attempts=0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"race_repetitions": 0}, "race_repetitions"),
        ({"race_offsets_s": ()}, "race_offsets_s"),
        ({"race_offsets_s": (0.0, float("nan"))}, "race_offsets_s"),
    ],
)
def test_race_parameters_are_bounded(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"funding_soak_s": -1.0}, "funding_soak_s"),
        ({"funding_soak_s": float("inf")}, "funding_soak_s"),
        ({"retry_after_probe_s": 0}, "retry_after_probe_s"),
        ({"snapshot_trials": 0}, "snapshot_trials"),
        ({"funding_poll_s": 0.0}, "funding_poll_s"),
    ],
)
def test_fault_recovery_parameters_are_bounded(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)
