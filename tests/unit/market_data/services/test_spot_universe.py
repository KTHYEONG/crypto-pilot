from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.market_data.services.spot_universe import (
    SPOT_PREFIX_MULTIPLIERS,
    resolve_spot_companions,
    seated_spans,
    spot_candidates,
)


def test_exact_spot_name_preferred_over_prefix_stripping() -> None:
    calls: list[str] = []

    def _probe(symbol: str, first: date, last: date) -> bool:
        calls.append(symbol)
        return symbol in ("PEPEUSDT", "EPEUSDT")

    companions = resolve_spot_companions({ "PEPEUSDT": (date(2023, 1, 1), date(2023, 2, 1))}, _probe)
    assert len(companions) == 1
    assert companions[0].spot_symbol == "PEPEUSDT"
    assert companions[0].price_multiplier == 1
    assert companions[0].status == "mapped"
    assert calls[0] == "PEPEUSDT"


def test_contract_unit_prefix_maps_with_multiplier() -> None:
    companions = resolve_spot_companions(
        {"1000PEPEUSDT": (date(2023, 1, 1), date(2023, 2, 1))},
        lambda symbol, first, last: symbol == "PEPEUSDT",
    )
    assert companions[0].spot_symbol == "PEPEUSDT"
    assert companions[0].price_multiplier == 1000

    companions = resolve_spot_companions(
        {"1000000MOGUSDT": (date(2023, 1, 1), date(2023, 2, 1))},
        lambda symbol, first, last: symbol == "MOGUSDT",
    )
    assert companions[0].spot_symbol == "MOGUSDT"
    assert companions[0].price_multiplier == 1_000_000


def test_longest_prefix_tried_first() -> None:
    assert spot_candidates("1000000MOGUSDT") == (
        ("1000000MOGUSDT", 1),
        ("MOGUSDT", 1_000_000),
        ("000MOGUSDT", 1_000),
    )
    assert SPOT_PREFIX_MULTIPLIERS[0][1] == 1_000_000


def test_no_spot_history_is_explicit_not_dropped() -> None:
    companions = resolve_spot_companions(
        {"NOPEUSDT": (date(2023, 1, 1), date(2023, 2, 1))},
        lambda symbol, first, last: False,
    )
    assert len(companions) == 1
    assert companions[0].status == "no_spot_history"
    assert companions[0].spot_symbol is None
    assert companions[0].price_multiplier == 1


def test_probe_errors_fail_closed() -> None:
    def _boom(symbol: str, first: date, last: date) -> bool:
        raise OSError("network unreachable")

    with pytest.raises(OSError, match="network unreachable"):
        resolve_spot_companions({"BTCUSDT": (date(2023, 1, 1), date(2023, 2, 1))}, _boom)


def test_delisted_pairs_not_filtered_by_current_listings() -> None:
    def _probe(symbol: str, first: date, last: date) -> bool:
        return symbol == "OLDUSDT" and first == date(2024, 1, 1)

    companions = resolve_spot_companions({"OLDUSDT": (date(2024, 1, 1), date(2024, 6, 1))}, _probe)
    assert companions[0].status == "mapped"
    assert companions[0].spot_symbol == "OLDUSDT"


def test_seated_spans_respect_window() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2022-12-01", "2023-02-10", freq="D", tz="UTC"))
    mask = pd.DataFrame(False, index=idx, columns=["AUSDT", "BUSDT"])
    mask.loc["2022-12-01":"2023-02-10", "AUSDT"] = True
    spans = seated_spans(mask, first=date(2023, 1, 1), last=date(2023, 2, 10))
    assert spans == {"AUSDT": (date(2023, 1, 1), date(2023, 2, 10))}


def test_non_perpetual_input_rejected() -> None:
    with pytest.raises(ValueError, match="USDT perpetual"):
        spot_candidates("BTCUSDT_250627")


def test_prefix_without_base_is_not_a_candidate() -> None:
    assert spot_candidates("1000USDT") == (("1000USDT", 1),)
