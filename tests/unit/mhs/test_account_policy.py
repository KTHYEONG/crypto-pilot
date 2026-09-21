from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.binance.venue_rules import VenueBracket, VenueRuleSnapshot, VenueSymbolRules
from src.mhs.account_policy import (
    ExposurePolicy,
    VenueLadder,
    build_venue_ladders,
    choose_exposure,
    maintenance_and_initial_margin,
    margin_exposure_cap,
)


def _ladder(
    floors: list[float], ratios: list[float], amounts: list[float], leverages: list[float]
) -> VenueLadder:
    return VenueLadder(
        floors=np.array(floors),
        ratios=np.array(ratios),
        amounts=np.array(amounts),
        leverages=np.array(leverages),
    )


def _growth_policy(**overrides: float | str) -> ExposurePolicy:
    base: dict[str, float | str] = {
        "kind": "growth",
        "exposure_max": 10.0,
        "exposure_step": 0.25,
        "unit_daily_mean": 0.0011,
        "unit_daily_sigma": 0.0094,
        "mean_haircut": 0.5,
        "shock_per_unit": 0.08,
        "margin_reserve": 0.10,
        "initial_margin_cap": 0.90,
        "impact_y": 0.6,
    }
    base.update(overrides)
    return ExposurePolicy(**base)  # type: ignore[arg-type]


def _snapshot() -> VenueRuleSnapshot:
    return VenueRuleSnapshot(
        captured_at=pd.Timestamp("2026-01-01", tz="UTC"),
        symbols={
            "BTCUSDT": VenueSymbolRules(
                symbol="BTCUSDT",
                brackets=(
                    VenueBracket(0.0, 50000.0, 0.01, 0.0, 50),
                    VenueBracket(50000.0, 250000.0, 0.02, 500.0, 20),
                ),
                step_size=0.001,
                min_notional=5.0,
            ),
            "ETHUSDT": VenueSymbolRules(
                symbol="ETHUSDT",
                brackets=(VenueBracket(0.0, 100000.0, 0.05, 10.0, 20),),
                step_size=0.01,
                min_notional=5.0,
            ),
        },
    )


def test_maintenance_and_initial_margin_follows_notional_tier() -> None:
    """Maintenance margin follows the notional tier."""
    ladder = _ladder([0.0, 5000.0], [0.01, 0.05], [0.0, 150.0], [20.0, 10.0])

    maintenance, initial = maintenance_and_initial_margin(np.array([8000.0]), (ladder,))

    assert maintenance[0] == pytest.approx(8000.0 * 0.05 - 150.0)
    assert initial[0] == pytest.approx(800.0)


def test_margin_exposure_cap_falls_as_equity_grows() -> None:
    """Margin cap falls as equity grows."""
    ladder = _ladder([0.0, 50000.0], [0.005, 0.20], [0.0, 0.0], [50.0, 5.0])
    policy = _growth_policy()
    weights = np.array([1.0])

    low = margin_exposure_cap(weights, 1e3, (ladder,), policy)
    high = margin_exposure_cap(weights, 1e6, (ladder,), policy)

    assert low == pytest.approx(10.0)
    assert high < low


def test_margin_exposure_cap_respects_initial_margin() -> None:
    """Cap respects initial margin."""
    ladder = _ladder([0.0], [0.001], [0.0], [2.0])
    policy = _growth_policy(shock_per_unit=0.0)
    weights = np.array([1.5, -1.5])

    cap = margin_exposure_cap(weights, 1000.0, (ladder, ladder), policy)

    assert cap <= 1.5
    assert (cap / 0.25).is_integer()


def test_margin_exposure_cap_zero_without_equity() -> None:
    """No equity leaves no margin budget."""
    ladder = _ladder([0.0], [0.01], [0.0], [10.0])

    assert margin_exposure_cap(np.array([1.0]), 0.0, (ladder,), _growth_policy()) == 0.0
    assert choose_exposure(np.array([1.0]), 0.0, np.zeros(1), np.array([1e9]), np.array([0.02]), (ladder,), _growth_policy()) == 0.0


def test_margin_exposure_cap_zero_when_no_rung_fits() -> None:
    """No rung fits inside the margin budget."""
    ladder = _ladder([0.0], [0.90], [0.0], [1.0])
    policy = _growth_policy()
    weights = np.array([2.0, 2.0])

    cap = margin_exposure_cap(weights, 1000.0, (ladder, ladder), policy)

    assert cap == 0.0
    assert choose_exposure(
        weights, 1000.0, np.zeros(2), np.array([1e9, 1e9]), np.array([0.02, 0.02]),
        (ladder, ladder), policy,
    ) == 0.0


def test_choose_exposure_growth_rung_never_exceeds_margin_cap() -> None:
    """Growth rung never exceeds margin cap."""
    ladder = _ladder([0.0], [0.01], [0.0], [10.0])
    policy = _growth_policy()
    weights = np.array([0.6, -0.4])

    cap = margin_exposure_cap(weights, 5000.0, (ladder, ladder), policy)
    chosen = choose_exposure(
        weights, 5000.0, np.zeros(2), np.array([1e7, 1e7]), np.array([0.02, 0.02]),
        (ladder, ladder), policy,
    )

    assert chosen <= cap


def test_choose_exposure_higher_haircut_never_raises_exposure() -> None:
    """Higher haircut never raises exposure."""
    ladder = _ladder([0.0], [0.01], [0.0], [10.0])
    weights = np.array([0.6, -0.4])
    ladders = (ladder, ladder)

    low = choose_exposure(
        weights, 5000.0, np.zeros(2), np.array([1e7, 1e7]), np.array([0.02, 0.02]),
        ladders, _growth_policy(mean_haircut=0.0),
    )
    high = choose_exposure(
        weights, 5000.0, np.zeros(2), np.array([1e7, 1e7]), np.array([0.02, 0.02]),
        ladders, _growth_policy(mean_haircut=0.5),
    )

    assert high <= low


def test_choose_exposure_impact_lowers_exposure_at_scale() -> None:
    """Impact lowers exposure at scale."""
    ladder = _ladder([0.0], [0.01], [0.0], [20.0])
    policy = _growth_policy()
    weights = np.array([1.0])

    small = choose_exposure(weights, 1e3, np.zeros(1), np.array([1e9]), np.array([0.02]), (ladder,), policy)
    large = choose_exposure(weights, 1e8, np.zeros(1), np.array([1e9]), np.array([0.02]), (ladder,), policy)

    assert large <= small


def test_choose_exposure_fixed_policy_returns_maximum() -> None:
    """Fixed policy returns its maximum."""
    ladder = _ladder([0.0], [0.01], [0.0], [10.0])
    policy = _growth_policy(kind="fixed", exposure_max=7.5)

    chosen = choose_exposure(
        np.array([1.0]), 1000.0, np.zeros(1), np.array([np.inf]), np.array([0.02]), (ladder,), policy
    )

    assert chosen == 7.5


def test_choose_exposure_ignores_non_finite_adv() -> None:
    """Non-finite ADV contributes zero impact."""
    ladder = _ladder([0.0], [0.01], [0.0], [20.0])
    policy = _growth_policy()
    weights = np.array([1.0])

    finite = choose_exposure(weights, 1e3, np.zeros(1), np.array([1e18]), np.array([0.02]), (ladder,), policy)
    broken = choose_exposure(weights, 1e3, np.zeros(1), np.array([np.nan]), np.array([0.02]), (ladder,), policy)

    assert broken >= finite


def test_build_venue_ladders_uses_punitive_fallback() -> None:
    """Symbols without a ladder use the snapshot's most punitive tier-1 ratio ladder."""
    ladders, fallback = build_venue_ladders(["BTCUSDT", "SOLUSDT"], _snapshot())

    assert fallback == ("SOLUSDT",)
    assert ladders[1].ratios[0] == pytest.approx(0.05)
    assert ladders[1].floors[0] == pytest.approx(0.0)
    assert ladders[0].ratios.tolist() == [0.01, 0.02]


def test_build_venue_ladders_rejects_empty_snapshot() -> None:
    """A snapshot without ladders cannot price margin."""
    empty = VenueRuleSnapshot(captured_at=pd.Timestamp("2026-01-01", tz="UTC"), symbols={})

    with pytest.raises(DataIntegrityError):
        build_venue_ladders(["BTCUSDT"], empty)
