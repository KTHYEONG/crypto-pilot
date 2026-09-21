from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.binance.venue_rules import VenueBracket, VenueRuleSnapshot, VenueSymbolRules
from src.mhs.account_ledger import AccountLedgerResult, AccountMarkPanels, replay_account
from src.mhs.account_policy import ExposurePolicy, UnitMoments


def _rules(
    step: float | None = 0.001,
    minimum: float | None = 5.0,
    ratio: float = 0.01,
    leverage: int = 10,
) -> VenueRuleSnapshot:
    return VenueRuleSnapshot(
        captured_at=pd.Timestamp("2026-01-01", tz="UTC"),
        symbols={
            "BTCUSDT": VenueSymbolRules(
                symbol="BTCUSDT",
                brackets=(VenueBracket(0.0, 1e12, ratio, 0.0, leverage),),
                step_size=step,
                min_notional=minimum,
            )
        },
    )


def _growth_policy(**overrides: float | str) -> ExposurePolicy:
    base: dict[str, float | str] = {
        "kind": "growth",
        "exposure_max": 10.0,
        "exposure_step": 0.25,
        "mean_haircut": 0.5,
        "prior_days": 730.0,
        "min_moment_days": 30,
        "shock_per_unit": 0.08,
        "margin_reserve": 0.10,
        "initial_margin_cap": 0.90,
        "impact_y": 0.6,
    }
    base.update(overrides)
    return ExposurePolicy(**base)  # type: ignore[arg-type]


def _fixed_policy(exposure_max: float = 1.0, impact_y: float = 0.0) -> ExposurePolicy:
    return _growth_policy(kind="fixed", exposure_max=exposure_max, impact_y=impact_y)


def _frames(
    dates: pd.DatetimeIndex,
    symbols: list[str],
    weights: list[list[float]],
    funding: list[list[float]] | None = None,
    adv: float = 1e9,
    sigma: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit = pd.DataFrame(weights, index=dates, columns=symbols, dtype="float64")
    zeros = [[0.0] * len(symbols) for _ in dates]
    funding_cum = pd.DataFrame(funding if funding is not None else zeros, index=dates, columns=symbols, dtype="float64")
    adv_frame = pd.DataFrame([[adv] * len(symbols) for _ in dates], index=dates, columns=symbols, dtype="float64")
    sigma_frame = pd.DataFrame([[sigma] * len(symbols) for _ in dates], index=dates, columns=symbols, dtype="float64")
    return unit, funding_cum, adv_frame, sigma_frame


def _panels(
    bar_times: list[pd.Timestamp],
    symbols: list[str],
    closes: list[list[float]],
    spread: float = 0.0,
    lows: list[list[float]] | None = None,
) -> AccountMarkPanels:
    index = pd.DatetimeIndex(bar_times)
    close = pd.DataFrame(closes, index=index, columns=symbols, dtype="float64")
    if lows is None:
        low = close * (1.0 - spread)
        high = close * (1.0 + spread)
    else:
        low = pd.DataFrame(lows, index=index, columns=symbols, dtype="float64")
        high = close.copy()
    return AccountMarkPanels(close=close, high=high, low=low)


def test_replay_account_skips_below_min_notional() -> None:
    """Step rounding and min-notional skip."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[0.0], [0.004]])
    marks = _panels(list(dates), ["BTCUSDT"], [[1.0], [1.0]])

    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(step=1.0, minimum=5.0),
        _fixed_policy(), capital=1000.0, taker_fee_bps=6.0,
    )

    assert result.skipped_orders == 1
    assert result.untraded_fraction == pytest.approx(1.0)
    assert result.daily_equity.iloc[1] == pytest.approx(1000.0)
    assert result.fee_paid == pytest.approx(0.0)


def test_replay_account_always_sends_full_close() -> None:
    """Full close always sent."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [0.0]])
    marks = _panels(list(dates), ["BTCUSDT"], [[100.0], [1.0]])

    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(),
        _fixed_policy(), capital=100.0, taker_fee_bps=6.0,
    )

    assert result.skipped_orders == 0
    assert result.fee_paid == pytest.approx(0.0606)
    assert result.daily_equity.iloc[1] == pytest.approx(0.9394)


def _wick_replay(wick_low: float) -> AccountLedgerResult:
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [1.0]])
    bars = [dates[0], dates[0] + pd.Timedelta(hours=12), dates[1]]
    marks = _panels(
        bars, ["BTCUSDT"], [[100.0], [100.0], [100.0]],
        lows=[[100.0], [wick_low], [100.0]],
    )
    return replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0, leverage=2),
        _fixed_policy(exposure_max=2.0), capital=1000.0, taker_fee_bps=6.0,
    )


def test_replay_account_liquidates_on_adverse_wick() -> None:
    """Liquidation on adverse wick."""
    result = _wick_replay(50.0)

    assert result.liquidated_at == pd.Timestamp("2026-01-01 12:00", tz="UTC")
    assert result.daily_equity.iloc[1] == 0.0
    assert result.initial_margin_breaches >= 1


def test_replay_account_no_liquidation_without_breach() -> None:
    """No liquidation without breach."""
    result = _wick_replay(99.0)

    assert result.liquidated_at is None
    assert bool((result.daily_equity > 0).all())


def test_replay_account_conserves_cash() -> None:
    """Cash conservation."""
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    symbols = ["BTCUSDT", "NOSUCH"]
    unit, funding, adv, sigma = _frames(
        dates, symbols, [[1.0, 0.0], [0.5, 0.0], [0.5, 0.0]],
        funding=[[0.0, 0.0], [0.0001, 0.0], [0.0002, 0.0]],
    )
    marks = _panels(list(dates), symbols, [[100.0, 10.0], [101.0, 10.0], [99.0, 10.0]])

    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _fixed_policy(exposure_max=2.0), capital=1000.0, taker_fee_bps=6.0,
        apply_order_filters=False,
    )

    closes = [100.0, 101.0, 99.0]
    fund_deltas = [0.0, 0.0001, 0.0001]
    weights = [1.0, 0.5, 0.5]
    cash, quantity, fees, funding_paid = 1000.0, 0.0, 0.0, 0.0
    price_pnl = 0.0
    for day in range(3):
        price = closes[day]
        marked = cash + quantity * price
        charge = quantity * price * fund_deltas[day]
        cash -= charge
        funding_paid += charge
        equity = marked - charge
        target = 2.0 * equity * weights[day]
        delta = target / price - quantity
        fee = 6e-4 * abs(delta) * price
        cash -= delta * price + fee
        fees += fee
        price_pnl -= delta * price
        quantity += delta
    price_pnl += quantity * closes[-1]
    final = cash + quantity * closes[-1]
    assert result.daily_equity.iloc[-1] == pytest.approx(final)
    assert final == pytest.approx(1000.0 + price_pnl - fees - funding_paid, rel=1e-9)
    assert result.impact_paid == pytest.approx(0.0)
    assert result.fee_paid == pytest.approx(fees)
    assert result.funding_paid == pytest.approx(funding_paid)
    assert result.missing_filter_symbols == ("NOSUCH",)
    assert result.fallback_ladder_symbols == ("NOSUCH",)


def test_replay_account_future_perturbation_leaves_past_unchanged() -> None:
    """Future perturbation leaves past exposure unchanged."""
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [1.0], [1.0]])
    base = _panels(list(dates), ["BTCUSDT"], [[100.0], [101.0], [102.0]], spread=0.001)
    shocked = AccountMarkPanels(
        close=pd.concat([base.close.iloc[:2], base.close.iloc[2:] * 1.5]),
        high=pd.concat([base.high.iloc[:2], base.high.iloc[2:] * 1.5]),
        low=pd.concat([base.low.iloc[:2], base.low.iloc[2:] * 1.5]),
    )
    kwargs = {"capital": 1000.0, "taker_fee_bps": 6.0}
    first = replay_account(unit, base, funding, adv, sigma, _rules(minimum=1.0), _fixed_policy(), **kwargs)
    second = replay_account(unit, shocked, funding, adv, sigma, _rules(minimum=1.0), _fixed_policy(), **kwargs)

    assert (first.daily_exposure.iloc[:2] == second.daily_exposure.iloc[:2]).all()
    assert (first.daily_equity.iloc[:2] == second.daily_equity.iloc[:2]).all()


def test_replay_account_filters_off_is_scale_free() -> None:
    """Filters off is scale-free."""
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(
        dates, ["BTCUSDT"], [[1.0], [1.0], [1.0]],
        funding=[[0.0], [0.0001], [0.0002]],
    )
    marks = _panels(list(dates), ["BTCUSDT"], [[100.0], [101.0], [102.0]], spread=0.001)
    policy = _growth_policy(impact_y=0.0)
    unit_equity = pd.Series([1000.0, 1005.0, 1010.0], index=dates, dtype="float64")
    kwargs = {"taker_fee_bps": 6.0, "apply_order_filters": False, "unit_equity": unit_equity}
    small = replay_account(unit, marks, funding, adv, sigma, _rules(minimum=0.0, ratio=1e-6, leverage=1000), policy, capital=1e3, **kwargs)
    large = replay_account(unit, marks, funding, adv, sigma, _rules(minimum=0.0, ratio=1e-6, leverage=1000), policy, capital=1e6, **kwargs)

    assert (small.daily_exposure == large.daily_exposure).all()
    np.testing.assert_allclose(large.daily_equity.to_numpy(), small.daily_equity.to_numpy() * 1000.0, rtol=1e-9)


def _valid_kwargs() -> dict[str, object]:
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [1.0]])
    marks = _panels(list(dates), ["BTCUSDT"], [[100.0], [101.0]], spread=0.001)
    return {
        "unit_weights": unit, "marks": marks, "funding_cum": funding, "adv": adv,
        "daily_sigma": sigma, "rules": _rules(), "policy": _fixed_policy(),
        "capital": 1000.0, "taker_fee_bps": 6.0,
    }


def test_replay_account_rejects_misaligned_inputs() -> None:
    """Misaligned indexes/columns, non-positive capital, or no entry row."""
    base = _valid_kwargs()
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit = base["unit_weights"]
    assert isinstance(unit, pd.DataFrame)

    bad_capital = dict(base, capital=0.0)
    with pytest.raises(DataIntegrityError):
        replay_account(**bad_capital)  # type: ignore[arg-type]

    with pytest.raises(DataIntegrityError):
        replay_account(**dict(base, unit_weights=unit.iloc[:0]))  # type: ignore[arg-type]

    adv = base["adv"]
    assert isinstance(adv, pd.DataFrame)
    with pytest.raises(DataIntegrityError):
        replay_account(**dict(base, adv=adv.rename(columns={"BTCUSDT": "ETHUSDT"})))  # type: ignore[arg-type]

    funding = base["funding_cum"]
    assert isinstance(funding, pd.DataFrame)
    shifted_index = funding.copy()
    shifted_index.index = dates + pd.Timedelta(days=1)
    with pytest.raises(DataIntegrityError):
        replay_account(**dict(base, funding_cum=shifted_index))  # type: ignore[arg-type]

    marks = base["marks"]
    assert isinstance(marks, AccountMarkPanels)
    shifted_high = AccountMarkPanels(
        close=marks.close, high=marks.high.set_axis(dates + pd.Timedelta(days=1)), low=marks.low
    )
    with pytest.raises(DataIntegrityError):
        replay_account(**dict(base, marks=shifted_high))  # type: ignore[arg-type]

    renamed_low = AccountMarkPanels(
        close=marks.close, high=marks.high, low=marks.low.rename(columns={"BTCUSDT": "ETHUSDT"})
    )
    with pytest.raises(DataIntegrityError):
        replay_account(**dict(base, marks=renamed_low))  # type: ignore[arg-type]

    wide_unit, wide_funding, wide_adv, wide_sigma = _frames(dates, ["BTCUSDT", "XRPUSDT"], [[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(DataIntegrityError):
        replay_account(**dict(base, unit_weights=wide_unit, funding_cum=wide_funding, adv=wide_adv, daily_sigma=wide_sigma))  # type: ignore[arg-type]

    early = pd.date_range("2025-12-31", periods=2, freq="D", tz="UTC")
    early_unit, early_funding, early_adv, early_sigma = _frames(early, ["BTCUSDT"], [[1.0], [1.0]])
    with pytest.raises(DataIntegrityError):
        replay_account(**dict(base, unit_weights=early_unit, funding_cum=early_funding, adv=early_adv, daily_sigma=early_sigma))  # type: ignore[arg-type]

    empty_rules = VenueRuleSnapshot(captured_at=pd.Timestamp("2026-01-01", tz="UTC"), symbols={})
    with pytest.raises(DataIntegrityError):
        replay_account(**dict(base, rules=empty_rules))  # type: ignore[arg-type]


def test_replay_account_liquidation_timestamp_is_absolute_on_later_day() -> None:
    """A breach inside a later day's segment reports that bar's timestamp, not a segment offset."""
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [1.0], [1.0]])
    bars = [dates[0], dates[1], dates[1] + pd.Timedelta(hours=6), dates[2]]
    marks = _panels(
        bars, ["BTCUSDT"], [[100.0], [100.0], [100.0], [100.0]],
        lows=[[100.0], [100.0], [50.0], [100.0]],
    )
    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0, leverage=2),
        _fixed_policy(exposure_max=2.0), capital=1000.0, taker_fee_bps=6.0,
    )
    assert result.liquidated_at == pd.Timestamp("2026-01-02 06:00", tz="UTC")
    assert result.daily_equity.iloc[0] > 0.0
    assert result.daily_equity.iloc[1] == 0.0


def test_replay_account_missing_sigma_charges_no_impact() -> None:
    """A non-finite daily sigma yields zero impact instead of poisoning cash with NaN."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[0.5], [0.5]], adv=1e6)
    sigma.iloc[:, :] = np.nan
    marks = _panels(list(dates), ["BTCUSDT"], [[100.0], [100.0]])
    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _fixed_policy(exposure_max=1.0, impact_y=0.6), capital=1000.0, taker_fee_bps=6.0,
    )
    assert result.impact_paid == 0.0
    assert bool(np.isfinite(result.daily_equity).all())


def test_replay_account_never_listed_symbol_does_not_poison_equity() -> None:
    """A universe symbol with all-NaN marks (not yet listed) has zero weight and zero delta;
    equity must stay finite from day one instead of 0.0 * NaN corrupting the whole account."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT", "NEWUSDT"], [[1.0, 0.0], [1.0, 0.0]])
    adv["NEWUSDT"] = np.nan
    sigma["NEWUSDT"] = np.nan
    marks = _panels(list(dates), ["BTCUSDT", "NEWUSDT"], [[100.0, np.nan], [100.0, np.nan]])

    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _growth_policy(), capital=1000.0, taker_fee_bps=6.0,
        unit_equity=pd.Series([1000.0, 1001.0], index=dates, dtype="float64"),
    )

    assert bool(np.isfinite(result.daily_equity).all())
    assert result.daily_equity.iloc[0] > 0.0
    assert result.daily_exposure.iloc[0] > 0.0


def test_choose_exposure_ignores_unheld_symbol_with_nan_sigma() -> None:
    """A zero-weight symbol's NaN sigma must not zero out every rung's growth score via 0*NaN."""
    from src.mhs.account_policy import build_venue_ladders, choose_exposure

    rules = _rules(minimum=0.0)
    ladders, _ = build_venue_ladders(["BTCUSDT", "NEWUSDT"], rules)
    weights = np.array([0.5, 0.0])
    adv = np.array([1e9, np.nan])
    sigma = np.array([0.02, np.nan])
    held = np.zeros(2)
    exposure = choose_exposure(
        weights, 10_000.0, held, adv, sigma, ladders, _growth_policy(),
        UnitMoments(mean=0.0011, sigma=0.0094, observations=1000),
    )
    assert exposure > 0.25


def _growth_frames(
    dates: pd.DatetimeIndex, weights: list[list[float]], closes: list[list[float]],
) -> tuple[pd.DataFrame, AccountMarkPanels, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], weights)
    marks = _panels(list(dates), ["BTCUSDT"], closes)
    return unit, marks, funding, adv, sigma


def _growth_equity(dates: pd.DatetimeIndex, values: list[float]) -> pd.Series:
    return pd.Series(values, index=dates, dtype="float64")


def test_replay_account_growth_requires_unit_equity() -> None:
    """Growth without unit equity fails closed."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit, marks, funding, adv, sigma = _growth_frames(dates, [[1.0], [1.0]], [[100.0], [101.0]])

    with pytest.raises(DataIntegrityError):
        replay_account(
            unit, marks, funding, adv, sigma, _rules(), _growth_policy(),
            capital=1000.0, taker_fee_bps=6.0,
        )


def test_replay_account_unit_equity_alignment_fails_closed() -> None:
    """Misaligned, non-finite, or non-positive unit equity fails closed."""
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    unit, marks, funding, adv, sigma = _growth_frames(
        dates, [[1.0], [1.0], [1.0]], [[100.0], [101.0], [102.0]]
    )
    kwargs = {"capital": 1000.0, "taker_fee_bps": 6.0, "policy": _growth_policy()}
    shifted = pd.Series([1000.0, 1001.0, 1002.0],
                        index=dates + pd.Timedelta(days=1), dtype="float64")
    with pytest.raises(DataIntegrityError):
        replay_account(unit, marks, funding, adv, sigma, _rules(), unit_equity=shifted, **kwargs)  # type: ignore[arg-type]
    nan_equity = _growth_equity(dates, [1000.0, float("nan"), 1002.0])
    with pytest.raises(DataIntegrityError):
        replay_account(unit, marks, funding, adv, sigma, _rules(), unit_equity=nan_equity, **kwargs)  # type: ignore[arg-type]
    zero_equity = _growth_equity(dates, [1000.0, 0.0, 1002.0])
    with pytest.raises(DataIntegrityError):
        replay_account(unit, marks, funding, adv, sigma, _rules(), unit_equity=zero_equity, **kwargs)  # type: ignore[arg-type]


def test_replay_account_today_unit_return_never_sizes_today() -> None:
    """Entries up to date d ignore unit returns realized at d and later."""
    dates = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    unit, marks, funding, adv, sigma = _growth_frames(
        dates, [[1.0]] * 5, [[100.0], [101.0], [102.0], [103.0], [104.0]]
    )
    base_values = [1000.0, 1005.0, 1003.0, 1010.0, 1015.0]
    fork_values = [1000.0, 1005.0, 1003.0, 900.0, 800.0]
    first = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0), _growth_policy(),
        capital=1000.0, taker_fee_bps=6.0, unit_equity=_growth_equity(dates, base_values),
    )
    second = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0), _growth_policy(),
        capital=1000.0, taker_fee_bps=6.0, unit_equity=_growth_equity(dates, fork_values),
    )

    assert (first.daily_exposure.iloc[:4] == second.daily_exposure.iloc[:4]).all()


def test_replay_account_early_days_use_smallest_rung() -> None:
    """Before min_moment_days of evidence every exposure is the smallest rung."""
    dates = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    unit, marks, funding, adv, sigma = _growth_frames(
        dates, [[1.0]] * 4, [[100.0], [101.0], [102.0], [103.0]]
    )
    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0, ratio=1e-6, leverage=1000),
        _growth_policy(), capital=1000.0, taker_fee_bps=6.0,
        unit_equity=_growth_equity(dates, [1000.0, 1001.0, 1002.0, 1003.0]),
    )

    assert (result.daily_exposure == 0.25).all()


def test_replay_account_fixed_replay_unchanged_by_unit_equity() -> None:
    """Fixed replay ignores unit equity bit-for-bit."""
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    unit, marks, funding, adv, sigma = _growth_frames(
        dates, [[1.0], [1.0], [1.0]], [[100.0], [101.0], [102.0]], 
    )
    policy = _fixed_policy()
    plain = replay_account(
        unit, marks, funding, adv, sigma, _rules(), policy,
        capital=1000.0, taker_fee_bps=6.0,
    )
    with_equity = replay_account(
        unit, marks, funding, adv, sigma, _rules(), policy,
        capital=1000.0, taker_fee_bps=6.0,
        unit_equity=_growth_equity(dates, [1000.0, 50.0, 2000.0]),
    )

    pd.testing.assert_series_equal(with_equity.daily_equity, plain.daily_equity)
