from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.binance.venue_rules import VenueBracket, VenueRuleSnapshot, VenueSymbolRules
from src.mhs.account_ledger import AccountLedgerResult, AccountMarkPanels, replay_account as _orig_replay_account
from src.mhs.account_policy import ExposurePolicy, UnitMoments


def replay_account(*args: object, **kwargs: object) -> AccountLedgerResult:
    """Regression shim: legacy calls without an explicit anchor replay at the entry labels."""
    if "anchor_times" not in kwargs:
        first = args[0] if args else kwargs.get("unit_weights")
        assert isinstance(first, pd.DataFrame)
        kwargs["anchor_times"] = pd.DatetimeIndex(first.index)
    return _orig_replay_account(*args, **kwargs)  # type: ignore[arg-type]


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
        "anchor_times": pd.DatetimeIndex(dates),
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
        replay_account(**dict(base, unit_weights=early_unit, funding_cum=early_funding, adv=early_adv, daily_sigma=early_sigma, anchor_times=pd.DatetimeIndex(early)))  # type: ignore[arg-type]

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


def _maker_single_day(
    closes: list[float],
    lows: list[float],
    highs: list[float],
    weights: list[list[float]] | None = None,
    window: int = 10,
    step: float | None = 0.001,
    ratio: float = 0.01,
    leverage: int = 10,
    filters: bool = True,
) -> AccountLedgerResult:
    dates = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    bars = [dates[0] + pd.Timedelta(minutes=3 * i) for i in range(len(closes))]
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], weights or [[1.0]])
    index = pd.DatetimeIndex(bars)
    marks = AccountMarkPanels(
        close=pd.DataFrame([[c] for c in closes], index=index, columns=["BTCUSDT"], dtype="float64"),
        high=pd.DataFrame([[h] for h in highs], index=index, columns=["BTCUSDT"], dtype="float64"),
        low=pd.DataFrame([[lo] for lo in lows], index=index, columns=["BTCUSDT"], dtype="float64"),
    )
    return replay_account(
        unit, marks, funding, adv, sigma, _rules(step=step, minimum=0.0, ratio=ratio, leverage=leverage),
        _fixed_policy(), capital=1000.0, taker_fee_bps=6.0, apply_order_filters=filters,
        execution="maker", maker_fee_bps=2.0, passive_window_bars=window,
    )


def test_replay_account_taker_execution_ignores_maker_params() -> None:
    """Default replay matches explicit taker execution with maker params attached."""
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [0.5], [0.5]])
    marks = _panels(list(dates), ["BTCUSDT"], [[100.0], [101.0], [99.0]], spread=0.001)

    plain = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _fixed_policy(exposure_max=2.0), capital=1000.0, taker_fee_bps=6.0,
    )
    explicit = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _fixed_policy(exposure_max=2.0), capital=1000.0, taker_fee_bps=6.0,
        execution="taker", maker_fee_bps=2.0, passive_window_bars=10,
    )

    pd.testing.assert_series_equal(explicit.daily_equity, plain.daily_equity)
    assert explicit.fee_paid == pytest.approx(plain.fee_paid)
    assert explicit.maker_fill_fraction == 0.0
    assert plain.maker_fill_fraction == 0.0


def test_replay_account_maker_fills_at_anchor_on_trade_through() -> None:
    """A buy whose window low trades strictly below the anchor fills as maker."""
    result = _maker_single_day([100.0, 100.0, 100.0], [100.0, 99.5, 100.0], [100.0, 100.0, 100.0])

    assert result.fee_paid == pytest.approx(10.0 * 100.0 * 2e-4)
    assert result.daily_equity.iloc[0] == pytest.approx(1000.0 - 10.0 * 100.0 * 2e-4)
    assert result.maker_fill_fraction == pytest.approx(1.0)


def test_replay_account_maker_touch_without_trade_through_falls_back() -> None:
    """A low exactly equal to the anchor is a touch, so the taker fallback applies."""
    result = _maker_single_day([100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0])

    assert result.fee_paid == pytest.approx(10.0 * 100.0 * 6e-4)
    assert result.maker_fill_fraction == pytest.approx(0.0)


def test_replay_account_maker_sell_fills_on_high_trade_through() -> None:
    """A sell whose window high trades strictly above the anchor fills as maker."""
    result = _maker_single_day(
        [100.0, 100.0], [100.0, 100.0], [100.0, 100.5], weights=[[-1.0]],
    )

    assert result.fee_paid == pytest.approx(10.0 * 100.0 * 2e-4)
    assert result.daily_equity.iloc[0] == pytest.approx(1000.0 - 10.0 * 100.0 * 2e-4)
    assert result.maker_fill_fraction == pytest.approx(1.0)


def test_replay_account_maker_unfilled_crosses_at_window_end_close() -> None:
    """Without penetration the buy crosses at the last window close with taker fees."""
    result = _maker_single_day([100.0, 102.0, 105.0], [100.0, 101.0, 103.0], [100.0, 102.0, 105.0])

    fee = 10.0 * 105.0 * 6e-4
    assert result.fee_paid == pytest.approx(fee)
    assert result.daily_equity.iloc[0] == pytest.approx(1000.0 - 10.0 * (105.0 - 100.0) - fee)
    assert result.maker_fill_fraction == pytest.approx(0.0)


def test_replay_account_maker_window_stays_within_holding_segment() -> None:
    """A penetration past the next entry never fills the current segment's order."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    bars = [dates[0], dates[0] + pd.Timedelta(minutes=3), dates[1], dates[1] + pd.Timedelta(minutes=3)]
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [1.0]])
    index = pd.DatetimeIndex(bars)
    marks = AccountMarkPanels(
        close=pd.DataFrame([[100.0], [101.0], [200.0], [50.0]], index=index, columns=["BTCUSDT"], dtype="float64"),
        high=pd.DataFrame([[100.0], [101.0], [200.0], [50.0]], index=index, columns=["BTCUSDT"], dtype="float64"),
        low=pd.DataFrame([[100.0], [101.0], [200.0], [50.0]], index=index, columns=["BTCUSDT"], dtype="float64"),
    )
    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _fixed_policy(), capital=1000.0, taker_fee_bps=6.0,
        execution="maker", maker_fee_bps=2.0, passive_window_bars=100,
    )

    fee = 10.0 * 101.0 * 6e-4
    assert result.daily_equity.iloc[0] == pytest.approx(1000.0 - 10.0 * (101.0 - 100.0) - fee)


def test_replay_account_maker_nan_bars_never_fill() -> None:
    """All-NaN window lows cannot print a maker fill; the fallback uses finite closes."""
    finite_fallback = _maker_single_day(
        [100.0, 102.0, 103.0],
        [100.0, float("nan"), float("nan")],
        [100.0, float("nan"), float("nan")],
        weights=[[0.01]],
        filters=False,
    )
    fee = 0.1 * 103.0 * 6e-4
    assert finite_fallback.fee_paid == pytest.approx(fee)
    assert finite_fallback.daily_equity.iloc[0] == pytest.approx(1000.0 - 0.1 * 3.0 - fee)
    assert finite_fallback.maker_fill_fraction == pytest.approx(0.0)

    anchor_fallback = _maker_single_day(
        [100.0, float("nan"), float("nan")],
        [100.0, float("nan"), float("nan")],
        [100.0, float("nan"), float("nan")],
        weights=[[0.01]],
        filters=False,
    )
    assert anchor_fallback.fee_paid == pytest.approx(0.1 * 100.0 * 6e-4)
    assert anchor_fallback.maker_fill_fraction == pytest.approx(0.0)


def test_replay_account_maker_nan_anchor_falls_back_to_finite_close() -> None:
    """A non-finite anchor never fills as maker and crosses at the last finite close."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    bars = [dates[0], dates[0] + pd.Timedelta(minutes=3)]
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0]])
    index = pd.DatetimeIndex(bars)
    marks = AccountMarkPanels(
        close=pd.DataFrame([[float("nan")], [50.0]], index=index, columns=["BTCUSDT"], dtype="float64"),
        high=pd.DataFrame([[float("nan")], [50.0]], index=index, columns=["BTCUSDT"], dtype="float64"),
        low=pd.DataFrame([[float("nan")], [49.0]], index=index, columns=["BTCUSDT"], dtype="float64"),
    )
    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _fixed_policy(), capital=1000.0, taker_fee_bps=6.0, apply_order_filters=False,
        execution="maker", maker_fee_bps=2.0, passive_window_bars=10,
    )

    assert result.fee_paid == pytest.approx(1000.0 * 50.0 * 6e-4)
    assert result.maker_fill_fraction == pytest.approx(0.0)


def test_replay_account_maker_min_notional_skip_unchanged() -> None:
    """Below-minimum orders are skipped without any fill judgment under maker."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[0.0], [0.004]])
    marks = _panels(list(dates), ["BTCUSDT"], [[1.0], [1.0]])
    kwargs = {"capital": 1000.0, "taker_fee_bps": 6.0}

    taker = replay_account(
        unit, marks, funding, adv, sigma, _rules(step=1.0, minimum=5.0),
        _fixed_policy(), **kwargs,
    )
    maker = replay_account(
        unit, marks, funding, adv, sigma, _rules(step=1.0, minimum=5.0),
        _fixed_policy(), execution="maker", maker_fee_bps=2.0, passive_window_bars=10, **kwargs,
    )

    assert maker.skipped_orders == taker.skipped_orders == 1
    assert maker.daily_equity.iloc[1] == pytest.approx(1000.0)


def test_replay_account_maker_future_bars_beyond_window_leave_past_unchanged() -> None:
    """Bars past the passive window cannot move equity realized before them."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    bars = [
        dates[0], dates[0] + pd.Timedelta(minutes=3), dates[0] + pd.Timedelta(minutes=6),
        dates[1], dates[1] + pd.Timedelta(minutes=3),
    ]
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [1.0]])
    index = pd.DatetimeIndex(bars)
    base_closes = [[100.0], [100.0], [100.0], [110.0], [112.0]]
    shocked_closes = [[100.0], [100.0], [100.0], [110.0], [168.0]]

    def _run(closes: list[list[float]]) -> pd.Series:
        frame = pd.DataFrame(closes, index=index, columns=["BTCUSDT"], dtype="float64")
        marks = AccountMarkPanels(close=frame, high=frame.copy(), low=frame.copy())
        return replay_account(
            unit, marks, funding, adv, sigma, _rules(minimum=0.0),
            _fixed_policy(), capital=1000.0, taker_fee_bps=6.0,
            execution="maker", maker_fee_bps=2.0, passive_window_bars=2,
        ).daily_equity

    assert _run(base_closes).iloc[0] == pytest.approx(_run(shocked_closes).iloc[0])


def test_replay_account_maker_fees_below_taker_on_identical_fills() -> None:
    """When every order rests to a maker fill at the anchor, fees sit below taker."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    bars = [dates[0], dates[0] + pd.Timedelta(minutes=3)]
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0]])
    index = pd.DatetimeIndex(bars)
    frame = pd.DataFrame([[100.0], [100.0]], index=index, columns=["BTCUSDT"], dtype="float64")
    marks = AccountMarkPanels(
        close=frame, high=frame.copy(),
        low=pd.DataFrame([[100.0], [99.0]], index=index, columns=["BTCUSDT"], dtype="float64"),
    )
    kwargs = {"capital": 1000.0, "taker_fee_bps": 6.0}
    maker = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _fixed_policy(), execution="maker", maker_fee_bps=2.0, passive_window_bars=10, **kwargs,
    )
    taker = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0), _fixed_policy(), **kwargs,
    )

    assert maker.maker_fill_fraction == pytest.approx(1.0)
    assert maker.fee_paid < taker.fee_paid
    assert maker.daily_equity.iloc[0] > taker.daily_equity.iloc[0]


def test_replay_account_maker_inactive_symbol_keeps_quantity() -> None:
    """A zero-delta symbol takes no fill judgment and leaves the maker fraction to actives."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    bars = [dates[0] + pd.Timedelta(minutes=3 * i) for i in range(3)]
    symbols = ["BTCUSDT", "ETHUSDT"]
    unit, funding, adv, sigma = _frames(dates, symbols, [[1.0, 0.0]])
    index = pd.DatetimeIndex(bars)
    marks = AccountMarkPanels(
        close=pd.DataFrame([[100.0, 10.0]] * 3, index=index, columns=symbols, dtype="float64"),
        high=pd.DataFrame([[100.0, 10.0]] * 3, index=index, columns=symbols, dtype="float64"),
        low=pd.DataFrame([[100.0, 10.0], [99.0, 10.0], [100.0, 10.0]], index=index, columns=symbols, dtype="float64"),
    )
    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _fixed_policy(), capital=1000.0, taker_fee_bps=6.0,
        execution="maker", maker_fee_bps=2.0, passive_window_bars=10,
    )

    assert result.fee_paid == pytest.approx(10.0 * 100.0 * 2e-4)
    assert result.maker_fill_fraction == pytest.approx(1.0)


def test_replay_account_maker_invalid_configuration_fails_closed() -> None:
    """Maker without valid fees and window, or with an unknown mode, fails closed."""
    dates = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0]])
    marks = _panels(list(dates), ["BTCUSDT"], [[100.0]])
    base = {
        "unit_weights": unit, "marks": marks, "funding_cum": funding, "adv": adv,
        "daily_sigma": sigma, "rules": _rules(minimum=0.0), "policy": _fixed_policy(),
        "capital": 1000.0, "taker_fee_bps": 6.0, "execution": "maker",
    }
    cases = [
        {"maker_fee_bps": None, "passive_window_bars": 10},
        {"maker_fee_bps": 2.0, "passive_window_bars": None},
        {"maker_fee_bps": 2.0, "passive_window_bars": 0},
        {"maker_fee_bps": 7.0, "passive_window_bars": 10},
        {"maker_fee_bps": float("nan"), "passive_window_bars": 10},
        {"maker_fee_bps": -1.0, "passive_window_bars": 10},
        {"maker_fee_bps": 2.0, "passive_window_bars": float("nan")},
    ]
    for case in cases:
        with pytest.raises(DataIntegrityError):
            replay_account(**{**base, **case})  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError):
        replay_account(**{**base, "execution": "peg"})  # type: ignore[arg-type]


def test_replay_account_maker_conserves_cash() -> None:
    """Maker cash conservation: final equity equals capital plus fill-aware price gains less costs."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    bars = [
        dates[0], dates[0] + pd.Timedelta(minutes=3), dates[0] + pd.Timedelta(minutes=6),
        dates[1], dates[1] + pd.Timedelta(minutes=3),
    ]
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [1.0]])
    index = pd.DatetimeIndex(bars)
    closes = [[100.0], [100.0], [100.0], [110.0], [112.0]]
    lows = [[100.0], [99.0], [100.0], [110.0], [110.0]]
    highs = [[100.0], [100.0], [100.0], [110.0], [112.0]]
    marks = AccountMarkPanels(
        close=pd.DataFrame(closes, index=index, columns=["BTCUSDT"], dtype="float64"),
        high=pd.DataFrame(highs, index=index, columns=["BTCUSDT"], dtype="float64"),
        low=pd.DataFrame(lows, index=index, columns=["BTCUSDT"], dtype="float64"),
    )
    result = replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0),
        _fixed_policy(), capital=1000.0, taker_fee_bps=6.0, apply_order_filters=False,
        execution="maker", maker_fee_bps=2.0, passive_window_bars=10,
    )

    first_qty = 1000.0 / 100.0
    first_fee = first_qty * 100.0 * 2e-4
    first_equity = 1000.0 - first_fee
    pre_second_equity = -first_fee + first_qty * 110.0
    second_qty = pre_second_equity / 110.0
    second_delta = second_qty - first_qty
    second_fee = abs(second_delta) * 110.0 * 2e-4
    expected_final = 1000.0 - first_qty * 100.0 - first_fee - second_delta * 110.0 - second_fee + second_qty * 110.0
    assert result.daily_equity.iloc[0] == pytest.approx(first_equity)
    assert result.daily_equity.iloc[-1] == pytest.approx(expected_final)
    assert result.fee_paid == pytest.approx(first_fee + second_fee)
    assert result.impact_paid == pytest.approx(0.0)
    assert result.funding_paid == pytest.approx(0.0)
    assert result.maker_fill_fraction == pytest.approx(1.0)


def _anchor_replay(
    unit: pd.DataFrame, marks: AccountMarkPanels, funding: pd.DataFrame,
    adv: pd.DataFrame, sigma: pd.DataFrame, anchors: pd.DatetimeIndex, **kwargs: object,
) -> AccountLedgerResult:
    base: dict[str, object] = {"capital": 1000.0, "taker_fee_bps": 6.0}
    base.update(kwargs)
    return _orig_replay_account(
        unit, marks, funding, adv, sigma, base.pop("rules", _rules(minimum=0.0)),  # type: ignore[arg-type]
        base.pop("policy", _fixed_policy()),  # type: ignore[arg-type]
        anchor_times=anchors, **base,  # type: ignore[arg-type]
    )


def test_replay_account_anchor_equal_to_entry_label_is_bit_identical() -> None:
    """Anchors on the entry labels reproduce the hand-computed pre-change ledger."""
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [0.5]])
    marks = _panels(list(dates), ["BTCUSDT"], [[100.0], [101.0]], spread=0.001)
    anchors = pd.DatetimeIndex(dates)
    # 1000 USDT 전액 매수(10.000개, 수수료 0.6) 후 101에서 절반으로 축소(4.997개, step 0.001 절사).
    expected_equity = [999.4, 1009.0968182]
    expected_fee = 0.9031818
    taker = _orig_replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0), _fixed_policy(),
        anchor_times=anchors, capital=1000.0, taker_fee_bps=6.0,
    )
    assert taker.daily_equity.to_list() == pytest.approx(expected_equity)
    assert taker.fee_paid == pytest.approx(expected_fee)
    assert taker.funding_paid == 0.0
    assert taker.liquidated_at is None
    # 앵커 뒤 대기 봉이 없으면 메이커 주문은 앵커 가격에서 테이커로 체결되어 같은 원장이 된다.
    maker = _orig_replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0), _fixed_policy(),
        anchor_times=anchors, capital=1000.0, taker_fee_bps=6.0,
        execution="maker", maker_fee_bps=2.0, passive_window_bars=10,
    )
    assert maker.daily_equity.to_list() == pytest.approx(expected_equity)
    assert maker.maker_fill_fraction == 0.0


def test_replay_account_earlier_anchor_prices_rebalance_at_anchor_close() -> None:
    entries = pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="UTC")])
    anchors = pd.DatetimeIndex([pd.Timestamp("2026-01-01 23:00", tz="UTC")])
    grid = pd.date_range(anchors[0], entries[0] + pd.Timedelta(days=1), freq="3min", tz="UTC")
    closes = [100.0 if ts < entries[0] else 110.0 for ts in grid]
    frame = pd.DataFrame([[c] for c in closes], index=grid, columns=["BTCUSDT"], dtype="float64")
    marks = AccountMarkPanels(close=frame, high=frame.copy(), low=frame.copy())
    unit, funding, adv, sigma = _frames(entries, ["BTCUSDT"], [[1.0]])
    result = _orig_replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0), _fixed_policy(),
        anchor_times=anchors, capital=1000.0, taker_fee_bps=0.0, apply_order_filters=False,
    )
    assert result.daily_equity.iloc[0] == pytest.approx(1000.0)


def test_replay_account_maker_window_starts_after_anchor() -> None:
    entries = pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="UTC")])
    anchor = pd.Timestamp("2026-01-01 23:00", tz="UTC")
    bars = [anchor, anchor + pd.Timedelta(minutes=3), anchor + pd.Timedelta(minutes=6)]
    unit, funding, adv, sigma = _frames(entries, ["BTCUSDT"], [[1.0]])
    anchors = pd.DatetimeIndex([anchor])

    def _marks(dip_bar: int) -> AccountMarkPanels:
        closes = [100.0, 100.0, 100.0]
        lows = [100.0, 100.0, 100.0]
        lows[dip_bar] = 99.0
        index = pd.DatetimeIndex(bars)
        return AccountMarkPanels(
            close=pd.DataFrame([[c] for c in closes], index=index, columns=["BTCUSDT"], dtype="float64"),
            high=pd.DataFrame([[100.0]] * 3, index=index, columns=["BTCUSDT"], dtype="float64"),
            low=pd.DataFrame([[v] for v in lows], index=index, columns=["BTCUSDT"], dtype="float64"),
        )

    filled = _orig_replay_account(
        unit, _marks(1), funding, adv, sigma, _rules(minimum=0.0), _fixed_policy(),
        anchor_times=anchors, capital=1000.0, taker_fee_bps=6.0,
        execution="maker", maker_fee_bps=2.0, passive_window_bars=1,
    )
    assert filled.maker_fill_fraction == pytest.approx(1.0)
    assert filled.fee_paid == pytest.approx(10.0 * 100.0 * 2e-4)
    crossed = _orig_replay_account(
        unit, _marks(2), funding, adv, sigma, _rules(minimum=0.0), _fixed_policy(),
        anchor_times=anchors, capital=1000.0, taker_fee_bps=6.0,
        execution="maker", maker_fee_bps=2.0, passive_window_bars=1,
    )
    assert crossed.maker_fill_fraction == pytest.approx(0.0)


def test_replay_account_funding_step_spans_anchor_to_anchor() -> None:
    anchors = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01 23:00", tz="UTC"), pd.Timestamp("2026-01-02 23:00", tz="UTC")]
    )
    entries = pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="UTC"), pd.Timestamp("2026-01-03", tz="UTC")])
    grid = pd.date_range(anchors[0], entries[-1] + pd.Timedelta(days=1), freq="3min", tz="UTC")
    frame = pd.DataFrame(100.0, index=grid, columns=["BTCUSDT"], dtype="float64")
    marks = AccountMarkPanels(close=frame, high=frame.copy(), low=frame.copy())
    unit, _, adv, sigma = _frames(entries, ["BTCUSDT"], [[1.0], [1.0]])
    funding = pd.DataFrame([[0.0], [0.001]], index=entries, columns=["BTCUSDT"], dtype="float64")
    result = _orig_replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0), _fixed_policy(),
        anchor_times=anchors, capital=1000.0, taker_fee_bps=0.0, apply_order_filters=False,
    )
    assert result.funding_paid == pytest.approx(10.0 * 100.0 * 0.001)


def test_replay_account_intraday_drawdown_sees_recovered_dip() -> None:
    anchors = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01 23:00", tz="UTC"), pd.Timestamp("2026-01-02 23:00", tz="UTC")]
    )
    entries = pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="UTC"), pd.Timestamp("2026-01-03", tz="UTC")])
    grid = pd.date_range(anchors[0], entries[-1] + pd.Timedelta(days=1), freq="3min", tz="UTC")
    closes = [100.0] * len(grid)
    mid = len(grid) // 4
    closes[mid] = 80.0
    frame = pd.DataFrame([[c] for c in closes], index=grid, columns=["BTCUSDT"], dtype="float64")
    low = frame.copy()
    low.iloc[mid, 0] = 80.0
    marks = AccountMarkPanels(close=frame, high=frame.copy(), low=low)
    unit, funding, adv, sigma = _frames(entries, ["BTCUSDT"], [[1.0], [1.0]])
    result = _orig_replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0, ratio=0.0, leverage=100),
        _fixed_policy(exposure_max=1.0), anchor_times=anchors,
        capital=1000.0, taker_fee_bps=0.0, apply_order_filters=False,
    )
    assert result.daily_equity.iloc[0] == pytest.approx(1000.0)
    assert result.daily_equity.iloc[1] == pytest.approx(1000.0)
    assert result.intraday_max_drawdown == pytest.approx(0.20)


def test_replay_account_intraday_drawdown_peak_carries_across_segments() -> None:
    anchors = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01 23:00", tz="UTC"),
            pd.Timestamp("2026-01-02 23:00", tz="UTC"),
            pd.Timestamp("2026-01-03 23:00", tz="UTC"),
        ]
    )
    entries = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-02", tz="UTC"), pd.Timestamp("2026-01-03", tz="UTC"), pd.Timestamp("2026-01-04", tz="UTC")]
    )
    grid = pd.date_range(anchors[0], entries[-1] + pd.Timedelta(days=1), freq="3min", tz="UTC")
    a1 = grid.get_loc(anchors[1])
    closes = [100.0] * len(grid)
    closes[a1 // 2] = 120.0
    closes[a1 + (len(grid) - a1) // 2] = 110.0
    frame = pd.DataFrame([[c] for c in closes], index=grid, columns=["BTCUSDT"], dtype="float64")
    marks = AccountMarkPanels(close=frame, high=frame.copy(), low=frame.copy())
    unit, funding, adv, sigma = _frames(entries, ["BTCUSDT"], [[1.0], [1.0], [1.0]])
    result = _orig_replay_account(
        unit, marks, funding, adv, sigma, _rules(minimum=0.0, ratio=0.0, leverage=100),
        _fixed_policy(exposure_max=1.0), anchor_times=anchors,
        capital=1000.0, taker_fee_bps=0.0, apply_order_filters=False,
    )
    assert result.intraday_max_drawdown == pytest.approx(1.0 - 1000.0 / 1200.0)


def test_replay_account_liquidation_sets_full_drawdown() -> None:
    result = _wick_replay(50.0)
    assert result.liquidated_at is not None
    assert result.intraday_max_drawdown == 1.0


def test_replay_account_misaligned_anchors_fail_closed() -> None:
    base = _valid_kwargs()
    dates = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    short = pd.DatetimeIndex(dates[:1])
    with pytest.raises(DataIntegrityError):
        _orig_replay_account(**{**base, "anchor_times": short})  # type: ignore[arg-type]
    dup = pd.DatetimeIndex([dates[0], dates[0]])
    with pytest.raises(DataIntegrityError):
        _orig_replay_account(**{**base, "anchor_times": dup})  # type: ignore[arg-type]
    off = pd.DatetimeIndex([dates[0], dates[1] + pd.Timedelta(minutes=1)])
    with pytest.raises(DataIntegrityError):
        _orig_replay_account(**{**base, "anchor_times": off})  # type: ignore[arg-type]


def test_replay_account_future_bars_never_change_past_entries() -> None:
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    anchors = pd.DatetimeIndex(dates)
    unit, funding, adv, sigma = _frames(dates, ["BTCUSDT"], [[1.0], [1.0], [1.0]])
    grid = pd.DatetimeIndex([*list(dates), dates[-1] + pd.Timedelta(hours=6)])
    closes = [[100.0], [101.0], [102.0], [103.0]]
    frame = pd.DataFrame(closes, index=grid, columns=["BTCUSDT"], dtype="float64")
    base_marks = AccountMarkPanels(close=frame, high=frame.copy(), low=frame.copy())
    shocked_frame = frame.copy()
    shocked_frame.iloc[2:, 0] = shocked_frame.iloc[2:, 0] * 2.5
    shocked_marks = AccountMarkPanels(
        close=shocked_frame, high=shocked_frame.copy(), low=shocked_frame.copy()
    )
    kwargs: dict[str, object] = {"capital": 1000.0, "taker_fee_bps": 6.0}
    first = _orig_replay_account(
        unit, base_marks, funding, adv, sigma, _rules(minimum=1.0), _fixed_policy(),
        anchor_times=anchors, **kwargs,  # type: ignore[arg-type]
    )
    second = _orig_replay_account(
        unit, shocked_marks, funding, adv, sigma, _rules(minimum=1.0), _fixed_policy(),
        anchor_times=anchors, **kwargs,  # type: ignore[arg-type]
    )
    assert first.daily_equity.iloc[:2].equals(second.daily_equity.iloc[:2])
