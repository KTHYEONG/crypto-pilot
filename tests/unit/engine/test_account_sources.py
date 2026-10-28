"""Account input assembly preserves release anchors and observable liquidity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.account_sources import assemble_account_inputs
from src.strategy.targets import FLOW_MOM_TOP20, StrategyTargets
from src.engine.strategy_backtest import StrategySourceContext
from src.core.resources import resolve_mhs_memory_budget


def test_account_inputs_share_entry_labels_without_using_entry_day_liquidity(tmp_path: Path) -> None:
    entries = pd.date_range("2021-02-01", periods=2, tz="UTC")
    releases = entries - pd.Timedelta(hours=1)
    weights = pd.DataFrame({"HELDUSDT": [0.5, -0.5], "IDLEUSDT": [0.0, 0.0]}, index=entries)
    candidate = StrategyTargets(
        target_weights=weights, signal_available_at=releases, strategy=FLOW_MOM_TOP20,
    )
    grid = pd.date_range(
        releases[0] - pd.Timedelta(minutes=3), entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left",
    )
    marks = tmp_path / "3m"
    marks.mkdir()
    pd.DataFrame({
        "timestamp": grid.as_unit("ns").asi8 // 1_000_000,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    }).to_parquet(marks / "HELDUSDT.parquet")
    daily_index = pd.date_range("2021-01-01", "2021-02-02", tz="UTC")
    close = pd.DataFrame(100.0, index=daily_index, columns=weights.columns)
    quote = pd.DataFrame(2e6, index=daily_index, columns=weights.columns)
    quote.loc[entries] = 2e8
    funding = pd.Series(
        [0.001, 0.002], index=pd.DatetimeIndex([releases[0] + pd.Timedelta(minutes=3), releases[1] - pd.Timedelta(minutes=3)]),
    )
    context = StrategySourceContext(
        census=tuple(weights.columns), root=str(tmp_path),
        funding_by_symbol={"HELDUSDT": funding}, funding_failures={},
        budget=resolve_mhs_memory_budget(None), daily_close=close, daily_quote_volume=quote,
    )
    unit, prices, cumulative, adv, sigma, anchors = assemble_account_inputs(candidate, context)
    pd.testing.assert_frame_equal(unit, weights[["HELDUSDT"]])
    expected_anchors = (releases - pd.Timedelta(minutes=3)).floor("3min")
    pd.testing.assert_index_equal(anchors, expected_anchors)
    assert prices.close.index[0] == expected_anchors[0]
    assert list(cumulative["HELDUSDT"]) == [0.0, 0.003]
    assert list(adv["HELDUSDT"]) == [2e6, 2e6]
    assert list(sigma["HELDUSDT"]) == [0.0, 0.0]
    for frame in (cumulative, adv, sigma):
        pd.testing.assert_index_equal(frame.index, entries)


def _assemble_trend(
    tmp_path: Path, name: str, releases: pd.DatetimeIndex, entries: pd.DatetimeIndex,
    *, garbage_factor: float = 1.0,
) -> tuple[tuple, pd.DatetimeIndex]:
    """Assemble one-symbol inputs on trending 3m bars; non-anchor bars are scaled by
    ``garbage_factor`` to prove post-release bars never inform anchor pricing."""
    sym = "HELDUSDT"
    weights = pd.DataFrame([[0.5]] * len(entries), index=entries, columns=[sym], dtype="float64")
    candidate = StrategyTargets(
        target_weights=weights, signal_available_at=releases, strategy=FLOW_MOM_TOP20,
    )
    anchors = (releases - pd.Timedelta(minutes=3)).floor("3min")
    grid = pd.date_range(anchors[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    close = 100.0 + np.arange(len(grid), dtype="float64")
    high = close + 1.0
    low = close - 1.0
    if garbage_factor != 1.0:
        perturbed = ~grid.isin(anchors)
        close = np.where(perturbed, close * garbage_factor, close)
        high = np.where(perturbed, high * garbage_factor, high)
        low = np.where(perturbed, low * garbage_factor, low)
    marks_dir = tmp_path / name / "3m"
    marks_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "timestamp": grid.as_unit("ns").asi8 // 1_000_000,
        "open": close, "high": high, "low": low, "close": close,
    }).to_parquet(marks_dir / f"{sym}.parquet")
    daily_index = pd.date_range(entries[0] - pd.Timedelta(days=30), entries[-1] + pd.Timedelta(days=1), tz="UTC")
    context = StrategySourceContext(
        census=(sym,), root=str(tmp_path / name),
        funding_by_symbol={}, funding_failures={},
        budget=resolve_mhs_memory_budget(None),
        daily_close=pd.DataFrame(100.0, index=daily_index, columns=[sym], dtype="float64"),
        daily_quote_volume=pd.DataFrame(2e6, index=daily_index, columns=[sym], dtype="float64"),
    )
    return assemble_account_inputs(candidate, context), grid


def test_anchor_is_bar_closing_at_on_grid_release(tmp_path: Path) -> None:
    entries = pd.DatetimeIndex(["2024-03-15"], tz="UTC")
    releases = pd.DatetimeIndex(["2024-03-14 23:00"], tz="UTC")
    (unit, marks, *_ , anchors), grid = _assemble_trend(tmp_path, "on_grid", releases, entries)
    assert anchors[0] == pd.Timestamp("2024-03-14 22:57", tz="UTC")
    assert marks.close.loc[anchors[0], "HELDUSDT"] == pytest.approx(100.0 + float(grid.get_loc(anchors[0])))


def test_off_grid_release_never_anchors_on_later_closing_bar(tmp_path: Path) -> None:
    entries = pd.DatetimeIndex(["2024-03-15"], tz="UTC")
    releases = pd.DatetimeIndex(["2024-03-14 23:01:10"], tz="UTC")
    (_unit, _marks, *_ , anchors), _grid = _assemble_trend(tmp_path, "off_grid", releases, entries)
    assert anchors[0] == pd.Timestamp("2024-03-14 22:57", tz="UTC")


def test_grid_starts_at_first_anchor(tmp_path: Path) -> None:
    entries = pd.DatetimeIndex(["2024-03-15", "2024-03-16"], tz="UTC")
    releases = pd.DatetimeIndex(["2024-03-14 23:00", "2024-03-15 23:00"], tz="UTC")
    (_unit, marks, *_ , anchors), _grid = _assemble_trend(tmp_path, "grid_start", releases, entries)
    assert marks.close.index[0] == anchors[0]
    assert all(anchor in marks.close.index for anchor in anchors)


def test_causality_perturbation_keeps_anchor_pricing_and_sizing(tmp_path: Path) -> None:
    from src.engine.account_ledger import replay_account
    from src.market_data.binance.venue_rules import VenueBracket, VenueRuleSnapshot, VenueSymbolRules
    from src.strategy.sizing import ExposurePolicy

    entries = pd.DatetimeIndex(["2024-03-15", "2024-03-16"], tz="UTC")
    releases = pd.DatetimeIndex(["2024-03-14 23:00", "2024-03-15 23:00"], tz="UTC")
    clean, _grid = _assemble_trend(tmp_path, "clean", releases, entries)
    perturbed, _ = _assemble_trend(tmp_path, "perturbed", releases, entries, garbage_factor=1e3)
    pd.testing.assert_index_equal(perturbed[-1], clean[-1])
    pd.testing.assert_series_equal(
        perturbed[1].close.loc[clean[-1], "HELDUSDT"], clean[1].close.loc[clean[-1], "HELDUSDT"],
    )
    rules = VenueRuleSnapshot(
        captured_at=pd.Timestamp("2024-01-01", tz="UTC"),
        symbols={"HELDUSDT": VenueSymbolRules(
            symbol="HELDUSDT", brackets=(VenueBracket(0.0, 1e12, 0.01, 0.0, 10),),
            step_size=0.001, min_notional=5.0,
        )},
    )
    policy = ExposurePolicy(
        kind="fixed", exposure_max=1.0, exposure_step=0.25, mean_haircut=0.5,
        prior_days=730.0, min_moment_days=30, shock_per_unit=0.08,
        margin_reserve=0.10, initial_margin_cap=0.90, impact_y=0.0,
    )
    kwargs = {"rules": rules, "policy": policy, "capital": 100000.0, "taker_fee_bps": 6.0,
              "apply_order_filters": False}
    first = replay_account(*clean[:5], anchor_times=clean[-1], **kwargs)
    second = replay_account(*perturbed[:5], anchor_times=perturbed[-1], **kwargs)
    pd.testing.assert_series_equal(first.daily_exposure, second.daily_exposure)
    pd.testing.assert_series_equal(first.daily_equity, second.daily_equity)
