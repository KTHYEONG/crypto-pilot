from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.portfolio.growth_strategy_library import (
    FAMILY_SIZE,
    RETIRED_STRATEGY_IDS,
    STRATEGY_REGISTRY,
    align_funding_bars,
    build_growth_strategy_weights,
    registry_definition,
    screen_growth_strategy_weights,
)

_SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT")
_MAX_POSITIONS = 5


def _grid(months: int = 3, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=months * 30 * 6, freq="4h", tz="UTC")


def _months(grid: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return sorted(pd.unique(grid - pd.offsets.MonthBegin(0)))


def _schedule(grid: pd.DatetimeIndex) -> dict[pd.Timestamp, tuple[str, ...]]:
    return dict.fromkeys(_months(grid), _SYMBOLS)


def _prices(grid: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    columns: dict[str, np.ndarray] = {}
    for i, symbol in enumerate(_SYMBOLS):
        columns[symbol] = 100.0 * (1.0 + np.cumsum(rng.normal(0.0, 0.01, len(grid))) * (1.0 + i))
    return pd.DataFrame(columns, index=grid)


def _taker(grid: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            symbol: 0.5 + 0.05 * np.sin(np.arange(len(grid)) / (7.0 + i))
            + 0.01 * rng.standard_normal(len(grid))
            for i, symbol in enumerate(_SYMBOLS)
        },
        index=grid,
    )


def _funding(grid: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            symbol: 1e-4 * np.sin(np.arange(len(grid)) / (5.0 + i))
            for i, symbol in enumerate(_SYMBOLS)
        },
        index=grid,
    )


def _zero(grid: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=grid, columns=list(_SYMBOLS))


def _late_funding(grid: pd.DatetimeIndex) -> pd.Series:
    """Funding history that starts after ``grid[0]`` and extends past window_end."""
    span_start = pd.Timestamp("2024-02-01", tz="UTC")
    times = [span_start + pd.Timedelta(hours=8 * i) for i in range(90)]
    times.append(grid[-1] + pd.Timedelta(days=30))
    return pd.Series(
        1e-4 * np.sin(np.arange(len(times))),
        index=pd.DatetimeIndex(times, tz="UTC"),
    )


class TestRegistry:
    def test_registry_is_exactly_four_families_and_twelve_variants(self) -> None:
        # GSD-04: registry is exactly four families and twelve variants.
        assert tuple(d.strategy_id for d in STRATEGY_REGISTRY) == (
            "funding_contrarian_v1",
            "taker_imbalance_v1",
            "vol_adjusted_trend_v1",
            "donchian_channel_position_v1",
        )
        assert tuple(d.windows for d in STRATEGY_REGISTRY) == (
            (42, 84, 168),
            (42, 84, 168),
            (42, 84, 180),
            (42, 84, 168),
        )
        assert FAMILY_SIZE == 12

    def test_unknown_strategy_id_raises_value_error(self) -> None:
        # GSD-04: unknown strategy id raises ValueError.
        grid = _grid()
        px = _prices(grid)
        with pytest.raises(ValueError, match="unknown growth strategy identity"):
            build_growth_strategy_weights(
                "not_a_strategy_v9", 42, _schedule(grid), px, _taker(grid), _zero(grid), _MAX_POSITIONS,
            )

    @pytest.mark.parametrize("retired", RETIRED_STRATEGY_IDS)
    def test_retired_identities_raise_value_error(self, retired: str) -> None:
        # GSD-04: retired identities cannot be reintroduced.
        grid = _grid()
        px = _prices(grid)
        with pytest.raises(ValueError, match="retired"):
            build_growth_strategy_weights(
                retired, 42, _schedule(grid), px, _taker(grid), _zero(grid), _MAX_POSITIONS,
            )
        with pytest.raises(ValueError, match="retired"):
            registry_definition(retired)

    def test_unregistered_window_raises_value_error(self) -> None:
        grid = _grid()
        px = _prices(grid)
        with pytest.raises(ValueError, match="not a registered window"):
            build_growth_strategy_weights(
                "taker_imbalance_v1", 37, _schedule(grid), px, _taker(grid), _zero(grid), _MAX_POSITIONS,
            )


class TestBuildWeights:
    @pytest.mark.parametrize(
        "strategy_id",
        ["taker_imbalance_v1", "vol_adjusted_trend_v1", "donchian_channel_position_v1"],
    )
    def test_price_only_weights_are_roster_only_and_dollar_neutral(self, strategy_id: str) -> None:
        # GSD-01: every valid score produces PIT-roster-only, dollar-neutral
        # weights; invalid rows (early bars without a full window) are zero.
        grid = _grid()
        px = _prices(grid)
        schedule = _schedule(grid)
        weights = build_growth_strategy_weights(
            strategy_id, 42, schedule, px, _taker(grid), _zero(grid), _MAX_POSITIONS,
        )
        assert weights.index.equals(px.index)
        assert list(weights.columns) == list(px.columns)
        assert np.allclose(weights.sum(axis=1).to_numpy(), 0.0)
        non_roster = [c for c in px.columns if c not in _SYMBOLS]
        if non_roster:
            assert np.allclose(weights[non_roster].to_numpy(), 0.0)
        early = weights.iloc[:41]
        assert np.allclose(early.to_numpy(), 0.0)

    def test_funding_contrarian_weights_are_dollar_neutral(self) -> None:
        grid = _grid()
        px = _prices(grid)
        weights = build_growth_strategy_weights(
            "funding_contrarian_v1", 42, _schedule(grid), px, _taker(grid), _funding(grid), _MAX_POSITIONS,
        )
        assert np.allclose(weights.sum(axis=1).to_numpy(), 0.0)
        assert not np.allclose(weights.iloc[100:].to_numpy(), 0.0)

    def test_roster_symbols_only_receive_weight(self) -> None:
        grid = _grid()
        px = _prices(grid)
        px["ZZZUSDT"] = 100.0
        schedule = _schedule(grid)
        weights = build_growth_strategy_weights(
            "donchian_channel_position_v1", 42, schedule, px, _taker(grid), _zero(grid), _MAX_POSITIONS,
        )
        assert np.allclose(weights["ZZZUSDT"].to_numpy(), 0.0)

    def test_funding_score_at_t_ignores_later_settlements(self) -> None:
        # GSD-02: funding score at t is unchanged when a later settlement changes.
        grid = _grid()
        px = _prices(grid)
        schedule = _schedule(grid)
        funding = _funding(grid)
        baseline = build_growth_strategy_weights(
            "funding_contrarian_v1", 42, schedule, px, _taker(grid), funding, _MAX_POSITIONS,
        )
        mutated = funding.copy()
        t = 100
        mutated.iloc[t + 5, :] += 0.01
        mutated.iloc[t, :] += 0.05
        changed = build_growth_strategy_weights(
            "funding_contrarian_v1", 42, schedule, px, _taker(grid), mutated, _MAX_POSITIONS,
        )
        assert np.allclose(baseline.iloc[t].to_numpy(), changed.iloc[t].to_numpy())

    def test_early_rows_before_full_window_are_zero(self) -> None:
        grid = _grid()
        px = _prices(grid)
        weights = build_growth_strategy_weights(
            "vol_adjusted_trend_v1", 42, _schedule(grid), px, _taker(grid), _zero(grid), _MAX_POSITIONS,
        )
        assert np.allclose(weights.iloc[:41].to_numpy(), 0.0)

    # GPR-01-POSITION-CAP
    @pytest.mark.parametrize("max_positions", [2, 3, 4, 5])
    def test_max_positions_bounds_active_symbols_on_every_bar(self, max_positions: int) -> None:
        grid = _grid()
        px = _prices(grid)
        weights = build_growth_strategy_weights(
            "taker_imbalance_v1", 42, _schedule(grid), px, _taker(grid), _zero(grid), max_positions,
        )
        active = np.count_nonzero(weights.to_numpy() != 0.0, axis=1)
        assert active.max() <= max_positions
        # gross long and short notional match exactly (dollar neutral)
        arr = weights.to_numpy()
        assert np.allclose(np.clip(arr, 0.0, None).sum(axis=1), np.clip(-arr, 0.0, None).sum(axis=1))
        # a fully warmed bar is active (the book fills) and never exceeds the cap
        assert active[100:].max() > 0
        assert np.allclose(weights.sum(axis=1).to_numpy(), 0.0)

    # GPR-01-POSITION-CAP
    @pytest.mark.parametrize("max_positions", [0, 1, -3])
    def test_max_positions_below_two_returns_all_zero(self, max_positions: int) -> None:
        grid = _grid()
        px = _prices(grid)
        weights = build_growth_strategy_weights(
            "taker_imbalance_v1", 42, _schedule(grid), px, _taker(grid), _zero(grid), max_positions,
        )
        assert np.allclose(weights.to_numpy(), 0.0)
        assert np.allclose(weights.sum(axis=1).to_numpy(), 0.0)


class TestFundingIntegrity:
    def test_missing_funding_invalidates_only_funding_candidate(self) -> None:
        # GSD-03: missing funding invalidates only funding_contrarian_v1.
        grid = _grid()
        px = _prices(grid)
        schedule = _schedule(grid)
        taker = _taker(grid)
        funding = _funding(grid)
        funding = funding.drop(columns=["AAAUSDT"])
        funding_screen = screen_growth_strategy_weights(
            "funding_contrarian_v1", 42, schedule, px, taker, funding, _MAX_POSITIONS,
        )
        assert funding_screen.status == "DATA_INVALID"
        assert "missing funding" in funding_screen.reason
        for strategy_id in ("taker_imbalance_v1", "vol_adjusted_trend_v1", "donchian_channel_position_v1"):
            screen = screen_growth_strategy_weights(
                strategy_id, 42, schedule, px, taker, funding, _MAX_POSITIONS,
            )
            assert screen.status == "SCREENED"

    def test_non_finite_funding_raises_data_integrity_error(self) -> None:
        grid = _grid()
        px = _prices(grid)
        schedule = _schedule(grid)
        funding = _funding(grid)
        # A genuine non-finite rate (inf) survives the per-column dropna used to
        # remove cross-symbol union NaN, so the finite contract still fails closed.
        funding.loc[grid[50], "BBBUSDT"] = np.inf
        with pytest.raises(DataIntegrityError, match="finite"):
            build_growth_strategy_weights(
                "funding_contrarian_v1", 42, schedule, px, _taker(grid), funding, _MAX_POSITIONS,
            )
        screen = screen_growth_strategy_weights(
            "funding_contrarian_v1", 42, schedule, px, _taker(grid), funding, _MAX_POSITIONS,
        )
        assert screen.status == "DATA_INVALID"

    def test_empty_funding_frame_invalidates_funding_candidate(self) -> None:
        grid = _grid()
        px = _prices(grid)
        empty = pd.DataFrame(index=grid)
        screen = screen_growth_strategy_weights(
            "funding_contrarian_v1", 42, _schedule(grid), px, _taker(grid), empty, _MAX_POSITIONS,
        )
        assert screen.status == "DATA_INVALID"

    def test_non_utc_funding_raises_data_integrity_error(self) -> None:
        grid = _grid()
        px = _prices(grid)
        funding = _funding(grid)
        funding.index = funding.index.tz_localize(None)
        with pytest.raises(DataIntegrityError, match="tz-aware"):
            build_growth_strategy_weights(
                "funding_contrarian_v1", 42, _schedule(grid), px, _taker(grid), funding, _MAX_POSITIONS,
            )

    def test_non_monotonic_funding_raises_data_integrity_error(self) -> None:
        grid = _grid()
        px = _prices(grid)
        funding = _funding(grid)
        funding = funding.iloc[::-1]
        with pytest.raises(DataIntegrityError, match="monotonic"):
            build_growth_strategy_weights(
                "funding_contrarian_v1", 42, _schedule(grid), px, _taker(grid), funding, _MAX_POSITIONS,
            )

    def test_unalignable_funding_raises_data_integrity_error(self) -> None:
        # A funding history that begins after the symbol's scheduled span is
        # unalignable: the roster period would otherwise be silently zero-funded.
        grid = _grid()
        px = _prices(grid)
        funding = _funding(grid).loc[grid[50]:]
        with pytest.raises(DataIntegrityError, match="alignable"):
            build_growth_strategy_weights(
                "funding_contrarian_v1", 42, _schedule(grid), px, _taker(grid), funding, _MAX_POSITIONS,
            )

    def test_price_only_candidates_ignore_garbage_funding(self) -> None:
        grid = _grid()
        px = _prices(grid)
        funding = _funding(grid)
        funding.loc[grid[50], "AAAUSDT"] = np.nan
        screen = screen_growth_strategy_weights(
            "donchian_channel_position_v1", 42, _schedule(grid), px, _taker(grid), funding, _MAX_POSITIONS,
        )
        assert screen.status == "SCREENED"


class TestAlignFundingBars:
    def test_settled_and_forward_buckets_differ_by_decision_boundary(self) -> None:
        grid = _grid(1)
        events = pd.Series(
            [1.0, 1.0, 1.0],
            index=pd.DatetimeIndex(
                [grid[0], grid[5] + pd.Timedelta(hours=1), grid[10]], tz="UTC",
            ),
        )
        raw = pd.DataFrame({"A": events})
        settled = align_funding_bars(raw, grid, forward=False)
        forward = align_funding_bars(raw, grid, forward=True)
        assert float(settled.loc[grid[0], "A"]) == 1.0
        assert float(forward.loc[grid[0], "A"]) == 0.0
        assert float(forward.loc[grid[5], "A"]) == 1.0
        assert float(forward.loc[grid[9], "A"]) == 1.0
        assert float(forward.loc[grid[10], "A"]) == 0.0

    def test_rejects_missing_utc_tz(self) -> None:
        grid = _grid(1)
        raw = pd.DataFrame({"A": [1.0]}, index=grid.tz_localize(None))
        with pytest.raises(DataIntegrityError, match="tz-aware"):
            align_funding_bars(raw, grid, forward=False)

    def test_rejects_non_utc_tz(self) -> None:
        grid = _grid(1)
        raw = pd.DataFrame({"A": [1.0]}, index=grid.tz_convert("Asia/Seoul"))
        with pytest.raises(DataIntegrityError, match="UTC"):
            align_funding_bars(raw, grid, forward=False)

    def test_rejects_non_finite_rates(self) -> None:
        grid = _grid(1)
        raw = pd.DataFrame({"A": [1.0, np.inf]}, index=grid[:2])
        with pytest.raises(DataIntegrityError, match="finite"):
            align_funding_bars(raw, grid, forward=False)

    def test_rejects_short_grid(self) -> None:
        raw = pd.DataFrame({"A": [1.0]}, index=_grid(1)[:1])
        with pytest.raises(DataIntegrityError, match="two bars"):
            align_funding_bars(raw, _grid(1)[:1], forward=False)

    def test_align_funding_bars_dropna_removes_cross_symbol_union_nan(self) -> None:
        # C1: pd.DataFrame(dict) unions disjoint per-symbol event timestamps so
        # each column carries non-adjacent NaN rows from the union; those rows
        # are dropped per column and must not trigger the finite check.
        grid = _grid(1)
        base = grid[0]
        offsets = [pd.Timedelta(hours=8 * i) for i in range(3)]
        index_a = pd.DatetimeIndex([base + o for o in offsets], tz="UTC")
        index_b = pd.DatetimeIndex(
            [base + o + pd.Timedelta(milliseconds=1) for o in offsets], tz="UTC",
        )
        raw = pd.DataFrame(
            {"AAA": pd.Series([1e-4] * 3, index=index_a),
             "BBB": pd.Series([2e-4] * 3, index=index_b)},
        )
        aligned = align_funding_bars(raw, grid, forward=False)
        assert np.isfinite(aligned.to_numpy(dtype=np.float64)).all()
        assert float(aligned["AAA"].sum()) == pytest.approx(3e-4)
        assert float(aligned["BBB"].sum()) == pytest.approx(6e-4)

    def test_align_funding_bars_still_raises_on_genuine_non_finite_value(self) -> None:
        # C1 fail-closed: inf (and coerced non-numeric values) survive dropna and
        # still violate the finite contract.
        grid = _grid(1)
        raw = pd.DataFrame({"A": [1.0, np.inf]}, index=grid[:2])
        with pytest.raises(DataIntegrityError, match="finite"):
            align_funding_bars(raw, grid, forward=False)
        raw_text = pd.DataFrame({"A": [1.0, "garbage"]}, index=grid[:2])
        with pytest.raises(DataIntegrityError, match="finite"):
            align_funding_bars(raw_text, grid, forward=False)

    def test_align_funding_bars_existing_shared_index_behavior_unchanged(self) -> None:
        # C1 regression: one shared index (no union NaN) keeps the legacy
        # bucketed sums byte-for-byte.
        grid = _grid(1)
        events = pd.Series(
            [1.0, 1.0, 1.0],
            index=pd.DatetimeIndex(
                [grid[0], grid[5] + pd.Timedelta(hours=1), grid[10]], tz="UTC",
            ),
        )
        raw = pd.DataFrame({"A": events, "B": events * 2.0})
        settled = align_funding_bars(raw, grid, forward=False)
        assert float(settled.loc[grid[0], "A"]) == 1.0
        assert float(settled.loc[grid[5], "A"]) == 1.0
        assert float(settled.loc[grid[10], "A"]) == 1.0
        assert float(settled.loc[grid[10], "B"]) == 2.0

    def test_align_funding_bars_symbol_spans_scopes_causal_window(self) -> None:
        # C2: a symbol listed after grid[0] whose funding covers its scheduled
        # span (with post-window data-collection events) is valid once scoped,
        # while the global-grid check still rejects it.
        grid = _grid(3)
        span_start = pd.Timestamp("2024-02-01", tz="UTC")
        span_end = pd.Timestamp("2024-02-01", tz="UTC")
        events = pd.Series(
            [1e-4] * 5,
            index=pd.DatetimeIndex(
                [
                    span_start,
                    span_start + pd.Timedelta(hours=8),
                    span_start + pd.Timedelta(hours=16),
                    span_start + pd.Timedelta(hours=24),
                    grid[-1] + pd.Timedelta(days=30),
                ],
                tz="UTC",
            ),
        )
        raw = pd.DataFrame({"AAA": events})
        symbol_spans = {"AAA": (span_start, span_end)}
        aligned = align_funding_bars(raw, grid, forward=False, symbol_spans=symbol_spans)
        assert np.isfinite(aligned.to_numpy(dtype=np.float64)).all()

    def test_align_funding_bars_symbol_spans_none_preserves_global_grid_check(self) -> None:
        grid = _grid(3)
        span_start = pd.Timestamp("2024-02-01", tz="UTC")
        events = pd.Series(
            [1e-4] * 5,
            index=pd.DatetimeIndex(
                [
                    span_start,
                    span_start + pd.Timedelta(hours=8),
                    span_start + pd.Timedelta(hours=16),
                    span_start + pd.Timedelta(hours=24),
                    grid[-1] + pd.Timedelta(days=30),
                ],
                tz="UTC",
            ),
        )
        raw = pd.DataFrame({"AAA": events})
        with pytest.raises(DataIntegrityError, match="alignable"):
            align_funding_bars(raw, grid, forward=False)

    def test_align_funding_bars_symbol_spans_still_fails_on_mid_series_gap(self) -> None:
        grid = _grid(3)
        span_start = pd.Timestamp("2024-02-01", tz="UTC")
        span_end = span_start + pd.Timedelta(days=30)
        times = [span_start + pd.Timedelta(hours=8 * i) for i in range(12)]
        times += [span_end + pd.Timedelta(hours=8 * i) for i in range(3)]
        raw = pd.DataFrame(
            {"AAA": pd.Series([1e-4] * len(times), index=pd.DatetimeIndex(times, tz="UTC"))},
        )
        symbol_spans = {"AAA": (span_start, span_end)}
        with pytest.raises(DataIntegrityError, match="gap"):
            align_funding_bars(raw, grid, forward=False, symbol_spans=symbol_spans)

    def test_build_growth_strategy_weights_derives_symbol_spans_from_schedule(self) -> None:
        # C2 wiring: build derives per-symbol spans from the schedule and scopes
        # the causal check, so a symbol listed mid-grid with post-window funding
        # events screens successfully instead of failing DATA_INVALID.
        grid = _grid(3)
        schedule = dict.fromkeys(
            [pd.Timestamp("2024-02-01", tz="UTC"), pd.Timestamp("2024-03-01", tz="UTC")],
            ("AAAUSDT", "BBBUSDT"),
        )
        px = _prices(grid)
        funding = pd.DataFrame({"AAAUSDT": _late_funding(grid), "BBBUSDT": _late_funding(grid)})
        screen = screen_growth_strategy_weights(
            "funding_contrarian_v1", 42, schedule, px, _taker(grid), funding, _MAX_POSITIONS,
        )
        assert screen.status == "SCREENED"
        assert screen.reason is None


class TestScreenResult:
    def test_screened_result_carries_weights_and_identity(self) -> None:
        grid = _grid()
        px = _prices(grid)
        screen = screen_growth_strategy_weights(
            "taker_imbalance_v1", 42, _schedule(grid), px, _taker(grid), _zero(grid), _MAX_POSITIONS,
        )
        assert screen.strategy_id == "taker_imbalance_v1"
        assert screen.parameter == 42
        assert screen.status == "SCREENED"
        assert screen.reason is None
        assert screen.weights.index.equals(px.index)

    def test_data_invalid_screen_returns_zero_weight_frame(self) -> None:
        grid = _grid()
        px = _prices(grid)
        empty = pd.DataFrame(index=grid)
        screen = screen_growth_strategy_weights(
            "funding_contrarian_v1", 42, _schedule(grid), px, _taker(grid), empty, _MAX_POSITIONS,
        )
        assert screen.status == "DATA_INVALID"
        assert screen.weights.index.equals(px.index)
        assert list(screen.weights.columns) == list(px.columns)
