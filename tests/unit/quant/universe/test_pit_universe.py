from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest

from src.quant.universe.pit_universe import (
    PitUniverseSpec,
    SymbolCoverage,
    build_universe_schedule,
    derive_backfill_candidates,
    earliest_admissible_start,
    symbol_partition,
)


def _coverage(
    symbols: list[str],
    first: pd.Timestamp | None = None,
    last: pd.Timestamp | None = None,
    bar_coverage: float = 1.0,
) -> list[SymbolCoverage]:
    if first is None:
        first = pd.Timestamp("2020-01-01", tz="UTC")
    if last is None:
        last = pd.Timestamp("2026-01-01", tz="UTC")
    return [
        SymbolCoverage(sym, first, last, bar_coverage)
        for sym in symbols
    ]


def _constant_liquidity(symbols: list[str], idx: pd.DatetimeIndex, value: float) -> dict[str, pd.Series]:
    return {sym: pd.Series(value, index=idx) for sym in symbols}


class TestPitUniverseSpec:
    # GEV2-03-START-DERIVED: defaults and validation are contract-locked.
    def test_defaults(self) -> None:
        spec = PitUniverseSpec()
        assert spec.universe_size == 20
        assert spec.max_positions == 5
        assert spec.seasoning_days == 365
        assert spec.liquidity_lookback_days == 30
        assert spec.min_bar_coverage == 0.99
        assert spec.dev_fraction == 0.80

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"universe_size": 2, "max_positions": 5},
            {"seasoning_days": 0},
            {"liquidity_lookback_days": 0},
            {"min_bar_coverage": 0.0},
            {"min_bar_coverage": 1.5},
            {"dev_fraction": 0.0},
            {"dev_fraction": 1.0},
        ],
    )
    def test_validation_raises(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="must"):
            PitUniverseSpec(**kwargs)


class TestSymbolCoverage:
    def test_valid(self) -> None:
        c = SymbolCoverage(
            "BTCUSDT",
            pd.Timestamp("2020-01-01", tz="UTC"),
            pd.Timestamp("2026-01-01", tz="UTC"),
            1.0,
        )
        assert c.symbol == "BTCUSDT"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"symbol": ""},
            {"bar_coverage": -0.1},
            {"bar_coverage": 1.1},
        ],
    )
    def test_validation_raises(self, kwargs: dict[str, object]) -> None:
        base = {
            "symbol": "A",
            "first_bar": pd.Timestamp("2020-01-01", tz="UTC"),
            "last_bar": pd.Timestamp("2026-01-01", tz="UTC"),
            "bar_coverage": 1.0,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match="must"):
            SymbolCoverage(**base)

    def test_rejects_tz_naive_timestamps(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            SymbolCoverage(
                "X",
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2026-01-01"),
                1.0,
            )

    def test_rejects_last_before_first(self) -> None:
        with pytest.raises(ValueError, match="not precede"):
            SymbolCoverage(
                "X",
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2020-01-01", tz="UTC"),
                1.0,
            )


class TestBuildUniverseSchedule:
    # GEV2-01-PIT-CAUSALITY
    def test_future_bars_never_enter_ranking(self) -> None:
        idx = pd.date_range("2020-12-01", periods=400, freq="4h", tz="UTC")
        rebalance = idx[-1]
        cov = [
            SymbolCoverage("AAA", idx[0], idx[-1], 1.0),
            SymbolCoverage("BBB", idx[0], idx[-1], 1.0),
        ]
        spec = PitUniverseSpec(universe_size=2, max_positions=2, seasoning_days=1)
        baseline = build_universe_schedule(
            cov, _constant_liquidity(["AAA", "BBB"], idx, 1.0), [rebalance], spec,
        )
        assert baseline[rebalance] == ("AAA", "BBB")

        injected = dict(_constant_liquidity(["AAA", "BBB"], idx, 1.0))
        injected["AAA"] = injected["AAA"].copy()
        injected["AAA"].iloc[-1] = 1e12
        causal = build_universe_schedule(cov, injected, [rebalance], spec)
        assert causal[rebalance] == baseline[rebalance]

    # GEV2-02-SURVIVORSHIP
    def test_delisted_symbol_keeps_contributing_history(self) -> None:
        idx = pd.date_range("2020-12-01", periods=400, freq="4h", tz="UTC")
        early = idx[100]
        late = idx[300]
        cov = [
            SymbolCoverage("AAA", idx[0], idx[-1], 1.0),
            SymbolCoverage("ZZZ", idx[0], early, 1.0),
        ]
        spec = PitUniverseSpec(universe_size=2, max_positions=2, seasoning_days=1)
        liq = _constant_liquidity(["AAA", "ZZZ"], idx, 1.0)
        schedule = build_universe_schedule(cov, liq, [early, late], spec)
        assert "ZZZ" in schedule[early]
        assert "ZZZ" not in schedule[late]

    # GEV2-03-START-DERIVED
    def test_eligibility_requires_seasoning(self) -> None:
        idx = pd.date_range("2020-12-01", periods=400, freq="4h", tz="UTC")
        date = idx[200]
        cov = _coverage(["AAA", "BBB"], first=idx[0], last=idx[-1])
        spec = PitUniverseSpec(universe_size=2, max_positions=2, seasoning_days=365)
        schedule = build_universe_schedule(cov, _constant_liquidity(["AAA", "BBB"], idx, 1.0), [date], spec)
        assert schedule[date] == ()

    def test_negative_or_nan_quote_volume_excludes_symbol(self) -> None:
        idx = pd.date_range("2020-12-01", periods=400, freq="4h", tz="UTC")
        date = idx[-1]
        cov = _coverage(["AAA", "BBB"])
        liq = _constant_liquidity(["AAA", "BBB"], idx, 1.0)
        liq["BBB"] = pd.Series(np.nan, index=idx)
        spec = PitUniverseSpec(universe_size=2, max_positions=2, seasoning_days=1)
        schedule = build_universe_schedule(cov, liq, [date], spec)
        assert "BBB" not in schedule[date]

    def test_ranks_by_descending_liquidity_with_lexicographic_tie_break(self) -> None:
        idx = pd.date_range("2020-12-01", periods=400, freq="4h", tz="UTC")
        date = idx[-1]
        cov = _coverage(["AAA", "BBB", "CCC"])
        liq = _constant_liquidity(["AAA", "BBB", "CCC"], idx, 1.0)
        liq["BBB"] = pd.Series(9.0, index=idx)
        spec = PitUniverseSpec(universe_size=3, max_positions=3, seasoning_days=1)
        schedule = build_universe_schedule(cov, liq, [date], spec)
        assert schedule[date] == ("BBB", "AAA", "CCC")


class TestEarliestAdmissibleStart:
    # GEV2-03-START-DERIVED
    def test_first_sustainable_date_is_returned(self) -> None:
        cov = _coverage(["S0", "S1", "S2"])
        dates = [
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-02-01", tz="UTC"),
        ]
        start = earliest_admissible_start(
            cov, dates, PitUniverseSpec(universe_size=3, max_positions=3),
        )
        assert start == pd.Timestamp("2021-01-01", tz="UTC")

    def test_none_when_pool_never_reaches_universe_size(self) -> None:
        cov = _coverage(["S0", "S1", "S2"])
        dates = [
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-02-01", tz="UTC"),
        ]
        assert earliest_admissible_start(
            cov, dates, PitUniverseSpec(universe_size=20, max_positions=5),
        ) is None

    def test_returns_later_date_when_early_pool_is_not_sustained(self) -> None:
        # S0 is seasoned from the start; S1/S2 only become eligible after 365d seasoning.
        cov = [
            SymbolCoverage("S0", pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"), 1.0),
            SymbolCoverage("S1", pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"), 1.0),
            SymbolCoverage("S2", pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"), 1.0),
        ]
        dates = [
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-01-01", tz="UTC"),
        ]
        spec = PitUniverseSpec(universe_size=3, max_positions=3)
        assert earliest_admissible_start(cov, dates, spec) == pd.Timestamp("2022-01-01", tz="UTC")

    def test_rejects_tz_naive_rebalance_dates(self) -> None:
        cov = _coverage(["S0"])
        with pytest.raises(ValueError, match="tz-aware"):
            earliest_admissible_start(cov, [pd.Timestamp("2021-01-01")], PitUniverseSpec())

    def test_rejects_non_monotonic_rebalance_dates(self) -> None:
        cov = _coverage(["S0"])
        dates = [
            pd.Timestamp("2021-02-01", tz="UTC"),
            pd.Timestamp("2021-01-01", tz="UTC"),
        ]
        with pytest.raises(ValueError, match="monotonic"):
            earliest_admissible_start(cov, dates, PitUniverseSpec())


class TestDeriveBackfillCandidates:
    # GEV2-02B-MINIMAL-BACKFILL-FILTER
    def test_union_of_all_schedule_rosters(self) -> None:
        idx = pd.date_range("2020-12-01", periods=400, freq="4h", tz="UTC")
        cov = _coverage(["AAA", "BBB", "CCC"])
        liq = {
            "AAA": pd.Series(1.0, index=idx),
            "BBB": pd.Series(9.0, index=idx),
            "CCC": pd.Series(0.5, index=idx),
        }
        spec = PitUniverseSpec(universe_size=2, max_positions=2, seasoning_days=1)
        dates = [idx[-1]]
        result = derive_backfill_candidates(cov, liq, dates, spec)
        assert result == ("AAA", "BBB")
        assert set(result) == set().union(*build_universe_schedule(cov, liq, dates, spec).values())

    def test_symbol_never_selected_is_absent(self) -> None:
        idx = pd.date_range("2020-12-01", periods=400, freq="4h", tz="UTC")
        cov = _coverage(["AAA", "BBB", "CCC"])
        liq = {
            "AAA": pd.Series(1.0, index=idx),
            "BBB": pd.Series(9.0, index=idx),
            "CCC": pd.Series(0.5, index=idx),
        }
        spec = PitUniverseSpec(universe_size=2, max_positions=2, seasoning_days=1)
        result = derive_backfill_candidates(cov, liq, [idx[-1]], spec)
        assert "CCC" not in result

    def test_set_union_semantics_across_dates(self) -> None:
        idx = pd.date_range("2020-12-01", periods=400, freq="4h", tz="UTC")
        cov = _coverage(["AAA", "BBB", "CCC"])
        liq = {
            "AAA": pd.Series(1.0, index=idx),
            "BBB": pd.Series(9.0, index=idx),
            "CCC": pd.Series(0.5, index=idx),
        }
        spec = PitUniverseSpec(universe_size=2, max_positions=2, seasoning_days=1)
        dates = [idx[100], idx[200], idx[-1]]
        schedule = build_universe_schedule(cov, liq, dates, spec)
        expected = tuple(sorted({sym for roster in schedule.values() for sym in roster}))
        assert derive_backfill_candidates(cov, liq, dates, spec) == expected
        assert derive_backfill_candidates(cov, liq, dates, spec) == (
            derive_backfill_candidates(cov, liq, dates, spec)
        )


class TestSymbolPartition:
    # GEV2-04-HOLDOUT-STABLE
    def test_pre_registered_examples(self) -> None:
        assert symbol_partition("BTCUSDT") == "dev"
        assert symbol_partition("ETHUSDT") == "holdout"
        assert symbol_partition("SOLUSDT") == "dev"

    def test_deterministic_across_calls(self) -> None:
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
            assert symbol_partition(symbol) == symbol_partition(symbol)

    def test_hash_semantics(self) -> None:
        symbol = "AAAUSDT"
        bucket = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16) % 100
        expected = "dev" if bucket < 80 else "holdout"
        assert symbol_partition(symbol) == expected

    def test_dev_share_is_close_to_dev_fraction(self) -> None:
        symbols = [f"SYM{i:03d}USDT" for i in range(200)]
        dev_share = sum(symbol_partition(s) == "dev" for s in symbols) / len(symbols)
        assert 0.70 <= dev_share <= 0.90

    def test_rejects_empty_symbol(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            symbol_partition("")


class TestNoLiteralStartDate:
    # GEV2-03-START-DERIVED: the module must derive its start, never hardcode one.
    def test_module_contains_no_literal_calendar_date(self) -> None:
        import re

        import src.quant.universe.pit_universe as pit_universe

        source = inspect.getsource(pit_universe)
        assert re.search(r"\b\d{4}-\d{2}-\d{2}\b", source) is None
