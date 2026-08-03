from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.application.research.growth import evaluation as ev


def _grid(n: int = 500, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="4h", tz="UTC")


def _schedule(grid: pd.DatetimeIndex) -> dict[pd.Timestamp, tuple[str, ...]]:
    months = pd.unique(grid.normalize() - pd.to_timedelta(grid.day - 1, unit="D"))
    return dict.fromkeys(
        (pd.Timestamp(m) for m in months), ("AAAUSDT", "BBBUSDT"),
    )


class TestSplitDevSchedule:
    def test_splits_chronologically_at_midpoint(self) -> None:
        dates = [
            pd.Timestamp(f"2024-{m:02d}-01", tz="UTC") for m in range(1, 13)
        ]
        schedule = dict.fromkeys(dates, ("AAAUSDT",))
        discovery, qualification = ev._split_dev_schedule(schedule)
        assert list(discovery) == dates[:6]
        assert list(qualification) == dates[6:]

    def test_odd_count_assigns_extra_month_to_qualification(self) -> None:
        dates = [
            pd.Timestamp(f"2024-{m:02d}-01", tz="UTC") for m in range(1, 9)
        ]
        schedule = dict.fromkeys(dates, ("AAAUSDT",))
        discovery, qualification = ev._split_dev_schedule(schedule)
        assert len(discovery) == 4
        assert len(qualification) == 4


class TestBuildForwardFunding:
    def test_aligns_present_columns_and_zero_fills_target(self) -> None:
        grid = _grid()
        raw = pd.DataFrame(
            {"AAAUSDT": [1e-4] * 3},
            index=grid[:3],
        )
        out = ev._build_forward_funding(raw, grid, ["AAAUSDT", "BBBUSDT"])
        assert out is not None
        assert list(out.columns) == ["AAAUSDT", "BBBUSDT"]
        assert out.index.equals(grid)
        # The event at the first bar is pre-decision and is never realized; the
        # events at bars 1 and 2 fall inside (bar[0], bar[1]] and (bar[1], bar[2]].
        assert out["AAAUSDT"].iloc[0] == 1e-4
        assert out["AAAUSDT"].iloc[1] == 1e-4
        assert out["AAAUSDT"].iloc[2] == 0.0
        assert np.allclose(out["BBBUSDT"].to_numpy(), 0.0)

    def test_returns_none_when_no_target_column_has_funding(self) -> None:
        grid = _grid()
        raw = pd.DataFrame({"ZZZUSDT": [1e-4]}, index=grid[:1])
        assert ev._build_forward_funding(raw, grid, ["AAAUSDT"]) is None


class TestBuildSettledFunding:
    def test_returns_empty_frame_when_funding_dir_missing(
        self, tmp_path: Path,
    ) -> None:
        grid = _grid()
        missing_dir = tmp_path / "does-not-exist"

        def _path(symbol: str) -> Path:
            return missing_dir / f"{symbol}.parquet"

        with patch.object(ev, "funding_path", _path):
            frame = ev._build_settled_funding(["AAAUSDT"], grid)
        assert list(frame.columns) == []

    def test_skips_symbols_whose_funding_cannot_be_loaded(
        self, tmp_path: Path,
    ) -> None:
        grid = _grid()

        def _path(symbol: str) -> Path:
            if symbol == "":
                return tmp_path
            raise ev.DataIntegrityError("no funding file")

        with patch.object(ev, "funding_path", _path):
            frame = ev._build_settled_funding(["AAAUSDT", "BBBUSDT"], grid)
        assert frame.empty

    def test_builds_frame_from_loaded_series(self, tmp_path: Path) -> None:
        grid = _grid()
        rates = pd.Series([1e-4, 2e-4, 3e-4], index=grid[:3])

        def _path(symbol: str) -> Path:
            return tmp_path / f"{symbol}.parquet"

        def _loader(path: Path) -> pd.Series:
            return rates if "AAAUSDT" in str(path) else rates * 2.0

        with (
            patch.object(ev, "funding_path", _path),
            patch.object(ev, "load_funding_rates", _loader),
        ):
            frame = ev._build_settled_funding(["AAAUSDT", "BBBUSDT"], grid)
        assert list(frame.columns) == ["AAAUSDT", "BBBUSDT"]
        assert len(frame) == 3
        assert np.allclose(frame["AAAUSDT"].to_numpy(), [1e-4, 2e-4, 3e-4])


class TestQualificationFoldGatePass:
    def test_fails_closed_on_span_shorter_than_one_fold(self) -> None:
        grid = _grid(n=500)  # ~83 days, well under the 6MS fold duration
        net = pd.Series(0.001, index=grid)
        assert ev._qualification_fold_gate_pass(net) is False

    def test_fails_closed_on_fewer_than_two_observations(self) -> None:
        grid = _grid(n=1)
        net = pd.Series([0.01], index=grid)
        assert ev._qualification_fold_gate_pass(net) is False

    def test_evenly_distributed_returns_pass(self) -> None:
        grid = _grid(n=2200, start="2023-01-01")  # ~1 year, 4h bars -> 2 folds
        net = pd.Series(0.0003, index=grid)
        assert ev._qualification_fold_gate_pass(net) is True

    def test_concentrated_returns_fail(self) -> None:
        grid = _grid(n=2200, start="2023-01-01")
        rets = np.zeros(len(grid))
        # all the gain lands in the first ~10% of bars; the rest is flat/dead.
        rets[:200] = 0.01
        net = pd.Series(rets, index=grid)
        assert ev._qualification_fold_gate_pass(net) is False


class TestFamilySelection:
    def _family(self, strategy_id: str, score: float, passed: bool = False) -> ev._FamilyScreen:
        return ev._FamilyScreen(
            strategy_id=strategy_id,
            chosen_parameter=42,
            chosen_score=score,
            passed=passed,
            parameter_scores={42.0: score},
        )

    def test_tiebreak_prefers_higher_score_then_smaller_id(self) -> None:
        families = [self._family("zebra_v1", 5.0), self._family("alpha_v1", 9.0), self._family("mid_v1", 9.0)]
        best = max(families, key=ev._family_tiebreak)
        assert best.strategy_id == "alpha_v1"

    def test_diagnostic_falsification_fails_closed_on_plateau(self) -> None:
        family = self._family("taker_imbalance_v1", 0.5)
        verdict = ev._diagnostic_falsification(family)
        assert verdict is not None
        assert verdict.passed is False
        assert verdict.binding_constraint == "plateau"

    def test_diagnostic_falsification_none_for_unscreened_family(self) -> None:
        family = ev._FamilyScreen(
            strategy_id="x", chosen_parameter=None, chosen_score=None,
            passed=False, parameter_scores={},
        )
        assert ev._diagnostic_falsification(family) is None


class TestScorecardHelpers:
    def test_empty_scorecard_carries_reason_and_family_size(self) -> None:
        card = ev._empty_scorecard("insufficient_data")
        assert card.family_size == ev.FAMILY_SIZE
        assert card.reason == "insufficient_data"
        assert card.entries == ()
        assert card.selected_strategy_id is None

    def test_scorecard_records_selected_family(self) -> None:
        entries: tuple[ev.GrowthCandidateScoreEntry, ...] = ()
        family = ev._FamilyScreen(
            strategy_id="donchian_channel_position_v1", chosen_parameter=84,
            chosen_score=1.5, passed=True, parameter_scores={84.0: 1.5},
        )
        card = ev._scorecard(entries, selected=family, reason=None)
        assert card.selected_strategy_id == "donchian_channel_position_v1"
        assert card.selected_parameter == 84
        assert card.reason is None


def _legacy_oos_t_stat(net: pd.Series) -> float:
    rets = net.dropna()
    if len(rets) < 10:
        return 0.0
    test = rets.iloc[len(rets) // 2 :]
    if len(test) < 2:
        return 0.0
    std = float(test.std())
    if std <= 0:
        return 0.0
    return float(test.mean() / std * np.sqrt(len(test)))


class TestOosTStat:
    def test_oos_t_stat_default_test_fraction_matches_legacy_half_split(self) -> None:
        rng = np.random.default_rng(0)
        net = pd.Series(rng.normal(0.0, 0.1, 20))
        expected = _legacy_oos_t_stat(net)
        assert ev._oos_t_stat(net) == pytest.approx(expected)
        assert ev._oos_t_stat(net, test_fraction=0.5) == pytest.approx(expected)

    def test_oos_t_stat_test_fraction_one_uses_full_series(self) -> None:
        rng = np.random.default_rng(1)
        net = pd.Series(rng.normal(0.0, 0.1, 20))
        rets = net.dropna()
        expected = float(rets.mean() / rets.std() * np.sqrt(len(rets)))
        assert ev._oos_t_stat(net, test_fraction=1.0) == pytest.approx(expected)

    def test_oos_t_stat_edge_cases_never_raise(self) -> None:
        assert ev._oos_t_stat(pd.Series([1.0] * 5)) == 0.0
        assert ev._oos_t_stat(pd.Series([np.nan] * 20)) == 0.0


class TestFamilyWindowCorrelation:
    def test_family_window_correlation_pairwise_values(self) -> None:
        rng = np.random.default_rng(0)
        index = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
        common = rng.normal(0.0, 1.0, 100)
        a = pd.Series(common + 0.1 * rng.normal(0.0, 1.0, 100), index=index)
        b = pd.Series(common + 0.1 * rng.normal(0.0, 1.0, 100), index=index)
        c = pd.Series(-common + 0.1 * rng.normal(0.0, 1.0, 100), index=index)
        result = ev.family_window_correlation({42: a, 84: b, 168: c})
        assert set(result) == {(42, 84), (42, 168), (84, 168)}
        assert all(-1.0 <= value <= 1.0 for value in result.values())
        assert result[(42, 84)] > 0.5
        assert result[(42, 168)] < -0.5

    def test_family_window_correlation_empty_for_single_window(self) -> None:
        net = pd.Series([1.0] * 20)
        assert ev.family_window_correlation({42: net}) == {}

    def test_family_window_correlation_skips_short_overlap(self) -> None:
        index = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
        rng = np.random.default_rng(2)
        a = pd.Series(rng.normal(0.0, 1.0, 100), index=index)
        b = pd.Series(rng.normal(0.0, 1.0, 5), index=index[:5])
        assert ev.family_window_correlation({42: a, 84: b}) == {}

    def test_family_window_correlation_inner_joins_non_null_index(self) -> None:
        index = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
        rng = np.random.default_rng(3)
        a = pd.Series(rng.normal(0.0, 1.0, 100), index=index)
        b = pd.Series(rng.normal(0.0, 1.0, 100), index=index)
        b.loc[index[20:30]] = np.nan
        result = ev.family_window_correlation({42: a, 84: b})
        assert (42, 84) in result
        assert -1.0 <= result[(42, 84)] <= 1.0


class TestScreenDiscoveryCandidates:
    def test_data_invalid_funding_rows_and_plateau_flag(self) -> None:
        grid = _grid()
        rng = np.random.default_rng(0)
        px = pd.DataFrame({
            "AAAUSDT": 100.0 + np.cumsum(rng.normal(0, 0.05, len(grid))),
            "BBBUSDT": 100.0 + np.cumsum(rng.normal(0, 0.05, len(grid))),
        }, index=grid)
        fwd = px.pct_change().fillna(0.0)
        taker = pd.DataFrame(0.5, index=grid, columns=px.columns)
        empty = pd.DataFrame(index=grid)
        schedule = _schedule(grid)
        bars = grid
        entries, families = ev._screen_discovery_candidates(
            schedule, px, fwd, taker, empty, bars,
        )
        by_id = {entry.strategy_id: entry for entry in entries}
        assert by_id["funding_contrarian_v1"].status == "DATA_INVALID"
        assert by_id["funding_contrarian_v1"].dev_discovery_score is None
        assert by_id["taker_imbalance_v1"].status == "SCREENED"
        family_map = {f.strategy_id: f for f in families}
        assert family_map["funding_contrarian_v1"].chosen_parameter is None
        assert family_map["funding_contrarian_v1"].passed is False
